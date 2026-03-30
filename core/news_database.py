"""
Financial news corpus — SEC EDGAR 8-K filings + Kaggle datasets.

Downloads and manages historical financial news headlines for sentiment
analysis. Two data sources are supported:

1. SEC EDGAR 8-K filings (primary, free, 10+ years, PIT-perfect):
   Every material event is filed as an 8-K with the SEC. The filing
   date is the exact point-in-time timestamp.

2. Kaggle datasets (supplementary, manual download):
   Pre-collected financial news headlines. Must be downloaded manually
   and placed in data/news/ as CSV files.

ARCHITECTURE:
    data/news/
        headlines_raw.parquet    ← Unified corpus (all sources merged)
        kaggle/                  ← Manually placed Kaggle CSVs
        edgar_cache/             ← Raw EDGAR 8-K data (cached)

Usage:
    # Download EDGAR 8-K filings
    python -m core.news_database --source edgar

    # Import a Kaggle CSV (must have 'date', 'ticker', 'headline' columns)
    python -m core.news_database --source kaggle --file data/news/kaggle/news.csv

    # Merge all available sources
    python -m core.news_database --source all
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

from core.config import PATHS, PROJECT_ROOT, setup_logging

logger = logging.getLogger(__name__)

NEWS_DIR = PATHS["news_dir"]
RAW_PATH = PATHS["news_raw_file"]
EDGAR_CACHE_DIR = NEWS_DIR / "edgar_cache"
KAGGLE_DIR = NEWS_DIR / "kaggle"

# SEC EDGAR requires a User-Agent header with a contact email
EDGAR_USER_AGENT = "DeepValueAI research@deepvalueai.dev"
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# 8-K item codes → human-readable descriptions (for FinBERT scoring)
ITEM_DESCRIPTIONS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Transfer",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def _load_dotenv() -> None:
    """Load variables from .env file into os.environ (no extra dependency)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# EDGAR 8-K Download
# ---------------------------------------------------------------------------

def _extract_ticker(display_names: list[str]) -> str | None:
    """Extract ticker symbol from EDGAR display_names field.

    The field looks like: 'APPLE INC  (AAPL)  (CIK 0000320193)'
    """
    for name in display_names:
        match = re.search(r"\(([A-Z]{1,5})\)", name)
        if match:
            return match.group(1)
    return None


def _items_to_headline(company: str, items: list[str]) -> str:
    """Convert 8-K item codes to a human-readable headline for FinBERT."""
    descriptions = []
    for item in items:
        desc = ITEM_DESCRIPTIONS.get(item)
        if desc:
            descriptions.append(desc)
    if not descriptions:
        descriptions = ["Corporate Filing"]
    return f"{company}: {'; '.join(descriptions)}"


def download_edgar_8k(
    start_year: int = 2016,
    end_year: int | None = None,
    tickers: set[str] | None = None,
    max_pages_per_quarter: int = 50,
) -> pd.DataFrame:
    """Download 8-K filing metadata from SEC EDGAR EFTS search API.

    Uses the EDGAR EFTS (full-text search) API to fetch 8-K filings.
    Each filing has an exact date and structured item descriptions that
    are converted to human-readable headlines for FinBERT scoring.

    Parameters
    ----------
    start_year : int
        First year to download.
    end_year : int or None
        Last year (inclusive). None = current year.
    tickers : set[str] or None
        If provided, only keep filings for these tickers.
    max_pages_per_quarter : int
        Maximum pagination depth per quarter (100 results/page).

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``headline``, ``source``.
    """
    import requests

    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year

    headers = {"User-Agent": EDGAR_USER_AGENT}
    all_filings: list[dict] = []

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            q_start = f"{year}-{(quarter - 1) * 3 + 1:02d}-01"
            q_end_month = quarter * 3
            q_end = f"{year}-{q_end_month:02d}-28"

            logger.info("Fetching EDGAR 8-K filings: %s Q%d...", year, quarter)
            quarter_count = 0

            for page in range(max_pages_per_quarter):
                try:
                    params = {
                        "q": '"8-K"',
                        "dateRange": "custom",
                        "startdt": q_start,
                        "enddt": q_end,
                        "forms": "8-K",
                        "from": page * 100,
                    }
                    response = requests.get(
                        EDGAR_EFTS_URL, headers=headers,
                        params=params, timeout=30,
                    )

                    if response.status_code != 200:
                        logger.warning(
                            "EDGAR returned status %d for %s Q%d page %d",
                            response.status_code, year, quarter, page,
                        )
                        break

                    data = response.json()
                    hits = data.get("hits", {}).get("hits", [])

                    if not hits:
                        break  # No more results

                    for hit in hits:
                        src = hit.get("_source", {})
                        filing_date = src.get("file_date", "")
                        display_names = src.get("display_names", [])
                        items = src.get("items", [])

                        ticker = _extract_ticker(display_names)
                        if not ticker:
                            continue
                        if tickers and ticker not in tickers:
                            continue

                        # Build company name (strip ticker/CIK from display)
                        company = (
                            display_names[0].split("(")[0].strip()
                            if display_names else "Unknown"
                        )
                        headline = _items_to_headline(company, items)

                        all_filings.append({
                            "ticker": ticker,
                            "date": filing_date,
                            "headline": headline,
                            "source": "edgar_8k",
                        })

                    quarter_count += len(hits)

                    # Respect SEC rate limits (max 10 requests/second)
                    time.sleep(0.15)

                    # If we got fewer than 100 results, no more pages
                    if len(hits) < 100:
                        break

                except Exception as exc:
                    logger.warning(
                        "Failed EDGAR fetch for %s Q%d page %d: %s",
                        year, quarter, page, exc,
                    )
                    break

            logger.info("  %s Q%d: %d filings", year, quarter, quarter_count)

    if not all_filings:
        logger.warning("No EDGAR 8-K filings could be fetched.")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df = pd.DataFrame(all_filings)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["ticker", "date", "headline"])

    logger.info(
        "EDGAR 8-K download complete: %d filings, %d tickers, %s → %s",
        len(df), df["ticker"].nunique(),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    return df


# ---------------------------------------------------------------------------
# Kaggle dataset import
# ---------------------------------------------------------------------------

def load_kaggle_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a Kaggle financial news CSV and normalize columns.

    Expected columns (flexible matching):
    - date/Date/published/timestamp → ``date``
    - ticker/stock/symbol/Ticker → ``ticker``
    - headline/title/text/news → ``headline``

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``headline``, ``source``.
    """
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("Loaded Kaggle CSV: %d rows from %s", len(df), csv_path)

    # Flexible column matching
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ("date", "published", "timestamp", "publishedat"):
            col_map[col] = "date"
        elif col_lower in ("ticker", "stock", "symbol", "stock_ticker"):
            col_map[col] = "ticker"
        elif col_lower in ("headline", "title", "text", "news", "description"):
            col_map[col] = "headline"

    if not all(v in col_map.values() for v in ("date", "ticker", "headline")):
        raise ValueError(
            f"Could not find required columns (date, ticker, headline) "
            f"in CSV columns: {list(df.columns)}. "
            f"Mapped: {col_map}"
        )

    df = df.rename(columns=col_map)
    df = df[["ticker", "date", "headline"]].copy()
    df["source"] = "kaggle"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker", "headline"])
    df = df.drop_duplicates(subset=["ticker", "date", "headline"])

    logger.info(
        "Kaggle import: %d headlines, %d tickers, %s → %s",
        len(df), df["ticker"].nunique(),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_headlines(df: pd.DataFrame, path: Path = RAW_PATH) -> None:
    """Save the headline corpus to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1_048_576
    logger.info("Saved headlines → %s (%.1f MB, %d rows)", path, mb, len(df))


def load_headlines(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the headline corpus from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Headlines not found: {path}\n"
            "Run first: python -m core.news_database --source edgar"
        )
    df = pd.read_parquet(path)
    logger.info(
        "Loaded headlines: %d rows, %d tickers",
        len(df), df["ticker"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build financial news headline corpus",
    )
    parser.add_argument(
        "--source",
        choices=["edgar", "kaggle", "all"],
        default="edgar",
        help="Data source: edgar (SEC 8-K), kaggle (CSV import), or all.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to Kaggle CSV file (required for --source kaggle).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="First year to fetch EDGAR data (default: 2016).",
    )
    args = parser.parse_args()

    chunks: list[pd.DataFrame] = []

    if args.source in ("edgar", "all"):
        edgar_df = download_edgar_8k(start_year=args.start_year)
        if not edgar_df.empty:
            chunks.append(edgar_df)

    if args.source in ("kaggle", "all"):
        if args.source == "kaggle" and not args.file:
            raise ValueError("--file is required for --source kaggle")

        if args.file:
            kaggle_df = load_kaggle_csv(args.file)
            chunks.append(kaggle_df)
        elif args.source == "all":
            # Auto-import any CSVs in the kaggle directory
            KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
            for csv_file in KAGGLE_DIR.glob("*.csv"):
                try:
                    kaggle_df = load_kaggle_csv(csv_file)
                    chunks.append(kaggle_df)
                except Exception as exc:
                    logger.warning("Failed to import %s: %s", csv_file, exc)

    if not chunks:
        logger.warning("No headlines were collected from any source.")
        return

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date", "headline"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    save_headlines(combined)

    logger.info(
        "News database built: %d headlines, %d tickers, %s → %s",
        len(combined), combined["ticker"].nunique(),
        combined["date"].min().strftime("%Y-%m-%d"),
        combined["date"].max().strftime("%Y-%m-%d"),
    )


if __name__ == "__main__":
    _load_dotenv()
    setup_logging()
    main()
