"""
Historical fundamental database — SimFin + Yahoo Finance.

Downloads quarterly financial statements (income, balance sheet, cash flow)
for all US-listed companies, computes fundamental ratios, and stores them
as a local Parquet file.

TWO DATA SOURCES (pick one):
    1. Yahoo Finance (default): no API key, S&P 500, last ~5 quarters.
    2. SimFin (optional):       free API key, ~3000 US stocks, 10+ years.

POINT-IN-TIME SAFE:
    Each row uses the Publish Date (when the report was actually filed
    with the SEC), so ``merge_asof`` with daily prices avoids look-ahead
    bias.  A report for Q4 2022 filed on 2023-02-15 is only "available"
    from 2023-02-15 onward in any simulation.

FEATURES STORED per (ticker, quarter):
    Pure ratios:      op_margin, revenue_growth, debt_equity,
                      current_ratio, cash_covers_debt
    TTM aggregates:   net_income_ttm, fcf_ttm, eps_ttm, earnings_growth
    (pe_ratio, peg_ratio, fcf_yield need stock prices →
     computed at merge time by ``merge_fundamentals_pit``)

ARCHITECTURE:
    data/fundamentals/
        simfin_cache/                  ← raw CSVs from SimFin (cached)
        fundamentals_features.parquet  ← clean, model-ready dataset

Usage:
    # Default: Yahoo Finance (no key needed, S&P 500)
    python -m core.fundamental_database

    # SimFin (more history, more tickers)
    python -m core.fundamental_database --source simfin --api-key YOUR_KEY

    # Force re-download (e.g. after a new quarter)
    python -m core.fundamental_database --refresh
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import PROJECT_ROOT, setup_logging

logger = logging.getLogger(__name__)


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
# Paths
# ---------------------------------------------------------------------------

FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "fundamentals"
SIMFIN_CACHE_DIR = FUNDAMENTALS_DIR / "simfin_cache"
DATASET_PATH = FUNDAMENTALS_DIR / "fundamentals_features.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den → NaN where den is 0, NaN, or result is ±inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den
    return result.replace([np.inf, -np.inf], np.nan)


def _reset_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex to regular columns."""
    if isinstance(df.index, pd.MultiIndex):
        return df.reset_index()
    return df.copy()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_simfin(api_key: str = "free") -> dict[str, pd.DataFrame]:
    """Download quarterly US financial statements from SimFin.

    Files are cached locally after the first download (~2-5 min).
    Subsequent calls load instantly from disk.

    Parameters
    ----------
    api_key : str
        SimFin API key.  Register for free at https://app.simfin.com/
        Use ``"free"`` for the limited public tier (no registration).

    Returns
    -------
    dict with keys: ``income``, ``balance``, ``cashflow``
    """
    try:
        import simfin as sf
    except ImportError as err:
        raise ImportError(
            "simfin is not installed.\n"
            "  pip install simfin\n"
            "Then get a free API key at https://app.simfin.com/"
        ) from err

    SIMFIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(api_key)
    sf.set_data_dir(str(SIMFIN_CACHE_DIR))

    logger.info("Downloading SimFin bulk data (US market)...")
    logger.info("  Cache directory: %s", SIMFIN_CACHE_DIR)

    datasets: dict[str, pd.DataFrame] = {}

    for name, loader in [
        ("income", lambda: sf.load_income(variant="quarterly", market="us")),
        ("balance", lambda: sf.load_balance(variant="quarterly", market="us")),
        ("cashflow", lambda: sf.load_cashflow(variant="quarterly", market="us")),
    ]:
        logger.info("  → %s ...", name)
        datasets[name] = loader()
        df = datasets[name]
        idx = df.index if isinstance(df.index, pd.MultiIndex) else None
        n_tickers = idx.get_level_values(0).nunique() if idx is not None else "?"
        logger.info("    %d rows, %s tickers", len(df), n_tickers)

    return datasets


# ---------------------------------------------------------------------------
# Download — Yahoo Finance (no API key needed)
# ---------------------------------------------------------------------------

# Mapping from yfinance line-item names → SimFin-compatible column names.
_YF_RENAME = {
    # Income statement
    "Total Revenue": "Revenue",
    "Operating Income": "Operating Income (Loss)",
    "Diluted Average Shares": "Shares (Diluted)",
    # Balance sheet
    "Current Assets": "Total Current Assets",
    "Current Liabilities": "Total Current Liabilities",
    "Stockholders Equity": "Total Equity",
    "Current Debt": "Short Term Debt",
    "Cash Cash Equivalents And Short Term Investments":
        "Cash, Cash Equivalents & Short Term Investments",
    # Cash flow
    "Operating Cash Flow": "Net Cash from Operating Activities",
    "Capital Expenditure": "Change in Fixed Assets & Intangibles",
}


def download_yfinance(
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download quarterly financial statements from Yahoo Finance.

    No API key required.  Downloads one ticker at a time
    (~5-10 min for the full S&P 500).

    Limitations vs SimFin:
      - Only the last ~5 quarters of history (not 10+ years).
      - No ``Publish Date`` — estimated as report_date + 45 days.

    Parameters
    ----------
    tickers : list[str] or None
        Tickers to download.  If None, downloads the full S&P 500.

    Returns
    -------
    dict with keys: ``income``, ``balance``, ``cashflow``
        Same format as ``download_simfin`` — compatible with
        ``build_dataset()``.
    """
    import time

    import yfinance as yf

    from core.config import API_SLEEP_SECONDS

    if tickers is None:
        from core.data_service import get_sp500_tickers
        tickers = get_sp500_tickers()

    logger.info(
        "Downloading fundamentals from Yahoo Finance (%d tickers)...",
        len(tickers),
    )

    inc_parts: list[pd.DataFrame] = []
    bal_parts: list[pd.DataFrame] = []
    cf_parts: list[pd.DataFrame] = []

    failed = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0 or i == 1:
            logger.info("  Progress: %d / %d", i, len(tickers))
        try:
            t = yf.Ticker(ticker)

            for attr, parts in [
                ("quarterly_income_stmt", inc_parts),
                ("quarterly_balance_sheet", bal_parts),
                ("quarterly_cashflow", cf_parts),
            ]:
                raw = getattr(t, attr, None)
                if raw is not None and not raw.empty:
                    # yfinance: rows = line items, cols = dates → transpose
                    df = raw.T.copy()
                    df["Ticker"] = ticker
                    df["Report Date"] = df.index
                    parts.append(df)

            time.sleep(API_SLEEP_SECONDS)
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("  %s failed: %s", ticker, e)

    if failed > 5:
        logger.warning("  ... and %d more tickers failed", failed - 5)

    # Concatenate all tickers
    income = pd.concat(inc_parts, ignore_index=True) if inc_parts else pd.DataFrame()
    balance = pd.concat(bal_parts, ignore_index=True) if bal_parts else pd.DataFrame()
    cashflow = pd.concat(cf_parts, ignore_index=True) if cf_parts else pd.DataFrame()

    logger.info(
        "Downloaded: income=%d rows, balance=%d rows, cashflow=%d rows",
        len(income), len(balance), len(cashflow),
    )

    if income.empty:
        raise RuntimeError("No data downloaded — check your internet connection")

    # Rename yfinance columns to SimFin-compatible names
    for df in [income, balance, cashflow]:
        applicable = {k: v for k, v in _YF_RENAME.items() if k in df.columns}
        df.rename(columns=applicable, inplace=True)

    # Add Fiscal Year / Period (estimated from Report Date)
    for df in [income, balance, cashflow]:
        if "Report Date" in df.columns and not df.empty:
            dt = pd.to_datetime(df["Report Date"])
            df.loc[:, "Fiscal Year"] = dt.dt.year.values
            df.loc[:, "Fiscal Period"] = ("Q" + ((dt.dt.month - 1) // 3 + 1).astype(str)).values

    return {"income": income, "balance": balance, "cashflow": cashflow}


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge statements and compute fundamental features.

    Returns one row per (ticker, fiscal quarter) with raw financials,
    computed ratios, and TTM aggregates.
    """
    income = _reset_multiindex(raw["income"])
    balance = _reset_multiindex(raw["balance"])
    cashflow = _reset_multiindex(raw["cashflow"])

    logger.info(
        "Building features — raw sizes: income=%d, balance=%d, cashflow=%d",
        len(income), len(balance), len(cashflow),
    )

    # ------------------------------------------------------------------
    # Keep quarterly data only (drop annual / TTM summary rows)
    # ------------------------------------------------------------------
    for df in [income, balance, cashflow]:
        if "Fiscal Period" in df.columns:
            df.drop(
                df[~df["Fiscal Period"].isin(["Q1", "Q2", "Q3", "Q4"])].index,
                inplace=True,
            )

    # ------------------------------------------------------------------
    # Select relevant columns from each statement
    # ------------------------------------------------------------------
    _INC_COLS = [
        "Ticker", "Report Date", "Publish Date",
        "Fiscal Year", "Fiscal Period",
        "Revenue", "Operating Income (Loss)",
        "Net Income", "Shares (Diluted)",
    ]
    _BAL_COLS = [
        "Ticker", "Report Date", "Fiscal Year", "Fiscal Period",
        "Total Current Assets", "Total Current Liabilities",
        "Total Equity", "Short Term Debt", "Long Term Debt",
        "Cash, Cash Equivalents & Short Term Investments",
    ]
    _CF_COLS = [
        "Ticker", "Report Date", "Fiscal Year", "Fiscal Period",
        "Net Cash from Operating Activities",
        "Change in Fixed Assets & Intangibles",
    ]

    income = income[[c for c in _INC_COLS if c in income.columns]].copy()
    balance = balance[[c for c in _BAL_COLS if c in balance.columns]].copy()
    cashflow = cashflow[[c for c in _CF_COLS if c in cashflow.columns]].copy()

    # ------------------------------------------------------------------
    # Merge on (Ticker, Report Date, Fiscal Year, Fiscal Period)
    # ------------------------------------------------------------------
    merge_keys = ["Ticker", "Report Date", "Fiscal Year", "Fiscal Period"]

    def _common_keys(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
        return [k for k in merge_keys if k in a.columns and k in b.columns]

    df = income.merge(balance, on=_common_keys(income, balance), how="outer")
    df = df.merge(cashflow, on=_common_keys(df, cashflow), how="outer")

    logger.info("  Merged: %d rows, %d tickers",
                len(df), df["Ticker"].nunique())

    # ------------------------------------------------------------------
    # Rename to clean, lowercase column names
    # ------------------------------------------------------------------
    df.rename(columns={
        "Ticker": "ticker",
        "Report Date": "report_date",
        "Publish Date": "publish_date",
        "Fiscal Year": "fiscal_year",
        "Fiscal Period": "fiscal_period",
        "Revenue": "revenue",
        "Operating Income (Loss)": "operating_income",
        "Net Income": "net_income",
        "Shares (Diluted)": "shares_diluted",
        "Total Current Assets": "current_assets",
        "Total Current Liabilities": "current_liabilities",
        "Total Equity": "total_equity",
        "Short Term Debt": "st_debt",
        "Long Term Debt": "lt_debt",
        "Cash, Cash Equivalents & Short Term Investments": "cash",
        "Net Cash from Operating Activities": "operating_cashflow",
        "Change in Fixed Assets & Intangibles": "capex",
    }, inplace=True)

    # ------------------------------------------------------------------
    # Derived raw columns
    # ------------------------------------------------------------------

    # Total debt = short-term + long-term (NaN only if both are NaN)
    st = df["st_debt"] if "st_debt" in df.columns else pd.Series(np.nan, index=df.index)
    lt = df["lt_debt"] if "lt_debt" in df.columns else pd.Series(np.nan, index=df.index)
    both_nan = st.isna() & lt.isna()
    df["total_debt"] = st.fillna(0) + lt.fillna(0)
    df.loc[both_nan, "total_debt"] = np.nan

    # Free cash flow = operating CF + capex (capex is negative in SimFin)
    if "operating_cashflow" in df.columns:
        capex = df["capex"] if "capex" in df.columns else pd.Series(0, index=df.index)
        df["fcf"] = df["operating_cashflow"] + capex.fillna(0)

    # ------------------------------------------------------------------
    # Ensure dates are datetime
    # ------------------------------------------------------------------
    df["report_date"] = pd.to_datetime(df["report_date"])
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"])
    else:
        # Fallback: most companies file within ~45 days of quarter end
        df["publish_date"] = df["report_date"] + pd.Timedelta(days=45)
        logger.warning("No Publish Date in data — estimating as report_date + 45 days")

    # ------------------------------------------------------------------
    # Sort, deduplicate, reset index
    # ------------------------------------------------------------------
    df.sort_values(["ticker", "report_date"], inplace=True)
    df.drop_duplicates(
        subset=["ticker", "report_date", "fiscal_period"], keep="last", inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    # ==================================================================
    # COMPUTED RATIOS (price-independent — stored in dataset)
    # ==================================================================

    # 1. Operating margin = operating_income / revenue
    df["op_margin"] = _safe_divide(df["operating_income"], df["revenue"])

    # 2. Revenue growth — YoY (same quarter vs 4 quarters ago)
    df["revenue_growth"] = df.groupby("ticker")["revenue"].pct_change(periods=4, fill_method=None)

    # 3. Debt-to-equity = total_debt / |total_equity|
    df["debt_equity"] = _safe_divide(df["total_debt"], df["total_equity"].abs())

    # 4. Current ratio = current_assets / current_liabilities
    df["current_ratio"] = _safe_divide(df["current_assets"], df["current_liabilities"])

    # 5. Cash covers debt = cash / total_debt
    df["cash_covers_debt"] = _safe_divide(df["cash"], df["total_debt"])

    # ==================================================================
    # TTM AGGREGATES (trailing 4 quarters — rolling sum)
    # ==================================================================
    for src, dst in [
        ("net_income", "net_income_ttm"),
        ("revenue", "revenue_ttm"),
        ("fcf", "fcf_ttm"),
    ]:
        if src in df.columns:
            df[dst] = (
                df.groupby("ticker")[src]
                .transform(lambda s: s.rolling(4, min_periods=4).sum())
            )

    # EPS TTM = net_income_ttm / diluted shares
    if "net_income_ttm" in df.columns and "shares_diluted" in df.columns:
        df["eps_ttm"] = _safe_divide(df["net_income_ttm"], df["shares_diluted"])

    # Earnings growth YoY (of TTM net income — used for PEG ratio)
    if "net_income_ttm" in df.columns:
        df["earnings_growth"] = (
            df.groupby("ticker")["net_income_ttm"].pct_change(periods=4, fill_method=None)
        )

    # ==================================================================
    # Final column ordering and cleanup
    # ==================================================================
    col_order = [
        # Identity
        "ticker", "report_date", "publish_date", "fiscal_year", "fiscal_period",
        # Raw financials
        "revenue", "operating_income", "net_income", "shares_diluted",
        "current_assets", "current_liabilities", "total_equity",
        "total_debt", "st_debt", "lt_debt", "cash",
        "operating_cashflow", "capex", "fcf",
        # Computed ratios
        "op_margin", "revenue_growth", "debt_equity",
        "current_ratio", "cash_covers_debt",
        # TTM aggregates
        "net_income_ttm", "revenue_ttm", "fcf_ttm", "eps_ttm",
        "earnings_growth",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.dropna(subset=["ticker", "report_date"], inplace=True)

    # Drop rows where ALL financial data is NaN (e.g. from unmatched
    # dates across statements during the outer merge)
    financial_cols = [c for c in ["revenue", "net_income", "total_equity"] if c in df.columns]
    if financial_cols:
        df.dropna(subset=financial_cols, how="all", inplace=True)

    logger.info(
        "Dataset built: %d rows, %d tickers, %d columns, "
        "date range %s -> %s",
        len(df),
        df["ticker"].nunique(),
        len(df.columns),
        df["report_date"].min().strftime("%Y-%m-%d"),
        df["report_date"].max().strftime("%Y-%m-%d"),
    )
    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> None:
    """Save the fundamental dataset to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1_048_576
    logger.info("Saved -> %s (%.1f MB)", path, mb)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the fundamental dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fundamental dataset not found: {path}\n"
            "Run first:  python -m core.fundamental_database"
        )
    df = pd.read_parquet(path)
    logger.info(
        "Loaded fundamentals: %d rows, %d tickers",
        len(df), df["ticker"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Point-in-time merge helper
# ---------------------------------------------------------------------------

def merge_fundamentals_pit(
    prices_df: pd.DataFrame,
    fund_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge daily prices with the most recent *available* fundamentals.

    Uses ``publish_date`` (not ``report_date``) so a report filed on
    2023-02-15 is only available from that date onward — no look-ahead
    bias.

    After the merge, price-dependent ratios (pe_ratio, peg_ratio,
    fcf_yield) are computed using the daily close price.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must have columns: ``ticker``, ``date``, ``close``.
    fund_df : pd.DataFrame or None
        Fundamental dataset.  If None, loads from default path.

    Returns
    -------
    pd.DataFrame
        ``prices_df`` enriched with: op_margin, revenue_growth,
        debt_equity, current_ratio, cash_covers_debt, pe_ratio,
        peg_ratio, fcf_yield.
    """
    if fund_df is None:
        fund_df = load_dataset()

    # Columns to carry from fundamentals into the merge
    merge_cols = [
        "ticker", "publish_date",
        "op_margin", "revenue_growth", "debt_equity",
        "current_ratio", "cash_covers_debt",
        "eps_ttm", "fcf_ttm", "earnings_growth", "shares_diluted",
    ]
    fund_slim = fund_df[[c for c in merge_cols if c in fund_df.columns]].copy()
    fund_slim["publish_date"] = (
        pd.to_datetime(fund_slim["publish_date"], utc=True).dt.tz_localize(None)
    )
    fund_slim = fund_slim.dropna(subset=["publish_date"])
    fund_slim.drop_duplicates(["ticker", "publish_date"], keep="last", inplace=True)

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)

    # Merge per ticker to avoid pandas global-sort requirement
    parts: list[pd.DataFrame] = []
    for ticker, price_grp in prices.groupby("ticker"):
        fund_grp = fund_slim[fund_slim["ticker"] == ticker]
        if fund_grp.empty:
            parts.append(price_grp)
            continue
        part = pd.merge_asof(
            price_grp.sort_values("date"),
            fund_grp.drop(columns="ticker").sort_values("publish_date"),
            left_on="date",
            right_on="publish_date",
            direction="backward",
        )
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Price-dependent ratios
    # ------------------------------------------------------------------
    if "close" in merged.columns and "eps_ttm" in merged.columns:
        merged["pe_ratio"] = _safe_divide(merged["close"], merged["eps_ttm"])
        merged["pe_ratio"] = merged["pe_ratio"].clip(-500, 500)

    if "pe_ratio" in merged.columns and "earnings_growth" in merged.columns:
        # PEG = P/E / (earnings growth in percentage points)
        eg_pct = merged["earnings_growth"] * 100
        merged["peg_ratio"] = _safe_divide(merged["pe_ratio"], eg_pct)
        merged["peg_ratio"] = merged["peg_ratio"].clip(-50, 50)

    if ("close" in merged.columns
            and "fcf_ttm" in merged.columns
            and "shares_diluted" in merged.columns):
        market_cap = merged["close"] * merged["shares_diluted"]
        merged["fcf_yield"] = _safe_divide(merged["fcf_ttm"], market_cap)

    # Drop intermediate columns (not needed downstream)
    for col in ["publish_date", "eps_ttm", "fcf_ttm",
                "earnings_growth", "shares_diluted"]:
        if col in merged.columns:
            merged.drop(columns=col, inplace=True)

    return merged


# ---------------------------------------------------------------------------
# Update (re-download + rebuild)
# ---------------------------------------------------------------------------

def update_dataset(api_key: str = "free") -> pd.DataFrame:
    """Re-download SimFin data and rebuild the full dataset.

    SimFin bulk files always contain the complete history, so updating
    means re-downloading the latest bulk files and re-processing.
    """
    import shutil

    if SIMFIN_CACHE_DIR.exists():
        shutil.rmtree(SIMFIN_CACHE_DIR)
        logger.info("Cleared SimFin cache for fresh download")

    raw = download_simfin(api_key=api_key)
    df = build_dataset(raw)
    save_dataset(df)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical fundamental database",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "simfin"],
        default="yfinance",
        help="Data source: yfinance (no key, ~5 quarters) "
             "or simfin (needs key, 10+ years). Default: yfinance",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help='SimFin API key. If omitted, reads SIMFIN_API_KEY from .env '
             'or environment. Register free at https://app.simfin.com/',
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete cached files and re-download everything",
    )
    args = parser.parse_args()

    # Resolve API key: CLI flag > .env file > environment variable
    api_key = args.api_key or os.environ.get("SIMFIN_API_KEY", "free")

    if args.refresh and args.source == "simfin":
        update_dataset(api_key=api_key)
    else:
        raw = (
            download_simfin(api_key=api_key)
            if args.source == "simfin"
            else download_yfinance()
        )
        df = build_dataset(raw)
        save_dataset(df)


if __name__ == "__main__":
    _load_dotenv()
    setup_logging()
    main()
