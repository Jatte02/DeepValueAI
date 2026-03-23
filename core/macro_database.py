"""
Macro economic data from FRED (Federal Reserve Economic Data).

Downloads key economic indicators (fed funds rate, unemployment, GDP,
CPI/inflation), handles point-in-time release dates, and stores as
a local Parquet file for merge into the training dataset.

POINT-IN-TIME SAFE:
    FRED's ``realtime_start`` metadata tells us exactly when each data
    point was first published. We use this as the merge key so that
    a GDP reading released on 2023-04-27 is only "available" from that
    date onward in any simulation.

SERIES DOWNLOADED:
    DFF       — Federal Funds Effective Rate (daily)
    UNRATE    — Unemployment Rate (monthly, ~5 day lag)
    GDP       — Real GDP (quarterly, ~30 day lag)
    CPIAUCSL  — Consumer Price Index (monthly, ~13 day lag)

FEATURES COMPUTED:
    fed_rate            — Raw federal funds rate
    fed_rate_change_3m  — 3-month change in fed rate
    unemployment        — Raw unemployment rate
    unemployment_trend  — 3-month change in unemployment
    gdp_growth          — Quarter-over-quarter GDP growth rate
    cpi_yoy             — Year-over-year CPI change (inflation)

Usage:
    python -m core.macro_database
    python -m core.macro_database --api-key YOUR_FRED_KEY
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import FRED_SERIES, PATHS, PROJECT_ROOT, setup_logging

logger = logging.getLogger(__name__)

MACRO_DIR = PATHS["macro_dir"]
MACRO_PATH = PATHS["macro_file"]


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
# Download from FRED
# ---------------------------------------------------------------------------

def download_fred(
    api_key: str,
    start_date: str = "2014-01-01",
) -> pd.DataFrame:
    """Download macro series from FRED and return a unified daily DataFrame.

    Each observation is tagged with its ``realtime_start`` date (when FRED
    first published it) for point-in-time safety.

    Parameters
    ----------
    api_key : str
        FRED API key. Register free at https://fred.stlouisfed.org/docs/api/api_key.html
    start_date : str
        Earliest observation date to fetch.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``release_date``, plus one column per feature
        (fed_rate, unemployment, gdp_growth, cpi_yoy, fed_rate_change_3m,
        unemployment_trend).
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series_data: dict[str, pd.DataFrame] = {}

    # Daily series (DFF) have too many vintage dates for get_series_all_releases.
    # Use get_series for daily data (release lag is negligible).
    daily_series = {"DFF"}

    for series_id, feature_name in FRED_SERIES.items():
        logger.info("Downloading FRED series: %s (%s)...", series_id, feature_name)
        try:
            if series_id in daily_series:
                # Daily data: use get_series (no vintage explosion)
                raw = fred.get_series(series_id, observation_start=start_date)
                if raw is None or raw.empty:
                    logger.warning("FRED series %s returned no data.", series_id)
                    continue
                obs = pd.DataFrame({
                    "date": raw.index,
                    "realtime_start": raw.index,   # same-day release
                    "value": raw.values,
                })
            else:
                # Monthly/quarterly: use all_releases for true PIT dates
                raw = fred.get_series_all_releases(series_id)
                if raw is None or raw.empty:
                    logger.warning("FRED series %s returned no data.", series_id)
                    continue
                # Already has columns: realtime_start, date, value
                obs = raw.reset_index(drop=True).copy()

            obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
            obs["date"] = pd.to_datetime(obs["date"])
            obs["realtime_start"] = pd.to_datetime(obs["realtime_start"])
            obs = obs.dropna(subset=["value"])

            # Keep first release only (point-in-time: what was known first)
            obs = obs.sort_values("realtime_start")
            obs = obs.drop_duplicates(subset=["date"], keep="first")
            obs = obs[obs["date"] >= pd.Timestamp(start_date)]

            series_data[series_id] = obs
            logger.info(
                "  %s: %d observations (%s → %s)",
                series_id, len(obs),
                obs["date"].min().strftime("%Y-%m-%d"),
                obs["date"].max().strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logger.warning("Failed to download FRED series %s: %s", series_id, exc)

    if not series_data:
        raise RuntimeError(
            "No FRED data could be downloaded. Check your API key and network."
        )

    return _build_macro_features(series_data)


def _build_macro_features(
    series_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute derived macro features from raw FRED series.

    Parameters
    ----------
    series_data : dict
        Mapping of FRED series ID → DataFrame with columns
        ``date``, ``realtime_start``, ``value``.

    Returns
    -------
    pd.DataFrame
        Daily macro features with ``date`` and ``release_date`` columns.
    """
    features: list[pd.DataFrame] = []

    # --- Federal Funds Rate (DFF) ---
    if "DFF" in series_data:
        dff = series_data["DFF"].copy()
        dff = dff.rename(columns={"value": "fed_rate"})
        dff["release_date"] = dff["realtime_start"]
        # 3-month change (approx 63 trading days for daily data)
        dff = dff.sort_values("date")
        dff["fed_rate_change_3m"] = dff["fed_rate"].diff(periods=63)
        features.append(dff[["date", "release_date", "fed_rate", "fed_rate_change_3m"]])

    # --- Unemployment (UNRATE) ---
    if "UNRATE" in series_data:
        unrate = series_data["UNRATE"].copy()
        unrate = unrate.rename(columns={"value": "unemployment"})
        unrate["release_date"] = unrate["realtime_start"]
        unrate = unrate.sort_values("date")
        # 3-month trend (3 monthly observations)
        unrate["unemployment_trend"] = unrate["unemployment"].diff(periods=3)
        features.append(
            unrate[["date", "release_date", "unemployment", "unemployment_trend"]]
        )

    # --- GDP Growth (GDP) ---
    if "GDP" in series_data:
        gdp = series_data["GDP"].copy()
        gdp = gdp.sort_values("date")
        gdp["release_date"] = gdp["realtime_start"]
        # Quarter-over-quarter growth rate
        gdp["gdp_growth"] = gdp["value"].pct_change()
        features.append(gdp[["date", "release_date", "gdp_growth"]])

    # --- CPI Year-over-Year (CPIAUCSL) ---
    if "CPIAUCSL" in series_data:
        cpi = series_data["CPIAUCSL"].copy()
        cpi = cpi.sort_values("date")
        cpi["release_date"] = cpi["realtime_start"]
        # YoY inflation (12 monthly observations)
        cpi["cpi_yoy"] = cpi["value"].pct_change(periods=12)
        features.append(cpi[["date", "release_date", "cpi_yoy"]])

    if not features:
        raise RuntimeError("No macro features could be computed.")

    # Merge all features on date using outer join
    result = features[0]
    for f in features[1:]:
        result = pd.merge(result, f, on=["date", "release_date"], how="outer")

    result = result.sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])
    result["release_date"] = pd.to_datetime(result["release_date"])

    return result


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_macro(df: pd.DataFrame, path: Path = MACRO_PATH) -> None:
    """Save the macro features to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1_048_576
    logger.info("Saved macro data → %s (%.1f MB)", path, mb)


def load_macro(path: Path = MACRO_PATH) -> pd.DataFrame:
    """Load the macro features from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Macro data not found: {path}\n"
            "Run first: python -m core.macro_database"
        )
    df = pd.read_parquet(path)
    logger.info("Loaded macro data: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Point-in-time merge helper
# ---------------------------------------------------------------------------

def merge_macro_pit(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge daily prices with the most recent available macro data.

    Uses ``release_date`` for PIT safety: a macro reading released
    on 2023-04-27 is only available from that date onward.

    For each macro feature, performs a backward ``merge_asof`` on
    ``release_date``, then forward-fills (economically correct:
    unemployment is 4.2% until the next release says otherwise).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must have a ``date`` column.
    macro_df : pd.DataFrame or None
        Macro features. If None, loads from default path.

    Returns
    -------
    pd.DataFrame
        ``prices_df`` enriched with 6 macro feature columns.
    """
    if macro_df is None:
        macro_df = load_macro()

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.as_unit("ns")

    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"]).dt.as_unit("ns")
    macro["release_date"] = pd.to_datetime(macro["release_date"]).dt.as_unit("ns")

    # Use release_date as the PIT key
    macro_cols = [
        "fed_rate", "fed_rate_change_3m",
        "unemployment", "unemployment_trend",
        "gdp_growth", "cpi_yoy",
    ]
    existing_cols = [c for c in macro_cols if c in macro.columns]
    macro_slim = macro[["release_date"] + existing_cols].copy()
    macro_slim = macro_slim.dropna(subset=existing_cols, how="all")
    macro_slim = macro_slim.sort_values("release_date")
    macro_slim = macro_slim.drop_duplicates(subset=["release_date"], keep="last")

    prices = prices.sort_values("date")

    merged = pd.merge_asof(
        prices,
        macro_slim,
        left_on="date",
        right_on="release_date",
        direction="backward",
    )

    # Drop the release_date column (not needed downstream)
    if "release_date" in merged.columns:
        merged.drop(columns="release_date", inplace=True)

    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download macro economic data from FRED",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="FRED API key. If omitted, reads FRED_API_KEY from .env or environment.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Provide via --api-key flag, "
            "FRED_API_KEY in .env, or FRED_API_KEY environment variable.\n"
            "Register free at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    df = download_fred(api_key=api_key)
    save_macro(df)

    logger.info(
        "Macro database built: %d rows, features: %s",
        len(df), [c for c in df.columns if c not in ("date", "release_date")],
    )


if __name__ == "__main__":
    _load_dotenv()
    setup_logging()
    main()
