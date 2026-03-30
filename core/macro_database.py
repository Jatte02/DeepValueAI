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

    Each series is stored as its own rows with ``release_date`` being
    the point-in-time date when the data was first published.  Different
    series have different release cadences (daily / monthly / quarterly),
    so they are stored **independently** and merged into prices one at a
    time inside ``merge_macro_pit``.

    The DataFrame contains one row per (series, release_date) combination.
    A ``feature`` column identifies which series the row belongs to.

    Parameters
    ----------
    series_data : dict
        Mapping of FRED series ID → DataFrame with columns
        ``date``, ``realtime_start``, ``value``.

    Returns
    -------
    pd.DataFrame
        Columns: ``release_date``, ``feature``, ``value``.
        Each row is one observation with its PIT release date.
    """
    rows: list[pd.DataFrame] = []

    # --- Federal Funds Rate (DFF) ---
    if "DFF" in series_data:
        dff = series_data["DFF"].copy()
        dff = dff.sort_values("date")
        # fed_rate: raw value
        fr = pd.DataFrame({
            "release_date": dff["realtime_start"].values,
            "feature": "fed_rate",
            "value": dff["value"].values,
        })
        rows.append(fr)
        # fed_rate_change_3m: 63 trading-day diff
        change = dff["value"].diff(periods=63)
        valid = change.notna()
        fc = pd.DataFrame({
            "release_date": dff.loc[valid, "realtime_start"].values,
            "feature": "fed_rate_change_3m",
            "value": change[valid].values,
        })
        rows.append(fc)

    # --- Unemployment (UNRATE) ---
    if "UNRATE" in series_data:
        unrate = series_data["UNRATE"].copy()
        unrate = unrate.sort_values("date")
        ur = pd.DataFrame({
            "release_date": unrate["realtime_start"].values,
            "feature": "unemployment",
            "value": unrate["value"].values,
        })
        rows.append(ur)
        # 3-month trend (3 monthly observations)
        trend = unrate["value"].diff(periods=3)
        valid = trend.notna()
        ut = pd.DataFrame({
            "release_date": unrate.loc[valid, "realtime_start"].values,
            "feature": "unemployment_trend",
            "value": trend[valid].values,
        })
        rows.append(ut)

    # --- GDP Growth (GDP) ---
    if "GDP" in series_data:
        gdp = series_data["GDP"].copy()
        gdp = gdp.sort_values("date")
        growth = gdp["value"].pct_change()
        valid = growth.notna()
        gg = pd.DataFrame({
            "release_date": gdp.loc[valid, "realtime_start"].values,
            "feature": "gdp_growth",
            "value": growth[valid].values,
        })
        rows.append(gg)

    # --- CPI Year-over-Year (CPIAUCSL) ---
    if "CPIAUCSL" in series_data:
        cpi = series_data["CPIAUCSL"].copy()
        cpi = cpi.sort_values("date")
        yoy = cpi["value"].pct_change(periods=12)
        valid = yoy.notna()
        cy = pd.DataFrame({
            "release_date": cpi.loc[valid, "realtime_start"].values,
            "feature": "cpi_yoy",
            "value": yoy[valid].values,
        })
        rows.append(cy)

    if not rows:
        raise RuntimeError("No macro features could be computed.")

    result = pd.concat(rows, ignore_index=True)
    result["release_date"] = pd.to_datetime(result["release_date"])
    result = result.sort_values("release_date").reset_index(drop=True)

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

    Each macro feature is merged independently using its own
    ``release_date`` as the PIT key.  This is necessary because
    different series have different release cadences:
        - DFF: daily (same-day release)
        - UNRATE: monthly (~5 day publication lag)
        - GDP: quarterly (~30 day lag)
        - CPIAUCSL: monthly (~13 day lag)

    A backward ``merge_asof`` on ``release_date`` means a value
    published on 2023-04-27 is only "visible" from that date onward.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must have a ``date`` column.
    macro_df : pd.DataFrame or None
        Long-format macro features (columns: release_date, feature, value).
        If None, loads from default path.

    Returns
    -------
    pd.DataFrame
        ``prices_df`` enriched with 6 macro feature columns.
    """
    if macro_df is None:
        macro_df = load_macro()

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.as_unit("ns")
    prices = prices.sort_values("date")

    macro = macro_df.copy()
    macro["release_date"] = pd.to_datetime(macro["release_date"]).dt.as_unit("ns")

    # Merge each feature independently (each has its own release dates)
    feature_names = macro["feature"].unique()
    for feat_name in feature_names:
        feat_data = macro[macro["feature"] == feat_name][["release_date", "value"]].copy()
        feat_data = feat_data.dropna(subset=["value"])
        feat_data = feat_data.sort_values("release_date")
        feat_data = feat_data.drop_duplicates(subset=["release_date"], keep="last")
        feat_data = feat_data.rename(columns={"value": feat_name})

        prices = pd.merge_asof(
            prices,
            feat_data,
            left_on="date",
            right_on="release_date",
            direction="backward",
        )
        # Drop release_date if merge_asof added it
        if "release_date" in prices.columns:
            prices.drop(columns="release_date", inplace=True)

    return prices


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

    features = df["feature"].unique().tolist()
    logger.info(
        "Macro database built: %d rows, features: %s",
        len(df), features,
    )


if __name__ == "__main__":
    _load_dotenv()
    setup_logging()
    main()
