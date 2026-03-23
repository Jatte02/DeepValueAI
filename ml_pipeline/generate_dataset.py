"""
Historical ETL pipeline — build the training dataset from scratch.

Downloads 5 years of daily OHLCV data for every S&P 500 constituent,
computes the 19 features defined in config.py, generates forward-looking
binary labels, and saves a single CSV ready for model training.

LABEL LOGIC:
    For each trading day *t* and a given ticker:
        1. Look at the next 60 trading days (PREDICTION_HORIZON_DAYS).
        2. Find the maximum closing price in that window.
        3. Compute the return: max_price / close_t - 1.
        4. If return >= 5% (MIN_RETURN_TARGET) → label = 1 (good buy).
           Else → label = 0 (bad buy).
    Rows without 60 future days of data are discarded (no label possible).

FUNDAMENTAL FEATURES:
    Historical quarterly fundamentals are merged point-in-time from the
    local fundamental database (built by core.fundamental_database).
    Each row gets the fundamentals that were actually available on that
    date (based on SEC filing / publish date), eliminating look-ahead
    bias. Price-dependent ratios (pe_ratio, peg_ratio, fcf_yield) are
    computed at merge time using the daily close price.

    This means the SAME trained model (all 19 features) can be used for
    both real-time screening and historical backtesting without bias.

Usage:
    python -m ml_pipeline.generate_dataset
"""

import logging

import numpy as np
import pandas as pd

from core.config import (
    FEATURE_COLUMNS,
    MIN_RETURN_TARGET,
    PATHS,
    PREDICTION_HORIZON_DAYS,
    SP500_MARKET_TICKER,
    TECHNICAL_FEATURES,
    setup_logging,
)
from core.data_service import (
    compute_technical_features,
    download_ohlcv,
    get_sp500_tickers,
)
from core.fundamental_database import merge_fundamentals_pit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def create_labels(
    close: pd.Series,
    horizon: int = PREDICTION_HORIZON_DAYS,
    min_return: float = MIN_RETURN_TARGET,
) -> pd.Series:
    """Generate binary labels from forward-looking maximum returns.

    For each row *i*, computes ``max(Close[i+1 : i+1+horizon]) / Close[i] - 1``.
    If that return >= *min_return*, label = 1; else label = 0.
    The last *horizon* rows get NaN (not enough future data).

    Uses a vectorized reverse-rolling-max trick instead of a Python loop:
        1. Shift Close by -1 so position *i* holds tomorrow's price.
        2. Reverse the series.
        3. Apply a backward rolling max (= forward rolling in original order).
        4. Reverse back.
    This is O(n) and avoids per-row iteration over ~1 000 rows × 500 tickers.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices with a DatetimeIndex.
    horizon : int
        Number of future trading days to look ahead.
    min_return : float
        Minimum return to qualify as a positive label.

    Returns
    -------
    pd.Series
        Binary labels (0.0 / 1.0) with NaN for rows lacking sufficient
        future data. Name is ``"label"``.
    """
    # Step 1: shift so position i holds Close[i+1]
    shifted = close.shift(-1)

    # Step 2–4: forward rolling max via reverse → rolling → reverse
    future_max = (
        shifted
        .iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .max()
        .iloc[::-1]
    )

    # Compute forward return and threshold
    forward_return = future_max / close - 1
    labels = (forward_return >= min_return).astype(float)
    labels[forward_return.isna()] = np.nan

    return labels.rename("label")


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    """Download S&P 500 data, compute features, create labels, save CSV.

    The full pipeline:
        1. Scrape the ~503 S&P 500 tickers from Wikipedia.
        2. Download 5 years of daily OHLCV data for every ticker + ^GSPC.
        3. For each ticker, compute 11 technical features and generate
           binary labels.
        4. Concatenate all tickers and merge historical fundamentals
           point-in-time (8 features from fundamental database).
        5. Save to ``PATHS["dataset_file"]``.

    Returns
    -------
    pd.DataFrame
        The full training dataset (also saved to disk).

    Raises
    ------
    RuntimeError
        If no data could be processed (network failure, API issues, etc.).
    """
    # ------------------------------------------------------------------
    # 1. Get ticker list
    # ------------------------------------------------------------------
    tickers = get_sp500_tickers()
    logger.info("Got %d S&P 500 tickers.", len(tickers))

    # ------------------------------------------------------------------
    # 2. Download OHLCV for all tickers + S&P 500 index
    # ------------------------------------------------------------------
    all_tickers = tickers + [SP500_MARKET_TICKER]
    ohlcv_data = download_ohlcv(all_tickers)

    market_df = ohlcv_data.pop(SP500_MARKET_TICKER, None)
    if market_df is None:
        logger.warning(
            "Could not download S&P 500 index (%s). "
            "market_trend feature will be NaN for all tickers.",
            SP500_MARKET_TICKER,
        )

    # ------------------------------------------------------------------
    # 3. Technical features + label creation per ticker
    # ------------------------------------------------------------------
    chunks: list[pd.DataFrame] = []
    skipped = 0

    for i, ticker in enumerate(tickers, start=1):
        if ticker not in ohlcv_data:
            skipped += 1
            continue

        try:
            # Compute 11 technical features from OHLCV data
            df = compute_technical_features(ohlcv_data[ticker], market_df)
            df = df.dropna(subset=TECHNICAL_FEATURES)

            if df.empty:
                skipped += 1
                continue

            # Generate forward-looking labels
            df["label"] = create_labels(df["Close"])
            df["ticker"] = ticker
            df["date"] = df.index
            df["close"] = df["Close"]  # needed for merge_fundamentals_pit

            # Drop rows without labels (last PREDICTION_HORIZON_DAYS rows)
            df = df.dropna(subset=["label"])

            if not df.empty:
                chunks.append(df)
            else:
                skipped += 1

        except Exception as exc:
            logger.warning("Failed to process %s: %s", ticker, exc)
            skipped += 1

        if i % 50 == 0:
            logger.info(
                "Progress: %d / %d tickers processed (%d chunks so far).",
                i, len(tickers), len(chunks),
            )

    logger.info(
        "Feature engineering complete: %d tickers succeeded, %d skipped.",
        len(chunks), skipped,
    )

    if not chunks:
        raise RuntimeError(
            "No data was generated. Check your network connection and "
            "Yahoo Finance API access."
        )

    # ------------------------------------------------------------------
    # 4. Concatenate and merge historical fundamentals
    # ------------------------------------------------------------------
    dataset = pd.concat(chunks, ignore_index=True)

    logger.info("Merging historical fundamentals (point-in-time)...")
    dataset = merge_fundamentals_pit(dataset)

    # Select only the columns needed for training
    cols_to_save = ["ticker", "date"] + FEATURE_COLUMNS + ["label"]
    dataset = dataset[cols_to_save]

    output_path = PATHS["dataset_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    n_positive = int((dataset["label"] == 1).sum())
    n_negative = int((dataset["label"] == 0).sum())
    n_total = n_positive + n_negative
    pct_positive = 100 * n_positive / n_total if n_total > 0 else 0

    logger.info(
        "Dataset saved to %s\n"
        "  Rows:     %d\n"
        "  Tickers:  %d\n"
        "  Positive: %d (%.1f%%)\n"
        "  Negative: %d (%.1f%%)",
        output_path,
        n_total,
        dataset["ticker"].nunique(),
        n_positive, pct_positive,
        n_negative, 100 - pct_positive,
    )

    return dataset


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    generate_dataset()
