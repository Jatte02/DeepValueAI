"""
S&P 500 opportunity screener.

This module scans all S&P 500 constituents in real time, runs the
production model (19 features: technical + fundamental), and returns
a ranked table of investment opportunities with metadata.

For each ticker the screener computes:
    - Model probability and binary buy signal
    - Signal strength (how far above the threshold)
    - Signal freshness (consecutive days the model has signaled "buy")
    - SMA headroom (margin before hitting the SMA buy ceiling)
    - Key fundamental metrics for quick assessment

DESIGN PRINCIPLE:
    The screener does NOT manage a portfolio or simulate trades.
    It answers one question: "Which S&P 500 stocks does the model
    like RIGHT NOW, and how confident is it?"

    Portfolio constraints (position sizing, cooldown, exposure limits)
    are the UI layer's concern — the screener provides raw rankings.
"""

import logging
import time

import numpy as np
import pandas as pd

from .config import (
    API_SLEEP_SECONDS,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURES,
    SMA_BUY_CEILING,
    PATHS,
)
from .data_service import (
    build_feature_row,
    download_ohlcv,
    get_sp500_tickers,
)
from .prediction_service import load_model, load_threshold, predict_proba

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal metadata
# ---------------------------------------------------------------------------

FRESHNESS_LOOKBACK_DAYS = 30  # How many days back to check for consecutive signals


def _compute_signal_metadata(
    df: pd.DataFrame,
    model,
    threshold: float,
    feature_list: list[str],
) -> dict[str, float]:
    """Compute signal metadata for a single ticker.

    Runs the model over the last N rows of the feature DataFrame to
    determine how long the buy signal has been active, and computes
    strength and SMA headroom from the latest row.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched OHLCV data for one ticker. Must contain all
        columns in ``feature_list`` plus ``sma_200`` and ``Close``.
    model : sklearn-compatible classifier
        Loaded model with a ``predict_proba`` method.
    threshold : float
        Probability threshold for a buy signal.
    feature_list : list[str]
        Feature columns the model expects (typically ``FEATURE_COLUMNS``).

    Returns
    -------
    dict[str, float]
        - ``probability``: model probability for the latest row
        - ``signal_strength``: 0.0–1.0 normalized distance above threshold
        - ``signal_freshness_days``: consecutive days the model has signaled buy
        - ``sma_headroom_pct``: % margin before hitting SMA buy ceiling
    """
    # ------------------------------------------------------------------
    # Step 1: Batch predict the last N rows.
    # We extract the feature matrix for the last N rows and run
    # predict_proba once. This is a single numpy operation — fast.
    # ------------------------------------------------------------------
    lookback = min(FRESHNESS_LOOKBACK_DAYS, len(df))
    recent = df.iloc[-lookback:]

    try:
        X = recent[feature_list].values
        probas = model.predict_proba(X)[:, 1]  # Probability of class 1 (buy)
    except Exception as exc:
        logger.warning("Batch predict failed for metadata: %s", exc)
        return {
            "probability": np.nan,
            "signal_strength": np.nan,
            "signal_freshness_days": 0,
            "sma_headroom_pct": np.nan,
        }

    # ------------------------------------------------------------------
    # Step 2: Probability and signal strength (latest row).
    #
    # Signal strength normalizes the probability into a 0–1 scale
    # ABOVE the threshold. This tells the user "how enthusiastic"
    # the model is, not just "buy or not".
    #
    # Example with threshold = 0.50:
    #   prob = 0.51 → strength = 0.02 (barely above, weak signal)
    #   prob = 0.75 → strength = 0.50 (solid signal)
    #   prob = 0.95 → strength = 0.90 (very strong signal)
    #   prob = 0.40 → strength = 0.00 (below threshold, no signal)
    # ------------------------------------------------------------------
    latest_prob = float(probas[-1])

    if latest_prob >= threshold and threshold < 1.0:
        signal_strength = (latest_prob - threshold) / (1.0 - threshold)
    else:
        signal_strength = 0.0

    # ------------------------------------------------------------------
    # Step 3: Signal freshness (consecutive buy days from today).
    #
    # We walk backwards from the most recent prediction. As soon as
    # we find a day where the model did NOT signal buy, we stop.
    # This counts how many consecutive days the signal has been active.
    #
    # freshness = 1 → signal appeared today (brand new opportunity)
    # freshness = 10 → signal has been active for 10 trading days
    # freshness = 0 → no signal today (shouldn't happen if we filter,
    #                  but we handle it for robustness)
    # ------------------------------------------------------------------
    buy_signals = probas >= threshold
    freshness = 0
    for is_buy in reversed(buy_signals):
        if is_buy:
            freshness += 1
        else:
            break

    # ------------------------------------------------------------------
    # Step 4: SMA headroom (how much room before hitting buy ceiling).
    #
    # The entry filter requires: price < SMA_200 × SMA_BUY_CEILING.
    # Headroom tells the user how close the price is to that ceiling.
    #
    # Example: SMA = 100, ceiling multiplier = 1.05, price = 95
    #   ceiling_price = 105
    #   headroom = (105 - 95) / 95 = 10.5%
    #
    # Negative headroom means price ALREADY exceeds the ceiling
    # (the stock fails the SMA filter).
    # ------------------------------------------------------------------
    latest_close = df["Close"].iloc[-1]
    latest_sma = df["sma_200"].iloc[-1]

    if pd.notna(latest_sma) and latest_sma > 0:
        ceiling_price = latest_sma * SMA_BUY_CEILING
        sma_headroom_pct = (ceiling_price - latest_close) / latest_close
    else:
        sma_headroom_pct = np.nan

    return {
        "probability": latest_prob,
        "signal_strength": round(signal_strength, 4),
        "signal_freshness_days": freshness,
        "sma_headroom_pct": round(sma_headroom_pct, 4) if pd.notna(sma_headroom_pct) else np.nan,
    }


# ---------------------------------------------------------------------------
# Single-ticker analysis
# ---------------------------------------------------------------------------

def _analyze_ticker(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    market_df: pd.DataFrame | None,
    model,
    threshold: float,
    feature_list: list[str],
) -> dict | None:
    """Run full analysis on a single ticker and return a result row.

    Orchestrates: feature engineering → model prediction → metadata
    computation → result assembly. Returns None if the ticker fails
    at any step (bad data, missing features, model error).

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., ``"AAPL"``).
    ohlcv_df : pd.DataFrame
        OHLCV data for this ticker.
    market_df : pd.DataFrame | None
        S&P 500 OHLCV data for the ``market_trend`` feature.
    model : sklearn-compatible classifier
        Loaded production model.
    threshold : float
        Probability threshold for buy signal.
    feature_list : list[str]
        Feature columns the model expects.

    Returns
    -------
    dict | None
        A flat dictionary with all screener columns, or None if the
        ticker could not be analyzed.
    """
    # ------------------------------------------------------------------
    # Step 1: Build features (11 technical + 8 fundamental).
    # build_feature_row already handles:
    #   - Technical indicator computation
    #   - Fundamental data fetching from Yahoo
    #   - Dropping warmup rows (first ~200 rows with NaN technicals)
    # ------------------------------------------------------------------
    try:
        df = build_feature_row(ticker, ohlcv_df, market_df=market_df)
    except Exception as exc:
        logger.warning("Feature engineering failed for %s: %s", ticker, exc)
        return None

    if df.empty or len(df) < FRESHNESS_LOOKBACK_DAYS:
        logger.warning(
            "Insufficient data for %s after feature engineering (%d rows) — skipping.",
            ticker, len(df),
        )
        return None

    # ------------------------------------------------------------------
    # Step 2: Compute signal metadata (probability, strength,
    # freshness, SMA headroom). This runs the model on the last
    # N rows in a single batch call.
    # ------------------------------------------------------------------
    metadata = _compute_signal_metadata(df, model, threshold, feature_list)

    if np.isnan(metadata["probability"]):
        logger.warning("Model prediction failed for %s — skipping.", ticker)
        return None

    # ------------------------------------------------------------------
    # Step 3: Apply entry filters.
    # We still include tickers that FAIL the filters in the result,
    # but mark them with passes_filters = False. This way the UI
    # can show "near misses" — stocks that the model likes but
    # don't quite meet the technical entry criteria yet.
    #
    # The two filters:
    #   1. Probability >= threshold (model says buy)
    #   2. Price < SMA_200 × SMA_BUY_CEILING (not overextended)
    # ------------------------------------------------------------------
    latest_close = df["Close"].iloc[-1]
    latest_sma = df["sma_200"].iloc[-1]

    passes_model = metadata["probability"] >= threshold
    passes_sma = (
        pd.notna(latest_sma)
        and latest_sma > 0
        and latest_close < latest_sma * SMA_BUY_CEILING
    )
    passes_filters = passes_model and passes_sma

    # ------------------------------------------------------------------
    # Step 4: Assemble the result row.
    # We build a flat dict rather than a dataclass because the final
    # output is a DataFrame — dicts convert directly via pd.DataFrame.
    # ------------------------------------------------------------------
    row = {
        "ticker": ticker,
        "close": round(latest_close, 2),
        "sma_200": round(latest_sma, 2) if pd.notna(latest_sma) else np.nan,

        # Signal
        "probability": metadata["probability"],
        "signal_strength": metadata["signal_strength"],
        "signal_freshness_days": metadata["signal_freshness_days"],
        "sma_headroom_pct": metadata["sma_headroom_pct"],
        "passes_filters": passes_filters,

        # Fundamentals — expose directly from the last row of df.
        # These are already computed by build_feature_row.
        **{feat: df[feat].iloc[-1] for feat in FUNDAMENTAL_FEATURES},
    }

    return row

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_sp500(
    model_path: str | None = None,
    threshold_path: str | None = None,
    tickers: list[str] | None = None,
    include_failing: bool = True,
) -> pd.DataFrame:
    """Scan S&P 500 stocks and return a ranked table of opportunities.

    This is the main entry point for the screener. It:
        1. Loads the production model and threshold.
        2. Downloads the S&P 500 ticker list (or uses a custom list).
        3. Downloads OHLCV data for all tickers + the market index.
        4. Runs each ticker through the full analysis pipeline.
        5. Returns a DataFrame sorted by probability descending.

    Parameters
    ----------
    model_path : str | None
        Path to the trained model file. Defaults to
        ``PATHS["model_file"]`` from config.
    threshold_path : str | None
        Path to the threshold file. Defaults to
        ``PATHS["threshold_file"]`` from config.
    tickers : list[str] | None
        Custom ticker list to scan. If None, downloads the full
        S&P 500 list from Wikipedia. Useful for testing with a
        small list (e.g., ``["AAPL", "MSFT", "GOOGL"]``).
    include_failing : bool
        If True (default), include tickers that don't pass entry
        filters (marked with ``passes_filters=False``). If False,
        only return tickers that pass all filters.

    Returns
    -------
    pd.DataFrame
        Ranked screener results with columns:
        - ``ticker``: symbol
        - ``close``: latest closing price
        - ``sma_200``: 200-day simple moving average
        - ``probability``: model buy probability
        - ``signal_strength``: 0–1 normalized strength above threshold
        - ``signal_freshness_days``: consecutive trading days with buy signal
        - ``sma_headroom_pct``: % margin before SMA buy ceiling
        - ``passes_filters``: bool, True if all entry conditions are met
        - 8 fundamental feature columns (pe_ratio, peg_ratio, etc.)

        Sorted by ``probability`` descending. Empty DataFrame if no
        tickers could be analyzed.

    Examples
    --------
    >>> # Quick test with 3 tickers
    >>> results = scan_sp500(tickers=["AAPL", "MSFT", "GOOGL"])
    >>> results.columns.tolist()[:4]
    ['ticker', 'close', 'sma_200', 'probability']

    >>> # Full scan (takes ~10-15 minutes due to data download)
    >>> results = scan_sp500()
    >>> passing = results[results.passes_filters]
    """
    # ------------------------------------------------------------------
    # Step 1: Load model and threshold.
    # load_model returns the model; threshold is loaded separately via
    # load_threshold(). If no threshold file exists, it falls back to
    # DEFAULT_THRESHOLD from config.
    # ------------------------------------------------------------------
    _model_path = model_path or str(PATHS["model_file"])
    _threshold_path = threshold_path or str(PATHS["threshold_file"])

    logger.info("Loading production model from %s...", _model_path)
    model = load_model(_model_path)
    threshold = load_threshold(_threshold_path)
    logger.info("Model loaded. Threshold: %.4f", threshold)

    feature_list = FEATURE_COLUMNS  # 19 features for production model

    # ------------------------------------------------------------------
    # Step 2: Get ticker list.
    # If no custom list is provided, download the full S&P 500.
    # We always add the market index (^GSPC) for the market_trend
    # feature, but we DON'T scan it — it's not a tradeable stock.
    # ------------------------------------------------------------------
    if tickers is None:
        tickers = get_sp500_tickers()
    else:
        tickers = list(tickers)  # Defensive copy
        logger.info("Using custom ticker list: %d tickers.", len(tickers))

    # ------------------------------------------------------------------
    # Step 3: Download OHLCV data for all tickers + market index.
    # We download everything in one batch call to download_ohlcv.
    # The market index is included in the download but processed
    # separately — it's needed for compute_technical_features but
    # is not itself a screener candidate.
    # ------------------------------------------------------------------
    all_symbols = tickers.copy()
    market_ticker = "^GSPC"
    if market_ticker not in all_symbols:
        all_symbols.append(market_ticker)

    logger.info("Downloading OHLCV data for %d symbols...", len(all_symbols))
    ohlcv_data = download_ohlcv(all_symbols)

    # Extract market data and remove it from the scan pool.
    market_df = ohlcv_data.pop(market_ticker, None)
    if market_df is None:
        logger.warning(
            "Could not download market index (%s). "
            "market_trend feature will be NaN for all tickers.",
            market_ticker,
        )

    # ------------------------------------------------------------------
    # Step 4: Analyze each ticker.
    # We iterate through the successfully downloaded tickers (not the
    # original list — some may have failed to download). Each ticker
    # goes through the full pipeline: features → prediction → metadata.
    # ------------------------------------------------------------------
    results: list[dict] = []
    scannable = [t for t in tickers if t in ohlcv_data]
    logger.info(
        "Analyzing %d tickers (%d failed to download)...",
        len(scannable), len(tickers) - len(scannable),
    )

    for i, ticker in enumerate(scannable, start=1):
        row = _analyze_ticker(
            ticker=ticker,
            ohlcv_df=ohlcv_data[ticker],
            market_df=market_df,
            model=model,
            threshold=threshold,
            feature_list=feature_list,
        )

        if row is not None:
            results.append(row)

        # Progress logging every 50 tickers
        if i % 50 == 0:
            logger.info("Screener progress: %d / %d tickers analyzed.", i, len(scannable))

    # ------------------------------------------------------------------
    # Step 5: Build result DataFrame and sort.
    # ------------------------------------------------------------------
    if not results:
        logger.warning("No tickers could be analyzed. Returning empty DataFrame.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Filter out non-passing tickers if requested
    if not include_failing:
        df_results = df_results[df_results["passes_filters"]].copy()

    # Sort by probability descending — highest conviction first
    df_results = df_results.sort_values("probability", ascending=False).reset_index(drop=True)

    # Summary stats
    n_passing = df_results["passes_filters"].sum()
    logger.info(
        "Scan complete: %d tickers analyzed, %d passing filters, %d total in results.",
        len(scannable), n_passing, len(df_results),
    )

    return df_results