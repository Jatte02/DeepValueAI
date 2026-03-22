"""
Data acquisition and feature engineering.

This module is the SINGLE source of truth for:
    - Downloading OHLCV data from Yahoo Finance
    - Downloading the S&P 500 constituent list
    - Computing technical indicators (SMA, Williams %R, RSI, MACD, ATR)
    - Extracting fundamental data (PE, margins, debt, cash flow)
    - Computing market trend context

Every other module imports from here instead of implementing its own
data logic. A fix or improvement here propagates everywhere.

DESIGN PRINCIPLE:
    All features are computed as raw numerical values. There are NO
    pre-filters that discard companies based on fundamentals. The model
    receives ALL data and learns what matters. This avoids discarding
    companies that fail one metric but excel in others.
"""

import logging
import time
from io import StringIO

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf

from .config import (
    API_SLEEP_SECONDS,
    ATR_LENGTH,
    DOWNLOAD_INTERVAL,
    DOWNLOAD_PERIOD,
    FUNDAMENTAL_FEATURES,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_LENGTH,
    SMA_LENGTH,
    SP500_MARKET_TICKER,
    SP500_WIKI_URL,
    TECHNICAL_FEATURES,
    VOLUME_SMA_LENGTH,
    WILLIAMS_R_LENGTH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S&P 500 ticker list
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Download the current S&P 500 constituent list from Wikipedia.

    Scrapes the first table on the Wikipedia "List of S&P 500 companies"
    page, extracts the 'Symbol' column, and normalizes tickers for
    Yahoo Finance compatibility (e.g., BRK.B → BRK-B).

    Returns
    -------
    list[str]
        Sorted list of ~503 ticker symbols (some companies have multiple
        share classes, so the count exceeds 500).

    Raises
    ------
    ConnectionError
        If Wikipedia is unreachable or returns a non-200 status.
    ValueError
        If the expected table structure is not found in the page.

    Examples
    --------
    >>> tickers = get_sp500_tickers()
    >>> "AAPL" in tickers
    True
    >>> len(tickers) > 490
    True
    """
    logger.info("Downloading S&P 500 ticker list from Wikipedia...")

    # ------------------------------------------------------------------
    # Step 1: Fetch the raw HTML from Wikipedia.
    # We use requests instead of letting pd.read_html fetch directly
    # because it gives us explicit control over error handling and
    # HTTP status codes. pd.read_html silently fails on some errors.
    # ------------------------------------------------------------------
    try:
        response = requests.get(SP500_WIKI_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to download S&P 500 list from Wikipedia: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Step 2: Parse the HTML table into a DataFrame.
    # pd.read_html returns a list of ALL tables found on the page.
    # The first table ([0]) is the current constituents list.
    # We wrap the HTML string in StringIO to avoid the pandas
    # FutureWarning about passing raw strings.
    # ------------------------------------------------------------------
    try:
        tables = pd.read_html(StringIO(response.text))
    except ValueError as exc:
        raise ValueError(
            "No HTML tables found on the Wikipedia S&P 500 page. "
            "The page structure may have changed."
        ) from exc

    if not tables:
        raise ValueError("Wikipedia returned an empty table list.")

    raw_df = tables[0]

    # ------------------------------------------------------------------
    # Step 3: Extract and clean ticker symbols.
    # The column is labeled "Symbol" in the Wikipedia table.
    # We validate it exists rather than using positional indexing,
    # which would break silently if Wikipedia reorders columns.
    # ------------------------------------------------------------------
    if "Symbol" not in raw_df.columns:
        raise ValueError(
            f"Expected 'Symbol' column not found. "
            f"Available columns: {list(raw_df.columns)}"
        )

    tickers = (
        raw_df["Symbol"]
        .str.strip()                     # Remove whitespace
        .str.replace(".", "-", regex=False)  # BRK.B → BRK-B (Yahoo format)
        .dropna()
        .unique()
        .tolist()
    )

    tickers.sort()
    logger.info("Retrieved %d S&P 500 tickers.", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# OHLCV data download
# ---------------------------------------------------------------------------

def download_ohlcv(
    tickers: list[str],
    period: str = DOWNLOAD_PERIOD,
    interval: str = DOWNLOAD_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """Download historical OHLCV data for a list of tickers from Yahoo Finance.

    Downloads each ticker individually using ``yf.Ticker().history()``
    rather than the bulk ``yf.download()`` API. This is intentionally
    slower but significantly more robust:

    - We get explicit per-ticker error handling and logging.
    - Bulk download silently drops tickers or returns partial data
      when Yahoo's servers hiccup, which corrupts downstream analysis
      without any warning.
    - We can apply a sleep between calls to respect rate limits.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download (e.g., ``["AAPL", "MSFT"]``).
    period : str
        Lookback period (default from config: ``"5y"``).
        Valid values: "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max".
    interval : str
        Bar interval (default from config: ``"1d"``).
        Valid values: "1d", "1wk", "1mo".

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → DataFrame with columns:
        ``[Open, High, Low, Close, Volume]``.
        Only tickers with at least 1 row of data are included.
        Tickers that failed to download are logged and skipped.

    Notes
    -----
    The returned DataFrames have a DatetimeIndex (trading dates).
    Dividends and stock splits columns are dropped — we only keep
    OHLCV because that's what the technical indicators need.

    Examples
    --------
    >>> data = download_ohlcv(["AAPL", "MSFT"], period="1y")
    >>> "AAPL" in data
    True
    >>> list(data["AAPL"].columns)
    ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    logger.info(
        "Downloading OHLCV data for %d tickers (period=%s, interval=%s)...",
        len(tickers), period, interval,
    )

    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        try:
            # ----------------------------------------------------------
            # yf.Ticker().history() returns a DataFrame with columns:
            # Open, High, Low, Close, Volume, Dividends, Stock Splits.
            # We only keep OHLCV — the rest is noise for our use case.
            # ----------------------------------------------------------
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, interval=interval)

            # ----------------------------------------------------------
            # Validation: Yahoo sometimes returns an empty DataFrame
            # for delisted tickers, tickers with no data in the period,
            # or during API hiccups. We skip these silently.
            # ----------------------------------------------------------
            if df.empty:
                logger.warning("Ticker %s returned empty data — skipping.", ticker)
                failed.append(ticker)
                continue

            # Keep only OHLCV columns and drop any rows with NaN in Close.
            # NaN in Close means the market was closed or data is corrupt.
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[ohlcv_cols].copy()
            df = df.dropna(subset=["Close"])

            if df.empty:
                logger.warning(
                    "Ticker %s has no valid Close prices after cleanup — skipping.",
                    ticker,
                )
                failed.append(ticker)
                continue

            result[ticker] = df

        except Exception as exc:
            # ----------------------------------------------------------
            # Broad except is intentional here. yfinance can raise
            # a variety of exceptions (KeyError, JSONDecodeError,
            # ConnectionError, etc.) depending on what Yahoo returns.
            # We don't want one bad ticker to kill a 500-ticker download.
            # ----------------------------------------------------------
            logger.warning("Failed to download %s: %s", ticker, exc)
            failed.append(ticker)

        # ----------------------------------------------------------
        # Progress logging every 50 tickers so the user knows
        # the process is alive during long runs (~500 tickers).
        # ----------------------------------------------------------
        if i % 50 == 0:
            logger.info("Progress: %d / %d tickers downloaded.", i, len(tickers))

        # Respect Yahoo Finance rate limits
        time.sleep(API_SLEEP_SECONDS)

    # Final summary
    logger.info(
        "Download complete: %d succeeded, %d failed out of %d total.",
        len(result), len(failed), len(tickers),
    )
    if failed:
        logger.warning("Failed tickers: %s", failed)

    return result

# ---------------------------------------------------------------------------
# Technical feature engineering
# ---------------------------------------------------------------------------

def compute_technical_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the 11 technical features to an OHLCV DataFrame.

    All indicators are computed using pandas_ta. The function modifies
    a COPY of the input — the original DataFrame is never mutated.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for a single ticker. Must contain columns:
        ``[Open, High, Low, Close, Volume]`` with a DatetimeIndex.
    market_df : pd.DataFrame | None
        OHLCV data for the market index (S&P 500). Used to compute
        the ``market_trend`` feature. If None, ``market_trend`` is set
        to NaN (useful for unit testing individual tickers).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with 11 new columns appended (see
        ``config.TECHNICAL_FEATURES`` for the full list).
    """
    df = df.copy()

    # --- 1. Williams %R (momentum oscillator) ---
    # Measures where today's close sits relative to the highest high
    # and lowest low over the lookback period.
    # Range: -100 (oversold) to 0 (overbought).
    # We use pandas_ta which returns a Series named "WILLr_{length}".
    df["williams_r"] = ta.willr(
        high=df["High"], low=df["Low"], close=df["Close"],
        length=WILLIAMS_R_LENGTH,
    )

    # --- 2. Williams %R buy signal (binary) ---
    # Detects when Williams %R crosses from deeply oversold (<-80) to
    # a recovery zone (>-40). This "snap back" often precedes a rally.
    # We look at the previous row to detect the crossing.
    prev_wr = df["williams_r"].shift(1)
    df["williams_r_signal"] = (
        (prev_wr < -80) & (df["williams_r"] > -40)
    ).astype(int)

    # --- 3. RSI (Relative Strength Index) ---
    # Classic momentum oscillator. 0-100 scale.
    # Below 30 = oversold, above 70 = overbought.
    df["rsi"] = ta.rsi(close=df["Close"], length=RSI_LENGTH)

    # --- 4. MACD Histogram ---
    # MACD = fast EMA - slow EMA. Signal = EMA of MACD.
    # Histogram = MACD - Signal. Positive = bullish momentum.
    # pandas_ta returns a DataFrame with 3 columns; we take the histogram.
    macd_result = ta.macd(
        close=df["Close"],
        fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL,
    )
    df["macd_histogram"] = macd_result.iloc[:, 2]  # 3rd column is histogram

    # --- 5. ATR Normalized (volatility as percentage of price) ---
    # Raw ATR is in dollar terms ($5 ATR means nothing without context).
    # Dividing by Close gives us a percentage: "this stock moves ~2%/day".
    # This makes ATR comparable across stocks at different price levels.
    atr = ta.atr(
        high=df["High"], low=df["Low"], close=df["Close"],
        length=ATR_LENGTH,
    )
    df["atr_normalized"] = atr / df["Close"]

    # --- 6. SMA Distance (price relative to 200-day moving average) ---
    # (Close - SMA_200) / SMA_200 → percentage distance.
    # Negative = price below SMA (potential value), positive = above.
    sma = ta.sma(close=df["Close"], length=SMA_LENGTH)
    df["sma_200"] = sma 
    df["sma_distance"] = (df["Close"] - sma) / sma

    # --- 7. Price vs SMA Trend (is price approaching or leaving SMA?) ---
    # We measure the 5-day change in sma_distance. If it's getting less
    # negative (approaching from below), that's a bullish signal.
    # Uses .diff(5) = value today minus value 5 days ago.
    df["price_vs_sma_trend"] = df["sma_distance"].diff(5)

    # --- 8. SMA Cross Below (binary: price recently crossed below SMA) ---
    # Detects when price was above SMA yesterday but below today.
    # This is a classic "support break" signal that value investors watch.
    price_above_sma = df["Close"] > sma
    df["sma_cross_below"] = (
        price_above_sma.shift(1) & ~price_above_sma
    ).astype(int)

    # --- 9. Relative Volume (today's volume vs 20-day average) ---
    # Values > 1 mean unusually high volume (institutional activity).
    # Values < 1 mean low interest. We cap at 5 to prevent outliers
    # from dominating the feature (e.g., earnings day = 10x normal volume).
    vol_sma = ta.sma(close=df["Volume"].astype(float), length=VOLUME_SMA_LENGTH)
    df["relative_volume"] = (df["Volume"] / vol_sma).clip(upper=5.0)

    # --- 10. Volume Trend (is average volume increasing or decreasing?) ---
    # 20-day slope of the volume moving average, normalized.
    # Positive = growing interest, negative = declining interest.
    df["volume_trend"] = vol_sma.pct_change(periods=20)

    # --- 11. Market Trend (is the overall market bullish?) ---
    # Binary: 1 if S&P 500 is above its own 200-day SMA, else 0.
    # This is a regime filter — most stocks do better in bull markets.
    if market_df is not None and not market_df.empty:
        market_sma = ta.sma(close=market_df["Close"], length=SMA_LENGTH)
        market_trend_series = (market_df["Close"] > market_sma).astype(int)
        # Reindex to match the ticker's dates (handles missing days)
        df["market_trend"] = market_trend_series.reindex(df.index, method="ffill")
    else:
        df["market_trend"] = np.nan

    return df

# ---------------------------------------------------------------------------
# Fundamental feature extraction
# ---------------------------------------------------------------------------

def get_fundamental_features(ticker: str) -> dict[str, float]:
    """Extract 8 fundamental features from Yahoo Finance for a single ticker.

    Uses the ``yf.Ticker().info`` dictionary, which contains the most
    recent snapshot of company financials (not historical — just the
    latest reported values).

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., ``"AAPL"``).

    Returns
    -------
    dict[str, float]
        Mapping of feature name → value for all 8 fundamental features.
        Missing values are returned as ``np.nan`` (never raises on
        missing data — the model handles NaN via imputation).

    Examples
    --------
    >>> feats = get_fundamental_features("AAPL")
    >>> "pe_ratio" in feats
    True
    """
    logger.debug("Fetching fundamentals for %s...", ticker)

    # ------------------------------------------------------------------
    # yf.Ticker().info returns a dict with 100+ keys. Not all keys
    # exist for every company (e.g., banks don't report "operatingMargins"
    # the same way tech companies do). Using .get(key, None) ensures
    # we never crash on a missing key — we just get None.
    # ------------------------------------------------------------------
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
        return {feat: np.nan for feat in FUNDAMENTAL_FEATURES}

    # ------------------------------------------------------------------
    # Helper: safely extract a numeric value from the info dict.
    # Yahoo sometimes returns strings, None, or 0 where we expect
    # a float. This function handles all those cases in one place
    # instead of repeating try/except for each feature.
    # ------------------------------------------------------------------
    def safe_get(key: str) -> float:
        """Return a float from info[key], or np.nan if anything goes wrong."""
        val = info.get(key)
        if val is None:
            return np.nan
        try:
            val = float(val)
            # Discard infinities — they break model training.
            return val if np.isfinite(val) else np.nan
        except (ValueError, TypeError):
            return np.nan

    # ------------------------------------------------------------------
    # Feature extraction: each feature maps to one or more Yahoo keys.
    # We compute derived features here rather than raw values because
    # ratios like cash_covers_debt and fcf_yield don't exist directly
    # in Yahoo's API — we have to calculate them from components.
    # ------------------------------------------------------------------

    # 1. PE Ratio — Price / Earnings.
    #    "trailingPE" = based on actual past 12 months earnings.
    #    We prefer trailing over forward because forward PE is an
    #    analyst estimate, not a fact.
    pe_ratio = safe_get("trailingPE")

    # 2. PEG Ratio — PE / expected growth rate.
    #    A PE of 30 is expensive for a slow grower but cheap for a
    #    company growing 40%/year. PEG normalizes for growth.
    #    PEG < 1 is classically considered undervalued.
    peg_ratio = safe_get("pegRatio")

    # 3. Operating Margin — what percentage of revenue is profit
    #    before interest and taxes. Measures operational efficiency.
    #    Yahoo returns this as a decimal (0.25 = 25%).
    op_margin = safe_get("operatingMargins")

    # 4. Revenue Growth — year-over-year revenue change.
    #    Also a decimal (0.10 = 10% growth). Negative = shrinking.
    revenue_growth = safe_get("revenueGrowth")

    # 5. Debt to Equity — total debt / total shareholder equity.
    #    High values (>2) mean the company is heavily leveraged.
    #    Yahoo provides this directly.
    debt_equity = safe_get("debtToEquity")

    # 6. Current Ratio — current assets / current liabilities.
    #    Above 1.0 = can pay short-term obligations. Below 1.0 = risky.
    current_ratio = safe_get("currentRatio")

    # 7. Cash Covers Debt — total cash / total debt.
    #    A value > 1 means the company could pay off ALL debt with
    #    cash on hand. We compute this ourselves because Yahoo
    #    doesn't provide it directly.
    total_cash = safe_get("totalCash")
    total_debt = safe_get("totalDebt")
    if np.isfinite(total_cash) and np.isfinite(total_debt) and total_debt > 0:
        cash_covers_debt = total_cash / total_debt
    else:
        cash_covers_debt = np.nan

    # 8. Free Cash Flow Yield — FCF / market cap.
    #    Measures how much real cash the business generates relative
    #    to its price. High FCF yield = you're paying less per dollar
    #    of cash generated. It's like dividend yield but better because
    #    it includes ALL free cash, not just what's paid as dividends.
    fcf = safe_get("freeCashflow")
    market_cap = safe_get("marketCap")
    if np.isfinite(fcf) and np.isfinite(market_cap) and market_cap > 0:
        fcf_yield = fcf / market_cap
    else:
        fcf_yield = np.nan

    return {
        "pe_ratio": pe_ratio,
        "peg_ratio": peg_ratio,
        "op_margin": op_margin,
        "revenue_growth": revenue_growth,
        "debt_equity": debt_equity,
        "current_ratio": current_ratio,
        "cash_covers_debt": cash_covers_debt,
        "fcf_yield": fcf_yield,
    }


# ---------------------------------------------------------------------------
# Combined feature pipeline
# ---------------------------------------------------------------------------

def build_feature_row(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a complete feature set for a single ticker.

    This is the main entry point that other modules call. It chains
    together the technical and fundamental feature computation into
    a single DataFrame ready for model consumption.

    The function:
        1. Computes all 11 technical features from OHLCV data.
        2. Fetches all 8 fundamental features from Yahoo Finance.
        3. Adds the fundamental values as constant columns (same value
           on every row, because fundamentals are a snapshot, not a
           time series).
        4. Drops rows where technical indicators are NaN due to warmup
           (the first ~200 rows have no SMA_200, for example).

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., ``"AAPL"``). Used to fetch fundamentals.
    ohlcv_df : pd.DataFrame
        OHLCV data for this ticker. Must have columns:
        ``[Open, High, Low, Close, Volume]`` with a DatetimeIndex.
    market_df : pd.DataFrame | None
        OHLCV data for the S&P 500 index. Passed through to
        ``compute_technical_features`` for the ``market_trend`` feature.

    Returns
    -------
    pd.DataFrame
        The original OHLCV data enriched with 19 feature columns
        (11 technical + 8 fundamental). Rows with NaN in any
        technical feature are dropped (warmup period).
        The DataFrame retains its DatetimeIndex.

    Examples
    --------
    >>> ohlcv = download_ohlcv(["AAPL"], period="2y")["AAPL"]
    >>> market = download_ohlcv(["^GSPC"], period="2y")["^GSPC"]
    >>> features = build_feature_row("AAPL", ohlcv, market)
    >>> features.shape[1] >= 24  # 5 OHLCV + 11 tech + 8 fund
    True
    """
    # ------------------------------------------------------------------
    # Step 1: Technical features (11 new columns added to df)
    # ------------------------------------------------------------------
    df = compute_technical_features(ohlcv_df, market_df=market_df)

    # ------------------------------------------------------------------
    # Step 2: Fundamental features (8 values from Yahoo snapshot)
    # ------------------------------------------------------------------
    fundamentals = get_fundamental_features(ticker)

    # Add each fundamental value as a constant column.
    # Every row gets the same value because fundamentals represent
    # the CURRENT state of the company, not a daily time series.
    # When we use this for backtesting later, we'll need to be aware
    # that this introduces a mild form of look-ahead bias — the
    # current fundamentals are applied to historical rows. This is
    # acceptable for a screening/decision tool but would need more
    # careful handling in rigorous academic backtesting.
    for feat_name, feat_value in fundamentals.items():
        df[feat_name] = feat_value

    # ------------------------------------------------------------------
    # Step 3: Drop warmup rows.
    # Technical indicators need N previous bars to compute. For example,
    # SMA_200 needs 200 bars, so the first 199 rows are NaN. Williams %R
    # needs 14 bars, RSI needs 14 bars, etc. The longest is SMA at 200,
    # so effectively the first ~200 rows will have NaN in sma_distance
    # and related features.
    #
    # We drop rows where ANY technical feature is NaN. We do NOT drop
    # on fundamental NaN because those are expected (some companies
    # simply don't report certain metrics) and the model handles them
    # via imputation during training.
    # ------------------------------------------------------------------

    df = df.dropna(subset=TECHNICAL_FEATURES)

    logger.debug(
        "Built features for %s: %d rows, %d columns.",
        ticker, len(df), len(df.columns),
    )

    return df