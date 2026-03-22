"""
Centralized configuration for DeepValue AI.

Every path, constant, model parameter and feature mapping lives here.
No other module should hardcode paths or magic numbers.

WHY THIS FILE EXISTS:
    In v1, the threshold was hardcoded as 0.62 in app.py while the
    trained model actually used 0.75. Paths like "../models/modelo.pkl"
    broke depending on which directory you ran from. Constants were
    scattered across 4 files with no explanation.

    Now everything is in ONE place. Change a parameter here and it
    propagates everywhere.
"""

from pathlib import Path
import logging


# ---------------------------------------------------------------------------
# Project paths (absolute — work from ANY working directory)
# ---------------------------------------------------------------------------
# Path(__file__)         → .../DeepValueAI/core/config.py
# .resolve()             → makes it absolute
# .parent                → .../DeepValueAI/core/
# .parent                → .../DeepValueAI/          (project root)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "models_dir": PROJECT_ROOT / "models",
    "data_dir": PROJECT_ROOT / "data",
    "model_file": PROJECT_ROOT / "models" / "best_model.pkl",
    "threshold_file": PROJECT_ROOT / "models" / "optimal_threshold.txt",
    "dataset_file": PROJECT_ROOT / "data" / "training_dataset.csv",
    "comparison_file": PROJECT_ROOT / "models" / "model_comparison.csv",
    "backtest_model_file": PROJECT_ROOT / "models" / "best_model_backtest.pkl",
    "backtest_threshold_file": PROJECT_ROOT / "models" / "optimal_threshold_backtest.txt",
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# These are the features our model will be trained on.
# Both training (ml_pipeline) and inference (prediction_service)
# use this SAME list — guaranteeing consistency.
#
# All names are English. No mapping needed since we train from scratch.

# --- Technical features (11) ---
# Derived from price, volume, and market data.
TECHNICAL_FEATURES = [
    "williams_r",               # Williams %R oscillator (momentum)
    "williams_r_signal",        # Binary: crossed from <-80 to >-40 (buy signal)
    "rsi",                      # Relative Strength Index (14-day)
    "macd_histogram",           # MACD histogram (trend change detection)
    "atr_normalized",           # ATR / Close price (volatility as percentage)
    "sma_distance",             # (Close - SMA_200) / SMA_200
    "price_vs_sma_trend",       # Is price approaching SMA? (5-day direction)
    "sma_cross_below",          # Binary: price crossed below SMA recently
    "relative_volume",          # Today's volume / 20-day average volume
    "volume_trend",             # Is average volume increasing? (20-day slope)
    "market_trend",             # 1 if S&P 500 > its SMA 200, else 0
]

# --- Fundamental features (8) ---
# Extracted from Yahoo Finance company data.
FUNDAMENTAL_FEATURES = [
    "pe_ratio",                 # Price / Earnings
    "peg_ratio",                # PE / Growth rate (value relative to growth)
    "op_margin",                # Operating margin (efficiency)
    "revenue_growth",           # Revenue growth rate (momentum)
    "debt_equity",              # Debt-to-equity ratio (leverage)
    "current_ratio",            # Current assets / current liabilities (liquidity)
    "cash_covers_debt",         # Cash / total debt (ability to pay off debt)
    "fcf_yield",                # Free cash flow / market cap (real cash generation)
]

# Combined list used by the model (order matters for training consistency)
FEATURE_COLUMNS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES


# ---------------------------------------------------------------------------
# Strategy parameters — Entry
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50          # Fallback if no threshold file exists yet
SMA_BUY_CEILING = 1.05           # Price must be < SMA_200 × this to buy


# ---------------------------------------------------------------------------
# Portfolio management
# ---------------------------------------------------------------------------
# These controls prevent over-concentration and enforce diversification.
# Without them, the system could dump all capital into one stock that
# keeps triggering buy signals at similar prices.
#
# Example with $100,000 portfolio:
#   - Each buy: 5% of portfolio = $5,000
#   - Max per ticker: 10% = $10,000 (so max 2 buys of same stock)
#   - Max open positions: 15 different stocks
#   - Cooldown: 22 trading days before buying MORE of the SAME stock
#     (buying a DIFFERENT stock has no cooldown)

POSITION_SIZE_PCT = 0.05          # Each buy = 5% of current portfolio value
MAX_TICKER_EXPOSURE_PCT = 0.10    # Max 10% of portfolio in a single stock
MAX_OPEN_POSITIONS = 15           # Max 15 different stocks at once
COOLDOWN_DAYS_PER_TICKER = 22     # Wait ~1 month before buying same stock again


# ---------------------------------------------------------------------------
# Strategy parameters — Exit
# ---------------------------------------------------------------------------
# Hybrid exit strategy: the position closes on whichever triggers FIRST.
#
# Example with stock bought at $300:
#   1. Stop Loss:          drops to $264 (-12%) → sell everything
#   2. Partial Take Profit: rises to $345 (+15%) → sell 33% of shares
#   3. Trailing Stop:       once at +5% ($315), trail -8% from peak
#                           peak reaches $400 → stop at $368
#                           price drops to $368 → sell remaining shares
#   4. Time Limit:          after 120 trading days, sell at market price

STOP_LOSS_PCT = 0.12                   # Sell all if price drops 12% from purchase
PARTIAL_TP_TRIGGER_PCT = 0.15          # Trigger partial sell at +15% gain
PARTIAL_TP_SELL_FRACTION = 0.33        # Sell 33% of position at trigger
TRAILING_STOP_ACTIVATION_PCT = 0.05    # Activate trailing stop after +5% gain
TRAILING_STOP_PCT = 0.08              # Trail 8% below the highest price reached
MAX_HOLDING_DAYS = 120                 # Force sell after 120 trading days


# ---------------------------------------------------------------------------
# Training target definition
# ---------------------------------------------------------------------------

PREDICTION_HORIZON_DAYS = 40     # Target: stock rises within N trading days
MIN_RETURN_TARGET = 0.10         # Target: stock rises at least 10%


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

DOWNLOAD_PERIOD = "5y"
DOWNLOAD_INTERVAL = "1d"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_MARKET_TICKER = "^GSPC"
API_SLEEP_SECONDS = 0.1          # Pause between yfinance calls


# ---------------------------------------------------------------------------
# Technical indicator parameters
# ---------------------------------------------------------------------------

SMA_LENGTH = 200
WILLIAMS_R_LENGTH = 14
RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LENGTH = 14
VOLUME_SMA_LENGTH = 20


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the entire application."""
    logging.basicConfig(format=LOG_FORMAT, level=level)