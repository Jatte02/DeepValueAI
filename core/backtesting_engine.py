"""
Historical backtesting engine for the DeepValue AI hybrid strategy.

This module simulates how the trading strategy would have performed
over historical data. It replays each trading day sequentially:
checking exit conditions on open positions first, then evaluating
entry signals for new buys.

LOOK-AHEAD BIAS PREVENTION:
    This engine uses the unified 34-feature model (technical + VIX +
    fundamental + macro + sentiment). All data sources are merged
    point-in-time: fundamentals via publish_date, macro via
    release_date, and sentiment with a 1-day lag. Each day only sees
    data that was actually available before that date. No look-ahead bias.

ARCHITECTURE:
    Three layers, each with a single responsibility:

    run_backtest()     → Orchestrates the full simulation.
    _process_day()     → Handles one day: exits first, then entries.
    PortfolioState     → Mutable state container (cash, positions, log).

    All strategy parameters come from config.py. Nothing is hardcoded.

USAGE:
    >>> from core.backtesting_engine import run_backtest
    >>> result = run_backtest(
    ...     tickers=["AAPL", "MSFT"],
    ...     start_date="2020-01-01",
    ...     end_date="2024-12-31",
    ...     initial_capital=100_000.0,
    ... )
    >>> result.metrics["total_return"]
    0.42  # 42% total return (example)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    COOLDOWN_DAYS_PER_TICKER,
    MAX_HOLDING_DAYS,
    MAX_OPEN_POSITIONS,
    MAX_TICKER_EXPOSURE_PCT,
    PARTIAL_TP_SELL_FRACTION,
    PARTIAL_TP_TRIGGER_PCT,
    PATHS,
    POSITION_SIZE_PCT,
    SMA_BUY_CEILING,
    SMA_LENGTH,
    SP500_MARKET_TICKER,
    STOP_LOSS_PCT,
    FEATURE_COLUMNS,
    TECHNICAL_FEATURES,
    TRAILING_STOP_ACTIVATION_PCT,
    TRAILING_STOP_PCT,
    VIX_TICKER,
)
from .data_service import compute_technical_features, download_ohlcv
from .fundamental_database import merge_fundamentals_pit
from .prediction_service import load_model, load_threshold, predict_proba

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open position in the portfolio.

    Each buy creates a new Position. A single ticker can have multiple
    Position objects if the strategy buys the same stock more than once
    (up to the MAX_TICKER_EXPOSURE_PCT limit).

    Attributes
    ----------
    ticker : str
        Stock symbol (e.g., "AAPL").
    entry_price : float
        Price at which the position was opened.
    shares : float
        Number of shares held. Can be fractional after partial sells.
    entry_date : pd.Timestamp
        Date the position was opened.
    days_held : int
        Trading days since entry. Incremented each day in the simulation.
    highest_price : float
        Highest price seen since entry. Used for trailing stop calculation.
        Initialized to entry_price and updated each day.
    trailing_stop_active : bool
        Whether the trailing stop has been activated (price rose ≥5%
        above entry). Once activated, it never deactivates.
    partial_tp_taken : bool
        Whether the partial take-profit has already been executed.
        The strategy only takes partial profit ONCE per position.
    """

    ticker: str
    entry_price: float
    shares: float
    entry_date: pd.Timestamp
    days_held: int = 0
    highest_price: float = 0.0
    trailing_stop_active: bool = False
    partial_tp_taken: bool = False

    def __post_init__(self) -> None:
        """Set highest_price to entry_price if not explicitly provided."""
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price


@dataclass
class PortfolioState:
    """Mutable container for the entire portfolio during simulation.

    This is the "blackboard" that ``_process_day()`` reads and writes.
    Every piece of mutable state lives here — no global variables,
    no hidden state in closures.

    Attributes
    ----------
    cash : float
        Available cash (starts at initial_capital).
    positions : list[Position]
        Currently open positions.
    trade_log : list[dict]
        Record of every executed trade (buys and sells).
        Each entry is a dict with keys: ticker, action, date, price,
        shares, reason, value.
    daily_values : list[dict]
        Snapshot of portfolio value at the end of each trading day.
        Each entry is a dict with keys: date, portfolio_value, cash,
        invested_value.
    last_buy_date : dict[str, pd.Timestamp]
        Tracks the last buy date for each ticker, used to enforce
        the 22-day cooldown period.
    """

    cash: float
    positions: list[Position] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    daily_values: list[dict[str, Any]] = field(default_factory=list)
    last_buy_date: dict[str, pd.Timestamp] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

def _check_exits(
    state: PortfolioState,
    date: pd.Timestamp,
    prices: dict[str, float],
) -> None:
    """Evaluate exit conditions on all open positions for a single day.

    Modifies *state* in place: sells shares, adds cash, logs trades,
    and removes closed positions.

    Exit conditions are checked in priority order. Full exits (sell
    everything) take precedence over partial exits. On any given day,
    at most ONE exit action fires per position:

        1. Stop Loss       → price fell 12% from entry    → sell ALL
        2. Trailing Stop    → price fell 8% from peak      → sell ALL
        3. Time Limit       → 120 trading days in position → sell ALL
        4. Partial TP       → price rose 15% from entry    → sell 33%
                              (only fires once per position)

    Parameters
    ----------
    state : PortfolioState
        The current portfolio state. Modified in place.
    date : pd.Timestamp
        The current trading day.
    prices : dict[str, float]
        Mapping of ticker → closing price for this day. Positions
        whose ticker is missing from *prices* are skipped (this can
        happen if a stock is halted or delisted mid-backtest).
    """
    surviving_positions: list[Position] = []

    for pos in state.positions:
        price = prices.get(pos.ticker)

        # ----------------------------------------------------------
        # If we don't have a price for this ticker today, keep the
        # position unchanged. This handles halted stocks, data gaps,
        # and tickers that disappear mid-backtest.
        # ----------------------------------------------------------
        if price is None:
            surviving_positions.append(pos)
            continue

        # ----------------------------------------------------------
        # Update daily tracking BEFORE checking exit conditions.
        # This ensures trailing stop uses TODAY's high, not yesterday's.
        # ----------------------------------------------------------
        pos.days_held += 1
        pos.highest_price = max(pos.highest_price, price)

        # Activate trailing stop if price has risen enough from entry.
        # Once activated, it NEVER deactivates (even if price drops
        # back below the activation level).
        if not pos.trailing_stop_active:
            gain_from_entry = (price - pos.entry_price) / pos.entry_price
            if gain_from_entry >= TRAILING_STOP_ACTIVATION_PCT:
                pos.trailing_stop_active = True

        # ----------------------------------------------------------
        # Calculate key ratios used by multiple exit checks.
        # ----------------------------------------------------------
        change_from_entry = (price - pos.entry_price) / pos.entry_price
        drop_from_peak = (pos.highest_price - price) / pos.highest_price

        # ===========================================================
        # EXIT CHECK 1: Stop Loss
        # The most important protection. If price dropped 12% from
        # our purchase price, sell everything immediately. This is
        # checked FIRST because capital preservation is priority #1.
        # ===========================================================
        if change_from_entry <= -STOP_LOSS_PCT:
            sell_value = pos.shares * price
            state.cash += sell_value
            state.trade_log.append({
                "ticker": pos.ticker,
                "action": "SELL",
                "date": date,
                "price": price,
                "shares": pos.shares,
                "reason": "stop_loss",
                "value": sell_value,
                "return_pct": change_from_entry,
            })
            logger.debug(
                "%s | STOP LOSS %s @ $%.2f (%.1f%% loss, %d days held)",
                date.date(), pos.ticker, price,
                change_from_entry * 100, pos.days_held,
            )
            # Position is NOT appended to surviving_positions → it's closed.
            continue

        # ===========================================================
        # EXIT CHECK 2: Trailing Stop
        # Only fires if the trailing stop has been activated (price
        # previously rose ≥5% from entry). Once active, we track the
        # highest price reached and sell if price drops 8% from that
        # peak. This locks in gains while giving the stock room to
        # fluctuate.
        #
        # Example: bought at $100, peak reached $150.
        #   Trailing stop level = $150 × (1 - 0.08) = $138.
        #   If price drops to $138 → sell.
        # ===========================================================
        if pos.trailing_stop_active and drop_from_peak >= TRAILING_STOP_PCT:
            sell_value = pos.shares * price
            state.cash += sell_value
            state.trade_log.append({
                "ticker": pos.ticker,
                "action": "SELL",
                "date": date,
                "price": price,
                "shares": pos.shares,
                "reason": "trailing_stop",
                "value": sell_value,
                "return_pct": change_from_entry,
            })
            logger.debug(
                "%s | TRAILING STOP %s @ $%.2f (%.1f%% from entry, "
                "peak was $%.2f, %d days held)",
                date.date(), pos.ticker, price,
                change_from_entry * 100, pos.highest_price, pos.days_held,
            )
            continue

        # ===========================================================
        # EXIT CHECK 3: Time Limit
        # If we've held this position for 120 trading days without
        # any other exit triggering, sell at market. This prevents
        # dead money — capital tied up in a stock that isn't moving.
        #
        # 120 trading days ≈ 6 calendar months.
        # ===========================================================
        if pos.days_held >= MAX_HOLDING_DAYS:
            sell_value = pos.shares * price
            state.cash += sell_value
            state.trade_log.append({
                "ticker": pos.ticker,
                "action": "SELL",
                "date": date,
                "price": price,
                "shares": pos.shares,
                "reason": "time_limit",
                "value": sell_value,
                "return_pct": change_from_entry,
            })
            logger.debug(
                "%s | TIME LIMIT %s @ $%.2f (%.1f%% return, %d days held)",
                date.date(), pos.ticker, price,
                change_from_entry * 100, pos.days_held,
            )
            continue

        # ===========================================================
        # EXIT CHECK 4: Partial Take Profit
        # If price rose 15% from entry and we haven't taken partial
        # profit yet, sell 33% of our shares and keep the rest.
        #
        # This is the ONLY exit that doesn't close the position.
        # After taking partial profit:
        #   - Remaining shares stay open (67% of original).
        #   - Trailing stop and time limit can still trigger later.
        #   - This same partial TP will NOT fire again (flag set).
        #
        # Why take partial profit? It's a risk management technique:
        # we lock in some gains while keeping upside exposure. If the
        # stock keeps rising, the trailing stop captures more. If it
        # reverses, we already banked 33% at a good price.
        # ===========================================================
        if not pos.partial_tp_taken and change_from_entry >= PARTIAL_TP_TRIGGER_PCT:
            shares_to_sell = pos.shares * PARTIAL_TP_SELL_FRACTION
            sell_value = shares_to_sell * price
            state.cash += sell_value
            state.trade_log.append({
                "ticker": pos.ticker,
                "action": "SELL_PARTIAL",
                "date": date,
                "price": price,
                "shares": shares_to_sell,
                "reason": "partial_tp",
                "value": sell_value,
                "return_pct": change_from_entry,
            })
            pos.shares -= shares_to_sell
            pos.partial_tp_taken = True
            logger.debug(
                "%s | PARTIAL TP %s: sold %.2f shares @ $%.2f "
                "(%.1f%% gain), %.2f shares remaining",
                date.date(), pos.ticker, shares_to_sell, price,
                change_from_entry * 100, pos.shares,
            )
            # Position survives — falls through to append below.

        # ----------------------------------------------------------
        # Position survived all exit checks → keep it.
        # ----------------------------------------------------------
        surviving_positions.append(pos)

    # Replace the positions list with only the survivors.
    state.positions = surviving_positions


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

def _check_entries(
    state: PortfolioState,
    date: pd.Timestamp,
    prices: dict[str, float],
    probabilities: dict[str, float],
    sma_values: dict[str, float],
    threshold: float,
) -> None:
    """Evaluate entry conditions for all candidate tickers on a single day.

    Modifies *state* in place: buys shares, deducts cash, logs trades,
    and updates cooldown tracking.

    ALL of the following conditions must be met for a buy to execute:

        1. Model probability > threshold
        2. Price < SMA_200 × 1.05 (technical confirmation)
        3. Portfolio has fewer than 15 open positions (unique tickers)
        4. Ticker exposure < 10% of portfolio value
        5. At least 22 trading days since last buy of THIS ticker
        6. Enough cash for the position (5% of portfolio value)

    Parameters
    ----------
    state : PortfolioState
        The current portfolio state. Modified in place.
    date : pd.Timestamp
        The current trading day.
    prices : dict[str, float]
        Mapping of ticker → closing price for this day.
    probabilities : dict[str, float]
        Mapping of ticker → model probability for this day.
        Only tickers present in this dict are evaluated as candidates.
    sma_values : dict[str, float]
        Mapping of ticker → SMA_200 value for this day. Used for the
        technical confirmation filter (price < SMA × 1.05).
    threshold : float
        Minimum probability required for a BUY signal.
    """
    # ------------------------------------------------------------------
    # Pre-compute portfolio-level metrics used by multiple checks.
    # These are computed ONCE per day, not per ticker.
    # ------------------------------------------------------------------
    portfolio_value = _compute_portfolio_value(state, prices)

    # Count unique tickers in open positions (not number of Position
    # objects, because one ticker can have multiple positions).
    unique_tickers_held = {pos.ticker for pos in state.positions}
    num_open_positions = len(unique_tickers_held)

    # Position size in dollars: 5% of current portfolio value.
    # This is dynamic — as the portfolio grows, each buy gets bigger.
    # As it shrinks, each buy gets smaller. Automatic risk scaling.
    target_buy_value = portfolio_value * POSITION_SIZE_PCT

    # ------------------------------------------------------------------
    # Sort candidates by probability (highest first).
    # When multiple stocks trigger on the same day, we want to buy
    # the highest-conviction signals first. Without sorting, the
    # order would be arbitrary (dict iteration order), which could
    # mean we fill our position slots with mediocre signals and miss
    # better ones.
    # ------------------------------------------------------------------
    candidates = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for ticker, prob in candidates:
        # =============================================================
        # CONDITION 1: Model probability exceeds threshold
        # =============================================================
        if prob <= threshold:
            # Candidates are sorted by prob descending, so once we
            # hit one below threshold, all remaining are below too.
            break

        price = prices.get(ticker)
        sma = sma_values.get(ticker)

        # Skip if price or SMA data is unavailable for this day.
        if price is None or sma is None:
            continue

        # =============================================================
        # CONDITION 2: Price < SMA_200 × 1.05
        # This is the technical confirmation filter. We only buy when
        # the price is near or below the 200-day moving average.
        # The 5% buffer (1.05) avoids being too strict — a stock
        # slightly above SMA can still be a good entry.
        #
        # WHY THIS FILTER EXISTS:
        # The model outputs a probability, but it doesn't know the
        # current price relative to the long-term trend. A stock
        # with great fundamentals trading 40% above its SMA is
        # likely overextended. This filter catches that.
        # =============================================================
        if sma > 0 and price >= sma * SMA_BUY_CEILING:
            continue

        # =============================================================
        # CONDITION 3: Not at max open positions (15 unique tickers)
        # =============================================================
        if num_open_positions >= MAX_OPEN_POSITIONS:
            logger.debug(
                "%s | SKIP %s: max positions reached (%d/%d)",
                date.date(), ticker, num_open_positions, MAX_OPEN_POSITIONS,
            )
            break  # No point checking more tickers today

        # =============================================================
        # CONDITION 4: Ticker exposure < 10% of portfolio value
        # Sum the CURRENT market value of all positions in this ticker.
        # If adding a new position would push exposure beyond 10%,
        # skip this ticker (but keep checking others).
        # =============================================================
        current_exposure = sum(
            pos.shares * prices.get(pos.ticker, pos.entry_price)
            for pos in state.positions
            if pos.ticker == ticker
        )
        max_exposure_value = portfolio_value * MAX_TICKER_EXPOSURE_PCT
        if current_exposure + target_buy_value > max_exposure_value:
            logger.debug(
                "%s | SKIP %s: exposure limit (current=$%.0f, "
                "buy=$%.0f, max=$%.0f)",
                date.date(), ticker, current_exposure,
                target_buy_value, max_exposure_value,
            )
            continue

        # =============================================================
        # CONDITION 5: Cooldown — 22 trading days since last buy
        # This prevents overconcentration by stopping the strategy
        # from repeatedly buying the same stock that keeps triggering
        # signals at similar prices. The cooldown is PER TICKER —
        # buying MSFT today doesn't block buying AAPL tomorrow.
        # =============================================================
        last_buy = state.last_buy_date.get(ticker)
        if last_buy is not None:
            # Count business days (Mon-Fri) between last buy and today.
            # This is more accurate than calendar days for a trading
            # cooldown, since weekends don't count as trading days.
            days_since = len(pd.bdate_range(last_buy, date)) - 1
            if days_since < COOLDOWN_DAYS_PER_TICKER:
                logger.debug(
                    "%s | SKIP %s: cooldown (%d/%d days)",
                    date.date(), ticker, days_since, COOLDOWN_DAYS_PER_TICKER,
                )
                continue

        # =============================================================
        # CONDITION 6: Enough cash for the position
        # We check this LAST because it's the least informative for
        # debugging. If we checked cash first, we'd never know if
        # the signal quality or filters were the real bottleneck.
        # =============================================================
        if state.cash < target_buy_value:
            logger.debug(
                "%s | SKIP %s: insufficient cash ($%.0f < $%.0f)",
                date.date(), ticker, state.cash, target_buy_value,
            )
            continue

        # =============================================================
        # ALL CONDITIONS PASSED → EXECUTE BUY
        # =============================================================
        shares_to_buy = target_buy_value / price
        buy_value = shares_to_buy * price  # Exactly target_buy_value

        state.cash -= buy_value
        state.positions.append(Position(
            ticker=ticker,
            entry_price=price,
            shares=shares_to_buy,
            entry_date=date,
        ))
        state.last_buy_date[ticker] = date
        state.trade_log.append({
            "ticker": ticker,
            "action": "BUY",
            "date": date,
            "price": price,
            "shares": shares_to_buy,
            "reason": "signal",
            "value": buy_value,
            "return_pct": 0.0,
        })

        # Update open position count (this ticker might be new).
        unique_tickers_held.add(ticker)
        num_open_positions = len(unique_tickers_held)

        logger.debug(
            "%s | BUY %s: %.2f shares @ $%.2f ($%.0f, prob=%.3f)",
            date.date(), ticker, shares_to_buy, price, buy_value, prob,
        )


def _compute_portfolio_value(
    state: PortfolioState,
    prices: dict[str, float],
) -> float:
    """Calculate the total portfolio value (cash + market value of positions).

    Parameters
    ----------
    state : PortfolioState
        Current portfolio state.
    prices : dict[str, float]
        Today's closing prices per ticker. If a ticker is missing,
        the position is valued at its entry price (conservative
        assumption for halted/missing stocks).

    Returns
    -------
    float
        Total portfolio value.
    """
    invested_value = sum(
        pos.shares * prices.get(pos.ticker, pos.entry_price)
        for pos in state.positions
    )
    return state.cash + invested_value

# ---------------------------------------------------------------------------
# Daily processing
# ---------------------------------------------------------------------------

def _process_day(
    state: PortfolioState,
    date: pd.Timestamp,
    feature_dfs: dict[str, pd.DataFrame],
    model,
    threshold: float,
) -> None:
    """Execute the full trading logic for a single day.

    This is the heartbeat of the simulation. For each trading day:

        1. Extract today's prices and SMA values from the feature DataFrames.
        2. Run the model in BATCH on all tickers with data today.
        3. Check exit conditions on all open positions.
        4. Check entry conditions for new buys.
        5. Record a daily portfolio snapshot.

    The order is critical: exits BEFORE entries. If we entered first,
    we might buy a stock and then immediately trigger a stop loss on
    a different position, distorting the cash available for the buy
    we just made. Exits first ensures we always know how much cash
    is truly available.

    Parameters
    ----------
    state : PortfolioState
        The current portfolio state. Modified in place.
    date : pd.Timestamp
        The current trading day.
    feature_dfs : dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame with OHLCV + 19 features
        (11 technical + 8 fundamental). Each DataFrame has a
        DatetimeIndex. Only rows up to *date* are used (no look-ahead).
    model
        A fitted scikit-learn compatible estimator (unified model,
        trained on all 19 features).
    threshold : float
        Minimum probability for a BUY signal.
    """
    # ------------------------------------------------------------------
    # Step 1: Collect today's data for all tickers that have it.
    # We build prices and SMA dicts, plus a batch DataFrame for
    # prediction. Doing one batch predict is ~30x faster than
    # calling predict_proba 500 times individually.
    # ------------------------------------------------------------------
    prices: dict[str, float] = {}
    sma_values: dict[str, float] = {}
    batch_rows: list[pd.DataFrame] = []
    batch_tickers: list[str] = []

    for ticker, df in feature_dfs.items():
        if date not in df.index:
            continue

        row = df.loc[[date]]  # Single-row DataFrame (not Series)
        prices[ticker] = row["Close"].iloc[0]

        # Recover SMA_200 from sma_distance:
        # sma_distance = (Close - SMA) / SMA → SMA = Close / (1 + sma_distance)
        sma_dist = row["sma_distance"].iloc[0]
        if pd.notna(sma_dist) and sma_dist != -1.0:
            sma_values[ticker] = prices[ticker] / (1.0 + sma_dist)

        # Accumulate rows for batch prediction.
        batch_rows.append(row)
        batch_tickers.append(ticker)

    # ------------------------------------------------------------------
    # Step 2: Batch prediction — one call for all tickers today.
    # ------------------------------------------------------------------
    probabilities: dict[str, float] = {}

    if batch_rows:
        batch_df = pd.concat(batch_rows, ignore_index=False)
        try:
            probs = predict_proba(batch_df, model, feature_list=FEATURE_COLUMNS)
            for ticker, prob in zip(batch_tickers, probs):
                probabilities[ticker] = float(prob)
        except (ValueError, Exception) as exc:
            # If batch prediction fails (e.g., unexpected NaN), fall
            # back to individual predictions so one bad ticker doesn't
            # kill the entire day.
            logger.debug(
                "%s | Batch prediction failed, falling back to individual: %s",
                date.date(), exc,
            )
            for ticker, row_df in zip(batch_tickers, batch_rows):
                try:
                    prob = predict_proba(
                        row_df, model, feature_list=FEATURE_COLUMNS,
                    )
                    probabilities[ticker] = float(prob[0])
                except Exception:
                    pass  # Skip this ticker silently

    # ------------------------------------------------------------------
    # Step 3: Check exits on open positions.
    # ------------------------------------------------------------------
    _check_exits(state, date, prices)

    # ------------------------------------------------------------------
    # Step 4: Check entries for new positions.
    # ------------------------------------------------------------------
    _check_entries(state, date, prices, probabilities, sma_values, threshold)

    # ------------------------------------------------------------------
    # Step 5: Record daily portfolio snapshot.
    # ------------------------------------------------------------------
    portfolio_value = _compute_portfolio_value(state, prices)
    invested_value = portfolio_value - state.cash

    state.daily_values.append({
        "date": date,
        "portfolio_value": portfolio_value,
        "cash": state.cash,
        "invested_value": invested_value,
    })


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable container for the complete results of a backtest run.

    Attributes
    ----------
    metrics : dict[str, float]
        The 22 performance metrics organized in 5 tiers.
    trades : pd.DataFrame
        Log of every executed trade (BUY, SELL, SELL_PARTIAL).
    equity_curve : pd.Series
        Daily portfolio value, indexed by date.
    benchmark_curve : pd.Series
        Daily S&P 500 buy-and-hold value, indexed by date.
    """

    metrics: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.Series
    benchmark_curve: pd.Series


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run a full historical backtest of the DeepValue AI strategy.

    This is the only function external code needs to call. It handles
    the entire pipeline:

        1. Load the backtest model and threshold.
        2. Download OHLCV data for all tickers + S&P 500 index.
        3. Compute technical features for each ticker.
        4. Simulate the strategy day by day.
        5. Compute the S&P 500 benchmark curve.
        6. Calculate all 22 performance metrics.
        7. Return a BacktestResult with everything packaged.

    Parameters
    ----------
    tickers : list[str] | None
        Tickers to include in the backtest universe. If None,
        downloads the full S&P 500 list from Wikipedia.
    start_date : str | None
        Start date for the simulation in "YYYY-MM-DD" format.
        If None, uses the earliest date common to all tickers.
    end_date : str | None
        End date for the simulation in "YYYY-MM-DD" format.
        If None, uses the latest date common to all tickers.
    initial_capital : float
        Starting cash (default: $100,000).

    Returns
    -------
    BacktestResult
        Complete results: metrics, trade log, equity curve, benchmark.

    Raises
    ------
    FileNotFoundError
        If the backtest model file does not exist.
        Message includes instructions to run ``make train``.
    RuntimeError
        If no valid OHLCV data could be downloaded for any ticker.
    """
    logger.info("=" * 60)
    logger.info("STARTING BACKTEST")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load the unified model (19 features).
    # ------------------------------------------------------------------
    model = load_model(path=PATHS["model_file"])
    threshold = load_threshold(path=PATHS["threshold_file"])
    logger.info("Model loaded. Threshold: %.4f", threshold)

    # ------------------------------------------------------------------
    # Step 2: Resolve the ticker universe.
    # ------------------------------------------------------------------
    if tickers is None:
        from .data_service import get_sp500_tickers
        tickers = get_sp500_tickers()

    logger.info("Backtest universe: %d tickers.", len(tickers))

    # ------------------------------------------------------------------
    # Step 3: Download OHLCV data.
    # We download the S&P 500 index and VIX separately — they're needed
    # for market_trend, VIX features, and the benchmark curve.
    # ------------------------------------------------------------------
    all_tickers = tickers + [SP500_MARKET_TICKER, VIX_TICKER]
    raw_data = download_ohlcv(all_tickers)

    if SP500_MARKET_TICKER not in raw_data:
        raise RuntimeError(
            f"Could not download S&P 500 index data ({SP500_MARKET_TICKER}). "
            "Cannot compute market_trend or benchmark curve."
        )

    market_df = raw_data.pop(SP500_MARKET_TICKER)

    vix_df = raw_data.pop(VIX_TICKER, None)
    if vix_df is None:
        logger.warning(
            "Could not download VIX (%s). VIX features will be NaN.",
            VIX_TICKER,
        )

    if not raw_data:
        raise RuntimeError(
            "No valid OHLCV data downloaded for any ticker. "
            "Check your internet connection and ticker symbols."
        )

    logger.info(
        "OHLCV data ready: %d tickers + market index.", len(raw_data),
    )

    # ------------------------------------------------------------------
    # Step 4: Compute features + merge all data sources.
    # Technical (11) + VIX (4) from OHLCV data.
    # Fundamentals (8) merged PIT from historical database.
    # Macro (6) merged PIT from FRED database.
    # Sentiment (5) merged with 1-day lag PIT.
    # ------------------------------------------------------------------
    parts: list[pd.DataFrame] = []

    for ticker, df in raw_data.items():
        featured = compute_technical_features(
            df, market_df=market_df, vix_df=vix_df,
        )
        featured = featured.dropna(subset=TECHNICAL_FEATURES)
        if not featured.empty:
            featured = featured.copy()
            featured["ticker"] = ticker
            featured["date"] = featured.index
            featured["close"] = featured["Close"]
            parts.append(featured)

    logger.info(
        "Technical + VIX features computed for %d tickers "
        "(%d dropped due to insufficient data).",
        len(parts), len(raw_data) - len(parts),
    )

    if not parts:
        raise RuntimeError(
            "No tickers have enough data after computing technical features. "
            "Try a longer download period."
        )

    # Merge all data sources point-in-time
    combined = pd.concat(parts, ignore_index=True)

    logger.info("Merging historical fundamentals (point-in-time)...")
    combined = merge_fundamentals_pit(combined)

    try:
        from .macro_database import merge_macro_pit
        logger.info("Merging FRED macro data (point-in-time)...")
        combined = merge_macro_pit(combined)
    except FileNotFoundError:
        logger.warning("Macro data not available. Macro features will be NaN.")

    try:
        from .sentiment_pipeline import merge_sentiment_pit
        logger.info("Merging sentiment scores (with 1-day lag)...")
        combined = merge_sentiment_pit(combined)
    except FileNotFoundError:
        logger.warning("Sentiment data not available. Sentiment features will be NaN.")

    # Ensure all expected feature columns exist
    import numpy as _np
    for col in FEATURE_COLUMNS:
        if col not in combined.columns:
            combined[col] = _np.nan

    # Split back into per-ticker DataFrames with DatetimeIndex
    feature_dfs: dict[str, pd.DataFrame] = {}
    for ticker, grp in combined.groupby("ticker"):
        grp = grp.copy()
        grp.index = pd.DatetimeIndex(grp["date"])
        feature_dfs[ticker] = grp

    logger.info(
        "Features ready for %d tickers (34 features per row).",
        len(feature_dfs),
    )

    if not feature_dfs:
        raise RuntimeError(
            "No tickers have enough data after computing technical features. "
            "Try a longer download period."
        )

    # ------------------------------------------------------------------
    # Step 5: Determine the simulation date range.
    # We use the intersection of all tickers' date ranges so that
    # every day in the simulation has data for ALL tickers.
    # Actually — that would be too restrictive (one ticker with a
    # short history would shrink the entire range). Instead, we use
    # the UNION: start at the earliest date any ticker has data,
    # end at the latest. Tickers without data on a given day are
    # simply skipped by _process_day.
    # ------------------------------------------------------------------
    all_dates = sorted(
        {date for df in feature_dfs.values() for date in df.index}
    )

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        all_dates = [d for d in all_dates if d.tz_localize(None) >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        all_dates = [d for d in all_dates if d.tz_localize(None) <= end_ts]

    if not all_dates:
        raise RuntimeError(
            f"No trading days in the specified range "
            f"({start_date} to {end_date})."
        )

    logger.info(
        "Simulation period: %s to %s (%d trading days).",
        all_dates[0].date(), all_dates[-1].date(), len(all_dates),
    )

    # ------------------------------------------------------------------
    # Step 6: Run the day-by-day simulation.
    # ------------------------------------------------------------------
    state = PortfolioState(cash=initial_capital)

    for i, date in enumerate(all_dates):
        _process_day(state, date, feature_dfs, model, threshold)

        # Progress logging every 250 days (~1 year).
        if (i + 1) % 250 == 0:
            current_value = state.daily_values[-1]["portfolio_value"]
            logger.info(
                "Simulation progress: day %d/%d | Portfolio: $%.0f",
                i + 1, len(all_dates), current_value,
            )

    logger.info(
        "Simulation complete. %d trades executed.", len(state.trade_log),
    )

    # ------------------------------------------------------------------
    # Step 7: Build the equity curve and benchmark curve.
    # ------------------------------------------------------------------
    equity_curve = pd.Series(
        {row["date"]: row["portfolio_value"] for row in state.daily_values},
        name="portfolio_value",
    )
    equity_curve.index.name = "date"

    # Benchmark: S&P 500 buy-and-hold with the same initial capital.
    # We compute how many "shares" of the index we could buy on day 1
    # and track the value over time.
    benchmark_curve = _build_benchmark_curve(
        market_df, all_dates, initial_capital,
    )

    # ------------------------------------------------------------------
    # Step 8: Build the trade log DataFrame.
    # ------------------------------------------------------------------
    trades = pd.DataFrame(state.trade_log)

    # ------------------------------------------------------------------
    # Step 9: Compute all 22 performance metrics.
    # ------------------------------------------------------------------
    metrics = _compute_metrics(equity_curve, benchmark_curve, trades)

    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info(
        "Total return: %.2f%% | Sharpe: %.2f | Max DD: %.2f%%",
        metrics.get("total_return", 0) * 100,
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown", 0) * 100,
    )
    logger.info("=" * 60)

    return BacktestResult(
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
    )


# ---------------------------------------------------------------------------
# Benchmark curve
# ---------------------------------------------------------------------------

def _build_benchmark_curve(
    market_df: pd.DataFrame,
    dates: list[pd.Timestamp],
    initial_capital: float,
) -> pd.Series:
    """Build a buy-and-hold equity curve for the S&P 500 index.

    Simulates investing all capital in the S&P 500 on the first day
    and holding through the entire backtest period.

    Parameters
    ----------
    market_df : pd.DataFrame
        OHLCV data for the S&P 500 (``^GSPC``).
    dates : list[pd.Timestamp]
        The trading dates from the simulation.
    initial_capital : float
        Starting capital (same as the strategy).

    Returns
    -------
    pd.Series
        Daily benchmark portfolio value, indexed by date.
    """
    # Build a lookup from calendar date → Close price (strips tz and time).
    price_lookup = {
        ts.date(): price
        for ts, price in market_df["Close"].items()
    }

    benchmark_values = {}
    shares = None

    for date in dates:
        cal_date = date.date() if hasattr(date, "date") else date
        if cal_date not in price_lookup:
            continue

        price = price_lookup[cal_date]

        # Buy on the first available day.
        if shares is None:
            shares = initial_capital / price

        # Store with ORIGINAL timestamp (same index as equity_curve).
        benchmark_values[date] = shares * price

    if not benchmark_values:
        # Fallback: return zeros so metrics don't crash.
        return pd.Series(0.0, index=dates, name="benchmark_value")

    result = pd.Series(benchmark_values, name="benchmark_value")
    result.index.name = "date"

    # Forward-fill to cover dates where market data might be missing.
    result = result.reindex(dates, method="ffill")

    return result

# ---------------------------------------------------------------------------
# Performance metrics (22 metrics, 5 tiers)
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR = 252
_RISK_FREE_RATE = 0.0  # Simplification: we use 0% for Sharpe/Sortino


def _compute_metrics(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    trades: pd.DataFrame,
) -> dict[str, float]:
    """Calculate all 22 performance metrics from backtest results.

    Metrics are organized in 5 tiers:

        Tier 1 — Return (5): How much money did we make?
        Tier 2 — Risk (5): How much pain did we endure?
        Tier 3 — Risk-adjusted (5): Was the pain worth the gain?
        Tier 4 — Trade quality (4): How good were individual trades?
        Tier 5 — Consistency (3): Was performance steady or erratic?

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.
    benchmark_curve : pd.Series
        Daily S&P 500 buy-and-hold value indexed by date.
    trades : pd.DataFrame
        Log of every trade with columns: ticker, action, date, price,
        shares, reason, value, return_pct.

    Returns
    -------
    dict[str, float]
        All 22 metrics. ``monthly_returns`` is a nested dict
        (``{str: float}``), all others are floats.
    """
    metrics: dict[str, float] = {}

    # Daily returns of the strategy (percentage change day to day).
    daily_returns = equity_curve.pct_change().dropna()

    # ------------------------------------------------------------------
    # TIER 1: Return metrics — "How much money did we make?"
    # ------------------------------------------------------------------
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    # 1. Total Return — simple (end - start) / start.
    #    A $100K portfolio that ends at $142K has total_return = 0.42.
    metrics["total_return"] = (final_value - initial_value) / initial_value

    # 2. Annualized Return — what's the equivalent yearly rate?
    #    Uses compound annual growth rate (CAGR) formula:
    #    CAGR = (final / initial)^(252 / trading_days) - 1
    #
    #    Why 252? That's the standard number of US trading days per year.
    #    Using calendar days would understate returns because weekends
    #    and holidays don't generate returns but DO add to the denominator.
    n_days = len(equity_curve)
    if n_days > 1:
        metrics["annualized_return"] = (
            (final_value / initial_value)
            ** (_TRADING_DAYS_PER_YEAR / (n_days - 1))
            - 1.0
        )
    else:
        metrics["annualized_return"] = 0.0

    # 3. ROI on Invested Capital — return relative to what we actually
    #    deployed, not just the initial capital sitting in cash.
    #    If we only ever invested 60% of our capital, ROI on invested
    #    tells us how well THAT 60% performed.
    #
    #    Sum of all buy values = total capital deployed over time.
    #    Sum of all sell values = total capital recovered.
    #    Plus current open position value at the end.
    if not trades.empty:
        total_bought = trades.loc[
            trades["action"] == "BUY", "value"
        ].sum()
        if total_bought > 0:
            net_profit = final_value - initial_value
            metrics["roi_on_invested"] = net_profit / total_bought
        else:
            metrics["roi_on_invested"] = 0.0
    else:
        metrics["roi_on_invested"] = 0.0

    # 4. Benchmark Return — S&P 500 buy-and-hold over the same period.
    bench_initial = benchmark_curve.iloc[0]
    bench_final = benchmark_curve.iloc[-1]
    metrics["benchmark_return"] = (
        (bench_final - bench_initial) / bench_initial
        if bench_initial > 0 else 0.0
    )

    # 5. Alpha — strategy return minus benchmark return.
    #    Positive alpha = we beat the market. Negative = we didn't.
    #    This is a simplified alpha (not Jensen's alpha which adjusts
    #    for beta). For a TFM, this is the clearest way to show
    #    whether the strategy adds value over passive investing.
    metrics["alpha"] = metrics["total_return"] - metrics["benchmark_return"]

    # ------------------------------------------------------------------
    # TIER 2: Risk metrics — "How much pain did we endure?"
    # ------------------------------------------------------------------

    # 6. Max Drawdown — worst peak-to-trough decline.
    #    If the portfolio went from $150K to $120K at its worst,
    #    max_drawdown = (150K - 120K) / 150K = 0.20 (20%).
    #    This is the single most important risk metric — it tells you
    #    the worst pain an investor would have actually felt.
    cumulative_max = equity_curve.cummax()
    drawdown_series = (equity_curve - cumulative_max) / cumulative_max
    metrics["max_drawdown"] = abs(drawdown_series.min())

    # 7. Max Drawdown Duration — how many trading days was the
    #    portfolio below its previous peak? Long drawdowns are
    #    psychologically devastating even if the depth is small.
    is_in_drawdown = equity_curve < cumulative_max
    if is_in_drawdown.any():
        # Find consecutive runs of True (in drawdown).
        # We create groups that increment when drawdown status changes.
        drawdown_groups = (~is_in_drawdown).cumsum()
        # For each group that IS in drawdown, count the days.
        drawdown_lengths = (
            is_in_drawdown
            .groupby(drawdown_groups)
            .sum()  # count of True values per group
        )
        metrics["max_drawdown_duration"] = float(drawdown_lengths.max())
    else:
        metrics["max_drawdown_duration"] = 0.0

    # 8. Volatility — annualized standard deviation of daily returns.
    #    Higher = more unpredictable daily swings.
    #    A volatility of 0.20 means daily returns have a std dev
    #    equivalent to 20% per year.
    metrics["volatility"] = (
        daily_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
    )

    # 9. Value at Risk (VaR) at 95% confidence — the worst daily
    #    loss you'd expect on 95% of days.
    #    "On a normal day, you won't lose more than X%."
    #    We use historical VaR (percentile method), not parametric,
    #    because return distributions are NOT normal (fat tails).
    metrics["value_at_risk"] = abs(float(np.percentile(daily_returns, 5)))

    # 10. Conditional VaR (CVaR / Expected Shortfall) — average loss
    #     on the WORST 5% of days. VaR says "you won't lose more than
    #     X on 95% of days", CVaR says "on the worst 5% of days, you
    #     lose Y on average". CVaR is always >= VaR.
    var_threshold = np.percentile(daily_returns, 5)
    tail_losses = daily_returns[daily_returns <= var_threshold]
    metrics["conditional_var"] = (
        abs(float(tail_losses.mean())) if len(tail_losses) > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # TIER 3: Risk-adjusted metrics — "Was the pain worth the gain?"
    # ------------------------------------------------------------------

    # 11. Sharpe Ratio — excess return per unit of total risk.
    #     Sharpe = (annualized_return - risk_free_rate) / volatility
    #     > 1.0 = good, > 2.0 = very good, > 3.0 = exceptional.
    #     We use risk_free = 0% for simplicity; in practice you'd
    #     subtract the Treasury rate.
    if metrics["volatility"] > 0:
        metrics["sharpe_ratio"] = (
            (metrics["annualized_return"] - _RISK_FREE_RATE)
            / metrics["volatility"]
        )
    else:
        metrics["sharpe_ratio"] = 0.0

    # 12. Sortino Ratio — like Sharpe but only penalizes DOWNSIDE
    #     volatility. Upside volatility is good (big wins), so why
    #     penalize it? Sortino only uses negative returns in the
    #     denominator.
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
        if len(downside_returns) > 0 else 0.0
    )
    if downside_std > 0:
        metrics["sortino_ratio"] = (
            (metrics["annualized_return"] - _RISK_FREE_RATE) / downside_std
        )
    else:
        metrics["sortino_ratio"] = 0.0

    # 13. Calmar Ratio — annualized return / max drawdown.
    #     Measures return per unit of worst-case pain.
    #     > 1.0 = decent, > 3.0 = excellent.
    if metrics["max_drawdown"] > 0:
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / metrics["max_drawdown"]
        )
    else:
        metrics["calmar_ratio"] = 0.0

    # 14. Omega Ratio — probability-weighted gains / losses relative
    #     to a threshold (we use 0%). Unlike Sharpe, Omega considers
    #     the ENTIRE distribution shape, not just mean and variance.
    #
    #     Omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))
    #     > 1.0 = gains outweigh losses.
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    metrics["omega_ratio"] = gains / losses if losses > 0 else 0.0

    # 15. Recovery Factor — total return / max drawdown.
    #     "How many times over did the strategy recover its worst loss?"
    #     A recovery factor of 3 means total gains were 3x the worst dip.
    if metrics["max_drawdown"] > 0:
        metrics["recovery_factor"] = (
            metrics["total_return"] / metrics["max_drawdown"]
        )
    else:
        metrics["recovery_factor"] = 0.0

    # ------------------------------------------------------------------
    # TIER 4: Trade quality — "How good were individual trades?"
    # ------------------------------------------------------------------

    # Extract completed trades (sells only — buys have return_pct = 0).
    sell_trades = trades.loc[
        trades["action"].isin(["SELL", "SELL_PARTIAL"])
    ].copy() if not trades.empty else pd.DataFrame()

    # 16. Win Rate — percentage of sells that were profitable.
    if len(sell_trades) > 0:
        wins = sell_trades["return_pct"] > 0
        metrics["win_rate"] = wins.sum() / len(sell_trades)
    else:
        metrics["win_rate"] = 0.0

    # 17. Profit Factor — gross profit / gross loss.
    #     Sum of all winning trade values / sum of all losing trade values.
    #     > 1.0 = profitable system. > 2.0 = very good.
    if len(sell_trades) > 0:
        gross_wins = (
            sell_trades.loc[sell_trades["return_pct"] > 0, "return_pct"].sum()
        )
        gross_losses = abs(
            sell_trades.loc[sell_trades["return_pct"] <= 0, "return_pct"].sum()
        )
        metrics["profit_factor"] = (
            gross_wins / gross_losses if gross_losses > 0 else 0.0
        )
    else:
        metrics["profit_factor"] = 0.0

    # 18. Average Win vs Average Loss — ratio of mean winning return
    #     to mean losing return. Tells you if your wins are bigger
    #     than your losses on average.
    #     Combined with win_rate, this gives the full picture:
    #     you can have low win_rate but still be profitable if
    #     avg_win >> avg_loss (trend following style).
    if len(sell_trades) > 0:
        avg_win = sell_trades.loc[
            sell_trades["return_pct"] > 0, "return_pct"
        ].mean()
        avg_loss = abs(sell_trades.loc[
            sell_trades["return_pct"] <= 0, "return_pct"
        ].mean())
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss > 0:
            metrics["avg_win_vs_avg_loss"] = avg_win / avg_loss
        else:
            metrics["avg_win_vs_avg_loss"] = 0.0
    else:
        metrics["avg_win_vs_avg_loss"] = 0.0

    # 19. Number of Trades — total executed trades (buys + sells).
    metrics["num_trades"] = len(trades) if not trades.empty else 0

    # ------------------------------------------------------------------
    # TIER 5: Consistency — "Was performance steady or erratic?"
    # ------------------------------------------------------------------

    # 20. Monthly Returns — return for each calendar month.
    #     Stored as a dict: {"2020-01": 0.03, "2020-02": -0.01, ...}
    #     This is the only non-scalar metric — used for visualization.
    monthly_equity = equity_curve.resample("ME").last()
    monthly_returns = monthly_equity.pct_change().dropna()
    metrics["monthly_returns"] = {
        str(date.date()): round(ret, 6)
        for date, ret in monthly_returns.items()
    }

    # 21. Positive Months Percentage — what fraction of months were
    #     positive? A strategy with 70%+ positive months feels much
    #     more comfortable to hold than one with 50%.
    if len(monthly_returns) > 0:
        metrics["positive_months_pct"] = (
            (monthly_returns > 0).sum() / len(monthly_returns)
        )
    else:
        metrics["positive_months_pct"] = 0.0

    # 22. Ulcer Index — measures the depth AND duration of drawdowns.
    #     Unlike max_drawdown (which is a single worst point), Ulcer
    #     Index captures the CUMULATIVE pain over the entire period.
    #
    #     Formula: sqrt(mean(drawdown_pct^2))
    #     Lower is better. < 5% is good, > 10% is painful.
    #
    #     Named because deep, prolonged drawdowns give investors ulcers.
    drawdown_pct = drawdown_series * 100  # Convert to percentage
    metrics["ulcer_index"] = float(
        np.sqrt((drawdown_pct ** 2).mean())
    )

    return metrics