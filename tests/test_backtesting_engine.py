"""Tests for core.backtesting_engine — positions, exits, entries, metrics."""

import numpy as np
import pandas as pd
import pytest

from core.backtesting_engine import (
    BacktestResult,
    Position,
    PortfolioState,
    _build_benchmark_curve,
    _check_entries,
    _check_exits,
    _compute_metrics,
    _compute_portfolio_value,
)
from core.config import (
    COOLDOWN_DAYS_PER_TICKER,
    MAX_HOLDING_DAYS,
    MAX_OPEN_POSITIONS,
    MAX_TICKER_EXPOSURE_PCT,
    PARTIAL_TP_SELL_FRACTION,
    PARTIAL_TP_TRIGGER_PCT,
    POSITION_SIZE_PCT,
    SMA_BUY_CEILING,
    STOP_LOSS_PCT,
    TRAILING_STOP_ACTIVATION_PCT,
    TRAILING_STOP_PCT,
)


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

class TestPosition:
    def test_default_highest_price_equals_entry(self):
        pos = Position(
            ticker="AAPL", entry_price=150.0, shares=10.0,
            entry_date=pd.Timestamp("2023-01-01"),
        )
        assert pos.highest_price == 150.0

    def test_default_fields(self):
        pos = Position(
            ticker="AAPL", entry_price=100.0, shares=5.0,
            entry_date=pd.Timestamp("2023-01-01"),
        )
        assert pos.days_held == 0
        assert pos.trailing_stop_active is False
        assert pos.partial_tp_taken is False

    def test_explicit_highest_price(self):
        pos = Position(
            ticker="AAPL", entry_price=100.0, shares=5.0,
            entry_date=pd.Timestamp("2023-01-01"),
            highest_price=120.0,
        )
        assert pos.highest_price == 120.0


# ---------------------------------------------------------------------------
# PortfolioState
# ---------------------------------------------------------------------------

class TestPortfolioState:
    def test_initial_state(self):
        state = PortfolioState(cash=100_000.0)
        assert state.cash == 100_000.0
        assert state.positions == []
        assert state.trade_log == []
        assert state.daily_values == []
        assert state.last_buy_date == {}


# ---------------------------------------------------------------------------
# _compute_portfolio_value
# ---------------------------------------------------------------------------

class TestComputePortfolioValue:
    def test_cash_only(self):
        state = PortfolioState(cash=50_000.0)
        assert _compute_portfolio_value(state, {}) == 50_000.0

    def test_cash_plus_positions(self):
        pos = Position(
            ticker="AAPL", entry_price=100.0, shares=10.0,
            entry_date=pd.Timestamp("2023-01-01"),
        )
        state = PortfolioState(cash=5_000.0, positions=[pos])
        value = _compute_portfolio_value(state, {"AAPL": 150.0})
        assert value == pytest.approx(5_000.0 + 10 * 150.0)

    def test_missing_price_uses_entry_price(self):
        pos = Position(
            ticker="AAPL", entry_price=100.0, shares=10.0,
            entry_date=pd.Timestamp("2023-01-01"),
        )
        state = PortfolioState(cash=5_000.0, positions=[pos])
        value = _compute_portfolio_value(state, {})
        assert value == pytest.approx(5_000.0 + 10 * 100.0)


# ---------------------------------------------------------------------------
# _check_exits
# ---------------------------------------------------------------------------

class TestCheckExits:
    def _make_state_with_position(
        self, entry_price=100.0, shares=10.0, days_held=0,
        trailing_active=False, partial_taken=False, highest_price=None,
    ) -> PortfolioState:
        pos = Position(
            ticker="AAPL",
            entry_price=entry_price,
            shares=shares,
            entry_date=pd.Timestamp("2023-01-01"),
            days_held=days_held,
            trailing_stop_active=trailing_active,
            partial_tp_taken=partial_taken,
            highest_price=highest_price or entry_price,
        )
        return PortfolioState(cash=50_000.0, positions=[pos])

    def test_stop_loss_triggers(self):
        """Price drops >12% from entry → sell all."""
        state = self._make_state_with_position(entry_price=100.0)
        drop_price = 100.0 * (1 - STOP_LOSS_PCT - 0.01)  # just past stop loss
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": drop_price})

        assert len(state.positions) == 0
        assert len(state.trade_log) == 1
        assert state.trade_log[0]["reason"] == "stop_loss"
        assert state.cash > 50_000.0  # got cash back

    def test_stop_loss_not_triggered_above(self):
        """Price drops 10% (within stop loss) → keep position."""
        state = self._make_state_with_position(entry_price=100.0)
        price = 100.0 * (1 - STOP_LOSS_PCT + 0.02)  # within threshold
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": price})

        assert len(state.positions) == 1
        assert len(state.trade_log) == 0

    def test_trailing_stop_triggers(self):
        """Price rose then fell 8% from peak → sell all."""
        state = self._make_state_with_position(
            entry_price=100.0,
            trailing_active=True,
            highest_price=130.0,
        )
        # 8%+ below peak of 130
        drop_price = 130.0 * (1 - TRAILING_STOP_PCT - 0.01)
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": drop_price})

        assert len(state.positions) == 0
        assert state.trade_log[0]["reason"] == "trailing_stop"

    def test_trailing_stop_not_triggered_when_inactive(self):
        """Trailing stop shouldn't fire if not yet activated.

        Use a price that is below the peak (so trailing stop *would*
        fire if active) but NOT high enough relative to entry to
        activate the trailing stop in the same tick.
        """
        state = self._make_state_with_position(
            entry_price=100.0,
            trailing_active=False,
            highest_price=105.0,
        )
        # Price = 96: below highest_price (would trigger trailing if active)
        # but gain_from_entry = -4% < TRAILING_STOP_ACTIVATION_PCT so it
        # doesn't activate, and -4% < STOP_LOSS_PCT (12%) so no stop loss.
        price = 96.0
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": price})

        assert len(state.positions) == 1

    def test_time_limit_triggers(self):
        """Position held for MAX_HOLDING_DAYS → sell all."""
        state = self._make_state_with_position(
            entry_price=100.0,
            days_held=MAX_HOLDING_DAYS - 1,  # will be incremented to MAX
        )
        date = pd.Timestamp("2023-06-01")

        _check_exits(state, date, {"AAPL": 105.0})

        assert len(state.positions) == 0
        assert state.trade_log[0]["reason"] == "time_limit"

    def test_partial_tp_triggers(self):
        """Price rose 15%+ from entry → sell 33% of shares."""
        state = self._make_state_with_position(entry_price=100.0, shares=100.0)
        price = 100.0 * (1 + PARTIAL_TP_TRIGGER_PCT + 0.01)
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": price})

        assert len(state.positions) == 1  # position survives
        remaining = state.positions[0].shares
        expected_remaining = 100.0 * (1 - PARTIAL_TP_SELL_FRACTION)
        assert remaining == pytest.approx(expected_remaining)
        assert state.positions[0].partial_tp_taken is True
        assert state.trade_log[0]["reason"] == "partial_tp"

    def test_partial_tp_only_fires_once(self):
        """After partial TP already taken, don't fire again."""
        state = self._make_state_with_position(
            entry_price=100.0, shares=67.0, partial_taken=True,
        )
        price = 100.0 * (1 + PARTIAL_TP_TRIGGER_PCT + 0.05)
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": price})

        assert len(state.trade_log) == 0  # no new trade

    def test_missing_price_keeps_position(self):
        """If no price data for ticker today, keep it unchanged."""
        state = self._make_state_with_position()
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {})  # no AAPL in prices

        assert len(state.positions) == 1
        assert len(state.trade_log) == 0

    def test_trailing_stop_activates(self):
        """Price rises 5%+ → trailing stop should activate."""
        state = self._make_state_with_position(entry_price=100.0)
        price = 100.0 * (1 + TRAILING_STOP_ACTIVATION_PCT + 0.01)
        date = pd.Timestamp("2023-02-01")

        _check_exits(state, date, {"AAPL": price})

        assert state.positions[0].trailing_stop_active is True


# ---------------------------------------------------------------------------
# _check_entries
# ---------------------------------------------------------------------------

class TestCheckEntries:
    def _make_entry_args(
        self,
        cash=100_000.0,
        positions=None,
        prob=0.80,
        threshold=0.50,
        price=95.0,
        sma=100.0,
    ):
        state = PortfolioState(cash=cash, positions=positions or [])
        date = pd.Timestamp("2023-03-01")
        prices = {"AAPL": price}
        probabilities = {"AAPL": prob}
        sma_values = {"AAPL": sma}
        return state, date, prices, probabilities, sma_values, threshold

    def test_buy_when_all_conditions_met(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args()
        _check_entries(state, date, prices, probs, smas, threshold)

        assert len(state.positions) == 1
        assert state.positions[0].ticker == "AAPL"
        assert len(state.trade_log) == 1
        assert state.trade_log[0]["action"] == "BUY"

    def test_no_buy_below_threshold(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            prob=0.30, threshold=0.50,
        )
        _check_entries(state, date, prices, probs, smas, threshold)
        assert len(state.positions) == 0

    def test_no_buy_above_sma_ceiling(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            price=110.0, sma=100.0,  # price > sma * 1.05
        )
        _check_entries(state, date, prices, probs, smas, threshold)
        assert len(state.positions) == 0

    def test_no_buy_max_positions_reached(self):
        # Fill up to MAX_OPEN_POSITIONS with different tickers
        existing = [
            Position(
                ticker=f"T{i}", entry_price=100.0, shares=10.0,
                entry_date=pd.Timestamp("2023-01-01"),
            )
            for i in range(MAX_OPEN_POSITIONS)
        ]
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            positions=existing,
        )
        _check_entries(state, date, prices, probs, smas, threshold)
        # Should not have added AAPL
        assert not any(p.ticker == "AAPL" for p in state.positions)

    def test_no_buy_insufficient_cash(self):
        # Start with a meaningful portfolio (existing position worth ~$100k)
        # but only $100 cash. Target buy = 5% of ~$100k = $5k > $100.
        existing = Position(
            ticker="MSFT", entry_price=200.0, shares=500.0,
            entry_date=pd.Timestamp("2023-01-01"),
        )
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            cash=100.0, positions=[existing],
        )
        prices = {"AAPL": 95.0, "MSFT": 200.0}
        _check_entries(state, date, prices, probs, smas, threshold)
        # Should not have bought AAPL — $100 cash < $5k target buy
        assert not any(p.ticker == "AAPL" for p in state.positions)

    def test_cooldown_prevents_buy(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args()
        # Set last buy date to yesterday (within cooldown)
        state.last_buy_date["AAPL"] = pd.Timestamp("2023-02-28")
        _check_entries(state, date, prices, probs, smas, threshold)
        assert len(state.positions) == 0

    def test_cooldown_expired_allows_buy(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args()
        # Set last buy date far in the past
        state.last_buy_date["AAPL"] = pd.Timestamp("2022-01-01")
        _check_entries(state, date, prices, probs, smas, threshold)
        assert len(state.positions) == 1

    def test_position_size_is_pct_of_portfolio(self):
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            cash=100_000.0, price=100.0,
        )
        _check_entries(state, date, prices, probs, smas, threshold)

        expected_value = 100_000.0 * POSITION_SIZE_PCT
        expected_shares = expected_value / 100.0
        assert state.positions[0].shares == pytest.approx(expected_shares)

    def test_ticker_exposure_limit(self):
        """Can't add more of AAPL if it would exceed MAX_TICKER_EXPOSURE_PCT."""
        # Create existing AAPL position worth ~9% of portfolio
        existing = Position(
            ticker="AAPL", entry_price=95.0,
            shares=(100_000 * MAX_TICKER_EXPOSURE_PCT - 100) / 95.0,
            entry_date=pd.Timestamp("2022-01-01"),
        )
        state, date, prices, probs, smas, threshold = self._make_entry_args(
            positions=[existing], cash=50_000.0,
        )
        # last buy date must be far away to not be blocked by cooldown
        state.last_buy_date["AAPL"] = pd.Timestamp("2020-01-01")
        _check_entries(state, date, prices, probs, smas, threshold)
        # Should not have added new position (exposure would exceed limit)
        assert len(state.positions) == 1


# ---------------------------------------------------------------------------
# _build_benchmark_curve
# ---------------------------------------------------------------------------

class TestBuildBenchmarkCurve:
    def test_returns_series(self):
        dates = pd.bdate_range("2023-01-02", periods=100, freq="B")
        market_df = pd.DataFrame(
            {"Close": np.linspace(4000, 4200, 100)},
            index=dates,
        )
        result = _build_benchmark_curve(market_df, list(dates), 100_000.0)
        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_first_day_equals_initial_capital(self):
        dates = pd.bdate_range("2023-01-02", periods=50, freq="B")
        market_df = pd.DataFrame(
            {"Close": np.linspace(4000, 4200, 50)},
            index=dates,
        )
        result = _build_benchmark_curve(market_df, list(dates), 100_000.0)
        assert result.iloc[0] == pytest.approx(100_000.0)

    def test_grows_with_market(self):
        dates = pd.bdate_range("2023-01-02", periods=50, freq="B")
        prices = np.linspace(100, 150, 50)  # 50% growth
        market_df = pd.DataFrame({"Close": prices}, index=dates)
        result = _build_benchmark_curve(market_df, list(dates), 100_000.0)
        # Last value should be ~150,000
        assert result.iloc[-1] == pytest.approx(150_000.0)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    @pytest.fixture()
    def simple_equity_and_benchmark(self):
        """Equity curve: 100k → 120k over 252 days. Benchmark: 100k → 110k."""
        dates = pd.bdate_range("2023-01-02", periods=252, freq="B")
        equity = pd.Series(
            np.linspace(100_000, 120_000, 252),
            index=dates,
            name="portfolio_value",
        )
        benchmark = pd.Series(
            np.linspace(100_000, 110_000, 252),
            index=dates,
            name="benchmark_value",
        )
        return equity, benchmark

    @pytest.fixture()
    def simple_trades(self):
        return pd.DataFrame([
            {"ticker": "AAPL", "action": "BUY", "date": pd.Timestamp("2023-01-03"),
             "price": 150.0, "shares": 33.0, "reason": "signal", "value": 5000, "return_pct": 0.0},
            {"ticker": "AAPL", "action": "SELL", "date": pd.Timestamp("2023-06-01"),
             "price": 170.0, "shares": 33.0, "reason": "trailing_stop", "value": 5610, "return_pct": 0.133},
            {"ticker": "MSFT", "action": "BUY", "date": pd.Timestamp("2023-02-01"),
             "price": 250.0, "shares": 20.0, "reason": "signal", "value": 5000, "return_pct": 0.0},
            {"ticker": "MSFT", "action": "SELL", "date": pd.Timestamp("2023-07-01"),
             "price": 240.0, "shares": 20.0, "reason": "stop_loss", "value": 4800, "return_pct": -0.04},
        ])

    def test_returns_dict(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert isinstance(metrics, dict)

    def test_total_return(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert metrics["total_return"] == pytest.approx(0.20, rel=0.01)

    def test_benchmark_return(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert metrics["benchmark_return"] == pytest.approx(0.10, rel=0.01)

    def test_alpha_positive(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert metrics["alpha"] > 0

    def test_max_drawdown_zero_for_linear_growth(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        # Linear growth = no drawdown
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_win_rate(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        # 1 winner (AAPL) out of 2 sells
        assert metrics["win_rate"] == pytest.approx(0.50)

    def test_num_trades(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert metrics["num_trades"] == 4  # 2 buys + 2 sells

    def test_sharpe_ratio_positive_for_growth(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert metrics["sharpe_ratio"] > 0

    def test_monthly_returns_is_dict(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        assert isinstance(metrics["monthly_returns"], dict)

    def test_all_22_metrics_present(self, simple_equity_and_benchmark, simple_trades):
        equity, benchmark = simple_equity_and_benchmark
        metrics = _compute_metrics(equity, benchmark, simple_trades)
        expected_keys = {
            "total_return", "annualized_return", "roi_on_invested",
            "benchmark_return", "alpha",
            "max_drawdown", "max_drawdown_duration", "volatility",
            "value_at_risk", "conditional_var",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "omega_ratio", "recovery_factor",
            "win_rate", "profit_factor", "avg_win_vs_avg_loss", "num_trades",
            "monthly_returns", "positive_months_pct", "ulcer_index",
        }
        assert expected_keys == set(metrics.keys())

    def test_empty_trades(self, simple_equity_and_benchmark):
        equity, benchmark = simple_equity_and_benchmark
        empty_trades = pd.DataFrame()
        metrics = _compute_metrics(equity, benchmark, empty_trades)
        assert metrics["win_rate"] == 0.0
        assert metrics["num_trades"] == 0


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_dataclass_fields(self):
        result = BacktestResult(
            metrics={"total_return": 0.10},
            trades=pd.DataFrame(),
            equity_curve=pd.Series(dtype=float),
            benchmark_curve=pd.Series(dtype=float),
        )
        assert result.metrics["total_return"] == 0.10
