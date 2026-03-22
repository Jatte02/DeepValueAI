"""Tests for core.screener_engine — signal metadata, ticker analysis, scan."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.config import (
    FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURES,
    SMA_BUY_CEILING,
    TECHNICAL_FEATURES,
)
from core.screener_engine import (
    _analyze_ticker,
    _compute_signal_metadata,
    scan_sp500,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal sklearn-compatible model for testing."""

    def __init__(self, fixed_proba: float = 0.80):
        self.fixed_proba = fixed_proba

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        col0 = np.full(n, 1 - self.fixed_proba)
        col1 = np.full(n, self.fixed_proba)
        return np.column_stack([col0, col1])

def _make_featured_df(n: int = 60, sma_200: float = 100.0, close: float = 95.0):
    """Build a minimal DataFrame with all columns the screener expects."""
    dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
    rng = np.random.default_rng(7)

    data = {
        "Open": close + rng.normal(0, 0.5, n),
        "High": close + rng.uniform(0.5, 2, n),
        "Low": close - rng.uniform(0.5, 2, n),
        "Close": np.full(n, close),
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "sma_200": np.full(n, sma_200),
    }
    # Add all technical features
    for feat in TECHNICAL_FEATURES:
        if feat not in data:
            data[feat] = rng.random(n)
    # Add all fundamental features
    for feat in FUNDAMENTAL_FEATURES:
        data[feat] = rng.random(n)

    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# _compute_signal_metadata
# ---------------------------------------------------------------------------

class TestComputeSignalMetadata:
    def test_returns_expected_keys(self):
        df = _make_featured_df()
        model = FakeModel(0.80)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)

        expected = {"probability", "signal_strength", "signal_freshness_days", "sma_headroom_pct"}
        assert set(result.keys()) == expected

    def test_probability_matches_model(self):
        df = _make_featured_df()
        model = FakeModel(0.75)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        assert result["probability"] == pytest.approx(0.75)

    def test_signal_strength_above_threshold(self):
        df = _make_featured_df()
        model = FakeModel(0.80)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        # strength = (0.80 - 0.50) / (1 - 0.50) = 0.60
        assert result["signal_strength"] == pytest.approx(0.60, abs=0.01)

    def test_signal_strength_zero_below_threshold(self):
        df = _make_featured_df()
        model = FakeModel(0.30)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        assert result["signal_strength"] == 0.0

    def test_freshness_all_buy(self):
        """Model always predicts 0.80 > 0.50 threshold → freshness = lookback."""
        df = _make_featured_df(n=60)
        model = FakeModel(0.80)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        # All 30 (FRESHNESS_LOOKBACK_DAYS) rows should be buy signals
        assert result["signal_freshness_days"] == 30

    def test_freshness_zero_when_no_buy(self):
        df = _make_featured_df()
        model = FakeModel(0.30)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        assert result["signal_freshness_days"] == 0

    def test_sma_headroom_positive_when_below_ceiling(self):
        # close=95, sma=100, ceiling=105
        df = _make_featured_df(close=95.0, sma_200=100.0)
        model = FakeModel(0.80)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        assert result["sma_headroom_pct"] > 0

    def test_sma_headroom_negative_when_above_ceiling(self):
        # close=110, sma=100, ceiling=105
        df = _make_featured_df(close=110.0, sma_200=100.0)
        model = FakeModel(0.80)
        result = _compute_signal_metadata(df, model, threshold=0.50, feature_list=FEATURE_COLUMNS)
        assert result["sma_headroom_pct"] < 0


# ---------------------------------------------------------------------------
# _analyze_ticker
# ---------------------------------------------------------------------------

class TestAnalyzeTicker:
    @patch("core.screener_engine.build_feature_row")
    def test_returns_dict_on_success(self, mock_build):
        df = _make_featured_df()
        mock_build.return_value = df
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "AAPL", df, market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert isinstance(result, dict)
        assert result["ticker"] == "AAPL"
        assert "probability" in result
        assert "passes_filters" in result

    @patch("core.screener_engine.build_feature_row")
    def test_returns_none_on_feature_error(self, mock_build):
        mock_build.side_effect = Exception("feature error")
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "BAD", pd.DataFrame(), market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert result is None

    @patch("core.screener_engine.build_feature_row")
    def test_returns_none_on_empty_df(self, mock_build):
        mock_build.return_value = pd.DataFrame()
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "EMPTY", pd.DataFrame(), market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert result is None

    @patch("core.screener_engine.build_feature_row")
    def test_passes_filters_when_conditions_met(self, mock_build):
        # close=95, sma=100, prob=0.80, threshold=0.50
        df = _make_featured_df(close=95.0, sma_200=100.0)
        mock_build.return_value = df
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "AAPL", df, market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert result["passes_filters"] == True

    @patch("core.screener_engine.build_feature_row")
    def test_fails_filters_when_price_above_sma_ceiling(self, mock_build):
        # close=110, sma=100, ceiling = 105 → fails SMA filter
        df = _make_featured_df(close=110.0, sma_200=100.0)
        mock_build.return_value = df
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "AAPL", df, market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert result["passes_filters"] == False

    @patch("core.screener_engine.build_feature_row")
    def test_fails_filters_when_below_threshold(self, mock_build):
        df = _make_featured_df(close=95.0, sma_200=100.0)
        mock_build.return_value = df
        model = FakeModel(0.30)

        result = _analyze_ticker(
            "AAPL", df, market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        assert result["passes_filters"] is False

    @patch("core.screener_engine.build_feature_row")
    def test_fundamentals_in_result(self, mock_build):
        df = _make_featured_df()
        mock_build.return_value = df
        model = FakeModel(0.80)

        result = _analyze_ticker(
            "AAPL", df, market_df=None,
            model=model, threshold=0.50, feature_list=FEATURE_COLUMNS,
        )
        for feat in FUNDAMENTAL_FEATURES:
            assert feat in result


# ---------------------------------------------------------------------------
# scan_sp500
# ---------------------------------------------------------------------------

class TestScanSP500:
    @patch("core.screener_engine.download_ohlcv")
    @patch("core.screener_engine.get_sp500_tickers")
    @patch("core.screener_engine.load_model")
    @patch("core.screener_engine.load_threshold")
    @patch("core.screener_engine.build_feature_row")
    def test_returns_dataframe(
        self, mock_build, mock_threshold, mock_model,
        mock_tickers, mock_download,
    ):
        mock_model.return_value = FakeModel(0.80)
        mock_threshold.return_value = 0.50

        featured_df = _make_featured_df()

        mock_build.return_value = featured_df
        mock_download.return_value = {
            "AAPL": featured_df,
            "^GSPC": featured_df,
        }

        result = scan_sp500(tickers=["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    @patch("core.screener_engine.download_ohlcv")
    @patch("core.screener_engine.load_model")
    @patch("core.screener_engine.load_threshold")
    @patch("core.screener_engine.build_feature_row")
    def test_sorted_by_probability(
        self, mock_build, mock_threshold, mock_model, mock_download,
    ):
        mock_model.return_value = FakeModel(0.80)
        mock_threshold.return_value = 0.50

        featured_df = _make_featured_df()
        mock_build.return_value = featured_df
        mock_download.return_value = {
            "AAPL": featured_df,
            "MSFT": featured_df,
            "^GSPC": featured_df,
        }

        result = scan_sp500(tickers=["AAPL", "MSFT"])
        if len(result) >= 2:
            probs = result["probability"].tolist()
            assert probs == sorted(probs, reverse=True)

    @patch("core.screener_engine.download_ohlcv")
    @patch("core.screener_engine.load_model")
    @patch("core.screener_engine.load_threshold")
    def test_empty_when_no_data(self, mock_threshold, mock_model, mock_download):
        mock_model.return_value = FakeModel(0.80)
        mock_threshold.return_value = 0.50
        mock_download.return_value = {"^GSPC": pd.DataFrame()}

        result = scan_sp500(tickers=["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
