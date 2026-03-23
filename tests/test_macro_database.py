"""Tests for core.macro_database — FRED data download, PIT merge."""

import numpy as np
import pandas as pd
import pytest

from core.macro_database import _build_macro_features, merge_macro_pit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fred_data():
    """Simulated raw FRED series data for testing."""
    dates_daily = pd.date_range("2020-01-01", periods=100, freq="B")
    dates_monthly = pd.date_range("2020-01-01", periods=10, freq="MS")
    dates_quarterly = pd.date_range("2020-01-01", periods=4, freq="QS")

    return {
        "DFF": pd.DataFrame({
            "date": dates_daily,
            "realtime_start": dates_daily,
            "value": np.linspace(0.25, 2.50, 100),
        }),
        "UNRATE": pd.DataFrame({
            "date": dates_monthly,
            "realtime_start": dates_monthly + pd.Timedelta(days=5),
            "value": np.linspace(3.5, 4.5, 10),
        }),
        "GDP": pd.DataFrame({
            "date": dates_quarterly,
            "realtime_start": dates_quarterly + pd.Timedelta(days=30),
            "value": [21000, 21500, 22000, 22500],
        }),
        "CPIAUCSL": pd.DataFrame({
            "date": dates_monthly,
            "realtime_start": dates_monthly + pd.Timedelta(days=13),
            "value": np.linspace(260, 270, 10),
        }),
    }


@pytest.fixture
def sample_macro_df():
    """Pre-built macro features for merge testing."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    return pd.DataFrame({
        "date": dates,
        "release_date": dates,
        "fed_rate": np.linspace(0.5, 2.0, 50),
        "fed_rate_change_3m": np.linspace(0.0, 0.5, 50),
        "unemployment": 4.0,
        "unemployment_trend": -0.1,
        "gdp_growth": 0.02,
        "cpi_yoy": 0.03,
    })


@pytest.fixture
def sample_prices_df():
    """Simple prices DataFrame for merge testing."""
    dates = pd.date_range("2020-01-10", periods=30, freq="B")
    tickers = ["AAPL"] * 15 + ["MSFT"] * 15
    return pd.DataFrame({
        "ticker": tickers,
        "date": list(dates[:15]) + list(dates[:15]),
        "close": np.random.default_rng(42).uniform(100, 200, 30),
    })


# ---------------------------------------------------------------------------
# _build_macro_features
# ---------------------------------------------------------------------------

class TestBuildMacroFeatures:
    def test_returns_dataframe(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        for col in ["date", "release_date", "fed_rate"]:
            assert col in result.columns

    def test_fed_rate_present(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        assert result["fed_rate"].notna().any()

    def test_gdp_growth_computed(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        assert "gdp_growth" in result.columns

    def test_cpi_yoy_computed(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        assert "cpi_yoy" in result.columns

    def test_raises_on_empty_data(self):
        with pytest.raises(RuntimeError, match="No macro features"):
            _build_macro_features({})


# ---------------------------------------------------------------------------
# merge_macro_pit
# ---------------------------------------------------------------------------

class TestMergeMacroPit:
    def test_preserves_row_count(self, sample_prices_df, sample_macro_df):
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        assert len(result) == len(sample_prices_df)

    def test_adds_macro_columns(self, sample_prices_df, sample_macro_df):
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        assert "fed_rate" in result.columns
        assert "unemployment" in result.columns

    def test_backward_merge_no_future_leak(self, sample_prices_df, sample_macro_df):
        """Macro data released after a price date should not appear."""
        # Create macro data with release_date far in the future
        future_macro = sample_macro_df.copy()
        future_macro["release_date"] = pd.Timestamp("2025-01-01")
        result = merge_macro_pit(sample_prices_df, macro_df=future_macro)
        # All macro features should be NaN (no data released before price dates)
        assert result["fed_rate"].isna().all()

    def test_does_not_add_release_date_column(self, sample_prices_df, sample_macro_df):
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        assert "release_date" not in result.columns
