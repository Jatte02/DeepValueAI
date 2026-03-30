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
    dates_monthly = pd.date_range("2019-01-01", periods=24, freq="MS")
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
            "value": np.linspace(3.5, 4.5, 24),
        }),
        "GDP": pd.DataFrame({
            "date": dates_quarterly,
            "realtime_start": dates_quarterly + pd.Timedelta(days=30),
            "value": [21000, 21500, 22000, 22500],
        }),
        "CPIAUCSL": pd.DataFrame({
            "date": dates_monthly,
            "realtime_start": dates_monthly + pd.Timedelta(days=13),
            "value": np.linspace(260, 270, 24),
        }),
    }


@pytest.fixture
def sample_macro_df():
    """Pre-built long-format macro features for merge testing."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    rows = []
    for d in dates:
        rows.append({"release_date": d, "feature": "fed_rate", "value": 1.5})
        rows.append({"release_date": d, "feature": "fed_rate_change_3m", "value": 0.1})
    # Add monthly unemployment
    for d in dates[::5]:
        rows.append({"release_date": d, "feature": "unemployment", "value": 4.0})
        rows.append({"release_date": d, "feature": "unemployment_trend", "value": -0.1})
    # Add quarterly GDP
    rows.append({"release_date": dates[10], "feature": "gdp_growth", "value": 0.02})
    rows.append({"release_date": dates[30], "feature": "gdp_growth", "value": 0.025})
    # Add monthly CPI
    for d in dates[::5]:
        rows.append({"release_date": d, "feature": "cpi_yoy", "value": 0.03})
    return pd.DataFrame(rows)


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

    def test_has_long_format_columns(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        for col in ["release_date", "feature", "value"]:
            assert col in result.columns

    def test_fed_rate_present(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        features = result["feature"].unique()
        assert "fed_rate" in features

    def test_gdp_growth_computed(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        features = result["feature"].unique()
        assert "gdp_growth" in features

    def test_cpi_yoy_computed(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        features = result["feature"].unique()
        assert "cpi_yoy" in features

    def test_unemployment_present(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        features = result["feature"].unique()
        assert "unemployment" in features

    def test_all_six_features_present(self, sample_fred_data):
        result = _build_macro_features(sample_fred_data)
        features = set(result["feature"].unique())
        expected = {
            "fed_rate", "fed_rate_change_3m",
            "unemployment", "unemployment_trend",
            "gdp_growth", "cpi_yoy",
        }
        assert expected == features

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
        assert "gdp_growth" in result.columns
        assert "cpi_yoy" in result.columns

    def test_all_six_macro_features_populated(self, sample_prices_df, sample_macro_df):
        """Each macro feature should have non-NaN values after merge."""
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        for col in ["fed_rate", "fed_rate_change_3m", "unemployment",
                     "unemployment_trend", "gdp_growth", "cpi_yoy"]:
            assert result[col].notna().any(), f"{col} is all NaN after merge"

    def test_backward_merge_no_future_leak(self, sample_prices_df):
        """Macro data released after a price date should not appear."""
        future_macro = pd.DataFrame({
            "release_date": [pd.Timestamp("2025-01-01")] * 2,
            "feature": ["fed_rate", "unemployment"],
            "value": [2.0, 5.0],
        })
        result = merge_macro_pit(sample_prices_df, macro_df=future_macro)
        assert result["fed_rate"].isna().all()
        assert result["unemployment"].isna().all()

    def test_does_not_add_release_date_column(self, sample_prices_df, sample_macro_df):
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        assert "release_date" not in result.columns

    def test_does_not_add_feature_column(self, sample_prices_df, sample_macro_df):
        result = merge_macro_pit(sample_prices_df, macro_df=sample_macro_df)
        assert "feature" not in result.columns
