"""Tests for core.data_service — feature engineering and data helpers.

Network-dependent functions (get_sp500_tickers, download_ohlcv,
get_fundamental_features) are tested via mocks.  compute_technical_features
and build_feature_row are tested with synthetic data.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.config import FUNDAMENTAL_FEATURES, TECHNICAL_FEATURES
from core.data_service import (
    build_feature_row,
    compute_technical_features,
    download_ohlcv,
    get_fundamental_features,
    get_sp500_tickers,
)


# ---------------------------------------------------------------------------
# compute_technical_features
# ---------------------------------------------------------------------------

class TestComputeTechnicalFeatures:
    def test_returns_all_technical_columns(self, sample_ohlcv_df, sample_market_df):
        result = compute_technical_features(sample_ohlcv_df, market_df=sample_market_df)
        for feat in TECHNICAL_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_does_not_mutate_input(self, sample_ohlcv_df):
        original_cols = list(sample_ohlcv_df.columns)
        compute_technical_features(sample_ohlcv_df)
        assert list(sample_ohlcv_df.columns) == original_cols

    def test_market_trend_nan_when_no_market_df(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df, market_df=None)
        assert result["market_trend"].isna().all()

    def test_market_trend_binary_when_market_provided(self, sample_ohlcv_df, sample_market_df):
        result = compute_technical_features(sample_ohlcv_df, market_df=sample_market_df)
        valid = result["market_trend"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_williams_r_range(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        valid = result["williams_r"].dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_williams_r_signal_binary(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        assert set(result["williams_r_signal"].dropna().unique()).issubset({0, 1})

    def test_rsi_range(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_sma_cross_below_binary(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        assert set(result["sma_cross_below"].dropna().unique()).issubset({0, 1})

    def test_relative_volume_capped_at_5(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        valid = result["relative_volume"].dropna()
        assert (valid <= 5.0).all()

    def test_sma_200_column_added(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        assert "sma_200" in result.columns

    def test_output_preserves_index(self, sample_ohlcv_df):
        result = compute_technical_features(sample_ohlcv_df)
        assert result.index.equals(sample_ohlcv_df.index)

    def test_warmup_rows_have_nans(self, sample_ohlcv_df):
        """First ~200 rows should have NaN in sma_distance (SMA_200 warmup)."""
        result = compute_technical_features(sample_ohlcv_df)
        assert result["sma_distance"].iloc[:190].isna().all()

    def test_later_rows_have_values(self, sample_ohlcv_df):
        """Rows after warmup should have valid values for all features."""
        result = compute_technical_features(sample_ohlcv_df)
        # Row 250 should have all technical features populated
        row = result.iloc[250]
        for feat in TECHNICAL_FEATURES:
            if feat != "market_trend":  # NaN when no market_df
                assert pd.notna(row[feat]), f"{feat} is NaN at row 250"


# ---------------------------------------------------------------------------
# get_sp500_tickers (mocked)
# ---------------------------------------------------------------------------

class TestGetSP500Tickers:
    @patch("core.data_service.requests.get")
    def test_returns_list_of_strings(self, mock_get):
        html = """
        <html><body>
        <table>
          <tr><th>Symbol</th><th>Name</th></tr>
          <tr><td>AAPL</td><td>Apple</td></tr>
          <tr><td>MSFT</td><td>Microsoft</td></tr>
          <tr><td>BRK.B</td><td>Berkshire</td></tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        tickers = get_sp500_tickers()
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)

    @patch("core.data_service.requests.get")
    def test_dot_replaced_with_dash(self, mock_get):
        html = """
        <html><body>
        <table>
          <tr><th>Symbol</th><th>Name</th></tr>
          <tr><td>BRK.B</td><td>Berkshire</td></tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        tickers = get_sp500_tickers()
        assert "BRK-B" in tickers
        assert "BRK.B" not in tickers

    @patch("core.data_service.requests.get")
    def test_sorted_output(self, mock_get):
        html = """
        <html><body>
        <table>
          <tr><th>Symbol</th><th>Name</th></tr>
          <tr><td>MSFT</td><td>Microsoft</td></tr>
          <tr><td>AAPL</td><td>Apple</td></tr>
          <tr><td>GOOGL</td><td>Google</td></tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        tickers = get_sp500_tickers()
        assert tickers == sorted(tickers)

    @patch("core.data_service.requests.get")
    def test_connection_error_raises(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("offline")
        with pytest.raises(ConnectionError):
            get_sp500_tickers()


# ---------------------------------------------------------------------------
# download_ohlcv (mocked)
# ---------------------------------------------------------------------------

class TestDownloadOHLCV:
    @patch("core.data_service.yf.Ticker")
    @patch("core.data_service.time.sleep")
    def test_returns_dict_of_dataframes(self, mock_sleep, mock_ticker_cls, sample_ohlcv_df):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_ohlcv_df
        mock_ticker_cls.return_value = mock_ticker

        result = download_ohlcv(["AAPL", "MSFT"])
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], pd.DataFrame)

    @patch("core.data_service.yf.Ticker")
    @patch("core.data_service.time.sleep")
    def test_empty_ticker_skipped(self, mock_sleep, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = download_ohlcv(["BAD"])
        assert "BAD" not in result

    @patch("core.data_service.yf.Ticker")
    @patch("core.data_service.time.sleep")
    def test_exception_skips_ticker(self, mock_sleep, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("API error")
        mock_ticker_cls.return_value = mock_ticker

        result = download_ohlcv(["FAIL"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_fundamental_features (mocked)
# ---------------------------------------------------------------------------

class TestGetFundamentalFeatures:
    @patch("core.data_service.yf.Ticker")
    def test_returns_all_fundamental_keys(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 25.0,
            "pegRatio": 1.5,
            "operatingMargins": 0.30,
            "revenueGrowth": 0.10,
            "debtToEquity": 50.0,
            "currentRatio": 1.8,
            "totalCash": 5e9,
            "totalDebt": 3e9,
            "freeCashflow": 2e9,
            "marketCap": 100e9,
        }
        mock_ticker_cls.return_value = mock_ticker

        feats = get_fundamental_features("AAPL")
        assert set(feats.keys()) == set(FUNDAMENTAL_FEATURES)

    @patch("core.data_service.yf.Ticker")
    def test_missing_keys_return_nan(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        feats = get_fundamental_features("AAPL")
        for key, val in feats.items():
            assert np.isnan(val), f"{key} should be NaN when info is empty"

    @patch("core.data_service.yf.Ticker")
    def test_cash_covers_debt_computed(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "totalCash": 10e9,
            "totalDebt": 5e9,
        }
        mock_ticker_cls.return_value = mock_ticker

        feats = get_fundamental_features("AAPL")
        assert feats["cash_covers_debt"] == pytest.approx(2.0)

    @patch("core.data_service.yf.Ticker")
    def test_fcf_yield_computed(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "freeCashflow": 5e9,
            "marketCap": 100e9,
        }
        mock_ticker_cls.return_value = mock_ticker

        feats = get_fundamental_features("AAPL")
        assert feats["fcf_yield"] == pytest.approx(0.05)

    @patch("core.data_service.yf.Ticker")
    def test_exception_returns_all_nan(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("network error")

        feats = get_fundamental_features("FAIL")
        for val in feats.values():
            assert np.isnan(val)

    @patch("core.data_service.yf.Ticker")
    def test_zero_debt_gives_nan_cash_covers_debt(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {"totalCash": 10e9, "totalDebt": 0.0}
        mock_ticker_cls.return_value = mock_ticker

        feats = get_fundamental_features("AAPL")
        assert np.isnan(feats["cash_covers_debt"])


# ---------------------------------------------------------------------------
# build_feature_row (mocked fundamentals, real technicals)
# ---------------------------------------------------------------------------

class TestBuildFeatureRow:
    @patch("core.data_service.get_fundamental_features")
    def test_returns_all_19_features(self, mock_fund, sample_ohlcv_df, sample_market_df):
        mock_fund.return_value = {f: 1.0 for f in FUNDAMENTAL_FEATURES}

        result = build_feature_row("AAPL", sample_ohlcv_df, market_df=sample_market_df)

        for feat in TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    @patch("core.data_service.get_fundamental_features")
    def test_warmup_rows_dropped(self, mock_fund, sample_ohlcv_df):
        mock_fund.return_value = {f: 1.0 for f in FUNDAMENTAL_FEATURES}

        result = build_feature_row("AAPL", sample_ohlcv_df)

        # After dropping warmup, no technical feature should have NaN
        for feat in TECHNICAL_FEATURES:
            if feat != "market_trend":
                assert result[feat].isna().sum() == 0, f"{feat} still has NaN"

    @patch("core.data_service.get_fundamental_features")
    def test_fundamental_nan_not_dropped(self, mock_fund, sample_ohlcv_df, sample_market_df):
        """Rows with NaN fundamentals should NOT be dropped."""
        mock_fund.return_value = {f: np.nan for f in FUNDAMENTAL_FEATURES}

        result = build_feature_row("AAPL", sample_ohlcv_df, market_df=sample_market_df)
        # Should still have rows (only technical NaN causes drops)
        assert len(result) > 0

    @patch("core.data_service.get_fundamental_features")
    def test_output_has_datetime_index(self, mock_fund, sample_ohlcv_df):
        mock_fund.return_value = {f: 1.0 for f in FUNDAMENTAL_FEATURES}

        result = build_feature_row("AAPL", sample_ohlcv_df)
        assert isinstance(result.index, pd.DatetimeIndex)
