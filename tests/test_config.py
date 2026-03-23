"""Tests for core.config — constants, paths, feature lists, logging."""

from pathlib import Path

from core.config import (
    COOLDOWN_DAYS_PER_TICKER,
    DEFAULT_THRESHOLD,
    DOWNLOAD_INTERVAL,
    DOWNLOAD_PERIOD,
    FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURES,
    MACRO_FEATURES,
    MAX_HOLDING_DAYS,
    MAX_OPEN_POSITIONS,
    MAX_TICKER_EXPOSURE_PCT,
    PARTIAL_TP_SELL_FRACTION,
    PARTIAL_TP_TRIGGER_PCT,
    PATHS,
    POSITION_SIZE_PCT,
    PROJECT_ROOT,
    SENTIMENT_FEATURES,
    SMA_BUY_CEILING,
    STOP_LOSS_PCT,
    TECHNICAL_FEATURES,
    TRAILING_STOP_ACTIVATION_PCT,
    TRAILING_STOP_PCT,
    VIX_FEATURES,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

class TestProjectRoot:
    def test_project_root_is_directory(self):
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_core(self):
        assert (PROJECT_ROOT / "core").is_dir()

    def test_project_root_contains_pyproject(self):
        assert (PROJECT_ROOT / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# PATHS dict
# ---------------------------------------------------------------------------

class TestPaths:
    def test_all_paths_are_path_objects(self):
        for key, val in PATHS.items():
            assert isinstance(val, Path), f"PATHS['{key}'] should be a Path"

    def test_all_paths_under_project_root(self):
        for key, val in PATHS.items():
            assert str(val).startswith(str(PROJECT_ROOT)), (
                f"PATHS['{key}'] is not under PROJECT_ROOT"
            )

    def test_expected_keys_exist(self):
        expected = {
            "models_dir", "data_dir", "model_file", "threshold_file",
            "dataset_file", "comparison_file",
            "backtest_model_file", "backtest_threshold_file",
            "fundamentals_dir", "fundamentals_file",
            "macro_dir", "macro_file",
            "news_dir", "news_raw_file", "sentiment_file",
        }
        assert expected == set(PATHS.keys())


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_technical_features_count(self):
        assert len(TECHNICAL_FEATURES) == 11

    def test_fundamental_features_count(self):
        assert len(FUNDAMENTAL_FEATURES) == 8

    def test_vix_features_count(self):
        assert len(VIX_FEATURES) == 4

    def test_macro_features_count(self):
        assert len(MACRO_FEATURES) == 6

    def test_sentiment_features_count(self):
        assert len(SENTIMENT_FEATURES) == 5

    def test_feature_columns_is_concatenation(self):
        assert FEATURE_COLUMNS == (
            TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES
            + VIX_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES
        )

    def test_feature_columns_total(self):
        assert len(FEATURE_COLUMNS) == 34

    def test_no_duplicate_features(self):
        assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)

    def test_key_technical_features_present(self):
        for feat in ("williams_r", "rsi", "macd_histogram", "sma_distance", "market_trend"):
            assert feat in TECHNICAL_FEATURES

    def test_key_fundamental_features_present(self):
        for feat in ("pe_ratio", "peg_ratio", "fcf_yield", "debt_equity"):
            assert feat in FUNDAMENTAL_FEATURES

    def test_key_vix_features_present(self):
        for feat in ("vix_level", "vix_regime"):
            assert feat in VIX_FEATURES

    def test_key_macro_features_present(self):
        for feat in ("fed_rate", "unemployment", "cpi_yoy"):
            assert feat in MACRO_FEATURES

    def test_key_sentiment_features_present(self):
        for feat in ("sentiment_mean", "news_volume"):
            assert feat in SENTIMENT_FEATURES


# ---------------------------------------------------------------------------
# Strategy parameters — sanity ranges
# ---------------------------------------------------------------------------

class TestStrategyParams:
    def test_default_threshold_in_range(self):
        assert 0 < DEFAULT_THRESHOLD < 1

    def test_sma_buy_ceiling_above_one(self):
        assert SMA_BUY_CEILING > 1.0

    def test_position_size_reasonable(self):
        assert 0 < POSITION_SIZE_PCT <= 0.20

    def test_max_ticker_exposure_gte_position_size(self):
        assert MAX_TICKER_EXPOSURE_PCT >= POSITION_SIZE_PCT

    def test_max_open_positions_positive(self):
        assert MAX_OPEN_POSITIONS > 0

    def test_cooldown_positive(self):
        assert COOLDOWN_DAYS_PER_TICKER > 0

    def test_stop_loss_in_range(self):
        assert 0 < STOP_LOSS_PCT < 1

    def test_partial_tp_trigger_positive(self):
        assert PARTIAL_TP_TRIGGER_PCT > 0

    def test_partial_tp_fraction_in_range(self):
        assert 0 < PARTIAL_TP_SELL_FRACTION < 1

    def test_trailing_stop_activation_positive(self):
        assert TRAILING_STOP_ACTIVATION_PCT > 0

    def test_trailing_stop_pct_positive(self):
        assert TRAILING_STOP_PCT > 0

    def test_max_holding_days_positive(self):
        assert MAX_HOLDING_DAYS > 0

    def test_download_period_valid(self):
        assert DOWNLOAD_PERIOD in ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")

    def test_download_interval_valid(self):
        assert DOWNLOAD_INTERVAL in ("1d", "1wk", "1mo")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_setup_logging_does_not_raise(self):
        setup_logging()
