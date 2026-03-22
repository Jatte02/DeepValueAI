"""Tests for core.prediction_service — model/threshold loading, inference, signals."""

import textwrap

import numpy as np
import pandas as pd
import pytest

from core.config import DEFAULT_THRESHOLD, FEATURE_COLUMNS
from core.prediction_service import (
    generate_signal,
    load_model,
    load_threshold,
    predict_proba,
)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(tmp_path / "nonexistent.pkl")

    def test_invalid_file_raises_runtime_error(self, tmp_path):
        bad_file = tmp_path / "bad.pkl"
        bad_file.write_text("not a pickle")
        with pytest.raises(RuntimeError, match="Failed to load model"):
            load_model(bad_file)

    def test_object_without_predict_proba_raises(self, tmp_path):
        import joblib
        dummy = {"not": "a model"}
        pkl = tmp_path / "dict.pkl"
        joblib.dump(dummy, pkl)
        with pytest.raises(RuntimeError, match="predict_proba"):
            load_model(pkl)

    def test_valid_model_loads_successfully(self, tmp_path, fake_model):
        import joblib
        pkl = tmp_path / "model.pkl"
        joblib.dump(fake_model, pkl)
        loaded = load_model(pkl)
        assert hasattr(loaded, "predict_proba")


# ---------------------------------------------------------------------------
# load_threshold
# ---------------------------------------------------------------------------

class TestLoadThreshold:
    def test_missing_file_returns_default(self, tmp_path):
        result = load_threshold(tmp_path / "missing.txt")
        assert result == DEFAULT_THRESHOLD

    def test_valid_threshold(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("0.72\n")
        assert load_threshold(f) == pytest.approx(0.72)

    def test_non_numeric_returns_default(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("abc")
        assert load_threshold(f) == DEFAULT_THRESHOLD

    def test_zero_returns_default(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("0.0")
        assert load_threshold(f) == DEFAULT_THRESHOLD

    def test_one_returns_default(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("1.0")
        assert load_threshold(f) == DEFAULT_THRESHOLD

    def test_negative_returns_default(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("-0.5")
        assert load_threshold(f) == DEFAULT_THRESHOLD

    def test_above_one_returns_default(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("1.5")
        assert load_threshold(f) == DEFAULT_THRESHOLD

    def test_boundary_just_above_zero(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("0.001")
        assert load_threshold(f) == pytest.approx(0.001)

    def test_boundary_just_below_one(self, tmp_path):
        f = tmp_path / "threshold.txt"
        f.write_text("0.999")
        assert load_threshold(f) == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def _make_feature_df(self, n_rows: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        data = {col: rng.random(n_rows) for col in FEATURE_COLUMNS}
        return pd.DataFrame(data)

    def test_returns_1d_array(self, fake_model):
        df = self._make_feature_df()
        probs = predict_proba(df, fake_model)
        assert probs.ndim == 1
        assert len(probs) == len(df)

    def test_values_match_model(self, fake_model):
        df = self._make_feature_df()
        probs = predict_proba(df, fake_model)
        np.testing.assert_allclose(probs, 0.80)

    def test_missing_column_raises(self, fake_model):
        df = self._make_feature_df().drop(columns=["rsi"])
        with pytest.raises(ValueError, match="missing.*feature"):
            predict_proba(df, fake_model)

    def test_custom_feature_list(self, fake_model):
        cols = ["a", "b", "c"]
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        probs = predict_proba(df, fake_model, feature_list=cols)
        assert len(probs) == 1

    def test_extra_columns_are_ignored(self, fake_model):
        df = self._make_feature_df()
        df["extra_col"] = 999
        probs = predict_proba(df, fake_model)
        assert len(probs) == len(df)


# ---------------------------------------------------------------------------
# generate_signal
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    def test_buy_above_threshold(self):
        result = generate_signal(0.80, threshold=0.70)
        assert result["signal"] == "BUY"

    def test_hold_below_threshold(self):
        result = generate_signal(0.50, threshold=0.70)
        assert result["signal"] == "HOLD"

    def test_buy_at_exact_threshold(self):
        result = generate_signal(0.70, threshold=0.70)
        assert result["signal"] == "BUY"

    def test_hold_confidence_is_na(self):
        result = generate_signal(0.30, threshold=0.70)
        assert result["confidence"] == "N/A"

    def test_confidence_low(self):
        result = generate_signal(0.72, threshold=0.70)
        assert result["confidence"] == "LOW"

    def test_confidence_medium(self):
        result = generate_signal(0.76, threshold=0.70)
        assert result["confidence"] == "MEDIUM"

    def test_confidence_high(self):
        result = generate_signal(0.86, threshold=0.70)
        assert result["confidence"] == "HIGH"

    def test_probability_passthrough(self):
        result = generate_signal(0.42, threshold=0.70)
        assert result["probability"] == 0.42

    def test_result_keys(self):
        result = generate_signal(0.80, threshold=0.50)
        assert set(result.keys()) == {"probability", "signal", "confidence"}

    @pytest.mark.parametrize("prob,threshold,expected_signal", [
        (0.00, 0.50, "HOLD"),
        (1.00, 0.50, "BUY"),
        (0.50, 0.50, "BUY"),
        (0.49, 0.50, "HOLD"),
        (0.99, 0.01, "BUY"),
    ])
    def test_parametrized_signals(self, prob, threshold, expected_signal):
        result = generate_signal(prob, threshold)
        assert result["signal"] == expected_signal
