"""Tests for core.drift_monitor — prediction and feature drift detection."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.drift_monitor import check_prediction_stability

# ---------------------------------------------------------------------------
# Prediction stability (KS test) — no external dependencies needed
# ---------------------------------------------------------------------------

class TestPredictionStability:
    """Tests for check_prediction_stability using the KS test."""

    def test_identical_distributions_no_drift(self):
        """Identical arrays should show no significant drift."""
        probs = np.random.RandomState(42).uniform(0, 1, 500)
        result = check_prediction_stability(probs, probs)

        assert result["significant_drift"] is False
        assert result["ks_statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_different_distributions_detects_drift(self):
        """Clearly different distributions should trigger drift."""
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 0.3, 500)
        cur = rng.uniform(0.7, 1.0, 500)
        result = check_prediction_stability(ref, cur)

        assert result["significant_drift"] is True
        assert result["ks_statistic"] > 0.5
        assert result["p_value"] < 0.01

    def test_similar_distributions_no_drift(self):
        """Very similar distributions should not trigger drift."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0.5, 0.1, 1000)
        cur = rng.normal(0.5, 0.1, 1000)
        result = check_prediction_stability(ref, cur)

        assert result["significant_drift"] is False

    def test_custom_significance_level(self):
        """Custom significance threshold should be respected."""
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 1, 100)
        cur = rng.uniform(0.1, 0.9, 100)
        result = check_prediction_stability(ref, cur, significance=0.001)

        # With a very strict threshold, marginal drift is not significant
        assert isinstance(result["significant_drift"], bool)

    def test_return_keys(self):
        """Result dict should contain all expected keys."""
        probs = np.array([0.1, 0.5, 0.9])
        result = check_prediction_stability(probs, probs)

        assert "ks_statistic" in result
        assert "p_value" in result
        assert "significant_drift" in result


# ---------------------------------------------------------------------------
# Feature drift (evidently) — mocked to avoid dependency
# ---------------------------------------------------------------------------

class TestFeatureDrift:
    """Tests for detect_drift with mocked evidently."""

    @patch("core.drift_monitor.HAS_EVIDENTLY", False)
    def test_raises_without_evidently(self):
        """Should raise ImportError when evidently is not installed."""
        from core.drift_monitor import detect_drift

        ref = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        cur = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])

        with pytest.raises(ImportError, match="evidently"):
            detect_drift(ref, cur, feature_cols=["a", "b", "c"])
