"""
Data and prediction drift monitoring for DeepValue AI.

Detects when the distribution of input features or model predictions
has shifted significantly from the training baseline. Drift signals
that the model may need retraining.

Two independent checks:
    1. Feature drift   — statistical comparison of feature distributions
                         using evidently's DataDriftPreset.
    2. Prediction drift — Kolmogorov-Smirnov test on predicted probabilities.

Usage:
    python -m core.drift_monitor
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import FEATURE_COLUMNS, PATHS

logger = logging.getLogger(__name__)

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False


# ---------------------------------------------------------------------------
# Feature drift (evidently)
# ---------------------------------------------------------------------------

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    report_path: str | Path | None = None,
) -> dict:
    """Compare feature distributions between reference and current data.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Baseline data (e.g. training set).
    current_data : pd.DataFrame
        New data to compare against baseline.
    feature_cols : list[str], optional
        Which columns to check. Defaults to FEATURE_COLUMNS.
    report_path : str or Path, optional
        Where to save the HTML report. Defaults to data/drift_report.html.

    Returns
    -------
    dict
        Keys: drifted, share_drifted_features, drifted_features, report_path.
    """
    if not HAS_EVIDENTLY:
        raise ImportError(
            "evidently is required for feature drift detection. "
            "Install it with: pip install evidently"
        )

    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    if report_path is None:
        report_path = PATHS["data_dir"] / "drift_report.html"
    report_path = Path(report_path)

    column_mapping = ColumnMapping(
        numerical_features=feature_cols,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data[feature_cols],
        current_data=current_data[feature_cols],
        column_mapping=column_mapping,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))
    logger.info("Drift report saved → %s", report_path)

    result_dict = report.as_dict()
    drift_result = result_dict["metrics"][0]["result"]

    drifted_features = [
        col for col, info in drift_result["drift_by_columns"].items()
        if info["drift_detected"]
    ]

    return {
        "drifted": drift_result["dataset_drift"],
        "share_drifted_features": drift_result["share_of_drifted_columns"],
        "drifted_features": drifted_features,
        "report_path": str(report_path),
    }


# ---------------------------------------------------------------------------
# Prediction drift (KS test — no external dependency)
# ---------------------------------------------------------------------------

def check_prediction_stability(
    y_probs_reference: np.ndarray,
    y_probs_current: np.ndarray,
    significance: float = 0.05,
) -> dict:
    """Compare prediction probability distributions using the KS test.

    Parameters
    ----------
    y_probs_reference : np.ndarray
        Predicted probabilities from the reference period.
    y_probs_current : np.ndarray
        Predicted probabilities from the current period.
    significance : float
        P-value threshold for declaring significant drift.

    Returns
    -------
    dict
        Keys: ks_statistic, p_value, significant_drift.
    """
    stat, p_value = ks_2samp(y_probs_reference, y_probs_current)

    logger.info(
        "Prediction stability — KS=%.4f, p=%.4f, drift=%s",
        stat, p_value, p_value < significance,
    )

    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "significant_drift": bool(p_value < significance),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset_path = PATHS["dataset_file"]
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        logger.error("Run first: python -m ml_pipeline.generate_dataset")
        raise SystemExit(1)

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Split 80/20 by date to simulate reference vs current
    cutoff = int(len(df) * 0.8)
    reference = df.iloc[:cutoff]
    current = df.iloc[cutoff:]

    logger.info("Reference: %d rows, Current: %d rows", len(reference), len(current))

    # Feature drift
    if HAS_EVIDENTLY:
        result = detect_drift(reference, current)
        logger.info("Dataset drift: %s", result["drifted"])
        logger.info("Drifted features (%.0f%%): %s",
                     result["share_drifted_features"] * 100,
                     result["drifted_features"])
        logger.info("Report: %s", result["report_path"])
    else:
        logger.warning("evidently not installed — skipping feature drift check.")

    # Prediction drift (mock probabilities from model)
    try:
        import joblib
        model = joblib.load(PATHS["model_file"])
        ref_probs = model.predict_proba(
            reference[FEATURE_COLUMNS].values
        )[:, 1]
        cur_probs = model.predict_proba(
            current[FEATURE_COLUMNS].values
        )[:, 1]
        stability = check_prediction_stability(ref_probs, cur_probs)
        logger.info("Prediction drift: %s (KS=%.4f, p=%.4f)",
                     stability["significant_drift"],
                     stability["ks_statistic"],
                     stability["p_value"])
    except FileNotFoundError:
        logger.warning("Model not found — skipping prediction drift check.")
