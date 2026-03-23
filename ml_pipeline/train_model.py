"""
Model training, comparison, and artifact export.

Trains four sklearn classifiers on the generated dataset, optimizes the
classification threshold for each via precision-recall analysis, selects
the best model by validation F1, and saves all artifacts:

    models/best_model.pkl               — unified model (34 features)
    models/optimal_threshold.txt        — optimal threshold
    models/model_comparison.csv         — metrics for every model x split

SPLITTING STRATEGY:
    Time-based split (NOT random) to respect the temporal nature of
    financial data. A gap of PREDICTION_HORIZON_DAYS (60 trading days)
    is inserted between train/val and val/test to prevent label leakage
    (labels look 60 days forward, so the last training labels would
    "peek" into the validation period without the gap).

        |--- train (70%) ---|-- gap --|--- val (15%) ---|-- gap --|--- test (15%) ---|

MODELS COMPARED:
    1. HistGradientBoosting — handles NaN natively, no imputation needed
    2. GradientBoosting     — Pipeline with median imputation
    3. RandomForest         — Pipeline with median imputation
    4. LogisticRegression   — Pipeline with imputation + scaling

    All are sklearn-native (no XGBoost/LightGBM dependency).
    HistGradientBoosting is the expected winner: it's essentially sklearn's
    built-in LightGBM implementation and handles missing fundamentals
    without any preprocessing.

UNIFIED MODEL:
    A single model with all 34 features (11 technical + 8 fundamental +
    4 VIX + 6 macro + 5 sentiment) is used for both real-time screening
    and historical backtesting. This is possible because the training
    dataset uses point-in-time data from all sources (publish_date for
    fundamentals, realtime_start for FRED, 1-day lag for sentiment).
    No look-ahead bias.

Usage:
    python -m ml_pipeline.train_model
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import (
    FEATURE_COLUMNS,
    PATHS,
    PREDICTION_HORIZON_DAYS,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit: 1 - TRAIN_RATIO - VAL_RATIO)


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically with gaps to prevent label leakage.

    Sorts all unique dates, then partitions them into train / val / test
    with a gap of ``PREDICTION_HORIZON_DAYS`` between each partition.
    The gap ensures that the forward-looking labels at the end of the
    training set don't peek into validation data (and similarly for
    val → test).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``"date"`` column (datetime or string parseable
        by pandas).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames, each a subset of *df*.
    """
    dates = sorted(df["date"].unique())
    n = len(dates)
    gap = PREDICTION_HORIZON_DAYS

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end + gap : val_end])
    test_dates = set(dates[val_end + gap :])

    train = df[df["date"].isin(train_dates)]
    val = df[df["date"].isin(val_dates)]
    test = df[df["date"].isin(test_dates)]

    return train, val, test


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.25,
) -> float:
    """Find the threshold that maximizes precision while keeping recall >= min_recall.

    Philosophy: quality over quantity.  We want the model to say BUY
    rarely, but when it does, it should be right.  This means we
    optimize for **precision** (accuracy of BUY signals) while ensuring
    recall stays above *min_recall* so the model still finds a
    reasonable number of opportunities.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    min_recall : float
        Minimum acceptable recall.  Thresholds that push recall below
        this value are discarded.

    Returns
    -------
    float
        Optimal threshold in (0, 1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Filter to thresholds that maintain minimum recall
    valid_mask = recalls[:-1] >= min_recall

    if not valid_mask.any():
        # Fallback: pick threshold with best F1 if no threshold meets min_recall
        denom = precisions[:-1] + recalls[:-1]
        f1_scores = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / denom, 0.0)
        best_idx = int(f1_scores.argmax())
        logger.warning(
            "No threshold achieves recall >= %.2f. "
            "Falling back to best F1 threshold.",
            min_recall,
        )
        return float(thresholds[best_idx])

    # Among valid thresholds, pick the one with highest precision
    valid_precisions = np.where(valid_mask, precisions[:-1], -1.0)
    best_idx = int(valid_precisions.argmax())

    logger.info(
        "Optimal threshold: %.4f (precision=%.4f, recall=%.4f)",
        thresholds[best_idx], precisions[best_idx], recalls[best_idx],
    )
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def get_candidate_models() -> dict:
    """Return the four candidate classifiers to compare.

    HistGradientBoosting handles NaN natively. The other three are
    wrapped in a Pipeline with ``SimpleImputer(strategy="median")``
    to fill NaN fundamentals before fitting.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name → sklearn-compatible estimator.
    """
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute classification metrics at a given probability threshold.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with ``predict_proba`` method.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True binary labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict[str, float]
        Metrics: threshold, accuracy, precision, recall, f1, roc_auc.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": round(threshold, 6),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_and_select(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_path,
    threshold_path,
) -> dict:
    """Train all candidates, optimize thresholds, select best, save.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns: ``date``, ``label``, and all
        columns in *feature_cols*.
    feature_cols : list[str]
        Which feature columns to use for training.
    model_path : Path
        Where to save the best model (.pkl).
    threshold_path : Path
        Where to save the optimal threshold (.txt).

    Returns
    -------
    dict
        ``{"best_model": name, "best_threshold": float, "results": [...]}``.
    """
    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    train, val, test = time_split(df)
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_val = val[feature_cols].values
    y_val = val["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values

    # ------------------------------------------------------------------
    # Train and evaluate each candidate
    # ------------------------------------------------------------------
    results: list[dict] = []
    best_f1 = -1.0
    best_model = None
    best_threshold = 0.5
    best_name = ""

    candidates = get_candidate_models()

    for name, model in candidates.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)

        # Optimal threshold on validation set
        y_val_prob = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_val_prob)

        # Metrics on validation
        val_metrics = evaluate_model(model, X_val, y_val, threshold)
        val_metrics["model"] = name
        val_metrics["split"] = "validation"

        # Metrics on test (same threshold — no re-optimizing on test!)
        test_metrics = evaluate_model(model, X_test, y_test, threshold)
        test_metrics["model"] = name
        test_metrics["split"] = "test"

        results.append(val_metrics)
        results.append(test_metrics)

        logger.info(
            "  %s — val F1=%.4f | test F1=%.4f | threshold=%.4f | AUC=%.4f",
            name, val_metrics["f1"], test_metrics["f1"],
            threshold, val_metrics["roc_auc"],
        )

        # Track best by validation F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model = model
            best_threshold = threshold
            best_name = name

    # ------------------------------------------------------------------
    # Save best model + threshold
    # ------------------------------------------------------------------
    logger.info(
        "Winner: %s (val F1=%.4f, threshold=%.4f)",
        best_name, best_f1, best_threshold,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info("Model saved → %s", model_path)

    threshold_path.write_text(f"{best_threshold:.6f}\n")
    logger.info("Threshold saved → %s", threshold_path)

    return {
        "best_model": best_name,
        "best_threshold": best_threshold,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_models() -> None:
    """Train a unified model with all 34 features and save artifacts.

    With point-in-time data from all sources (fundamentals, macro,
    sentiment), a single model serves both real-time screening and
    backtesting without look-ahead bias. The model is saved to both
    production and backtest paths for backward compatibility.
    """
    import shutil

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = PATHS["dataset_file"]
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run first: python -m ml_pipeline.generate_dataset"
        )

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    logger.info(
        "Loaded dataset: %d rows, %d tickers.",
        len(df), df["ticker"].nunique(),
    )

    # Class balance summary
    positive_pct = df["label"].mean() * 100
    logger.info(
        "Class balance: %.1f%% positive (1), %.1f%% negative (0).",
        positive_pct, 100 - positive_pct,
    )

    # ------------------------------------------------------------------
    # Unified model — 34 features (technical + fundamental + VIX + macro + sentiment)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING UNIFIED MODEL (34 features)")
    logger.info("=" * 60)

    results = train_and_select(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        model_path=PATHS["model_file"],
        threshold_path=PATHS["threshold_file"],
    )

    # ------------------------------------------------------------------
    # Copy to backtest paths (backward compatibility)
    # ------------------------------------------------------------------
    shutil.copy2(PATHS["model_file"], PATHS["backtest_model_file"])
    shutil.copy2(PATHS["threshold_file"], PATHS["backtest_threshold_file"])
    logger.info("Copied model to backtest paths for backward compatibility.")

    # ------------------------------------------------------------------
    # Save comparison table
    # ------------------------------------------------------------------
    comparison = pd.DataFrame(results["results"])
    comparison["variant"] = "unified_34feat"
    comparison_path = PATHS["comparison_file"]
    comparison.to_csv(comparison_path, index=False)
    logger.info("Comparison table saved -> %s", comparison_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(
        "  Unified model: %s (threshold=%.4f)",
        results["best_model"], results["best_threshold"],
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    train_models()
