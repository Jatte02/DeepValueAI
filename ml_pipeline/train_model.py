"""
Model training, comparison, and artifact export.

Trains three sklearn classifiers on the generated dataset, optimizes the
classification threshold for each via precision-recall analysis, selects
the best model by validation F1, and saves all artifacts:

    models/best_model.pkl               — unified model (34 features)
    models/optimal_threshold.txt        — optimal threshold
    models/model_comparison.csv         — metrics for every model x split

SPLITTING STRATEGY:
    Two-phase temporal approach:

    Phase 1 — Model selection via expanding-window temporal CV (3 folds).
    The first 85% of dates are split into 4 chunks; each fold trains on
    all chunks up to i and validates on chunk i+1.  A gap of
    PREDICTION_HORIZON_DAYS separates train from val in each fold.

    Phase 2 — Final training on 70% / 15% val / 15% test (with gaps).
    All models are retrained and evaluated; the CV winner is saved.

        |--- train (70%) ---|-- gap --|--- val (15%) ---|-- gap --|--- test (15%) ---|

MODELS COMPARED:
    1. HistGradientBoosting — handles NaN natively, no imputation needed
    2. RandomForest         — Pipeline with median imputation
    3. LogisticRegression   — Pipeline with imputation + scaling

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
import shutil

import joblib
import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
from sklearn.base import clone
from sklearn.ensemble import (
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
N_CV_FOLDS = 3


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


def time_series_cv_splits(
    df: pd.DataFrame,
    n_folds: int = N_CV_FOLDS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate expanding-window temporal CV folds from the training portion.

    Uses the first 85% of dates (same as train+val in ``time_split``),
    divided into ``n_folds + 1`` chunks.  Each fold trains on all chunks
    up to *i* and validates on chunk *i+1*, with a gap of
    ``PREDICTION_HORIZON_DAYS`` between them to prevent label leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``"date"`` column.
    n_folds : int
        Number of CV folds to generate.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train, val) DataFrames for each fold.
    """
    dates = sorted(df["date"].unique())
    n = len(dates)
    gap = PREDICTION_HORIZON_DAYS

    # Use first 85% for CV (same dates that time_split uses for train+val)
    cv_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    cv_dates = dates[:cv_end]
    n_cv = len(cv_dates)

    # Divide into (n_folds + 1) chunks
    chunk_size = n_cv // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = chunk_size * (i + 1)
        val_start = train_end + gap
        val_end = chunk_size * (i + 2)

        if val_start >= n_cv:
            break
        val_end = min(val_end, n_cv)

        train_set = set(cv_dates[:train_end])
        val_set = set(cv_dates[val_start:val_end])

        if not val_set:
            break

        folds.append((
            df[df["date"].isin(train_set)],
            df[df["date"].isin(val_set)],
        ))

    return folds


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
# Hyperparameter tuning
# ---------------------------------------------------------------------------

HGBC_PARAM_SPACE = {
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_leaf": [10, 20, 30, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
}

N_TUNING_ITER = 15


def tune_hgbc(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> dict:
    """Find best HGBC hyperparameters via randomized temporal CV search.

    Samples ``N_TUNING_ITER`` random combinations from ``HGBC_PARAM_SPACE``,
    evaluates each with the pre-computed temporal CV folds, and returns the
    parameter dict with the highest mean-std F1 score.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (unused directly, kept for API consistency).
    feature_cols : list[str]
        Feature columns to train on.
    folds : list[tuple[pd.DataFrame, pd.DataFrame]]
        Pre-computed (train, val) temporal CV folds.

    Returns
    -------
    dict
        Best hyperparameter combination.
    """
    import random
    rng = random.Random(42)

    best_score = -1.0
    best_params: dict = {}

    logger.info("Tuning HGBC: %d random searches x %d folds ...", N_TUNING_ITER, len(folds))

    for i in range(N_TUNING_ITER):
        params = {k: rng.choice(v) for k, v in HGBC_PARAM_SPACE.items()}

        template = HistGradientBoostingClassifier(
            **params,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )

        fold_f1s = []
        for train_fold, val_fold in folds:
            model = clone(template)
            model.fit(
                train_fold[feature_cols].values,
                train_fold["label"].values,
            )
            y_prob = model.predict_proba(val_fold[feature_cols].values)[:, 1]
            thr = find_optimal_threshold(val_fold["label"].values, y_prob)
            met = evaluate_model(
                model, val_fold[feature_cols].values,
                val_fold["label"].values, thr,
            )
            fold_f1s.append(met["f1"])

        score = float(np.mean(fold_f1s) - np.std(fold_f1s))

        if score > best_score:
            best_score = score
            best_params = params
            logger.info(
                "  [%d/%d] New best: score=%.4f %s",
                i + 1, N_TUNING_ITER, score, params,
            )

    logger.info("Best HGBC params (score=%.4f): %s", best_score, best_params)
    return best_params


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def get_candidate_models(hgbc_params: dict | None = None) -> dict:
    """Return the candidate classifiers to compare.

    HistGradientBoosting handles NaN natively. The others are
    wrapped in a Pipeline with ``SimpleImputer(strategy="median")``
    to fill NaN fundamentals before fitting.

    Parameters
    ----------
    hgbc_params : dict | None
        If provided, override default HGBC hyperparameters with
        tuned values from ``tune_hgbc``.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name → sklearn-compatible estimator.
    """
    hgbc_defaults = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    }
    if hgbc_params:
        hgbc_defaults.update(hgbc_params)

    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            **hgbc_defaults,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
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
    """Train candidates with temporal CV, select best, retrain, save.

    Two-phase approach:
        1. **CV phase**: expanding-window temporal cross-validation across
           ``N_CV_FOLDS`` folds to compute a robust average F1 per model.
           The model with the highest mean CV F1 is selected.
        2. **Final phase**: all models are retrained on the full training
           split (70%), thresholds optimized on validation (15%), and
           evaluated on the held-out test set (15%).  The winner from
           phase 1 is saved.

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
    # Phase 0: Hyperparameter tuning for HGBC
    # ------------------------------------------------------------------
    folds = time_series_cv_splits(df)
    logger.info("Temporal CV: %d folds", len(folds))
    for i, (tr, va) in enumerate(folds):
        logger.info("  Fold %d: train=%d, val=%d", i + 1, len(tr), len(va))

    hgbc_params = tune_hgbc(df, feature_cols, folds)
    candidates = get_candidate_models(hgbc_params=hgbc_params)

    # ------------------------------------------------------------------
    # MLflow: start parent run
    # ------------------------------------------------------------------
    mlflow_active = False
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("DeepValueAI")
            mlflow.start_run(run_name="train_and_select")
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_cv_folds", len(folds))
            mlflow.log_param("dataset_rows", len(df))
            mlflow_active = True
        except Exception:
            logger.warning("MLflow logging failed to start — continuing without it.")

    # ------------------------------------------------------------------
    # Phase 1: Temporal CV for model selection
    # ------------------------------------------------------------------
    cv_scores: dict[str, float] = {}

    for name, template in candidates.items():
        logger.info("CV: %s ...", name)
        fold_f1s = []
        for train_fold, val_fold in folds:
            model = clone(template)
            model.fit(
                train_fold[feature_cols].values,
                train_fold["label"].values,
            )
            y_prob = model.predict_proba(val_fold[feature_cols].values)[:, 1]
            thr = find_optimal_threshold(val_fold["label"].values, y_prob)
            met = evaluate_model(
                model, val_fold[feature_cols].values,
                val_fold["label"].values, thr,
            )
            fold_f1s.append(met["f1"])

        avg_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        # Penalize instability: mean - std favors consistent models
        score = avg_f1 - std_f1
        cv_scores[name] = score
        logger.info(
            "  %s — CV score=%.4f (mean=%.4f - std=%.4f) folds=%s",
            name, score, avg_f1, std_f1, [round(f, 4) for f in fold_f1s],
        )

    best_name = max(cv_scores, key=cv_scores.get)
    logger.info("CV winner: %s (score=%.4f)", best_name, cv_scores[best_name])

    # ------------------------------------------------------------------
    # Phase 2: Final train/val/test — all models for comparison
    # ------------------------------------------------------------------
    train, val, test = time_split(df)
    logger.info(
        "Final split — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    if mlflow_active:
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("val_size", len(val))
        mlflow.log_param("test_size", len(test))

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_val = val[feature_cols].values
    y_val = val["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values

    results: list[dict] = []
    trained_models: dict = {}
    trained_thresholds: dict[str, float] = {}

    for name, template in candidates.items():
        logger.info("Training %s (final) ...", name)
        model = clone(template)
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

        trained_models[name] = model
        trained_thresholds[name] = threshold

        # MLflow: log each candidate as a nested run
        if mlflow_active:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_name", name)
                mlflow.log_param("threshold", round(threshold, 6))
                mlflow.log_metric("cv_score", cv_scores.get(name, 0))
                for k, v in val_metrics.items():
                    if isinstance(v, float):
                        mlflow.log_metric(f"val_{k}", v)
                for k, v in test_metrics.items():
                    if isinstance(v, float):
                        mlflow.log_metric(f"test_{k}", v)

        logger.info(
            "  %s — CV F1=%.4f | val F1=%.4f | test F1=%.4f | threshold=%.4f | AUC=%.4f",
            name, cv_scores.get(name, 0), val_metrics["f1"], test_metrics["f1"],
            threshold, test_metrics["roc_auc"],
        )

    # ------------------------------------------------------------------
    # Save CV winner
    # ------------------------------------------------------------------
    best_model = trained_models[best_name]
    best_threshold = trained_thresholds[best_name]

    logger.info(
        "Winner: %s (CV F1=%.4f, threshold=%.4f)",
        best_name, cv_scores[best_name], best_threshold,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info("Model saved → %s", model_path)

    threshold_path.write_text(f"{best_threshold:.6f}\n")
    logger.info("Threshold saved → %s", threshold_path)

    # MLflow: log winner model artifact and close parent run
    if mlflow_active:
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("best_threshold", round(best_threshold, 6))
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.end_run()

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

    # Replace infinities with NaN so all models can handle them
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    n_tickers = df["ticker"].nunique()
    logger.info("Loaded dataset: %d rows, %d tickers.", len(df), n_tickers)

    # Class balance summary
    positive_pct = df["label"].mean() * 100
    logger.info(
        "Class balance: %.1f%% positive (1), %.1f%% negative (0).",
        positive_pct, 100 - positive_pct,
    )

    # Log dataset metadata to MLflow (if available)
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("DeepValueAI")
        except Exception:
            pass

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
