"""
Model loading and inference.

This module is the SINGLE source of truth for:
    - Loading the trained model from disk
    - Loading the optimal classification threshold
    - Generating probability predictions from feature DataFrames
    - Converting probabilities into actionable BUY / HOLD signals

Every module that needs a prediction imports from here. The analyzer,
screener, and backtesting engine all go through predict_proba() and
generate_signal() — none of them touch the model file directly.

DESIGN DECISIONS:
    1. Pure functions, no classes. The caller controls when to load
       the model and passes it explicitly. This keeps the module
       stateless, testable, and consistent with data_service.py.

    2. The model is treated as a generic sklearn estimator. We call
       .predict_proba() and nothing else. This means ANY classifier
       that implements the sklearn API works: XGBoost, LightGBM,
       Random Forest, Logistic Regression, etc. The module doesn't
       know or care which one was selected during training.

    3. Feature validation happens HERE, not in the caller. Before
       predicting, we verify the DataFrame has exactly the 19 columns
       the model expects, in the correct order. This catches silent
       bugs where a feature was renamed or dropped upstream.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    TECHNICAL_FEATURES,
    PATHS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: Path | None = None):
    """Load a trained sklearn-compatible model from disk.

    Parameters
    ----------
    path : Path | None
        Path to the ``.pkl`` file. If None, uses the default path
        from ``config.PATHS["model_file"]``.

    Returns
    -------
    sklearn estimator
        Any object that implements ``.predict_proba()``. Could be
        an XGBoost classifier, Random Forest, LightGBM, a Pipeline,
        or even a VotingClassifier — we don't care which.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist. This means training hasn't
        been run yet (``make train``).
    RuntimeError
        If the file exists but cannot be deserialized (corrupted file,
        incompatible sklearn version, etc.).

    Examples
    --------
    >>> model = load_model()
    >>> hasattr(model, "predict_proba")
    True
    """
    if path is None:
        path = PATHS["model_file"]

    path = Path(path)  # ensure Path type even if caller passes a string

    # ------------------------------------------------------------------
    # Explicit existence check BEFORE calling joblib.load().
    # Why not just let joblib raise its own error? Because joblib's
    # FileNotFoundError message is generic: "No such file or directory".
    # Our message tells the user exactly what's wrong and how to fix it.
    # ------------------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Have you trained the model yet? Run: make train"
        )

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {path}. The file may be corrupted "
            f"or incompatible with the current scikit-learn version.\n"
            f"Try retraining: make train\n"
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Validate that the loaded object is actually a model.
    # joblib.load() can deserialize ANY Python object — it could be a
    # dict, a list, or someone's cat photo if the wrong file got saved.
    # We check for predict_proba because that's the ONLY method we call.
    # ------------------------------------------------------------------
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"The loaded object from {path} does not have a predict_proba() "
            f"method. Expected an sklearn-compatible classifier, but got: "
            f"{type(model).__name__}"
        )

    logger.info("Model loaded from %s (%s).", path, type(model).__name__)
    return model

# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------

def load_threshold(path: Path | None = None) -> float:
    """Load the optimal classification threshold from disk.

    The threshold is a single float (e.g., 0.72) saved as plain text
    during training. It represents the probability cutoff: if the model
    says P(stock goes up) > threshold, we generate a BUY signal.

    WHY A SEPARATE FILE?
        The threshold is NOT part of the model itself. sklearn models
        always predict at 0.50 by default. The optimal threshold is
        found AFTER training by analyzing the precision-recall curve
        on the validation set. Storing it separately means we can
        re-optimize the threshold without retraining the entire model.

    Parameters
    ----------
    path : Path | None
        Path to the threshold text file. If None, uses the default
        from ``config.PATHS["threshold_file"]``.

    Returns
    -------
    float
        The threshold value, guaranteed to be in the range (0, 1).

    Examples
    --------
    >>> threshold = load_threshold()
    >>> 0 < threshold < 1
    True
    """
    if path is None:
        path = PATHS["threshold_file"]

    path = Path(path)

    # ------------------------------------------------------------------
    # If the file doesn't exist, return the default threshold.
    # This is NOT an error — it just means training hasn't happened yet.
    # We log a warning so the user knows they're running with a fallback,
    # but we don't crash. This lets the app start and display the UI
    # even before training (useful for development and demos).
    # ------------------------------------------------------------------
    if not path.exists():
        logger.warning(
            "Threshold file not found: %s. Using default threshold: %.2f. "
            "Run 'make train' to generate the optimal threshold.",
            path, DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # The file contains a single line with a float, e.g. "0.72".
    # We read it, strip whitespace/newlines, and convert to float.
    # ------------------------------------------------------------------
    try:
        raw_text = path.read_text().strip()
        threshold = float(raw_text)
    except ValueError as exc:
        logger.warning(
            "Could not parse threshold from %s (content: '%s'). "
            "Using default: %.2f. Original error: %s",
            path, raw_text, DEFAULT_THRESHOLD, exc,
        )
        return DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # Sanity check: a threshold must be between 0 and 1 (exclusive).
    # Values outside this range indicate a corrupted file or a bug
    # in the training pipeline. We don't crash — we fall back to
    # the default — but we log loudly so it gets investigated.
    # ------------------------------------------------------------------
    if not (0 < threshold < 1):
        logger.warning(
            "Threshold %.4f from %s is outside valid range (0, 1). "
            "Using default: %.2f.",
            threshold, path, DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD

    logger.info("Threshold loaded: %.4f (from %s).", threshold, path)
    return threshold

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_proba(
    df: pd.DataFrame,
    model,
    feature_list: list[str] | None = None,
) -> np.ndarray:
    """Return the positive-class probability for each row in *df*.

    Extracts the features in the exact order expected by the model,
    then calls ``model.predict_proba()``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain (at minimum) all columns in *feature_list*.
    model
        A fitted scikit-learn compatible estimator with a
        ``predict_proba`` method.
    feature_list : list[str] | None
        Which feature columns to extract and in what order.
        Defaults to ``config.FEATURE_COLUMNS`` (19 features).
        Pass ``config.TECHNICAL_FEATURES`` (11 features) for the
        backtest model, which was trained without fundamentals.

    Returns
    -------
    np.ndarray
        1-D array of probabilities (one per row).

    Raises
    ------
    ValueError
        If any required column is missing from *df*.
    """
    if feature_list is None:
        feature_list = FEATURE_COLUMNS

    missing = [col for col in feature_list if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing {len(missing)} required feature(s): {missing}"
        )

    X = df[feature_list].values
    probabilities = model.predict_proba(X)[:, 1]

    logger.debug(
        "Predicted %d rows. Prob range: [%.4f, %.4f].",
        len(probabilities), probabilities.min(), probabilities.max(),
    )
    return probabilities

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(
    probability: float,
    threshold: float,
) -> dict[str, float | str]:
    """Convert a single probability into a BUY or HOLD signal.

    This is intentionally a simple function that operates on ONE
    probability at a time (not an array). The caller decides how to
    loop — the analyzer calls it once, the screener calls it 500 times,
    the backtesting engine calls it for every row in a 5-year history.

    The function does NOT apply the SMA filter (price < SMA × 1.05).
    That filter depends on price data which lives in the caller's
    context. This function only knows about probability vs threshold.
    Mixing price logic here would violate single responsibility.

    Parameters
    ----------
    probability : float
        Model's predicted probability that the stock will rise.
        Expected range: [0.0, 1.0].
    threshold : float
        Minimum probability required to generate a BUY signal.
        Expected range: (0.0, 1.0).

    Returns
    -------
    dict[str, float | str]
        Always contains:
        - ``"probability"`` : float — the input probability (passed
          through for convenience so the caller doesn't need to track
          it separately).
        - ``"signal"`` : str — either ``"BUY"`` or ``"HOLD"``.
        - ``"confidence"`` : str — ``"HIGH"``, ``"MEDIUM"``, or
          ``"LOW"`` based on how far the probability is from the
          threshold (see below).

    Notes
    -----
    Confidence tiers are defined relative to the threshold:
        - HIGH:   probability >= threshold + 0.15
        - MEDIUM: probability >= threshold + 0.05
        - LOW:    probability >= threshold (but below MEDIUM)

    For HOLD signals, confidence is always ``"N/A"`` because the
    concept doesn't apply — we're not buying.

    Examples
    --------
    >>> generate_signal(0.85, threshold=0.70)
    {'probability': 0.85, 'signal': 'BUY', 'confidence': 'HIGH'}

    >>> generate_signal(0.50, threshold=0.70)
    {'probability': 0.50, 'signal': 'HOLD', 'confidence': 'N/A'}
    """
    # ------------------------------------------------------------------
    # Core decision: is the probability above the threshold?
    # This single comparison is the heart of the entire system.
    # Everything else — data download, feature engineering, model
    # training — exists to make THIS comparison meaningful.
    # ------------------------------------------------------------------
    if probability >= threshold:
        signal = "BUY"

        # --------------------------------------------------------------
        # Confidence tiers add nuance to the binary BUY signal.
        #
        # WHY THIS MATTERS:
        #   A BUY at 0.71 (barely above a 0.70 threshold) is very
        #   different from a BUY at 0.92. Both are "buy", but the
        #   second one has much higher model conviction. The screener
        #   can use this to sort results, and the user can use it to
        #   decide how much attention to give each signal.
        #
        #   The tiers are relative to the threshold (not absolute)
        #   so they adapt automatically if the threshold changes.
        # --------------------------------------------------------------
        distance = probability - threshold

        if distance >= 0.15:
            confidence = "HIGH"
        elif distance >= 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        signal = "HOLD"
        confidence = "N/A"

    return {
        "probability": probability,
        "signal": signal,
        "confidence": confidence,
    }