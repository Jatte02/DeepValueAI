"""
NLP sentiment scoring pipeline — FinBERT on financial headlines.

Scores each headline using FinBERT (ProsusAI/finbert), aggregates
to daily ticker-level features, and stores the results as Parquet.

PIPELINE:
    1. Load raw headlines from ``news_database``
    2. Run FinBERT inference (GPU-accelerated on RTX 5070)
    3. Compute sentiment score: P(positive) - P(negative) ∈ [-1, +1]
    4. Aggregate to daily ticker-level features:
       - sentiment_mean: average sentiment of the day
       - sentiment_std: disagreement between headlines
       - news_volume: number of headlines
       - sentiment_max: most positive headline
       - sentiment_min: most negative headline
    5. Save aggregated scores as Parquet

POINT-IN-TIME SAFETY:
    Sentiment features are lagged by ``SENTIMENT_LAG_DAYS`` (1 business
    day by default) to prevent look-ahead bias from after-hours news.

DEVICE SELECTION:
    Automatically uses CUDA if available, otherwise falls back to CPU.
    RTX 5070 (12 GB VRAM) can handle FinBERT comfortably at batch_size=256.

Usage:
    # Score all headlines (GPU recommended)
    python -m core.sentiment_pipeline

    # Score with custom batch size
    python -m core.sentiment_pipeline --batch-size 128

    # Force CPU
    python -m core.sentiment_pipeline --device cpu
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (
    PATHS,
    SENTIMENT_FFILL_LIMIT,
    SENTIMENT_LAG_DAYS,
    setup_logging,
)

logger = logging.getLogger(__name__)

SENTIMENT_PATH = PATHS["sentiment_file"]

# FinBERT model from HuggingFace
FINBERT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_BATCH_SIZE = 256
MAX_TOKEN_LENGTH = 128


# ---------------------------------------------------------------------------
# FinBERT inference
# ---------------------------------------------------------------------------

def score_headlines(
    headlines: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """Score a list of headlines using FinBERT.

    Returns a sentiment score for each headline:
    ``P(positive) - P(negative)`` ∈ [-1, +1].

    Parameters
    ----------
    headlines : list[str]
        Raw headline texts.
    batch_size : int
        Batch size for GPU inference.
    device : str or None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).

    Returns
    -------
    np.ndarray
        1-D array of sentiment scores, same length as ``headlines``.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading FinBERT (%s) on %s for %d headlines...",
        FINBERT_MODEL_NAME, device, len(headlines),
    )

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.to(device)
    model.eval()

    scores = np.empty(len(headlines), dtype=np.float32)

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        # FinBERT labels: [positive, negative, neutral]
        sentiment = probs[:, 0] - probs[:, 1]  # P(pos) - P(neg)
        scores[i : i + len(batch)] = sentiment

        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info(
                "  Scored %d / %d headlines (%.1f%%)",
                i + len(batch), len(headlines),
                100 * (i + len(batch)) / len(headlines),
            )

    logger.info("FinBERT scoring complete: %d headlines scored.", len(scores))
    return scores


# ---------------------------------------------------------------------------
# Aggregation to daily ticker-level features
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    headlines_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate headline-level scores to daily ticker-level features.

    Before aggregation, each headline date is mapped to the **next
    trading day it would impact**:
        - Weekday news (Mon-Fri) → impacts next business day
          (the +1 BDay lag is applied later in merge_sentiment_pit)
        - Weekend news (Sat/Sun) → mapped to Monday before aggregation
          so that Saturday and Sunday news are combined with Monday's
          headlines for a unified daily score

    This ensures weekend/holiday news is correctly attributed to the
    first trading day when the market can react.

    Parameters
    ----------
    headlines_df : pd.DataFrame
        Must have columns: ``ticker``, ``date``, ``sentiment_score``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``sentiment_mean``,
        ``sentiment_std``, ``news_volume``, ``sentiment_max``,
        ``sentiment_min``.
    """
    df = headlines_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Map weekend dates to the next Monday (next trading day)
    # Saturday (dayofweek=5) → +2 days, Sunday (dayofweek=6) → +1 day
    dow = df["date"].dt.dayofweek
    df.loc[dow == 5, "date"] += pd.Timedelta(days=2)  # Sat → Mon
    df.loc[dow == 6, "date"] += pd.Timedelta(days=1)  # Sun → Mon

    agg = df.groupby(["ticker", "date"]).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        news_volume=("sentiment_score", "count"),
        sentiment_max=("sentiment_score", "max"),
        sentiment_min=("sentiment_score", "min"),
    ).reset_index()

    # Fill NaN std (happens when only 1 headline per day) with 0
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)

    logger.info(
        "Aggregated to %d ticker-day rows from %d headlines.",
        len(agg), len(df),
    )
    return agg


# ---------------------------------------------------------------------------
# PIT merge helper
# ---------------------------------------------------------------------------

def merge_sentiment_pit(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge daily sentiment features into prices with PIT lag.

    Applies a lag of ``SENTIMENT_LAG_DAYS`` (default: 1 business day)
    so that headlines from day T only appear in features for day T+1.
    Days with no news get forward-filled for up to
    ``SENTIMENT_FFILL_LIMIT`` days, then decay to 0 (neutral).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must have columns: ``ticker``, ``date``.
    sentiment_df : pd.DataFrame or None
        Daily aggregated sentiment. If None, loads from default path.

    Returns
    -------
    pd.DataFrame
        ``prices_df`` enriched with 5 sentiment feature columns.
    """
    if sentiment_df is None:
        sentiment_df = load_sentiment()

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["date"])

    # Apply PIT lag: shift sentiment date forward by SENTIMENT_LAG_DAYS
    sent["date"] = sent["date"] + pd.tseries.offsets.BDay(SENTIMENT_LAG_DAYS)

    # Merge on (ticker, date)
    sentiment_cols = [
        "sentiment_mean", "sentiment_std", "news_volume",
        "sentiment_max", "sentiment_min",
    ]
    merged = prices.merge(
        sent[["ticker", "date"] + sentiment_cols],
        on=["ticker", "date"],
        how="left",
    )

    # Forward-fill sentiment within each ticker (limited window)
    for col in sentiment_cols:
        if col == "news_volume":
            # No-news days get 0, not forward-filled
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged.groupby("ticker")[col].ffill(limit=SENTIMENT_FFILL_LIMIT)
            # After ffill limit, fill remaining NaN with 0 (neutral)
            merged[col] = merged[col].fillna(0.0)

    return merged


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_sentiment(df: pd.DataFrame, path: Path = SENTIMENT_PATH) -> None:
    """Save aggregated daily sentiment to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1_048_576
    logger.info("Saved sentiment → %s (%.1f MB, %d rows)", path, mb, len(df))


def load_sentiment(path: Path = SENTIMENT_PATH) -> pd.DataFrame:
    """Load aggregated daily sentiment from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment scores not found: {path}\n"
            "Run first: python -m core.sentiment_pipeline"
        )
    df = pd.read_parquet(path)
    logger.info(
        "Loaded sentiment: %d ticker-day rows, %d tickers",
        len(df), df["ticker"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> pd.DataFrame:
    """Run the full NLP pipeline: load headlines → score → aggregate → save.

    Parameters
    ----------
    batch_size : int
        FinBERT batch size.
    device : str or None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).

    Returns
    -------
    pd.DataFrame
        Aggregated daily sentiment features.
    """
    from core.news_database import load_headlines

    # Step 1: Load raw headlines
    headlines_df = load_headlines()
    logger.info("Loaded %d headlines for scoring.", len(headlines_df))

    # Step 2: Score with FinBERT
    headline_texts = headlines_df["headline"].tolist()
    scores = score_headlines(headline_texts, batch_size=batch_size, device=device)
    headlines_df["sentiment_score"] = scores

    # Step 3: Aggregate to daily ticker-level
    daily = aggregate_daily_sentiment(headlines_df)

    # Step 4: Save
    save_sentiment(daily)

    logger.info(
        "Sentiment pipeline complete: %d ticker-day rows, %d tickers.",
        len(daily), daily["ticker"].nunique(),
    )
    return daily


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment scoring on financial headlines",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for FinBERT inference (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Force device (default: auto-detect).",
    )
    args = parser.parse_args()

    run_pipeline(batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    setup_logging()
    main()
