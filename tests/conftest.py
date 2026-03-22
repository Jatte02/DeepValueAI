"""Shared fixtures for the DeepValueAI test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate ~300 rows of synthetic OHLCV data with a DatetimeIndex.

    Prices start at 100 and follow a random walk.  Volume is random
    between 1M and 5M.  The data is deterministic (fixed seed) so
    tests are reproducible.
    """
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")

    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)  # keep prices positive

    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 1.0)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture()
def sample_market_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Market index DataFrame with the same dates as sample_ohlcv_df.

    Prices drift upward to simulate a bull-market index.
    """
    rng = np.random.default_rng(99)
    n = len(sample_ohlcv_df)
    close = 4000.0 + np.cumsum(rng.normal(0.5, 5, n))
    close = np.maximum(close, 100.0)

    high = close + rng.uniform(5, 20, n)
    low = close - rng.uniform(5, 20, n)
    open_ = close + rng.normal(0, 3, n)
    volume = rng.integers(2_000_000_000, 5_000_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=sample_ohlcv_df.index,
    )


class FakeModel:
    """Minimal sklearn-compatible model for testing.

    Always returns ``fixed_proba`` for the positive class.
    """

    def __init__(self, fixed_proba: float = 0.80):
        self.fixed_proba = fixed_proba

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        col0 = np.full(n, 1 - self.fixed_proba)
        col1 = np.full(n, self.fixed_proba)
        return np.column_stack([col0, col1])


@pytest.fixture()
def fake_model() -> FakeModel:
    """A FakeModel that always predicts probability 0.80."""
    return FakeModel(0.80)


@pytest.fixture()
def fake_model_low() -> FakeModel:
    """A FakeModel that always predicts probability 0.30 (below any threshold)."""
    return FakeModel(0.30)
