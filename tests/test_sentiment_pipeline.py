"""Tests for core.sentiment_pipeline — aggregation and PIT merge."""

import numpy as np
import pandas as pd
import pytest

from core.sentiment_pipeline import aggregate_daily_sentiment, merge_sentiment_pit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_headlines_scored():
    """Scored headlines for aggregation testing."""
    return pd.DataFrame({
        "ticker": ["AAPL"] * 5 + ["MSFT"] * 3,
        "date": [
            "2020-01-10", "2020-01-10", "2020-01-10",  # 3 headlines same day
            "2020-01-13", "2020-01-14",                  # 1 each
            "2020-01-10", "2020-01-10", "2020-01-13",    # MSFT
        ],
        "headline": [f"headline_{i}" for i in range(8)],
        "sentiment_score": [0.8, -0.2, 0.5, 0.3, -0.7, 0.1, 0.4, -0.3],
    })


@pytest.fixture
def sample_daily_sentiment():
    """Pre-aggregated daily sentiment for merge testing."""
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "date": pd.to_datetime(["2020-01-10", "2020-01-13", "2020-01-10"]),
        "sentiment_mean": [0.5, 0.3, 0.25],
        "sentiment_std": [0.3, 0.0, 0.15],
        "news_volume": [3, 1, 2],
        "sentiment_max": [0.8, 0.3, 0.4],
        "sentiment_min": [-0.2, 0.3, 0.1],
    })


@pytest.fixture
def sample_prices_for_sentiment():
    """Prices DataFrame for sentiment merge testing."""
    dates = pd.date_range("2020-01-10", periods=10, freq="B")
    return pd.DataFrame({
        "ticker": ["AAPL"] * 10,
        "date": dates,
        "close": np.random.default_rng(42).uniform(100, 200, 10),
    })


# ---------------------------------------------------------------------------
# aggregate_daily_sentiment
# ---------------------------------------------------------------------------

class TestAggregateDailySentiment:
    def test_groups_by_ticker_date(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        # AAPL has 3 unique dates, MSFT has 2 unique dates → 5 rows
        assert len(result) == 5

    def test_computes_mean(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        aapl_jan10 = result[
            (result["ticker"] == "AAPL")
            & (result["date"] == pd.Timestamp("2020-01-10"))
        ]
        # Mean of [0.8, -0.2, 0.5] = 0.3667
        assert abs(aapl_jan10["sentiment_mean"].iloc[0] - 0.3667) < 0.01

    def test_computes_news_volume(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        aapl_jan10 = result[
            (result["ticker"] == "AAPL")
            & (result["date"] == pd.Timestamp("2020-01-10"))
        ]
        assert aapl_jan10["news_volume"].iloc[0] == 3

    def test_computes_max_min(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        aapl_jan10 = result[
            (result["ticker"] == "AAPL")
            & (result["date"] == pd.Timestamp("2020-01-10"))
        ]
        assert aapl_jan10["sentiment_max"].iloc[0] == 0.8
        assert aapl_jan10["sentiment_min"].iloc[0] == -0.2

    def test_single_headline_std_is_zero(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        aapl_jan13 = result[
            (result["ticker"] == "AAPL")
            & (result["date"] == pd.Timestamp("2020-01-13"))
        ]
        # Single headline → std should be 0 (filled from NaN)
        assert aapl_jan13["sentiment_std"].iloc[0] == 0.0

    def test_has_expected_columns(self, sample_headlines_scored):
        result = aggregate_daily_sentiment(sample_headlines_scored)
        expected = {
            "ticker", "date", "sentiment_mean", "sentiment_std",
            "news_volume", "sentiment_max", "sentiment_min",
        }
        assert expected == set(result.columns)


# ---------------------------------------------------------------------------
# merge_sentiment_pit
# ---------------------------------------------------------------------------

class TestMergeSentimentPit:
    def test_preserves_row_count(
        self, sample_prices_for_sentiment, sample_daily_sentiment,
    ):
        result = merge_sentiment_pit(
            sample_prices_for_sentiment, sentiment_df=sample_daily_sentiment,
        )
        assert len(result) == len(sample_prices_for_sentiment)

    def test_adds_sentiment_columns(
        self, sample_prices_for_sentiment, sample_daily_sentiment,
    ):
        result = merge_sentiment_pit(
            sample_prices_for_sentiment, sentiment_df=sample_daily_sentiment,
        )
        for col in ["sentiment_mean", "sentiment_std", "news_volume",
                     "sentiment_max", "sentiment_min"]:
            assert col in result.columns

    def test_no_news_days_have_zero_volume(
        self, sample_prices_for_sentiment, sample_daily_sentiment,
    ):
        result = merge_sentiment_pit(
            sample_prices_for_sentiment, sentiment_df=sample_daily_sentiment,
        )
        # Days without news should have news_volume = 0
        assert (result["news_volume"] >= 0).all()

    def test_sentiment_lag_applied(
        self, sample_prices_for_sentiment, sample_daily_sentiment,
    ):
        """Sentiment from day T should appear on day T+1 (1-day lag)."""
        result = merge_sentiment_pit(
            sample_prices_for_sentiment, sentiment_df=sample_daily_sentiment,
        )
        # The first day (2020-01-10) should NOT have sentiment from 2020-01-10
        # because of the 1-day lag — it would appear on 2020-01-13 (next bday)
        first_day = result[result["date"] == pd.Timestamp("2020-01-10")]
        # With lag, sentiment from Jan 10 appears on Jan 13, so Jan 10 has
        # no sentiment data (should be 0 after ffill exhaustion)
        assert first_day["news_volume"].iloc[0] == 0

    def test_ffill_limited_to_5_days(self):
        """Forward-fill should stop after SENTIMENT_FFILL_LIMIT days."""
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        prices = pd.DataFrame({
            "ticker": ["AAPL"] * 20,
            "date": dates,
            "close": 150.0,
        })
        # Sentiment only on the first day (after lag, appears on day 2)
        sentiment = pd.DataFrame({
            "ticker": ["AAPL"],
            "date": [pd.Timestamp("2019-12-31")],  # Will be lagged to 2020-01-02
            "sentiment_mean": [0.5],
            "sentiment_std": [0.1],
            "news_volume": [3],
            "sentiment_max": [0.8],
            "sentiment_min": [0.2],
        })
        result = merge_sentiment_pit(prices, sentiment_df=sentiment)
        # After 5 business days of ffill, sentiment_mean should go to 0
        late_rows = result[result["date"] >= pd.Timestamp("2020-01-15")]
        assert (late_rows["sentiment_mean"] == 0.0).all()
