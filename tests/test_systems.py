import numpy as np
import pytest

from fuzzy_analytics import (
    CustomerSegmentationSystem,
    MarketingFuzzySystem,
    ShareholderValueOptimizer,
    TradingFuzzySystem,
)
from fuzzy_analytics.trading import ACTIONS


@pytest.fixture(scope="module")
def trading():
    return TradingFuzzySystem()


class TestTrading:
    @pytest.mark.parametrize(
        "rsi, macd, expected",
        [
            (20, 0.5, "strong_buy"),
            (50, 0.5, "buy"),
            (50, 0.0, "hold"),
            (50, -0.5, "sell"),
            (85, -0.5, "strong_sell"),
            (20, -0.5, "hold"),
        ],
    )
    def test_actions(self, trading, rsi, macd, expected):
        result = trading.evaluate(rsi=rsi, macd_histogram=macd, volatility=0.1)
        assert result["action"] == expected
        assert 0.0 <= result["confidence"] <= 1.0

    def test_signal_is_monotonic_in_rsi(self, trading):
        signals = [trading.evaluate(rsi=r, macd_histogram=0.0, volatility=0.1)["signal"] for r in range(0, 101, 5)]
        assert all(a >= b - 1e-9 for a, b in zip(signals, signals[1:]))

    def test_high_volatility_dampens_signal(self, trading):
        calm = trading.evaluate(rsi=20, macd_histogram=0.5, volatility=0.1)["signal"]
        wild = trading.evaluate(rsi=20, macd_histogram=0.5, volatility=0.9)["signal"]
        assert 0 < wild < calm

    def test_custom_rsi_thresholds(self):
        aggressive = TradingFuzzySystem(rsi_oversold=20, rsi_overbought=80)
        default = TradingFuzzySystem()
        assert aggressive.evaluate(rsi=30, macd_histogram=0, volatility=0.1)["signal"] < \
            default.evaluate(rsi=30, macd_histogram=0, volatility=0.1)["signal"]

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            TradingFuzzySystem(rsi_oversold=60)

    def test_evaluate_prices(self, trading):
        prices = np.linspace(100, 80, 60)
        result = trading.evaluate_prices(prices)
        assert result["action"] in ACTIONS
        assert set(result["indicators"]) == {"rsi", "macd_histogram", "volatility"}
        # A steady decline is oversold while MACD has flattened out
        assert result["indicators"]["rsi"] < 30


@pytest.fixture(scope="module")
def marketing():
    return MarketingFuzzySystem()


@pytest.fixture(scope="module")
def optimizer():
    return ShareholderValueOptimizer()


class TestMarketing:
    @pytest.mark.parametrize(
        "engagement, conversion, roi, expected",
        [
            (0.9, 0.7, 4.0, "excellent"),
            (0.5, 0.25, 1.5, "average"),
            (0.0, 0.0, 0.0, "poor"),
        ],
    )
    def test_quality(self, marketing, engagement, conversion, roi, expected):
        result = marketing.evaluate_campaign(engagement_rate=engagement, conversion_rate=conversion, roi=roi)
        assert result["quality"] == expected
        assert result["recommendation"]

    def test_roi_above_universe_still_counts_as_high(self, marketing):
        capped = marketing.evaluate_campaign(engagement_rate=0.9, conversion_rate=0.7, roi=5.0)
        huge = marketing.evaluate_campaign(engagement_rate=0.9, conversion_rate=0.7, roi=50.0)
        assert capped == huge


class TestSegmentation:
    @pytest.mark.parametrize(
        "recency, frequency, spend, expected",
        [
            (5, 25, 9000, "champion"),
            (80, 10, 3000, "loyal"),
            (400, 1, 100, "at_risk"),
        ],
    )
    def test_segments(self, recency, frequency, spend, expected):
        result = CustomerSegmentationSystem().segment_customer(
            recency_days=recency, purchase_frequency=frequency, annual_spend=spend
        )
        assert result["segment"] == expected


class TestShareholderValue:
    def _score(self, optimizer, *args):
        keys = ("trading_performance", "marketing_roi", "customer_retention", "market_share_growth")
        return optimizer.optimize(**dict(zip(keys, args)))

    def test_ratings_ordered(self, optimizer):
        best = self._score(optimizer, 0.9, 4.0, 0.95, 0.2)
        middle = self._score(optimizer, 0.5, 1.5, 0.75, 0.0)
        worst = self._score(optimizer, 0.1, 0.3, 0.4, -0.1)
        assert best["score"] > middle["score"] > worst["score"]
        assert best["rating"] == "excellent"
        assert worst["rating"] == "poor"
        assert 0 <= worst["score"] <= best["score"] <= 100

    def test_actions_target_weak_metrics(self, optimizer):
        result = self._score(optimizer, 0.9, 4.0, 0.3, 0.2)
        assert len(result["actions"]) == 1
        assert "retention" in result["actions"][0]
        assert self._score(optimizer, 0.9, 4.0, 0.95, 0.2)["actions"] == []
