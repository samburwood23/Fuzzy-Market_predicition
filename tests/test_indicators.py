import numpy as np
import pytest

from fuzzy_analytics.indicators import (
    TradingIndicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
)


def test_rsi_extremes_and_flat():
    assert calculate_rsi(range(1, 30)) == 100.0
    assert calculate_rsi(range(30, 1, -1)) == 0.0
    assert calculate_rsi([5.0] * 20) == 50.0
    assert calculate_rsi([1.0]) == 50.0


def test_rsi_known_value():
    # 4 gains of 1 and 1 loss of 1 within the seed window -> RS = 4
    assert calculate_rsi([1, 2, 3, 2, 3, 4]) == pytest.approx(80.0)


def test_macd_sign_follows_trend():
    up = np.linspace(100, 150, 60)
    macd, _, _ = calculate_macd(up)
    assert macd > 0
    macd, _, _ = calculate_macd(up[::-1])
    assert macd < 0


def test_volatility_is_annualised():
    rng = np.random.default_rng(42)
    daily_sigma = 0.01
    prices = 100 * np.exp(np.cumsum(rng.normal(0, daily_sigma, 5000)))
    vol = calculate_volatility(prices, period=4999)
    assert vol == pytest.approx(daily_sigma * np.sqrt(252), rel=0.05)


def test_volatility_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        calculate_volatility([1.0, 0.0, 2.0])


def test_bollinger_band_ordering():
    middle, upper, lower = calculate_bollinger_bands([1, 2, 3, 4, 5])
    assert lower < middle < upper
    assert middle == pytest.approx(3.0)


def test_sma():
    assert calculate_sma([1, 2, 3, 4], period=2) == 3.5


def test_atr_constant_range():
    assert calculate_atr([11] * 20, [9] * 20, [10] * 20) == pytest.approx(2.0)


def test_namespace_class():
    assert TradingIndicators.calculate_rsi([1, 2]) == 100.0
