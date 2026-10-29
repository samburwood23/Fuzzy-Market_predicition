"""
Technical indicator functions.

This module contains common financial indicator calculations used in
trading strategies.  These indicators can be combined with fuzzy
logic to derive higher level trading signals.  All functions accept
any iterable of floats (lists, NumPy arrays, pandas Series) and return
the most recent indicator value.  If the input is shorter than the
requested window the functions operate on whatever data is available.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Compute the exponential moving average (EMA) series of a 1D array."""
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_sma(prices: Iterable[float], period: int = 20) -> float:
    """Compute the simple moving average of the last ``period`` prices."""
    p = _as_array(prices)
    if p.size == 0:
        return 0.0
    return float(np.mean(p[-period:]))


def calculate_ema(prices: Iterable[float], period: int = 20) -> float:
    """Compute the latest exponential moving average value."""
    p = _as_array(prices)
    if p.size == 0:
        return 0.0
    return float(_ema(p, period)[-1])


def calculate_rsi(prices: Iterable[float], period: int = 14) -> float:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    The RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.  Values below 30 typically
    indicate oversold conditions; values above 70 indicate overbought
    conditions.

    Returns
    -------
    float
        The RSI value between 0 and 100.  50 is returned when there is
        insufficient data or the price has not moved at all.
    """
    p = _as_array(prices)
    if p.size < 2:
        return 50.0

    deltas = np.diff(p)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with a simple average, then apply Wilder's smoothing to the rest.
    window = min(period, len(deltas))
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    for gain, loss in zip(gains[window:], losses[window:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def calculate_macd(prices: Iterable[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Compute the Moving Average Convergence Divergence (MACD) indicator.

    The MACD is the difference between a fast and a slow exponential
    moving average.  A signal line is computed as an EMA of the MACD
    line.  The histogram represents the distance between the MACD and
    the signal line.

    Returns
    -------
    Tuple[float, float, float]
        The latest MACD value, signal value and histogram value
        (MACD - signal), all in price units.
    """
    p = _as_array(prices)
    if p.size < 2:
        return 0.0, 0.0, 0.0
    macd_line = _ema(p, fast) - _ema(p, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])


def calculate_volatility(prices: Iterable[float], period: int = 20, periods_per_year: int = 252) -> float:
    """Compute annualised historical volatility from log returns.

    Parameters
    ----------
    prices : Iterable[float]
        Sequence of strictly positive price observations.
    period : int, optional
        Number of recent returns to include (default 20).
    periods_per_year : int, optional
        Sampling frequency used for annualisation (252 for daily bars,
        52 for weekly, 12 for monthly).

    Returns
    -------
    float
        Estimated volatility expressed as a fraction (e.g. 0.25 for 25%).
    """
    p = _as_array(prices)
    if p.size < 3:
        return 0.0
    if np.any(p <= 0):
        raise ValueError("Volatility requires strictly positive prices.")
    returns = np.diff(np.log(p[-(period + 1):]))
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def calculate_bollinger_bands(prices: Iterable[float], window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands for a sequence of prices.

    Returns
    -------
    Tuple[float, float, float]
        ``(middle_band, upper_band, lower_band)`` for the most recent window.
    """
    p = _as_array(prices)
    if p.size == 0:
        return 0.0, 0.0, 0.0
    rolling = p[-window:]
    middle = np.mean(rolling)
    std = np.std(rolling)
    return float(middle), float(middle + num_std * std), float(middle - num_std * std)


def calculate_atr(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 14) -> float:
    """Compute the Average True Range (ATR) using Wilder's smoothing.

    The true range of a bar is the largest of: high − low,
    |high − previous close| and |low − previous close|.
    """
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    if not h.size == l.size == c.size:
        raise ValueError("high, low and close must have the same length.")
    if h.size == 0:
        return 0.0
    prev_close = np.concatenate(([c[0]], c[:-1]))
    true_range = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    window = min(period, true_range.size)
    atr = np.mean(true_range[:window])
    for tr in true_range[window:]:
        atr = (atr * (period - 1) + tr) / period
    return float(atr)


class TradingIndicators:
    """Namespace grouping the indicator functions for convenient access.

    >>> TradingIndicators.calculate_rsi([1, 2, 3, 2, 3, 4])
    80.0
    """

    calculate_sma = staticmethod(calculate_sma)
    calculate_ema = staticmethod(calculate_ema)
    calculate_rsi = staticmethod(calculate_rsi)
    calculate_macd = staticmethod(calculate_macd)
    calculate_volatility = staticmethod(calculate_volatility)
    calculate_bollinger_bands = staticmethod(calculate_bollinger_bands)
    calculate_atr = staticmethod(calculate_atr)
