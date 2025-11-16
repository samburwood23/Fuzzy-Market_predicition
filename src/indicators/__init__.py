"""
Technical indicator functions.

This package contains common financial indicator calculations used in
trading strategies.  These indicators can be combined with fuzzy
logic to derive higher level trading signals.  All functions accept
iterables of floats and return computed indicator values.  If the
input length is shorter than the required window the functions will
operate on whatever data is available.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def calculate_rsi(prices: Iterable[float], period: int = 14) -> float:
    """Compute the Relative Strength Index (RSI).

    The RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.  Values below 30 typically
    indicate oversold conditions; values above 70 indicate overbought
    conditions.

    Parameters
    ----------
    prices : Iterable[float]
        Sequence of price observations.
    period : int, optional
        Lookback period for the RSI calculation (default 14).

    Returns
    -------
    float
        The RSI value between 0 and 100.
    """
    # Convert to numpy array
    p = np.asarray(list(prices), dtype=float)
    if p.size < 2:
        return 50.0  # Neutral RSI if insufficient data

    # Compute price changes
    deltas = np.diff(p)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Use simple moving average of gains/losses
    window = min(period, len(gains))
    if window == 0:
        return 50.0
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])

    # Handle zero loss to avoid division by zero
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average (EMA) of a 1D array.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    span : int
        Span parameter for EMA; analogous to lookback period.

    Returns
    -------
    np.ndarray
        The EMA series of the same length as ``data``.
    """
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_macd(prices: Iterable[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Compute the Moving Average Convergence Divergence (MACD) indicator.

    The MACD is the difference between a fast and a slow exponential
    moving average.  A signal line is computed as an EMA of the MACD
    line.  The histogram represents the distance between the MACD and
    the signal line.

    Parameters
    ----------
    prices : Iterable[float]
        Sequence of price observations.
    fast : int, optional
        Span for the fast EMA (default 12).
    slow : int, optional
        Span for the slow EMA (default 26).
    signal : int, optional
        Span for the signal line EMA (default 9).

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the latest MACD value, the latest signal
        value and the latest histogram value (MACD - signal).
    """
    p = np.asarray(list(prices), dtype=float)
    if p.size < 2:
        return 0.0, 0.0, 0.0
    # Compute EMAs
    ema_fast = _ema(p, fast)
    ema_slow = _ema(p, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])


def calculate_volatility(prices: Iterable[float], period: int = 10) -> float:
    """Compute historical volatility as the standard deviation of returns.

    Volatility is a measure of how much the price fluctuates.  Here
    volatility is computed as the standard deviation of the percentage
    returns over the specified lookback period.  The result is
    annualised to facilitate comparison across different sampling
    frequencies.

    Parameters
    ----------
    prices : Iterable[float]
        Sequence of price observations.
    period : int, optional
        Number of recent observations to include (default 10).

    Returns
    -------
    float
        Estimated volatility expressed as a fraction (e.g. 0.25 for 25%).
    """
    p = np.asarray(list(prices), dtype=float)
    if p.size < 2:
        return 0.0
    # Use the most recent period
    p = p[-period:]
    # Compute log returns
    returns = np.diff(np.log(p))
    if returns.size == 0:
        return 0.0
    volatility = np.std(returns) * np.sqrt(len(returns))  # simple annualisation
    return float(volatility)


def calculate_bollinger_bands(prices: Iterable[float], window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands for a sequence of prices.

    Bollinger Bands consist of a moving average and upper/lower bands
    defined by a specified number of standard deviations away from
    that average.

    Parameters
    ----------
    prices : Iterable[float]
        Sequence of price observations.
    window : int, optional
        Rolling window length for the moving average (default 20).
    num_std : float, optional
        Number of standard deviations for the bands (default 2.0).

    Returns
    -------
    Tuple[float, float, float]
        A tuple (middle_band, upper_band, lower_band) representing
        the most recent values of the moving average and the upper
        and lower Bollinger Bands.
    """
    p = np.asarray(list(prices), dtype=float)
    if p.size == 0:
        return 0.0, 0.0, 0.0
    window = min(window, p.size)
    rolling = p[-window:]
    middle = np.mean(rolling)
    std = np.std(rolling)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return float(middle), float(upper), float(lower)
