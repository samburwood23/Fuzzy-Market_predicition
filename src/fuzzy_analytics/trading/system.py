"""
system.py
---------

This module provides ``TradingFuzzySystem``, a Mamdani fuzzy inference
system that turns three technical indicators into a trading signal:

* ``rsi`` – Relative Strength Index (0–100)
* ``macd_histogram`` – MACD histogram expressed as a percentage of the
  current price, so that the same thresholds work for any asset
* ``volatility`` – annualised volatility as a fraction (0.25 = 25%)

The output ``signal`` ranges from −1 (strong sell) to +1 (strong buy).
It is mapped to one of five actions together with a confidence score.
High volatility pulls the signal towards ``hold``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Union

from ..core.fuzzy_variable import FuzzyVariable, FuzzySet
from ..core.fuzzy_rule import FuzzyRule
from ..core.fuzzy_system import FuzzyInferenceSystem
from ..core import membership_functions as mf
from ..indicators import TradingIndicators, calculate_macd, calculate_rsi, calculate_volatility

ACTIONS = ("strong_sell", "sell", "hold", "buy", "strong_buy")


class TradingFuzzySystem:
    """Fuzzy inference system producing buy/sell/hold signals.

    Parameters
    ----------
    rsi_oversold : float, optional
        RSI level below which the market is considered oversold (default 30).
    rsi_overbought : float, optional
        RSI level above which the market is considered overbought (default 70).
    """

    def __init__(self, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0) -> None:
        if not 10.0 <= rsi_oversold < 50.0 < rsi_overbought <= 90.0:
            raise ValueError("Require 10 <= rsi_oversold < 50 < rsi_overbought <= 90.")

        # Input: RSI (0–100)
        self.rsi = FuzzyVariable("rsi", (0.0, 100.0))
        self.rsi.add_set("oversold", FuzzySet(mf.trapezoidal(0.0, 0.0, rsi_oversold - 5, rsi_oversold + 10)))
        self.rsi.add_set("neutral", FuzzySet(mf.triangular(rsi_oversold, 50.0, rsi_overbought)))
        self.rsi.add_set("overbought", FuzzySet(mf.trapezoidal(rsi_overbought - 10, rsi_overbought + 5, 100.0, 100.0)))

        # Input: MACD histogram as % of price
        self.macd = FuzzyVariable("macd_histogram", (-1.0, 1.0))
        self.macd.add_set("bearish", FuzzySet(mf.trapezoidal(-1.0, -1.0, -0.25, 0.0)))
        self.macd.add_set("neutral", FuzzySet(mf.triangular(-0.1, 0.0, 0.1)))
        self.macd.add_set("bullish", FuzzySet(mf.trapezoidal(0.0, 0.25, 1.0, 1.0)))

        # Input: annualised volatility (0–1)
        self.volatility = FuzzyVariable("volatility", (0.0, 1.0))
        self.volatility.add_set("low", FuzzySet(mf.trapezoidal(0.0, 0.0, 0.1, 0.2)))
        self.volatility.add_set("medium", FuzzySet(mf.triangular(0.15, 0.3, 0.45)))
        self.volatility.add_set("high", FuzzySet(mf.trapezoidal(0.35, 0.5, 1.0, 1.0)))

        # Output: signal (−1 strong sell … +1 strong buy)
        self.signal = FuzzyVariable("signal", (-1.0, 1.0))
        self.signal.add_set("strong_sell", FuzzySet(mf.trapezoidal(-1.0, -1.0, -0.8, -0.5)))
        self.signal.add_set("sell", FuzzySet(mf.triangular(-0.7, -0.4, -0.1)))
        self.signal.add_set("hold", FuzzySet(mf.triangular(-0.25, 0.0, 0.25)))
        self.signal.add_set("buy", FuzzySet(mf.triangular(0.1, 0.4, 0.7)))
        self.signal.add_set("strong_buy", FuzzySet(mf.trapezoidal(0.5, 0.8, 1.0, 1.0)))

        rsi, macd, vol, out = self.rsi, self.macd, self.volatility, self.signal
        rules = [
            # Momentum and mean reversion agree
            FuzzyRule([(rsi, "oversold"), (macd, "bullish")], (out, "strong_buy")),
            FuzzyRule([(rsi, "overbought"), (macd, "bearish")], (out, "strong_sell")),
            # One indicator leans, the other is neutral
            FuzzyRule([(rsi, "oversold"), (macd, "neutral")], (out, "buy")),
            FuzzyRule([(rsi, "neutral"), (macd, "bullish")], (out, "buy")),
            FuzzyRule([(rsi, "overbought"), (macd, "neutral")], (out, "sell")),
            FuzzyRule([(rsi, "neutral"), (macd, "bearish")], (out, "sell")),
            # Nothing to act on
            FuzzyRule([(rsi, "neutral"), (macd, "neutral")], (out, "hold")),
            # Indicators disagree
            FuzzyRule([(rsi, "oversold"), (macd, "bearish")], (out, "hold"), weight=0.8),
            FuzzyRule([(rsi, "overbought"), (macd, "bullish")], (out, "hold"), weight=0.8),
            # Turbulent markets favour caution
            FuzzyRule([(vol, "high")], (out, "hold"), weight=0.7),
        ]

        self.system = FuzzyInferenceSystem(
            input_variables=[rsi, macd, vol],
            output_variable=out,
            rules=rules,
            universe_resolution=401,
        )

    def evaluate(self, *, rsi: float, macd_histogram: float, volatility: float) -> Dict[str, Union[str, float]]:
        """Evaluate market conditions and return a trading signal.

        Parameters
        ----------
        rsi : float
            Relative Strength Index (0–100).
        macd_histogram : float
            MACD histogram as a percentage of the current price.
        volatility : float
            Annualised volatility as a fraction.

        Returns
        -------
        Dict[str, Union[str, float]]
            ``signal`` (crisp value in [−1, 1]), ``action`` (one of
            ``strong_sell``, ``sell``, ``hold``, ``buy``, ``strong_buy``)
            and ``confidence`` (membership of ``signal`` in that action, 0–1).
        """
        signal = self.system.evaluate({"rsi": rsi, "macd_histogram": macd_histogram, "volatility": volatility})
        action, confidence = self.signal.best_set(signal)
        return {"signal": signal, "action": action, "confidence": confidence}

    def evaluate_prices(self, prices: Iterable[float], periods_per_year: int = 252) -> Dict[str, object]:
        """Compute indicators from a price history and evaluate them.

        Parameters
        ----------
        prices : Iterable[float]
            Closing prices, oldest first.  At least ~35 observations are
            recommended so the MACD has warmed up.
        periods_per_year : int, optional
            Sampling frequency used to annualise volatility.

        Returns
        -------
        Dict[str, object]
            The result of :meth:`evaluate` plus an ``indicators`` entry
            holding the computed inputs.
        """
        prices = list(prices)
        if len(prices) < 3:
            raise ValueError("At least 3 prices are required.")
        _, _, histogram = calculate_macd(prices)
        indicators = {
            "rsi": calculate_rsi(prices),
            "macd_histogram": float(100.0 * histogram / prices[-1]),
            "volatility": calculate_volatility(prices, periods_per_year=periods_per_year),
        }
        result: Dict[str, object] = dict(self.evaluate(**indicators))
        result["indicators"] = indicators
        return result


__all__ = ["ACTIONS", "TradingFuzzySystem", "TradingIndicators"]
