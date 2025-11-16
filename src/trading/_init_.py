"""
Trading module for fuzzy analytics.

This package provides the ``TradingFuzzySystem`` class which wraps a
Mamdani fuzzy inference system tailored for financial trading
applications.  It defines fuzzy variables for typical trading
indicators such as RSI, MACD histogram and volatility, constructs a
set of intuitive fuzzy rules, and exposes a convenient API for
evaluating trading conditions to produce buy/sell/hold signals along
with a confidence score.

Example
-------
>>> from fuzzy_analytics.trading import TradingFuzzySystem
>>> system = TradingFuzzySystem()
>>> result = system.evaluate(rsi=45, macd_histogram=0.05, volatility=0.2)
>>> result['action']  # e.g. 'buy'
>>> result['confidence']  # e.g. 0.63
"""

from .system import TradingFuzzySystem, TradingIndicators

__all__ = ["TradingFuzzySystem", "TradingIndicators"]
