"""
fuzzy_analytics package.

This package provides a high‑level interface for fuzzy logic based
analytics in trading, marketing and shareholder value optimisation.
Importing this package exposes the main systems for immediate use:

* ``TradingFuzzySystem`` – generate trading signals from market data.
* ``TradingIndicators`` – compute common technical indicators.
* ``MarketingFuzzySystem`` – evaluate marketing campaigns and provide recommendations.
* ``CustomerSegmentationSystem`` – perform fuzzy customer segmentation.
* ``ShareholderValueOptimizer`` – synthesise trading and marketing performance
  into a shareholder value score with strategic guidance.

Example
-------
>>> from fuzzy_analytics import TradingFuzzySystem, MarketingFuzzySystem
>>> trading_system = TradingFuzzySystem()
>>> marketing_system = MarketingFuzzySystem()
"""

from .trading import TradingFuzzySystem, TradingIndicators
from .marketing import MarketingFuzzySystem, CustomerSegmentationSystem
from .shareholder_value import ShareholderValueOptimizer

__all__ = [
    "TradingFuzzySystem",
    "TradingIndicators",
    "MarketingFuzzySystem",
    "CustomerSegmentationSystem",
    "ShareholderValueOptimizer",
]
