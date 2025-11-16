"""
shareholder_value module.

This package defines a fuzzy system for evaluating overall
shareholder value by combining trading performance, marketing ROI,
risk considerations and market share growth.  The resulting score is
scaled between 0 and 100 and accompanied by a set of actionable
recommendations.
"""

from .system import ShareholderValueOptimizer

__all__ = ["ShareholderValueOptimizer"]
