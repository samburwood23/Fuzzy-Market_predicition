"""
Marketing module for fuzzy analytics.

This package defines fuzzy systems for evaluating marketing campaign
quality and segmenting customers.  The fuzzy marketing system
ingests campaign metrics such as engagement rate, conversion rate
and ROI to output a qualitative assessment and recommended action.
The customer segmentation system uses RFM‑style inputs (recency,
frequency and monetary value) to assign customers to segments.
"""

from .system import MarketingFuzzySystem, CustomerSegmentationSystem

__all__ = ["MarketingFuzzySystem", "CustomerSegmentationSystem"]
