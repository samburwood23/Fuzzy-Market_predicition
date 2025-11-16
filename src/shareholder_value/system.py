"""
system.py
---------

This module provides two classes:

* ``MarketingFuzzySystem`` – evaluate marketing campaign performance.
* ``CustomerSegmentationSystem`` – segment customers based on RFM inputs.

Both classes leverage a generic Mamdani fuzzy inference engine.  They
define input and output variables, membership functions and fuzzy
rules appropriate to their respective domains.  Results are
presented in human‑readable form with qualitative labels and
recommendations.
"""

from __future__ import annotations

from typing import Dict

from ..core.fuzzy_variable import FuzzyVariable, FuzzySet
from ..core.fuzzy_rule import FuzzyRule
from ..core.fuzzy_system import FuzzyInferenceSystem
from ..core import membership_functions as mf


class MarketingFuzzySystem:
    """Fuzzy inference system for marketing campaign evaluation.

    Inputs consist of:
      - ``engagement_rate`` (0–1): fraction of the audience that engaged
      - ``conversion_rate`` (0–1): fraction of engaged users who converted
      - ``roi`` (0–∞): return on investment ratio

    The output variable ``quality`` ranges from 0 (poor) to 1
    (excellent).  After defuzzification, the crisp quality score is
    mapped to a qualitative label (``poor``, ``average``, ``good`` or
    ``excellent``) and an associated recommendation.
    """

    def __init__(self) -> None:
        # Input: engagement rate (0–1)
        self.engagement = FuzzyVariable("engagement_rate", (0.0, 1.0))
        self.engagement.add_set("low", FuzzySet(mf.triangular(0.0, 0.1, 0.3), label="low"))
        self.engagement.add_set("medium", FuzzySet(mf.triangular(0.2, 0.5, 0.8), label="medium"))
        self.engagement.add_set("high", FuzzySet(mf.triangular(0.6, 0.8, 1.0), label="high"))

        # Input: conversion rate (0–1)
        self.conversion = FuzzyVariable("conversion_rate", (0.0, 1.0))
        self.conversion.add_set("low", FuzzySet(mf.triangular(0.0, 0.05, 0.15), label="low"))
        self.conversion.add_set("medium", FuzzySet(mf.triangular(0.1, 0.25, 0.4), label="medium"))
        self.conversion.add_set("high", FuzzySet(mf.triangular(0.3, 0.6, 1.0), label="high"))

        # Input: ROI (0–∞).  We cap at 5 for the universe to handle typical ranges
        self.roi = FuzzyVariable("roi", (0.0, 5.0))
        self.roi.add_set("low", FuzzySet(mf.triangular(0.0, 0.5, 1.0), label="low"))
        self.roi.add_set("medium", FuzzySet(mf.triangular(0.8, 1.5, 2.5), label="medium"))
        self.roi.add_set("high", FuzzySet(mf.triangular(2.0, 3.5, 5.0), label="high"))

        # Output: quality (0–1)
        self.quality = FuzzyVariable("quality", (0.0, 1.0))
        self.quality.add_set("poor", FuzzySet(mf.triangular(0.0, 0.1, 0.3), label="poor"))
        self.quality.add_set("average", FuzzySet(mf.triangular(0.25, 0.4, 0.55), label="average"))
        self.quality.add_set("good", FuzzySet(mf.triangular(0.5, 0.7, 0.85), label="good"))
        self.quality.add_set("excellent", FuzzySet(mf.triangular(0.8, 0.9, 1.0), label="excellent"))

        # Build rules
        rules = []
        # Excellent campaigns: all inputs high
        rules.append(
            FuzzyRule(
                antecedents=[(self.engagement, "high"), (self.conversion, "high"), (self.roi, "high")],
                consequent=(self.quality, "excellent"),
            )
        )
        # Good campaigns: high engagement and medium conversion/roi
        rules.append(
            FuzzyRule(
                antecedents=[(self.engagement, "high"), (self.conversion, "medium"), (self.roi, "medium")],
                consequent=(self.quality, "good"),
            )
        )
        rules.append(
            FuzzyRule(
                antecedents=[(self.engagement, "medium"), (self.conversion, "high"), (self.roi, "medium")],
                consequent=(self.quality, "good"),
            )
        )
        # Average campaigns: medium engagement and medium conversion and medium roi
        rules.append(
            FuzzyRule(
                antecedents=[(self.engagement, "medium"), (self.conversion, "medium"), (self.roi, "medium")],
                consequent=(self.quality, "average"),
            )
        )
        # Poor campaigns: low engagement and low conversion
        rules.append(
            FuzzyRule(
                antecedents=[(self.engagement, "low"), (self.conversion, "low")],
                consequent=(self.quality, "poor"),
            )
        )
        # Poor campaigns: high spend (low roi) with low engagement or conversion
        rules.append(
            FuzzyRule(
                antecedents=[(self.roi, "low"), (self.engagement, "low")],
                consequent=(self.quality, "poor"),
                weight=0.8,
            )
        )
        rules.append(
            FuzzyRule(
                antecedents=[(self.roi, "low"), (self.conversion, "low")],
                consequent=(self.quality, "poor"),
                weight=0.8,
            )
        )

        # Good campaigns: high ROI with either high engagement or high conversion
        rules.append(
            FuzzyRule(
                antecedents=[(self.roi, "high"), (self.engagement, "high"), (self.conversion, "medium")],
                consequent=(self.quality, "good"),
            )
        )
        rules.append(
            FuzzyRule(
                antecedents=[(self.roi, "high"), (self.conversion, "high"), (self.engagement, "medium")],
                consequent=(self.quality, "good"),
            )
        )

        self.system = FuzzyInferenceSystem(
            input_variables=[self.engagement, self.conversion, self.roi],
            output_variable=self.quality,
            rules=rules,
            universe_resolution=200,
        )

        # Map quality labels to recommended actions
        self.recommendations = {
            "poor": "Reevaluate campaign strategy and increase targeting efforts.",
            "average": "Optimize specific elements to improve conversion.",
            "good": "Maintain momentum and consider scaling successful tactics.",
            "excellent": "Double down on successful channels and reinvest gains.",
        }

    def evaluate_campaign(self, *, engagement_rate: float, conversion_rate: float, roi: float) -> Dict[str, str]:
        """Evaluate a marketing campaign and return quality and recommendation.

        Parameters
        ----------
        engagement_rate : float
            Engagement rate between 0 and 1.
        conversion_rate : float
            Conversion rate between 0 and 1.
        roi : float
            Return on investment ratio (0–∞).

        Returns
        -------
        Dict[str, str]
            Dictionary with ``quality`` label and ``recommendation`` text.
        """
        inputs = {
            "engagement_rate": float(engagement_rate),
            "conversion_rate": float(conversion_rate),
            "roi": float(roi),
        }
        crisp_output = self.system.evaluate(inputs)
        memberships = {name: fs.membership(crisp_output) for name, fs in self.quality.sets.items()}
        best_quality = max(memberships.items(), key=lambda item: item[1])[0]
        return {
            "quality": best_quality,
            "recommendation": self.recommendations[best_quality],
        }


class CustomerSegmentationSystem:
    """Fuzzy system for customer segmentation using RFM features.

    Inputs:
      - ``recency_days``: days since last purchase (0–365 or more)
      - ``purchase_frequency``: number of purchases in a period (e.g. per year)
      - ``annual_spend``: total spend over a year in monetary units

    Output variable ``segment`` is mapped to categories such as
    ``champion``, ``loyal``, ``potential_loyalist`` or ``at_risk``.  The
    defuzzified value is mapped to the segment with the highest
    membership degree.
    """

    def __init__(self) -> None:
        # Input: recency in days (0–365)
        self.recency = FuzzyVariable("recency_days", (0.0, 365.0))
        self.recency.add_set("very_recent", FuzzySet(mf.triangular(0.0, 15.0, 45.0), label="very_recent"))
        self.recency.add_set("recent", FuzzySet(mf.triangular(30.0, 90.0, 150.0), label="recent"))
        self.recency.add_set("not_recent", FuzzySet(mf.triangular(120.0, 240.0, 365.0), label="not_recent"))

        # Input: purchase frequency (0–30+)
        self.frequency = FuzzyVariable("purchase_frequency", (0.0, 30.0))
        self.frequency.add_set("low", FuzzySet(mf.triangular(0.0, 1.0, 5.0), label="low"))
        self.frequency.add_set("medium", FuzzySet(mf.triangular(4.0, 10.0, 16.0), label="medium"))
        self.frequency.add_set("high", FuzzySet(mf.triangular(12.0, 20.0, 30.0), label="high"))

        # Input: annual spend (0–10000+).  We cap at 10000 for the universe
        self.spend = FuzzyVariable("annual_spend", (0.0, 10000.0))
        self.spend.add_set("low", FuzzySet(mf.triangular(0.0, 500.0, 2000.0), label="low"))
        self.spend.add_set("medium", FuzzySet(mf.triangular(1500.0, 3500.0, 6000.0), label="medium"))
        self.spend.add_set("high", FuzzySet(mf.triangular(5000.0, 7500.0, 10000.0), label="high"))

        # Output: segment (0–1)
        self.segment = FuzzyVariable("segment", (0.0, 1.0))
        self.segment.add_set("at_risk", FuzzySet(mf.triangular(0.0, 0.1, 0.3), label="at_risk"))
        self.segment.add_set("potential_loyalist", FuzzySet(mf.triangular(0.25, 0.4, 0.55), label="potential_loyalist"))
        self.segment.add_set("loyal", FuzzySet(mf.triangular(0.5, 0.65, 0.8), label="loyal"))
        self.segment.add_set("champion", FuzzySet(mf.triangular(0.75, 0.9, 1.0), label="champion"))

        # Build rules
        rules = []
        # Champions: very recent purchase, high frequency, high spend
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "very_recent"), (self.frequency, "high"), (self.spend, "high")],
                consequent=(self.segment, "champion"),
            )
        )
        # Loyal: recent purchase, medium/high frequency, medium/high spend
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "recent"), (self.frequency, "medium"), (self.spend, "medium")],
                consequent=(self.segment, "loyal"),
            )
        )
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "recent"), (self.frequency, "high"), (self.spend, "medium")],
                consequent=(self.segment, "loyal"),
            )
        )
        # Potential loyalist: recent purchase, medium frequency, low/medium spend
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "recent"), (self.frequency, "medium"), (self.spend, "low")],
                consequent=(self.segment, "potential_loyalist"),
            )
        )
        # At risk: not recent purchase, low frequency
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "not_recent"), (self.frequency, "low")],
                consequent=(self.segment, "at_risk"),
            )
        )
        # At risk: not recent purchase and low spend
        rules.append(
            FuzzyRule(
                antecedents=[(self.recency, "not_recent"), (self.spend, "low")],
                consequent=(self.segment, "at_risk"),
                weight=0.8,
            )
        )
        # Champion: high spend and high frequency even if recency medium
        rules.append(
            FuzzyRule(
                antecedents=[(self.spend, "high"), (self.frequency, "high"), (self.recency, "recent")],
                consequent=(self.segment, "champion"),
                weight=0.8,
            )
        )

        self.system = FuzzyInferenceSystem(
            input_variables=[self.recency, self.frequency, self.spend],
            output_variable=self.segment,
            rules=rules,
            universe_resolution=200,
        )

    def segment_customer(self, *, recency_days: float, purchase_frequency: float, annual_spend: float) -> Dict[str, str]:
        """Segment a customer using fuzzy logic.

        Parameters
        ----------
        recency_days : float
            Days since the customer's last purchase.
        purchase_frequency : float
            Number of purchases in the time horizon (e.g. per year).
        annual_spend : float
            Total spend in the time horizon.

        Returns
        -------
        Dict[str, str]
            Dictionary with ``segment`` label and a brief description.
        """
        inputs = {
            "recency_days": float(recency_days),
            "purchase_frequency": float(purchase_frequency),
            "annual_spend": float(annual_spend),
        }
        crisp_output = self.system.evaluate(inputs)
        memberships = {name: fs.membership(crisp_output) for name, fs in self.segment.sets.items()}
        best_segment = max(memberships.items(), key=lambda item: item[1])[0]
        descriptions = {
            "champion": "Recent, frequent, and high‑value customer.",
            "loyal": "Regular customer with consistent purchases.",
            "potential_loyalist": "Emerging customer with growth potential.",
            "at_risk": "Customer with declining activity and spend.",
        }
        return {"segment": best_segment, "description": descriptions[best_segment]}
