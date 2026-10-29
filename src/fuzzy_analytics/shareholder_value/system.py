"""
system.py
---------

This module provides ``ShareholderValueOptimizer``, a fuzzy inference
system that synthesises business performance metrics into a single
shareholder value score between 0 and 100:

* ``trading_performance`` (0–1): share of profitable or positive trading signals
* ``marketing_roi`` (0–5+): return on marketing spend as a multiple
* ``customer_retention`` (0–1): share of customers retained over the period
* ``market_share_growth`` (−0.2–0.3+): change in market share as a fraction

The score is mapped to a qualitative rating and accompanied by
recommended actions targeting the weakest metrics.
"""

from __future__ import annotations

from typing import Dict, List, Union

from ..core.fuzzy_variable import FuzzyVariable, FuzzySet
from ..core.fuzzy_rule import FuzzyRule
from ..core.fuzzy_system import FuzzyInferenceSystem
from ..core import membership_functions as mf


class ShareholderValueOptimizer:
    """Fuzzy system scoring overall shareholder value."""

    RECOMMENDATIONS = {
        "poor": "Value is being eroded; prioritise a turnaround of the weakest areas.",
        "fair": "Value is stable but under‑performing; target improvements in lagging metrics.",
        "good": "Value creation is healthy; reinforce strengths and address remaining gaps.",
        "excellent": "Exceptional value creation; sustain momentum and reinvest in growth.",
    }

    # Advice for each input when it sits in its weakest fuzzy set
    WEAKNESS_ACTIONS = {
        ("trading_performance", "weak"): "Review trading strategy: tighten risk limits and reassess signal thresholds.",
        ("marketing_roi", "low"): "Reallocate marketing spend from low‑ROI channels to proven performers.",
        ("customer_retention", "low"): "Invest in customer retention through loyalty programmes and service quality.",
        ("market_share_growth", "declining"): "Address market share loss by reviewing pricing and competitive positioning.",
    }

    def __init__(self) -> None:
        self.trading = FuzzyVariable("trading_performance", (0.0, 1.0))
        self.trading.add_set("weak", FuzzySet(mf.trapezoidal(0.0, 0.0, 0.25, 0.45)))
        self.trading.add_set("moderate", FuzzySet(mf.triangular(0.3, 0.5, 0.7)))
        self.trading.add_set("strong", FuzzySet(mf.trapezoidal(0.55, 0.75, 1.0, 1.0)))

        self.roi = FuzzyVariable("marketing_roi", (0.0, 5.0))
        self.roi.add_set("low", FuzzySet(mf.trapezoidal(0.0, 0.0, 0.5, 1.2)))
        self.roi.add_set("medium", FuzzySet(mf.triangular(0.8, 1.8, 2.8)))
        self.roi.add_set("high", FuzzySet(mf.trapezoidal(2.2, 3.5, 5.0, 5.0)))

        self.retention = FuzzyVariable("customer_retention", (0.0, 1.0))
        self.retention.add_set("low", FuzzySet(mf.trapezoidal(0.0, 0.0, 0.5, 0.7)))
        self.retention.add_set("medium", FuzzySet(mf.triangular(0.6, 0.75, 0.9)))
        self.retention.add_set("high", FuzzySet(mf.trapezoidal(0.8, 0.9, 1.0, 1.0)))

        self.growth = FuzzyVariable("market_share_growth", (-0.2, 0.3))
        self.growth.add_set("declining", FuzzySet(mf.trapezoidal(-0.2, -0.2, -0.05, 0.0)))
        self.growth.add_set("flat", FuzzySet(mf.triangular(-0.03, 0.0, 0.05)))
        self.growth.add_set("growing", FuzzySet(mf.trapezoidal(0.02, 0.1, 0.3, 0.3)))

        self.score = FuzzyVariable("score", (0.0, 100.0))
        self.score.add_set("poor", FuzzySet(mf.trapezoidal(0.0, 0.0, 15.0, 35.0)))
        self.score.add_set("fair", FuzzySet(mf.triangular(25.0, 45.0, 60.0)))
        self.score.add_set("good", FuzzySet(mf.triangular(50.0, 68.0, 85.0)))
        self.score.add_set("excellent", FuzzySet(mf.trapezoidal(75.0, 90.0, 100.0, 100.0)))

        trading, roi, retention, growth, out = self.trading, self.roi, self.retention, self.growth, self.score
        rules = [
            # Combined conditions carry the most weight
            FuzzyRule([(trading, "strong"), (roi, "high"), (retention, "high")], (out, "excellent")),
            FuzzyRule([(trading, "strong"), (retention, "high"), (growth, "growing")], (out, "excellent")),
            FuzzyRule([(roi, "high"), (retention, "high"), (growth, "growing")], (out, "excellent")),
            FuzzyRule([(trading, "strong"), (roi, "medium")], (out, "good")),
            FuzzyRule([(trading, "moderate"), (roi, "high")], (out, "good")),
            FuzzyRule([(retention, "high"), (growth, "growing")], (out, "good"), weight=0.5),
            FuzzyRule([(trading, "moderate"), (roi, "medium"), (retention, "medium")], (out, "fair")),
            FuzzyRule([(retention, "medium"), (growth, "flat")], (out, "fair"), weight=0.8),
            FuzzyRule([(trading, "weak"), (roi, "low")], (out, "poor")),
            FuzzyRule([(retention, "low"), (growth, "declining")], (out, "poor")),
            FuzzyRule([(trading, "weak"), (retention, "low")], (out, "poor")),
        ]
        # Weaker single‑metric rules so every input always influences the score
        for variable, (low, mid, high) in [
            (trading, ("weak", "moderate", "strong")),
            (roi, ("low", "medium", "high")),
            (retention, ("low", "medium", "high")),
            (growth, ("declining", "flat", "growing")),
        ]:
            rules.append(FuzzyRule([(variable, low)], (out, "poor"), weight=0.3))
            rules.append(FuzzyRule([(variable, mid)], (out, "fair"), weight=0.3))
            rules.append(FuzzyRule([(variable, high)], (out, "good"), weight=0.3))

        self.system = FuzzyInferenceSystem(
            input_variables=[trading, roi, retention, growth],
            output_variable=out,
            rules=rules,
            universe_resolution=501,
        )

    def optimize(
        self,
        *,
        trading_performance: float,
        marketing_roi: float,
        customer_retention: float,
        market_share_growth: float,
    ) -> Dict[str, Union[str, float, List[str]]]:
        """Score shareholder value and recommend actions.

        Returns
        -------
        Dict[str, Union[str, float, List[str]]]
            ``score`` (0–100), ``rating`` (``poor``, ``fair``, ``good`` or
            ``excellent``), an overall ``recommendation`` and a list of
            specific ``actions`` for metrics that are clearly weak.
        """
        inputs = {
            "trading_performance": trading_performance,
            "marketing_roi": marketing_roi,
            "customer_retention": customer_retention,
            "market_share_growth": market_share_growth,
        }
        score = self.system.evaluate(inputs)
        rating, _ = self.score.best_set(score)

        actions = []
        for (name, weak_set), action in self.WEAKNESS_ACTIONS.items():
            variable = self.system.input_variables[name]
            if variable.membership(weak_set, variable.clip(inputs[name])) >= 0.5:
                actions.append(action)

        return {
            "score": score,
            "rating": rating,
            "recommendation": self.RECOMMENDATIONS[rating],
            "actions": actions,
        }
