"""
fuzzy_rule.py
-------------

This module defines the ``FuzzyRule`` class used to represent fuzzy
if–then rules.  A fuzzy rule consists of a set of antecedents and a
consequent.  Each antecedent associates a fuzzy variable with one of
its fuzzy sets (e.g. ``RSI is high``).  Multiple antecedents are
combined using the logical AND operator.  The consequent similarly
associates an output variable with a fuzzy set.

The evaluation of a fuzzy rule proceeds by computing the degree of
membership of each antecedent for given crisp input values and then
combining these degrees using a t‑norm (minimum) to form the degree
of firing.  The rule contributes to the aggregated fuzzy output by
truncating (clipping) the consequent membership function at the
firing strength.

This implementation supports a single consequent and uses the minimum
t‑norm for antecedent aggregation.  OR conditions can be expressed by
writing one rule per alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .fuzzy_variable import FuzzyVariable
from .membership_functions import Numeric

Antecedent = Tuple[FuzzyVariable, str]
Consequent = Tuple[FuzzyVariable, str]


@dataclass
class FuzzyRule:
    """Represent a fuzzy if–then rule.

    Attributes
    ----------
    antecedents : List[Antecedent]
        A non‑empty list of antecedents, each a tuple of (variable, set_name).
    consequent : Consequent
        The consequent of the rule, also (variable, set_name).
    weight : float
        Rule weight in [0, 1]; defaults to 1.  Allows certain rules to
        have greater or lesser influence.
    """

    antecedents: List[Antecedent]
    consequent: Consequent
    weight: float = 1.0

    def __post_init__(self) -> None:
        # Catch typos in set names when the rule is built rather than
        # when it is first evaluated.
        if not self.antecedents:
            raise ValueError("A fuzzy rule needs at least one antecedent.")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Rule weight must be in [0, 1], got {self.weight}.")
        for variable, set_name in [*self.antecedents, self.consequent]:
            if set_name not in variable:
                raise ValueError(f"Set '{set_name}' not defined for variable '{variable.name}'.")

    def evaluate(self, inputs: Dict[str, float]) -> float:
        """Return the firing strength of the rule (between 0 and 1).

        Parameters
        ----------
        inputs : Dict[str, float]
            A mapping from variable name to crisp input value.
        """
        degrees = []
        for variable, set_name in self.antecedents:
            if variable.name not in inputs:
                raise ValueError(f"Missing input for variable '{variable.name}'.")
            degrees.append(variable.membership(set_name, inputs[variable.name]))
        # Use min t-norm to combine antecedents
        return min(degrees) * self.weight

    def implication(self, inputs: Dict[str, float]) -> Callable[[Numeric], Numeric]:
        """Return the consequent membership function clipped at the firing strength.

        The returned function accepts a scalar or an array of output values.
        """
        firing_strength = self.evaluate(inputs)
        output_variable, output_set_name = self.consequent

        def clipped_membership(x: Numeric) -> Numeric:
            return np.minimum(firing_strength, output_variable.membership(output_set_name, x))

        return clipped_membership

    def __str__(self) -> str:
        conditions = " AND ".join(f"{v.name} is {s}" for v, s in self.antecedents)
        output_variable, output_set_name = self.consequent
        text = f"IF {conditions} THEN {output_variable.name} is {output_set_name}"
        return text if self.weight == 1.0 else f"{text} (weight {self.weight:g})"
