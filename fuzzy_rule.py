"""
fuzzy_rule.py
-------------

This module defines the ``FuzzyRule`` class used to represent fuzzy
if–then rules.  A fuzzy rule consists of a set of antecedents and a
consequent.  Each antecedent associates a fuzzy variable with one of
its fuzzy sets (e.g. ``RSI is high``).  Multiple antecedents are
combined using the logical AND operator by default.  The consequent
similarly associates an output variable with a fuzzy set.

The evaluation of a fuzzy rule proceeds by computing the degree of
membership of each antecedent for given crisp input values and then
combining these degrees using a t‑norm (minimum) to form the degree
of firing.  The rule contributes to the aggregated fuzzy output by
truncating (clipping) the consequent membership function at the
firing strength.

For simplicity this implementation supports only a single consequent
and uses the minimum t‑norm for antecedent aggregation.  Extending
this class to support OR operations or other t‑norms is left as an
exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

from .fuzzy_variable import FuzzyVariable


Antecedent = Tuple[FuzzyVariable, str]
Consequent = Tuple[FuzzyVariable, str]


@dataclass
class FuzzyRule:
    """Represent a fuzzy if–then rule.

    Attributes
    ----------
    antecedents : List[Antecedent]
        A list of antecedents, each a tuple of (variable, set_name).
    consequent : Consequent
        The consequent of the rule, also (variable, set_name).
    weight : float
        Optional rule weight in [0, 1]; defaults to 1.  Allows certain
        rules to have greater or lesser influence.
    """

    antecedents: List[Antecedent]
    consequent: Consequent
    weight: float = 1.0

    def evaluate(self, inputs: Dict[str, float]) -> float:
        """Evaluate the rule for the given crisp inputs.

        Parameters
        ----------
        inputs : Dict[str, float]
            A mapping from variable name to crisp input value.

        Returns
        -------
        float
            The firing strength of the rule (between 0 and 1).
        """
        degrees = []
        for variable, set_name in self.antecedents:
            if variable.name not in inputs:
                raise ValueError(f"Missing input for variable '{variable.name}'.")
            x = inputs[variable.name]
            degree = variable.membership(set_name, x)
            degrees.append(degree)
        # Use min t-norm to combine antecedents
        firing_strength = min(degrees) * self.weight if degrees else 0.0
        return firing_strength

    def implication(self, inputs: Dict[str, float]) -> Callable[[float], float]:
        """Return a clipped consequent membership function.

        For Mamdani inference, the consequent membership function is
        truncated at the firing strength of the antecedents.

        Parameters
        ----------
        inputs : Dict[str, float]
            Crisp inputs for evaluating antecedents.

        Returns
        -------
        Callable[[float], float]
            A function mapping an output value to a membership degree
            clipped at the firing strength.
        """
        firing_strength = self.evaluate(inputs)
        output_variable, output_set_name = self.consequent
        # Retrieve the membership function of the output set
        def clipped_membership(x: float) -> float:
            return min(firing_strength, output_variable.membership(output_set_name, x))
        return clipped_membership
