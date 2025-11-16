"""
fuzzy_system.py
---------------

This module provides a basic Mamdani fuzzy inference system.  A fuzzy
system defines a set of input variables, an output variable and a
collection of rules linking inputs to the output.  Given crisp
inputs the system computes the aggregated fuzzy output by
superimposing the contributions of each rule and then defuzzifies
the result using the centroid (center of gravity) method.

The implementation here assumes a single output variable.  For
systems with multiple outputs you may instantiate separate fuzzy
systems for each output or extend this class accordingly.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np

from .fuzzy_variable import FuzzyVariable
from .fuzzy_rule import FuzzyRule


class FuzzyInferenceSystem:
    """A Mamdani type fuzzy inference system.

    Parameters
    ----------
    input_variables : List[FuzzyVariable]
        List of fuzzy variables representing the system inputs.
    output_variable : FuzzyVariable
        The output variable of the system.
    rules : List[FuzzyRule]
        A collection of fuzzy rules mapping inputs to the output.
    universe_resolution : int, optional
        Number of samples of the output universe used for defuzzification.
    """

    def __init__(
        self,
        input_variables: List[FuzzyVariable],
        output_variable: FuzzyVariable,
        rules: List[FuzzyRule],
        universe_resolution: int = 1000,
    ) -> None:
        self.input_variables = {v.name: v for v in input_variables}
        self.output_variable = output_variable
        self.rules = rules
        self.universe_resolution = max(10, universe_resolution)

    def evaluate(self, inputs: Dict[str, float]) -> float:
        """Evaluate the fuzzy system for given crisp inputs.

        This method performs Mamdani inference: each rule's consequent
        membership function is clipped at the rule's firing strength and
        all the clipped membership functions are aggregated by taking
        their pointwise maximum.  The aggregated fuzzy set is then
        defuzzified using the centroid method.

        Parameters
        ----------
        inputs : Dict[str, float]
            Mapping of input variable names to crisp values.

        Returns
        -------
        float
            The crisp output value obtained by defuzzification.
        """
        # Verify that all required inputs are provided
        for name in self.input_variables:
            if name not in inputs:
                raise ValueError(f"Missing input for variable '{name}'.")

        # Build the universe of discourse for the output variable
        start, end = self.output_variable.domain
        x_values = np.linspace(start, end, self.universe_resolution)
        aggregated = np.zeros_like(x_values, dtype=float)

        # Aggregate contributions from each rule
        for rule in self.rules:
            clipped = np.array([rule.implication(inputs)(x) for x in x_values])
            aggregated = np.maximum(aggregated, clipped)

        # Defuzzify using the centroid (center of gravity)
        numerator = np.trapz(aggregated * x_values, x_values)
        denominator = np.trapz(aggregated, x_values)
        if denominator == 0:
            # No rules fired; default to midpoint of domain
            return (start + end) / 2.0
        return numerator / denominator
