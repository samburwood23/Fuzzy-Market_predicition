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
systems for each output.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
    clip_inputs : bool, optional
        If true (the default), inputs outside a variable's domain are
        clamped to the nearest edge so that they still fire the edge sets.
    """

    def __init__(
        self,
        input_variables: List[FuzzyVariable],
        output_variable: FuzzyVariable,
        rules: List[FuzzyRule],
        universe_resolution: int = 1000,
        clip_inputs: bool = True,
    ) -> None:
        self.input_variables = {v.name: v for v in input_variables}
        self.output_variable = output_variable
        self.rules = rules
        self.universe_resolution = max(10, universe_resolution)
        self.clip_inputs = clip_inputs

        for rule in rules:
            if rule.consequent[0] is not output_variable:
                raise ValueError(f"Rule consequent must use output variable '{output_variable.name}': {rule}")
            for variable, _ in rule.antecedents:
                if variable.name not in self.input_variables:
                    raise ValueError(f"Rule uses unknown input variable '{variable.name}': {rule}")

        start, end = output_variable.domain
        self._universe = np.linspace(start, end, self.universe_resolution)

    def _prepare_inputs(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Validate inputs and clip them to their domains if configured."""
        prepared = {}
        for name, variable in self.input_variables.items():
            if name not in inputs:
                raise ValueError(f"Missing input for variable '{name}'.")
            value = float(inputs[name])
            if not np.isfinite(value):
                raise ValueError(f"Input '{name}' must be a finite number, got {inputs[name]!r}.")
            prepared[name] = variable.clip(value) if self.clip_inputs else value
        return prepared

    def rule_activations(self, inputs: Dict[str, float]) -> List[Tuple[FuzzyRule, float]]:
        """Return each rule paired with its firing strength for ``inputs``.

        Useful for explaining why the system produced a given output.
        """
        prepared = self._prepare_inputs(inputs)
        return [(rule, rule.evaluate(prepared)) for rule in self.rules]

    def evaluate(self, inputs: Dict[str, float]) -> float:
        """Evaluate the fuzzy system for given crisp inputs.

        Each rule's consequent membership function is clipped at the
        rule's firing strength and the clipped functions are aggregated
        by taking their pointwise maximum.  The aggregated fuzzy set is
        then defuzzified using the centroid method.  If no rule fires,
        the midpoint of the output domain is returned.
        """
        x_values = self._universe
        aggregated = np.zeros_like(x_values)
        for rule, strength in self.rule_activations(inputs):
            if strength <= 0.0:
                continue
            output_variable, set_name = rule.consequent
            clipped = np.minimum(strength, output_variable.membership(set_name, x_values))
            aggregated = np.maximum(aggregated, clipped)

        denominator = np.trapezoid(aggregated, x_values)
        if denominator == 0:
            start, end = self.output_variable.domain
            return (start + end) / 2.0
        return float(np.trapezoid(aggregated * x_values, x_values) / denominator)
