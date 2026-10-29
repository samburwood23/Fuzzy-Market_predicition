"""
fuzzy_variable.py
-----------------

This module defines classes for representing fuzzy variables and fuzzy
sets.  A fuzzy variable encapsulates one or more fuzzy sets
(membership functions) that describe linguistic terms such as
"low", "medium", "high".  Fuzzy sets assign degrees of membership to
crisp numeric values based on their underlying membership functions.

The ``FuzzyVariable`` class stores a dictionary of named fuzzy sets
together with the domain (minimum and maximum values) the variable is
defined over.  Inputs outside the domain can be clipped back into it
with :meth:`FuzzyVariable.clip`, which the inference engine does
automatically so that out‑of‑range values still fire the edge sets.

The ``FuzzySet`` is a lightweight wrapper around a membership
function.  It provides a uniform interface and can store optional
parameters such as descriptive labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

from .membership_functions import Numeric


@dataclass
class FuzzySet:
    """Represent a fuzzy set via a membership function.

    A fuzzy set is defined by a membership function that maps a
    numeric input (scalar or array) to a degree of membership in the
    interval [0, 1].
    """

    func: Callable[[Numeric], Numeric]
    label: str = ""
    params: Tuple[float, ...] = field(default_factory=tuple)

    def membership(self, x: Numeric) -> Numeric:
        """Return the degree of membership of ``x`` in this fuzzy set.

        Scalars produce a ``float``; arrays produce an array of degrees.
        """
        y = self.func(x)
        return float(y) if np.ndim(y) == 0 else np.asarray(y, dtype=float)


class FuzzyVariable:
    """A fuzzy variable composed of multiple fuzzy sets.

    Fuzzy variables model linguistic concepts by grouping together
    fuzzy sets representing qualitative labels such as ``low``,
    ``medium`` and ``high``.  Each set has an associated membership
    function.
    """

    def __init__(self, name: str, domain: Tuple[float, float]):
        low, high = domain
        if low >= high:
            raise ValueError(f"Domain for '{name}' must satisfy min < max, got {domain}.")
        self.name = name
        self.domain = (float(low), float(high))
        self.sets: Dict[str, FuzzySet] = {}

    def add_set(self, name: str, fuzzy_set: FuzzySet) -> None:
        """Add a named fuzzy set (e.g. ``"low"``) to the variable."""
        self.sets[name] = fuzzy_set

    def membership(self, set_name: str, x: Numeric) -> Numeric:
        """Return the membership degree of ``x`` for the fuzzy set ``set_name``."""
        if set_name not in self.sets:
            raise ValueError(f"Set '{set_name}' not defined for variable '{self.name}'.")
        return self.sets[set_name].membership(x)

    def fuzzy_values(self, x: float) -> Dict[str, float]:
        """Compute membership degrees for all fuzzy sets at ``x``."""
        return {label: fs.membership(x) for label, fs in self.sets.items()}

    def best_set(self, x: float) -> Tuple[str, float]:
        """Return the name and degree of the set ``x`` belongs to most strongly."""
        return max(self.fuzzy_values(x).items(), key=lambda item: item[1])

    def clip(self, x: float) -> float:
        """Clamp ``x`` to the variable's domain."""
        low, high = self.domain
        return min(max(float(x), low), high)

    def __contains__(self, item: str) -> bool:
        return item in self.sets

    def __repr__(self) -> str:
        sets = ', '.join(self.sets.keys())
        return f"FuzzyVariable(name='{self.name}', sets=[{sets}])"
