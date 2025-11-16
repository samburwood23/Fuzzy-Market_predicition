"""
fuzzy_variable.py
-----------------

This module defines classes for representing fuzzy variables and fuzzy
sets.  A fuzzy variable encapsulates one or more fuzzy sets
(membership functions) that describe linguistic terms such as
"low", "medium", "high".  Fuzzy sets assign degrees of membership to
crisp numeric values based on their underlying membership functions.

The ``FuzzyVariable`` class stores a dictionary of named fuzzy sets.
Each fuzzy set is represented by a callable taking a numeric input
and returning a value between 0 and 1 indicating the degree of
membership.  Additional metadata, such as the domain (minimum and
maximum values), can also be stored for documentation purposes.

The ``FuzzySet`` is a lightweight wrapper around a membership
function.  It provides a uniform interface and can store optional
parameters such as the type of membership function or descriptive
labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Tuple


@dataclass
class FuzzySet:
    """Represent a fuzzy set via a membership function.

    A fuzzy set is defined by a membership function that maps a
    numeric input to a degree of membership in the interval [0, 1].
    Additional attributes, such as a descriptive label or parameters,
    may be stored for reference.
    """

    func: Callable[[float], float]
    label: str = ""
    params: Tuple[float, ...] = field(default_factory=tuple)

    def membership(self, x: float) -> float:
        """Return the degree of membership of ``x`` in this fuzzy set.

        Parameters
        ----------
        x : float
            The crisp input value.

        Returns
        -------
        float
            A number between 0 and 1 representing the membership degree.
        """
        return float(self.func(x))


class FuzzyVariable:
    """A fuzzy variable composed of multiple fuzzy sets.

    Fuzzy variables model linguistic concepts by grouping together
    fuzzy sets representing qualitative labels such as ``low``,
    ``medium`` and ``high``.  Each set has an associated membership
    function.  The variable stores the domain of values over which it
    is defined, although the domain is not enforced when evaluating
    membership functions.
    """

    def __init__(self, name: str, domain: Tuple[float, float]):
        self.name = name
        self.domain = domain
        self.sets: Dict[str, FuzzySet] = {}

    def add_set(self, name: str, fuzzy_set: FuzzySet) -> None:
        """Add a named fuzzy set to the variable.

        Parameters
        ----------
        name : str
            The linguistic label for the fuzzy set (e.g. ``"low"``).
        fuzzy_set : FuzzySet
            The fuzzy set instance to associate with the label.
        """
        self.sets[name] = fuzzy_set

    def membership(self, set_name: str, x: float) -> float:
        """Return the membership degree of ``x`` for the fuzzy set ``set_name``.

        Parameters
        ----------
        set_name : str
            The name of the fuzzy set within this variable.
        x : float
            The crisp input value.

        Returns
        -------
        float
            The membership degree of ``x`` in the specified fuzzy set.
        """
        if set_name not in self.sets:
            raise ValueError(f"Set '{set_name}' not defined for variable '{self.name}'.")
        return self.sets[set_name].membership(x)

    def fuzzy_values(self, x: float) -> Dict[str, float]:
        """Compute membership degrees for all fuzzy sets at ``x``.

        Parameters
        ----------
        x : float
            Crisp input value.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping set names to membership degrees.
        """
        return {label: fs.membership(x) for label, fs in self.sets.items()}

    def __contains__(self, item: str) -> bool:
        return item in self.sets

    def __repr__(self) -> str:
        sets = ', '.join(self.sets.keys())
        return f"FuzzyVariable(name='{self.name}', sets=[{sets}])"
