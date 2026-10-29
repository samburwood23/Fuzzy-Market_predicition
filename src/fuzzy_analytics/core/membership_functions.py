"""
membership_functions.py
-----------------------

This module defines a collection of common membership functions used in
fuzzy logic systems. Membership functions map a real‑valued input into
a degree of membership between 0 and 1.  The shapes of the
membership functions determine how input values are interpreted by the
fuzzy system.

**Triangular membership function**
  Defined by three points ``(a, b, c)``.  The function returns 0
  outside the interval ``[a, c]``, rises linearly from ``a`` to ``b``
  where it reaches 1, and then falls linearly to 0 at ``c``.  When
  ``b`` equals ``a`` or ``c`` the triangle becomes a shoulder.

**Trapezoidal membership function**
  Defined by four points ``(a, b, c, d)``.  The function returns 0
  outside the interval ``[a, d]``, rises linearly from ``a`` to ``b``,
  stays at 1 between ``b`` and ``c``, and then falls linearly to 0 at
  ``d``.  If ``b`` equals ``a`` or ``c`` equals ``d`` then a shoulder
  is formed, which is the usual way to cover the edges of a domain.

**Gaussian membership function**
  Defined by a mean ``m`` and a standard deviation ``sigma``.  The
  function follows the normal bell curve.  It is smooth and
  differentiable everywhere.

**Sigmoid membership function**
  Defined by a midpoint ``m`` and a slope parameter ``s``.  The
  function transitions from 0 to 1 around ``m``.  Positive slopes
  produce an increasing function; negative slopes produce a decreasing
  function.

Every function returned here accepts either a scalar or a NumPy array.
Scalars produce a ``float``; arrays produce an array of the same shape.
Accepting arrays lets the inference engine evaluate a whole output
universe in a single call.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

Numeric = Union[float, np.ndarray]
MembershipFunction = Callable[[Numeric], Numeric]


def _result(y: np.ndarray) -> Numeric:
    """Return a ``float`` for 0‑d results and an array otherwise."""
    return float(y) if y.ndim == 0 else y


def trapezoidal(a: float, b: float, c: float, d: float) -> MembershipFunction:
    """Return a trapezoidal membership function.

    Parameters
    ----------
    a : float
        Left foot of the trapezoid.  Values below ``a`` have membership 0.
    b : float
        Left shoulder where the membership rises to 1.  If ``b == a``
        then the left edge is vertical and ``a`` itself has membership 1.
    c : float
        Right shoulder where the membership begins to fall.  If
        ``c == d`` then the right edge is vertical and ``d`` itself has
        membership 1.
    d : float
        Right foot of the trapezoid.  Values above ``d`` have membership 0.

    Raises
    ------
    ValueError
        If the points are not ordered ``a <= b <= c <= d``.
    """
    if not a <= b <= c <= d:
        raise ValueError(f"trapezoidal requires a <= b <= c <= d, got {(a, b, c, d)}")

    def func(x: Numeric) -> Numeric:
        x = np.asarray(x, dtype=float)
        rise = np.ones_like(x) if b == a else (x - a) / (b - a)
        fall = np.ones_like(x) if d == c else (d - x) / (d - c)
        y = np.clip(np.minimum(rise, fall), 0.0, 1.0)
        # Vertical edges have slope "1 everywhere"; zero out what lies beyond them.
        if a == b:
            y = np.where(x < a, 0.0, y)
        if c == d:
            y = np.where(x > d, 0.0, y)
        return _result(y)

    return func


def triangular(a: float, b: float, c: float) -> MembershipFunction:
    """Return a triangular membership function.

    Parameters
    ----------
    a : float
        Left base of the triangle.  Values below ``a`` have membership 0.
    b : float
        Peak of the triangle.  At ``b`` the membership reaches 1.
    c : float
        Right base of the triangle.  Values above ``c`` have membership 0.

    Raises
    ------
    ValueError
        If the points are not ordered ``a <= b <= c``.
    """
    if not a <= b <= c:
        raise ValueError(f"triangular requires a <= b <= c, got {(a, b, c)}")
    return trapezoidal(a, b, b, c)


def gaussian(m: float, sigma: float) -> MembershipFunction:
    """Return a Gaussian membership function.

    Parameters
    ----------
    m : float
        Mean of the Gaussian curve.
    sigma : float
        Standard deviation controlling the width of the curve.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive for gaussian membership functions")

    def func(x: Numeric) -> Numeric:
        x = np.asarray(x, dtype=float)
        # Scaled so the peak is 1 when x == m
        return _result(np.exp(-0.5 * ((x - m) / sigma) ** 2))

    return func


def sigmoid(m: float, s: float) -> MembershipFunction:
    """Return a sigmoid membership function.

    Parameters
    ----------
    m : float
        Midpoint (inflection point) of the sigmoid.
    s : float
        Slope parameter.  Positive ``s`` yields an increasing function; negative
        ``s`` yields a decreasing function.
    """

    def func(x: Numeric) -> Numeric:
        x = np.asarray(x, dtype=float)
        # The tanh form equals 1 / (1 + exp(-s(x - m))) but cannot overflow.
        return _result(0.5 * (1.0 + np.tanh(0.5 * s * (x - m))))

    return func
