"""
membership_functions.py
-----------------------

This module defines a collection of common membership functions used in
fuzzy logic systems. Membership functions map a real‑valued input into
a degree of membership between 0 and 1.  The shapes of the
membership functions determine how input values are interpreted by the
fuzzy system.  The functions implemented here are intentionally
simple; they can be extended or replaced as needed for more complex
applications.

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
  is formed.

**Gaussian membership function**
  Defined by a mean ``m`` and a standard deviation ``sigma``.  The
  function follows the normal bell curve.  It is smooth and
  differentiable everywhere.  For fuzzy systems that need smooth
  transitions this shape is often preferred.

**Sigmoid membership function**
  Defined by a midpoint ``m`` and a slope parameter ``s``.  The
  function transitions from 0 to 1 around ``m``.  Positive slopes
  produce an increasing function; negative slopes produce a decreasing
  function.

Each membership function is implemented as a simple Python function
accepting a numeric input and returning a float between 0 and 1.
"""

import math
from typing import Callable


def triangular(a: float, b: float, c: float) -> Callable[[float], float]:
    """Return a triangular membership function.

    Parameters
    ----------
    a : float
        Left base of the triangle.  Values at or below ``a`` have
        membership 0.
    b : float
        Peak of the triangle.  At ``b`` the membership reaches 1.
    c : float
        Right base of the triangle.  Values at or above ``c`` have
        membership 0.

    Returns
    -------
    Callable[[float], float]
        A function that computes the triangular membership for an
        input ``x``.
    """

    def func(x: float) -> float:
        if a == b and b == c:
            # Degenerate triangle: constant membership of 1 at a point
            return float(x == a)
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if (b - a) != 0 else 0.0
        # x > b
        return (c - x) / (c - b) if (c - b) != 0 else 0.0

    return func


def trapezoidal(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
    """Return a trapezoidal membership function.

    Parameters
    ----------
    a : float
        Left foot of the trapezoid.  Values at or below ``a`` have
        membership 0.
    b : float
        Left shoulder where the membership rises to 1.  If ``b == a``
        then the left edge is vertical.
    c : float
        Right shoulder where the membership begins to fall.  If
        ``c == d`` then the right edge is vertical.
    d : float
        Right foot of the trapezoid.  Values at or above ``d`` have
        membership 0.

    Returns
    -------
    Callable[[float], float]
        A function computing the trapezoidal membership for input ``x``.
    """

    def func(x: float) -> float:
        if a == b:
            left_slope = float('inf')  # Vertical rise
        else:
            left_slope = 1.0 / (b - a)
        if c == d:
            right_slope = float('inf')  # Vertical drop
        else:
            right_slope = 1.0 / (d - c)

        if x <= a or x >= d:
            return 0.0
        if x < b:
            return (x - a) * left_slope
        if x <= c:
            return 1.0
        # x > c and x < d
        return (d - x) * right_slope

    return func


def gaussian(m: float, sigma: float) -> Callable[[float], float]:
    """Return a Gaussian membership function.

    Parameters
    ----------
    m : float
        Mean of the Gaussian curve.
    sigma : float
        Standard deviation controlling the width of the curve.

    Returns
    -------
    Callable[[float], float]
        A function computing the Gaussian membership for input ``x``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive for gaussian membership functions")

    def func(x: float) -> float:
        # Gaussian function scaled so the peak is 1 when x == m
        return math.exp(-0.5 * ((x - m) / sigma) ** 2)

    return func


def sigmoid(m: float, s: float) -> Callable[[float], float]:
    """Return a sigmoid membership function.

    Parameters
    ----------
    m : float
        Midpoint (inflection point) of the sigmoid.
    s : float
        Slope parameter.  Positive ``s`` yields an increasing function; negative
        ``s`` yields a decreasing function.

    Returns
    -------
    Callable[[float], float]
        A function computing the sigmoid membership for input ``x``.
    """

    def func(x: float) -> float:
        # Avoid overflow for large exponentials by using a stable formula
        exponent = -s * (x - m)
        # When exponent is very large positive, e**exponent might overflow,
        # but the ratio simplifies to 0 or 1 accordingly.
        try:
            return 1.0 / (1.0 + math.exp(exponent))
        except OverflowError:
            return 0.0 if exponent > 0 else 1.0

    return func
