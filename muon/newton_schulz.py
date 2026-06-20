"""Newton-Schulz iteration and polynomial stability checks.

The degree-5 NS polynomial: f(x) = ax + bx³ + cx⁵
Applied element-wise to singular values after Frobenius normalization.
"""

import numpy as np


def iterate_polynomial(x0, a, b, c, K=5):
    """Apply f(x) = ax + bx³ + cx⁵ for K iterations.

    Parameters
    ----------
    x0 : float or array
        Initial value(s).
    a, b, c : float
        Polynomial coefficients.
    K : int
        Number of iterations (default 5).

    Returns
    -------
    x : float or array
        Value after K iterations.
    """
    x = x0
    for _ in range(K):
        x = a * x + b * x**3 + c * x**5
    return x


def is_stable(a, b, c, epsilon=1e-3, n_pts=200):
    """Check if (a,b,c) satisfies the 5-iteration stability constraints.

    Two conditions must hold for all x in [epsilon, 1]:
      1. After 5 iterations: 0.5 ≤ f⁵(x) ≤ 1.5
      2. No intermediate orbit escape: |fᵏ(x)| ≤ 2 for k=1..5

    Parameters
    ----------
    a, b, c : float
        Polynomial coefficients.
    epsilon : float
        Lower bound of test range (default 1e-3).
    n_pts : int
        Number of log-spaced test points in [epsilon, 1].

    Returns
    -------
    ok : bool
        True if both constraints pass.
    """
    xs = np.logspace(np.log10(epsilon), 0, n_pts)
    x = xs.copy()
    for _ in range(5):
        x = a * x + b * x**3 + c * x**5
        if not np.isfinite(x).all():
            return False
        if (np.abs(x) > 2.0).any():
            return False
    return bool((x >= 0.5).all() and (x <= 1.5).all())


def orbit_max(a, b, c, n_pts=200):
    """Maximum absolute value of any intermediate iterate over [1e-3, 1].

    Useful for debugging: coefficients with orbit_max > 2.0 are unstable.

    Parameters
    ----------
    a, b, c : float
    n_pts : int

    Returns
    -------
    max_val : float
        Maximum |fᵏ(x)| for k=1..5, x ∈ [1e-3, 1]. inf if diverged.
    """
    xs = np.logspace(-3, 0, n_pts)
    worst = np.abs(xs)
    x = xs.copy()
    for _ in range(5):
        x = a * x + b * x**3 + c * x**5
        if not np.isfinite(x).all():
            return float('inf')
        worst = np.maximum(worst, np.abs(x))
    return float(worst.max())
