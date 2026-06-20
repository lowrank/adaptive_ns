"""Coefficient search: find stable (a,b,c) triples for Newton-Schulz.

The search follows the 5-iteration band criterion from
``ns_coeff_search_algorithm.pdf``: sample triples, apply the scalar polynomial on
a log-spaced grid in ``[1e-3, 1]`` for five iterations, and keep candidates whose
final values land in ``[0.5, 1.5]`` without orbit escape.
"""

import numpy as np
from .coeff_table import (
    DEFAULT_COEFF_TABLE,
    FAST_NS,
    JORDAN_NS,
    STANDARD_NS,
)
from .newton_schulz import is_stable

def _batch_stability_mask(a, b, c, xs, K=5, band=(0.5, 1.5), orbit_bound=2.0):
    """Vectorized stability check for candidate coefficient arrays."""
    x = np.broadcast_to(xs, (len(a), len(xs))).copy()
    max_orbit = np.zeros(len(a), dtype=float)

    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(K):
            x2 = x * x
            x = a[:, None] * x + b[:, None] * x * x2 + c[:, None] * x * x2 * x2
            max_orbit = np.maximum(max_orbit, np.max(np.abs(x), axis=1))

    finite = np.isfinite(x).all(axis=1)
    lo, hi = band
    final_min = np.min(x, axis=1)
    final_max = np.max(x, axis=1)
    return finite & (max_orbit <= orbit_bound) & (final_min >= lo) & (final_max <= hi)


def search_coefficients(n_samples=50000, a_range=(1.5, 5.5),
                        b_range=(-7.5, 0.0), c_range=(0.5, 4.5),
                        seed=0, n_pts=300, batch_size=8192):
    """Random search for stable (a,b,c) that maximizes a.

    For each randomly sampled triple, check the strict 5-iteration band
    criterion from the PDF. Keep the stable candidate with the largest a.

    Parameters
    ----------
    n_samples : int
        Number of random triples to test.
    a_range, b_range, c_range : tuple
        Search ranges.
    seed : int or None
        Random seed. The default is deterministic so benchmarks are
        reproducible.
    n_pts : int
        Number of log-spaced grid points in [1e-3, 1].
    batch_size : int
        Number of sampled triples evaluated per vectorized batch.

    Returns
    -------
    best : tuple (a, b, c)
        The stable triple with the largest a found. If no candidate is found,
        returns STANDARD_NS as a conservative fallback.
    """
    rng = np.random.default_rng(seed)
    xs = np.logspace(-3, 0, n_pts)
    best = None
    remaining = int(n_samples)

    while remaining > 0:
        size = min(batch_size, remaining)
        remaining -= size

        a = rng.uniform(a_range[0], a_range[1], size)
        b = rng.uniform(b_range[0], b_range[1], size)
        c = rng.uniform(c_range[0], c_range[1], size)

        ok = _batch_stability_mask(a, b, c, xs)
        if ok.any():
            idxs = np.flatnonzero(ok)
            idx = idxs[np.argmax(a[idxs])]
            candidate = (float(a[idx]), float(b[idx]), float(c[idx]))
            if best is None or candidate[0] > best[0]:
                best = candidate

    return best if best is not None else STANDARD_NS


def find_fast_coeffs(n_samples=200000):
    """Find the fastest stable coefficient triple (the "FastNS").

    Uses a wide search to maximize 'a' subject to stability.
    """
    return search_coefficients(
        n_samples=n_samples,
        a_range=(1.5, 5.5),
        b_range=(-7.5, 0.0),
        c_range=(0.5, 4.5),
    )


def build_coeff_table(a_values, n_samples=50000, seed=0, width=0.03):
    """Build a dense lookup table for aggressive adaptive scheduling.

    For each target, call search_coefficients restricted to a narrow range
    around that value. Targets below roughly 3.47 cannot satisfy the strict
    5-iteration lower band from x=1e-3, so this helper is intended for the
    aggressive part of the adaptive table.

    Parameters
    ----------
    a_values : list of float
        Target a values.
    n_samples : int
        Random triples tested per target.
    seed : int
        Base seed used to derive one deterministic stream per target.
    width : float
        Half-width of the target-specific a search interval.

    Returns
    -------
    table : dict
        {a_target: (a, b, c)} mapping each target a to a stable triple.
    """
    table = {}
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(a_values))
    for a_target, child_seed in zip(a_values, child_seeds):
        a_target = float(a_target)
        a_lo, a_hi = max(1.5, a_target - width), min(5.5, a_target + width)
        best = search_coefficients(
            n_samples=n_samples,
            a_range=(a_lo, a_hi),
            b_range=(-7.5, 0.0),
            c_range=(0.5, 4.5),
            seed=int(child_seed.generate_state(1)[0]),
        )
        if not is_stable(*best, n_pts=300):
            raise RuntimeError(
                f"no stable coefficients found near a={a_target:.2f}; "
                "increase n_samples or widen the search"
            )
        table[round(a_target, 2)] = best
    return table
