#!/usr/bin/env python3
"""Build a dense Newton-Schulz coefficient table.

This implements the random-sampling procedure from ns_coeff_search_algorithm.pdf.
It first runs one global search over the aggressive a-range, buckets stable
samples by rounded a, then uses targeted refinement for missing or dense-grid
validation failures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muon.coeff_search import _batch_stability_mask, search_coefficients  # noqa: E402
from muon.newton_schulz import iterate_polynomial, orbit_max  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=float, default=3.50)
    parser.add_argument('--stop', type=float, default=3.89)
    parser.add_argument('--step', type=float, default=0.01)
    parser.add_argument('--samples', type=int, default=300000)
    parser.add_argument('--refine-samples', type=int, default=80000)
    parser.add_argument('--seed', type=int, default=202605)
    parser.add_argument('--batch-size', type=int, default=20000)
    parser.add_argument('--grid-points', type=int, default=300)
    parser.add_argument('--validation-points', type=int, default=5000)
    return parser.parse_args()


def target_values(start, stop, step):
    n = int(round((stop - start) / step)) + 1
    return [round(start + step * i, 2) for i in range(n)]


def validate(coeffs, n_pts):
    xs = np.logspace(-3, 0, n_pts)
    y = iterate_polynomial(xs, *coeffs, K=5)
    orbit = orbit_max(*coeffs, n_pts=n_pts)
    margin = min(float(y.min() - 0.5), float(1.5 - y.max()), float(2.0 - orbit))
    return margin >= 0.0, margin, float(y.min()), float(y.max()), float(orbit)


def main():
    args = parse_args()
    targets = target_values(args.start, args.stop, args.step)
    table = {target: None for target in targets}

    rng = np.random.default_rng(args.seed)
    xs = np.logspace(-3, 0, args.grid_points)
    remaining = args.samples
    stable_count = 0
    a_lo = args.start - 3 * args.step
    a_hi = args.stop + 3 * args.step

    while remaining > 0:
        size = min(args.batch_size, remaining)
        remaining -= size
        a = rng.uniform(a_lo, a_hi, size)
        b = rng.uniform(-7.5, 0.0, size)
        c = rng.uniform(0.5, 4.5, size)
        ok = _batch_stability_mask(a, b, c, xs)
        stable_count += int(ok.sum())
        for idx in np.flatnonzero(ok):
            key = round(float(a[idx]), 2)
            if key in table and (table[key] is None or a[idx] > table[key][0]):
                table[key] = (float(a[idx]), float(b[idx]), float(c[idx]))

    print(f'# global stable samples: {stable_count}', file=sys.stderr)

    for key in targets:
        coeffs = table[key]
        ok = False
        if coeffs is not None:
            ok, *_ = validate(coeffs, args.validation_points)
        if coeffs is None or not ok:
            coeffs = search_coefficients(
                n_samples=args.refine_samples,
                a_range=(key - args.step / 2, key + args.step / 2),
                b_range=(-7.5, 0.0),
                c_range=(0.5, 4.5),
                seed=args.seed + int(round(1000 * key)),
                n_pts=1000,
                batch_size=max(1000, args.batch_size // 4),
            )
            ok, *_ = validate(coeffs, args.validation_points)
            if not ok:
                raise RuntimeError(f'could not validate key {key:.2f}: {coeffs}')
            table[key] = coeffs

    print('MUON_COEFF_TABLE = {')
    for target in targets:
        coeffs = table[target]
        ok, margin, y_min, y_max, orbit = validate(coeffs, args.validation_points)
        a, b, c = coeffs
        print(
            f'    {target:.2f}: ({a:.6f}, {b:.6f}, {c:.6f}), '
            f'# f5=[{y_min:.3f}, {y_max:.3f}] orbit={orbit:.3f} margin={margin:.4f}'
        )
    print('}')


if __name__ == '__main__':
    main()
