#!/usr/bin/env python3
"""Rank fallback endpoints for a FastNS-to-safe table schedule.

This script mines the prepared Muon coefficient table and checks which endpoint
keys can be reached from the FastNS table key by scheduling only the scalar
target a. The actual Newton-Schulz coefficients are always selected from the
prepared table by nearest-lower lookup. The score is intentionally simple and
scalar-map based; it is a starting point for later energy-weighted spectrum
objectives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muon.coeff_table import FAST_NS_TABLE_KEY, JORDAN_NS, MUON_COEFF_TABLE  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-points', type=int, default=5000)
    parser.add_argument('--top-k', type=int, default=15)
    return parser.parse_args()


def scalar_metrics(coeffs, xs):
    a, b, c = coeffs
    x = xs.copy()
    orbit = float(np.max(np.abs(x)))
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(5):
            x = a * x + b * x**3 + c * x**5
            if not np.isfinite(x).all():
                return None
            orbit = max(orbit, float(np.max(np.abs(x))))
    idx_1e2 = int(np.searchsorted(xs, 1e-2))
    return {
        'lo': float(x.min()),
        'hi': float(x.max()),
        'orbit': orbit,
        'amp_1e3': float(x[0] / xs[0]),
        'amp_1e2': float(x[idx_1e2] / xs[idx_1e2]),
        'tail_ratio': float(x[0] / x.max()),
    }


def table_path_metrics(start_key, end_key, table, xs):
    lo_key, hi_key = sorted((float(start_key), float(end_key)))
    rows = []
    for key in sorted(table):
        if lo_key <= float(key) <= hi_key:
            metrics = scalar_metrics(table[key], xs)
            if metrics is None:
                return None
            rows.append(metrics)
    if not rows:
        return None
    return {
        'lo': min(row['lo'] for row in rows),
        'hi': max(row['hi'] for row in rows),
        'orbit': max(row['orbit'] for row in rows),
        'entries': len(rows),
    }


def endpoint_score(key, metrics):
    # Prefer lower overshoot/orbit and larger lower-tail lift. This is not a
    # final scientific objective; it is a reproducible scalar-map heuristic.
    return (2.0 * metrics['tail_ratio'] + metrics['lo']
            - 2.0 * (metrics['hi'] - 1.0) - (metrics['orbit'] - 1.0)
            - 0.02 * abs(float(key) - FAST_NS_TABLE_KEY))


def main():
    args = parse_args()
    xs = np.logspace(-3, 0, args.grid_points)
    rows = []
    for key, coeffs in MUON_COEFF_TABLE.items():
        if key < 3.5 or key > FAST_NS_TABLE_KEY:
            continue
        metrics = scalar_metrics(coeffs, xs)
        path = table_path_metrics(FAST_NS_TABLE_KEY, key, MUON_COEFF_TABLE, xs)
        if metrics is None or path is None:
            continue
        strict_path = path['lo'] >= 0.5 and path['hi'] <= 1.5 and path['orbit'] <= 2.0
        if not strict_path:
            continue
        rows.append((endpoint_score(key, metrics), key, coeffs, metrics, path))

    rows.sort(reverse=True, key=lambda row: row[0])
    print('rank,key,a,b,c,score,f5_min,f5_max,orbit,amp_1e-3,amp_1e-2,tail_ratio,path_min,path_max,path_orbit,path_entries')
    for rank, (score, key, coeffs, metrics, path) in enumerate(rows[:args.top_k], start=1):
        a, b, c = coeffs
        print(
            f'{rank},{key:.2f},{a:.6f},{b:.6f},{c:.6f},{score:.6f},'
            f"{metrics['lo']:.6f},{metrics['hi']:.6f},{metrics['orbit']:.6f},"
            f"{metrics['amp_1e3']:.6f},{metrics['amp_1e2']:.6f},{metrics['tail_ratio']:.6f},"
            f"{path['lo']:.6f},{path['hi']:.6f},{path['orbit']:.6f},{path['entries']}"
        )
    jordan = scalar_metrics(JORDAN_NS, xs)
    jordan_path = table_path_metrics(FAST_NS_TABLE_KEY, 3.44, MUON_COEFF_TABLE, xs)
    print('# Jordan reference:', jordan, 'path=', jordan_path, file=sys.stderr)


if __name__ == '__main__':
    main()
