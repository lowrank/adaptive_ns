#!/usr/bin/env python3
"""Compute aggregate and per-problem ratio tables for a benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='adaptive_ns_suite')
    return parser.parse_args()


def read_summary(results_dir: Path):
    rows = []
    with (results_dir / 'summary.csv').open() as f:
        for row in csv.DictReader(f):
            numeric_keys = [
                'final_mean', 'final_std', 'min_mean', 'min_std',
                'spikes_mean', 'spikes_std', 'a_final_mean', 'a_final_std',
                'trigger_step_mean', 'trigger_step_std', 'trigger_rate',
                'current_spike_count_mean', 'current_spike_count_std',
                'previous_spike_count_mean', 'previous_spike_count_std',
                'spike_count_delta_mean', 'spike_count_delta_std',
            ]
            for key in numeric_keys:
                row[key] = float(row[key]) if key in row and row[key] != '' else float('nan')
            row['runs'] = int(row['runs'])
            rows.append(row)
    return rows


def write_ratio_tables(rows: list[dict], results_dir: Path):
    problems = list(dict.fromkeys(row['problem'] for row in rows))
    ratio_rows = []
    for problem in problems:
        problem_rows = [row for row in rows if row['problem'] == problem]
        best_final = min(row['final_mean'] for row in problem_rows)
        best_min = min(row['min_mean'] for row in problem_rows)
        for row in problem_rows:
            ratio_rows.append({
                'problem': problem,
                'method': row['method'],
                'final_mean': row['final_mean'],
                'final_ratio': row['final_mean'] / best_final,
                'min_mean': row['min_mean'],
                'min_ratio': row['min_mean'] / best_min,
                'is_final_best': row['final_mean'] == best_final,
            })

    fields = ['problem', 'method', 'final_mean', 'final_ratio', 'min_mean', 'min_ratio', 'is_final_best']
    with (results_dir / 'final_loss_ratios.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ratio_rows)
    with (results_dir / 'final_loss_ratios.json').open('w') as f:
        json.dump(ratio_rows, f, indent=2)
    return ratio_rows


def write_aggregate(rows: list[dict], results_dir: Path):
    problems = list(dict.fromkeys(row['problem'] for row in rows))
    methods = list(dict.fromkeys(row['method'] for row in rows))
    best_final = {problem: min(row['final_mean'] for row in rows if row['problem'] == problem) for problem in problems}
    best_min = {problem: min(row['min_mean'] for row in rows if row['problem'] == problem) for problem in problems}
    aggregate = []
    for method in methods:
        method_rows = [row for row in rows if row['method'] == method]
        final_ratios = [row['final_mean'] / best_final[row['problem']] for row in method_rows]
        min_ratios = [row['min_mean'] / best_min[row['problem']] for row in method_rows]
        finite_a = [row['a_final_mean'] for row in method_rows if not math.isnan(row['a_final_mean'])]
        finite_trigger = [row['trigger_step_mean'] for row in method_rows if not math.isnan(row.get('trigger_step_mean', float('nan')))]
        finite_trigger_rate = [row['trigger_rate'] for row in method_rows if not math.isnan(row.get('trigger_rate', float('nan')))]
        aggregate.append({
            'method': method,
            'geomean_final_ratio': math.exp(sum(math.log(x) for x in final_ratios) / len(final_ratios)),
            'geomean_min_ratio': math.exp(sum(math.log(x) for x in min_ratios) / len(min_ratios)),
            'final_wins': sum(abs(row['final_mean'] - best_final[row['problem']]) < 1e-15 for row in method_rows),
            'mean_spikes': sum(row['spikes_mean'] for row in method_rows) / len(method_rows),
            'mean_a_final': sum(finite_a) / len(finite_a) if finite_a else float('nan'),
            'mean_trigger_step': sum(finite_trigger) / len(finite_trigger) if finite_trigger else float('nan'),
            'mean_trigger_rate': sum(finite_trigger_rate) / len(finite_trigger_rate) if finite_trigger_rate else float('nan'),
        })

    with (results_dir / 'aggregate.json').open('w') as f:
        json.dump(aggregate, f, indent=2)
    with (results_dir / 'aggregate.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)
    return aggregate


def main():
    args = parse_args()
    results_dir = ROOT / 'results' / args.run_name
    if not (results_dir / 'summary.csv').exists():
        raise FileNotFoundError(f'missing summary.csv in {results_dir}')
    rows = read_summary(results_dir)
    write_ratio_tables(rows, results_dir)
    aggregate = write_aggregate(rows, results_dir)
    print(f'Wrote aggregate and final-loss ratio tables in {results_dir}')
    for row in aggregate:
        a = row['mean_a_final']
        a_text = '--' if math.isnan(a) else f'{a:.2f}'
        trigger = row.get('mean_trigger_step', float('nan'))
        trigger_text = '--' if math.isnan(trigger) else f'{trigger:.1f}'
        print(
            f"{row['method']:<22s} final_ratio={row['geomean_final_ratio']:.3f} "
            f"min_ratio={row['geomean_min_ratio']:.3f} wins={row['final_wins']} "
            f"spikes={row['mean_spikes']:.2f} a={a_text} trigger={trigger_text}"
        )


if __name__ == '__main__':
    main()
