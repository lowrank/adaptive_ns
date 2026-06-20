#!/usr/bin/env python3
"""Regenerate benchmark results, summaries, figures, optional diagnostics, and the report.

Default run:
    python3 scripts/generate_all_experiments.py

Fast smoke run:
    python3 scripts/generate_all_experiments.py --run-name smoke --epochs 5 --n-inits 1 --hidden 16 --skip-report-compile
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='adaptive_ns_suite')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-inits', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--adaptive-a-init', type=float, default=3.87)
    parser.add_argument('--adaptive-transition-start', type=int, default=-1, help='Fixed transition start; negative uses the spike-frequency trigger.')
    parser.add_argument('--adaptive-transition-steps', type=int, default=100)
    parser.add_argument('--adaptive-spike-window', type=int, default=40)
    parser.add_argument('--adaptive-spike-threshold', type=float, default=1.25)
    parser.add_argument('--adaptive-spike-ema-beta', type=float, default=0.98)
    parser.add_argument('--adaptive-min-spikes', type=int, default=2)
    parser.add_argument('--adaptive-spike-count-margin', type=int, default=1)
    parser.add_argument('--adam-lr', type=float, default=1e-3)
    parser.add_argument('--adamw-lr', type=float, default=1e-3)
    parser.add_argument('--adamw-weight-decay', type=float, default=1e-2)
    parser.add_argument('--clean', action='store_true', default=True, help='Remove this run_name under results/ and figures/ before running.')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--skip-benchmark', action='store_true', help='Only regenerate summaries/report from existing results.')
    parser.add_argument('--skip-report-compile', action='store_true')
    parser.add_argument('--rebuild-coeff-table', action='store_true', help='Print a regenerated coefficient table before experiments.')
    parser.add_argument('--skip-spectrum-diagnostics', action='store_true', help='Skip the singular-value spectrum diagnostic run.')
    parser.add_argument('--spectrum-run-name', default=None, help='Run name for spectrum diagnostics. Defaults to spectrum_diagnostics for the standard suite.')
    parser.add_argument('--spectrum-n-inits', type=int, default=None, help='Initializations for spectrum diagnostics. Defaults to min(3, n_inits).')
    parser.add_argument('--spectrum-methods', nargs='*', default=['jordan', 'fastns', 'adaptive'], help='Method keys for spectrum diagnostics.')
    parser.add_argument('--spectrum-log-epochs', nargs='*', type=int, default=None, help='Epochs at which to log spectra.')
    return parser.parse_args()


def run(cmd: list[str]):
    print('+ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def clean_run(run_name: str):
    for parent in ['results', 'figures']:
        path = ROOT / parent / run_name
        if path.exists():
            print(f'Removing {path}')
            shutil.rmtree(path)


def main():
    args = parse_args()
    spectrum_run_name = args.spectrum_run_name
    if spectrum_run_name is None:
        spectrum_run_name = 'spectrum_diagnostics' if args.run_name == 'adaptive_ns_suite' else f'{args.run_name}_spectrum'
    spectrum_n_inits = args.spectrum_n_inits if args.spectrum_n_inits is not None else min(3, args.n_inits)

    if args.rebuild_coeff_table:
        run([sys.executable, 'scripts/build_coeff_table.py'])

    if args.clean and not args.skip_benchmark:
        clean_run(args.run_name)

    if not args.skip_benchmark:
        run([
            sys.executable, '-m', 'benchmarks.run_suite',
            '--run-name', args.run_name,
            '--epochs', str(args.epochs),
            '--n-inits', str(args.n_inits),
            '--hidden', str(args.hidden),
            '--batch-size', str(args.batch_size),
            '--threads', str(args.threads),
            '--adaptive-a-init', str(args.adaptive_a_init),
            '--adaptive-transition-start', str(args.adaptive_transition_start),
            '--adaptive-transition-steps', str(args.adaptive_transition_steps),
            '--adaptive-spike-window', str(args.adaptive_spike_window),
            '--adaptive-spike-threshold', str(args.adaptive_spike_threshold),
            '--adaptive-spike-ema-beta', str(args.adaptive_spike_ema_beta),
            '--adaptive-min-spikes', str(args.adaptive_min_spikes),
            '--adaptive-spike-count-margin', str(args.adaptive_spike_count_margin),
            '--adam-lr', str(args.adam_lr),
            '--adamw-lr', str(args.adamw_lr),
            '--adamw-weight-decay', str(args.adamw_weight_decay),
        ])

    run([sys.executable, 'scripts/summarize_results.py', '--run-name', args.run_name])

    if not args.skip_spectrum_diagnostics and not args.skip_benchmark:
        spectrum_cmd = [
            sys.executable, 'scripts/spectrum_diagnostics.py',
            '--run-name', spectrum_run_name,
            '--epochs', str(args.epochs),
            '--n-inits', str(spectrum_n_inits),
            '--hidden', str(args.hidden),
            '--batch-size', str(args.batch_size),
            '--threads', str(args.threads),
            '--adaptive-a-init', str(args.adaptive_a_init),
            '--adaptive-transition-start', str(args.adaptive_transition_start),
            '--adaptive-transition-steps', str(args.adaptive_transition_steps),
            '--adaptive-spike-window', str(args.adaptive_spike_window),
            '--adaptive-spike-threshold', str(args.adaptive_spike_threshold),
            '--adaptive-spike-ema-beta', str(args.adaptive_spike_ema_beta),
            '--adaptive-min-spikes', str(args.adaptive_min_spikes),
            '--adaptive-spike-count-margin', str(args.adaptive_spike_count_margin),
            '--methods', *args.spectrum_methods,
        ]
        if args.spectrum_log_epochs is not None:
            spectrum_cmd.extend(['--log-epochs', *[str(epoch) for epoch in args.spectrum_log_epochs]])
        if not args.clean:
            spectrum_cmd.append('--no-clean')
        run(spectrum_cmd)

    report_cmd = [sys.executable, 'scripts/generate_report.py', '--run-name', args.run_name]
    if not args.skip_report_compile:
        report_cmd.append('--compile')
    run(report_cmd)

    print('\nDone.')
    print(f'Results: {ROOT / "results" / args.run_name}')
    print(f'Figures: {ROOT / "figures" / args.run_name}')
    if not args.skip_spectrum_diagnostics:
        print(f'Spectrum results: {ROOT / "results" / spectrum_run_name}')
        print(f'Spectrum figures: {ROOT / "figures" / spectrum_run_name}')
    print(f'Report:  {ROOT / "docs" / "adaptive_dense_muon_report.pdf"}')


if __name__ == '__main__':
    main()
