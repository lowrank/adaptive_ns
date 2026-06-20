"""Command-line entrypoint for the adaptive Muon benchmark suite."""

from __future__ import annotations

import argparse

from benchmarks.optimizer_factory import default_optimizer_specs, select_optimizers
from benchmarks.problems import select_problems
from benchmarks.runner import BenchmarkConfig, run_benchmark_suite


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='adaptive_ns_suite')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-inits', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
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
    parser.add_argument('--problems', nargs='*', default=None)
    parser.add_argument('--optimizers', nargs='*', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(
        run_name=args.run_name,
        epochs=args.epochs,
        n_inits=args.n_inits,
        hidden=args.hidden,
        batch_size=args.batch_size,
        threads=args.threads,
        device=args.device,
        adaptive_a_init=args.adaptive_a_init,
        adaptive_transition_start=args.adaptive_transition_start,
        adaptive_transition_steps=args.adaptive_transition_steps,
        adaptive_spike_window=args.adaptive_spike_window,
        adaptive_spike_threshold=args.adaptive_spike_threshold,
        adaptive_spike_ema_beta=args.adaptive_spike_ema_beta,
        adaptive_min_spikes=args.adaptive_min_spikes,
        adaptive_spike_count_margin=args.adaptive_spike_count_margin,
        adam_lr=args.adam_lr,
        adamw_lr=args.adamw_lr,
        adamw_weight_decay=args.adamw_weight_decay,
    )
    problems = select_problems(args.problems)
    optimizers = default_optimizer_specs(
        adaptive_a_init=args.adaptive_a_init,
        adaptive_transition_start=args.adaptive_transition_start,
        adaptive_transition_steps=args.adaptive_transition_steps,
        adaptive_spike_window=args.adaptive_spike_window,
        adaptive_spike_threshold=args.adaptive_spike_threshold,
        adaptive_spike_ema_beta=args.adaptive_spike_ema_beta,
        adaptive_min_spikes=args.adaptive_min_spikes,
        adaptive_spike_count_margin=args.adaptive_spike_count_margin,
        adam_lr=args.adam_lr,
        adamw_lr=args.adamw_lr,
        adamw_weight_decay=args.adamw_weight_decay,
    )
    optimizers = select_optimizers(optimizers, args.optimizers)
    run_benchmark_suite(config=config, problems=problems, optimizers=optimizers)


if __name__ == '__main__':
    main()
