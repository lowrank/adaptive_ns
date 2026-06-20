"""Benchmark runner for adaptive Newton-Schulz Muon experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from benchmarks.optimizer_factory import OptimizerSpec
from benchmarks.plotting import (
    plot_final_loss_bars,
    plot_final_ratio_heatmap,
    plot_loss_curves,
    plot_target,
)
from benchmarks.problems import ProblemSpec
from muon.adaptive_optimizer import AdaptiveMuon, ScheduledFamilyMuon
from muon.train import MLP


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    run_name: str = 'adaptive_ns_suite'
    epochs: int = 500
    n_inits: int = 20
    hidden: int = 32
    batch_size: int = 256
    threads: int = 1
    device: str = 'cpu'
    adaptive_a_init: float = 3.87
    adaptive_transition_start: int = -1
    adaptive_transition_steps: int = 100
    adaptive_spike_window: int = 40
    adaptive_spike_threshold: float = 1.25
    adaptive_spike_ema_beta: float = 0.98
    adaptive_min_spikes: int = 2
    adaptive_spike_count_margin: int = 1
    adam_lr: float = 1e-3
    adamw_lr: float = 1e-3
    adamw_weight_decay: float = 1e-2


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slugify(text: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in text.lower()).strip('_')


def spike_count(losses: np.ndarray) -> int:
    """Count loss excursions above 3x the running best after warmup."""
    running_min = np.minimum.accumulate(losses)
    start = min(100, len(losses))
    return int(np.sum(losses[start:] > 3.0 * running_min[start:]))


def train_one(
    problem: ProblemSpec,
    optimizer: OptimizerSpec,
    *,
    seed: int,
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train one model initialization and return loss, a, trigger, and spike traces."""
    torch.manual_seed(seed)
    model = MLP(hidden=config.hidden).to(device)
    opt = optimizer.factory(model)
    losses = np.empty(config.epochs, dtype=np.float64)
    a_values = np.full(config.epochs, np.nan, dtype=np.float64)
    trigger_values = np.full(config.epochs, np.nan, dtype=np.float64)
    current_spike_values = np.full(config.epochs, np.nan, dtype=np.float64)
    previous_spike_values = np.full(config.epochs, np.nan, dtype=np.float64)

    for ep in range(config.epochs):
        xb = torch.rand(config.batch_size, 1, device=device) * 2 - 1
        yb = problem.target_fn(xb)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        lv = float(loss.item())
        losses[ep] = lv
        if isinstance(opt, AdaptiveMuon):
            opt.step(loss_val=lv)
            a_values[ep] = opt.a_target
            trigger_step = getattr(opt, 'trigger_step', None)
            current_spikes = getattr(opt, 'current_spike_count', None)
            previous_spikes = getattr(opt, 'previous_spike_count', None)
            if trigger_step is not None:
                trigger_values[ep] = float(trigger_step)
            if current_spikes is not None:
                current_spike_values[ep] = float(current_spikes)
            if previous_spikes is not None:
                previous_spike_values[ep] = float(previous_spikes)
        else:
            opt.step()

    return losses, a_values, trigger_values, current_spike_values, previous_spike_values


def summarize(
    problem_name: str,
    optimizer_name: str,
    curves: list[np.ndarray],
    a_traces: list[np.ndarray],
    trigger_traces: list[np.ndarray],
    current_spike_traces: list[np.ndarray],
    previous_spike_traces: list[np.ndarray],
) -> dict:
    arr = np.asarray(curves)
    a_arr = np.asarray(a_traces)
    final = arr[:, -1]
    best = arr.min(axis=1)
    spikes = np.asarray([spike_count(row) for row in arr], dtype=float)
    trigger_arr = np.asarray(trigger_traces)
    current_spike_arr = np.asarray(current_spike_traces)
    previous_spike_arr = np.asarray(previous_spike_traces)
    finite_a = a_arr[:, -1][np.isfinite(a_arr[:, -1])]
    trigger_final = trigger_arr[:, -1] if trigger_arr.size else np.asarray([])
    finite_trigger = trigger_final[np.isfinite(trigger_final)]
    current_spike_final = current_spike_arr[:, -1] if current_spike_arr.size else np.asarray([])
    previous_spike_final = previous_spike_arr[:, -1] if previous_spike_arr.size else np.asarray([])
    finite_current_spikes = current_spike_final[np.isfinite(current_spike_final)]
    finite_previous_spikes = previous_spike_final[np.isfinite(previous_spike_final)]
    finite_spike_delta = (current_spike_final - previous_spike_final)[np.isfinite(current_spike_final - previous_spike_final)] if current_spike_final.size and previous_spike_final.size else np.asarray([])
    return {
        'problem': problem_name,
        'method': optimizer_name,
        'runs': int(arr.shape[0]),
        'final_mean': float(final.mean()),
        'final_std': float(final.std(ddof=0)),
        'min_mean': float(best.mean()),
        'min_std': float(best.std(ddof=0)),
        'spikes_mean': float(spikes.mean()),
        'spikes_std': float(spikes.std(ddof=0)),
        'a_final_mean': float(finite_a.mean()) if finite_a.size else float('nan'),
        'a_final_std': float(finite_a.std(ddof=0)) if finite_a.size else float('nan'),
        'trigger_step_mean': float(finite_trigger.mean()) if finite_trigger.size else float('nan'),
        'trigger_step_std': float(finite_trigger.std(ddof=0)) if finite_trigger.size else float('nan'),
        'trigger_rate': float(finite_trigger.size / arr.shape[0]) if arr.shape[0] else float('nan'),
        'current_spike_count_mean': float(finite_current_spikes.mean()) if finite_current_spikes.size else float('nan'),
        'current_spike_count_std': float(finite_current_spikes.std(ddof=0)) if finite_current_spikes.size else float('nan'),
        'previous_spike_count_mean': float(finite_previous_spikes.mean()) if finite_previous_spikes.size else float('nan'),
        'previous_spike_count_std': float(finite_previous_spikes.std(ddof=0)) if finite_previous_spikes.size else float('nan'),
        'spike_count_delta_mean': float(finite_spike_delta.mean()) if finite_spike_delta.size else float('nan'),
        'spike_count_delta_std': float(finite_spike_delta.std(ddof=0)) if finite_spike_delta.size else float('nan'),
    }


def write_summary(summary_rows: list[dict], results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        'problem', 'method', 'runs', 'final_mean', 'final_std', 'min_mean',
        'min_std', 'spikes_mean', 'spikes_std', 'a_final_mean', 'a_final_std',
        'trigger_step_mean', 'trigger_step_std', 'trigger_rate',
        'current_spike_count_mean', 'current_spike_count_std',
        'previous_spike_count_mean', 'previous_spike_count_std',
        'spike_count_delta_mean', 'spike_count_delta_std',
    ]
    with (results_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    with (results_dir / 'summary.json').open('w') as f:
        json.dump(summary_rows, f, indent=2)


def write_method_manifest(optimizers: Iterable[OptimizerSpec], results_dir: Path):
    manifest = [
        {'name': opt.name, 'slug': opt.slug, 'description': opt.description}
        for opt in optimizers
    ]
    with (results_dir / 'optimizers.json').open('w') as f:
        json.dump(manifest, f, indent=2)


def write_problem_manifest(problems: Iterable[ProblemSpec], results_dir: Path):
    manifest = [
        {
            'name': problem.name,
            'description': problem.description,
            'formula': problem.formula,
        }
        for problem in problems
    ]
    with (results_dir / 'problems.json').open('w') as f:
        json.dump(manifest, f, indent=2)


def final_loss_ratio_rows(summary_rows: list[dict]) -> list[dict]:
    """Compute final/min-loss ratios to the best method on each problem."""
    problems = list(dict.fromkeys(row['problem'] for row in summary_rows))
    rows = []
    for problem in problems:
        problem_rows = [row for row in summary_rows if row['problem'] == problem]
        best_final = min(row['final_mean'] for row in problem_rows)
        best_min = min(row['min_mean'] for row in problem_rows)
        for row in problem_rows:
            rows.append({
                'problem': problem,
                'method': row['method'],
                'final_mean': row['final_mean'],
                'final_ratio': row['final_mean'] / best_final,
                'min_mean': row['min_mean'],
                'min_ratio': row['min_mean'] / best_min,
                'is_final_best': row['final_mean'] == best_final,
            })
    return rows


def write_ratio_tables(summary_rows: list[dict], results_dir: Path) -> list[dict]:
    ratio_rows = final_loss_ratio_rows(summary_rows)
    fields = ['problem', 'method', 'final_mean', 'final_ratio', 'min_mean', 'min_ratio', 'is_final_best']
    with (results_dir / 'final_loss_ratios.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ratio_rows)
    with (results_dir / 'final_loss_ratios.json').open('w') as f:
        json.dump(ratio_rows, f, indent=2)
    return ratio_rows


def run_benchmark_suite(
    *,
    config: BenchmarkConfig,
    problems: list[ProblemSpec],
    optimizers: list[OptimizerSpec],
) -> list[dict]:
    """Run the benchmark suite and write artifacts to results/ and figures/."""
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    root = project_root()
    results_dir = root / 'results' / config.run_name
    figures_dir = root / 'figures' / config.run_name
    curves_npz_dir = results_dir / 'curves_npz'
    targets_dir = figures_dir / 'targets'
    loss_dir = figures_dir / 'loss_curves'
    curves_npz_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / 'config.json').open('w') as f:
        json.dump(asdict(config), f, indent=2)
    write_method_manifest(optimizers, results_dir)
    write_problem_manifest(problems, results_dir)

    summary_rows: list[dict] = []
    print(
        f'Benchmark suite: problems={len(problems)}, methods={len(optimizers)}, '
        f'n_inits={config.n_inits}, epochs={config.epochs}, hidden={config.hidden}'
    )
    print(f'Results: {results_dir}')
    print(f'Figures: {figures_dir}')

    for problem_idx, problem in enumerate(problems):
        print(f'\nProblem: {problem.name}', flush=True)
        plot_target(problem.name, problem.target_fn, targets_dir / f'{problem.name}.png', device)
        method_curves: dict[str, list[np.ndarray]] = {}

        for optimizer in optimizers:
            curves = []
            a_traces = []
            trigger_traces = []
            current_spike_traces = []
            previous_spike_traces = []
            for init_idx in range(config.n_inits):
                seed = 10000 * problem_idx + init_idx + 123
                losses, a_values, trigger_values, current_spike_values, previous_spike_values = train_one(
                    problem,
                    optimizer,
                    seed=seed,
                    config=config,
                    device=device,
                )
                curves.append(losses)
                a_traces.append(a_values)
                trigger_traces.append(trigger_values)
                current_spike_traces.append(current_spike_values)
                previous_spike_traces.append(previous_spike_values)

            row = summarize(problem.name, optimizer.name, curves, a_traces, trigger_traces, current_spike_traces, previous_spike_traces)
            summary_rows.append(row)
            method_curves[optimizer.name] = curves
            np.savez_compressed(
                curves_npz_dir / f'{problem.name}_{optimizer.slug}.npz',
                losses=np.asarray(curves),
                a_values=np.asarray(a_traces),
                trigger_steps=np.asarray(trigger_traces),
                current_spike_counts=np.asarray(current_spike_traces),
                previous_spike_counts=np.asarray(previous_spike_traces),
            )
            print(
                f'  {optimizer.name:<22s} final={row["final_mean"]:.3e} '
                f'min={row["min_mean"]:.3e} spikes={row["spikes_mean"]:.1f} '
                f'a_final={row["a_final_mean"]:.2f} trigger={row["trigger_step_mean"]:.1f}',
                flush=True,
            )

        plot_loss_curves(problem.name, method_curves, loss_dir / f'{problem.name}_loss.png')
        write_summary(summary_rows, results_dir)
        write_ratio_tables(summary_rows, results_dir)

    write_summary(summary_rows, results_dir)
    ratio_rows = write_ratio_tables(summary_rows, results_dir)
    plot_final_loss_bars(summary_rows, figures_dir / 'final_loss_by_problem.png')
    plot_final_ratio_heatmap(ratio_rows, figures_dir / 'final_loss_ratio_heatmap.png')
    print(f'\nWrote {results_dir / "summary.csv"}')
    return summary_rows
