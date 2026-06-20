#!/usr/bin/env python3
"""Generate Muon singular-value spectrum diagnostics.

The diagnostic logs singular values of the Frobenius-normalized Muon momentum
matrix, i.e. the matrix to which the Newton-Schulz polynomial is applied. It is
intended to test whether later-stage spectra become more spread or anisotropic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.problems import select_problems  # noqa: E402
from muon.adaptive_optimizer import AdaptiveMuon, ScheduledFamilyMuon  # noqa: E402
from muon.coeff_table import FAST_NS, JORDAN_NS, MUON_COEFF_TABLE, STANDARD_NS  # noqa: E402
from muon.newton_schulz import iterate_polynomial  # noqa: E402
from muon.train import MLP  # noqa: E402

METHODS = {
    'standard': ('Standard Muon', 'standard_muon', STANDARD_NS),
    'jordan': ('Jordan Muon', 'jordan_muon', JORDAN_NS),
    'fastns': ('FastNS Muon', 'fastns_muon', FAST_NS),
    'adaptive': ('Adaptive Dense Muon', 'adaptive_dense_muon', None),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='spectrum_diagnostics')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-inits', type=int, default=3)
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
    parser.add_argument('--methods', nargs='*', default=['jordan', 'fastns', 'adaptive'], choices=sorted(METHODS))
    parser.add_argument('--problems', nargs='*', default=None)
    parser.add_argument('--log-epochs', nargs='*', type=int, default=None)
    parser.add_argument('--clean', action='store_true', default=True)
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    return parser.parse_args()


def default_log_epochs(epochs: int) -> list[int]:
    candidates = [0, 25, 100, epochs // 2, epochs - 1]
    return sorted({ep for ep in candidates if 0 <= ep < epochs})


def make_optimizer(
    method_key: str,
    model: torch.nn.Module,
    logger,
    log_steps: set[int],
    adaptive_a_init: float,
    adaptive_transition_start: int,
    adaptive_transition_steps: int,
    adaptive_spike_window: int,
    adaptive_spike_threshold: float,
    adaptive_spike_ema_beta: float,
    adaptive_min_spikes: int,
    adaptive_spike_count_margin: int,
):
    name, _slug, coeffs = METHODS[method_key]
    if coeffs is None:
        _ = adaptive_a_init
        return ScheduledFamilyMuon(
            model,
            transition_start=adaptive_transition_start,
            transition_steps=adaptive_transition_steps,
            spike_window=adaptive_spike_window,
            spike_threshold=adaptive_spike_threshold,
            spike_ema_beta=adaptive_spike_ema_beta,
            min_spikes=adaptive_min_spikes,
            spike_count_margin=adaptive_spike_count_margin,
            spectrum_logger=logger,
            spectrum_log_steps=log_steps,
        )

    a, b, c = coeffs
    opt = AdaptiveMuon(
        model,
        a_init=a,
        coeff_table={a: (a, b, c)},
        spectrum_logger=logger,
        spectrum_log_steps=log_steps,
    )
    opt.a_min = a
    opt.a_max = a
    opt.step_up = 0.0
    opt.step_down = 0.0
    return opt


def spectrum_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    clipped = np.maximum(values, 1e-12)
    energy = clipped * clipped
    total = float(energy.sum())
    if total <= 0:
        return {
            'n_sv': int(values.size),
            'sv_min': 0.0,
            'sv_p10': 0.0,
            'sv_median': 0.0,
            'sv_p90': 0.0,
            'sv_max': 0.0,
            'top_energy_frac': 0.0,
            'effective_rank': 0.0,
            'participation_rank': 0.0,
            'tiny_frac': 1.0,
        }
    probs = energy / total
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    participation = total * total / float(np.sum(energy * energy))
    return {
        'n_sv': int(values.size),
        'sv_min': float(values.min()),
        'sv_p10': float(np.quantile(values, 0.10)),
        'sv_median': float(np.quantile(values, 0.50)),
        'sv_p90': float(np.quantile(values, 0.90)),
        'sv_max': float(values.max()),
        'top_energy_frac': float(probs.max()),
        'effective_rank': float(np.exp(entropy)),
        'participation_rank': float(participation),
        'tiny_frac': float(np.mean(values < 1e-3)),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(metric_rows: list[dict]) -> list[dict]:
    keys = ['problem', 'method', 'slug', 'epoch', 'layer_idx', 'shape']
    metric_names = [
        'sv_min', 'sv_p10', 'sv_median', 'sv_p90', 'sv_max',
        'top_energy_frac', 'effective_rank', 'participation_rank', 'tiny_frac',
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in metric_rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(keys, key))
        item['n_records'] = len(rows)
        for metric in metric_names:
            vals = np.asarray([float(row[metric]) for row in rows], dtype=float)
            item[metric + '_mean'] = float(vals.mean())
            item[metric + '_std'] = float(vals.std(ddof=0))
        out.append(item)
    return out


def plot_scalar_maps(out_dir: Path):
    xs = np.logspace(-3, 0, 600)
    curves = [
        ('Standard', STANDARD_NS, '#1f77b4'),
        ('Jordan', JORDAN_NS, '#d62728'),
        ('FastNS / start', FAST_NS, '#2ca02c'),
        ('Table max 3.89', MUON_COEFF_TABLE[3.89], '#9467bd'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for label, coeffs, color in curves:
        y = iterate_polynomial(xs, *coeffs, K=5)
        axes[0].plot(xs, y, label=label, color=color, lw=1.8)
        axes[1].plot(xs, y / xs, label=label, color=color, lw=1.8)
    for ax in axes:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('initial singular value')
    axes[0].axhspan(0.5, 1.5, color='gray', alpha=0.12, label='target band')
    axes[0].set_ylabel('$f^5(x)$')
    axes[0].set_title('Five-step NS scalar maps')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('$f^5(x)/x$')
    axes[1].set_title('Net amplification factor')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'ns_scalar_maps.png', dpi=180)
    plt.close(fig)


def plot_histograms(spectrum_rows: list[dict], out_dir: Path, log_epochs: list[int]):
    if not spectrum_rows:
        return
    early, late = log_epochs[0], log_epochs[-1]
    problems = list(dict.fromkeys(row['problem'] for row in spectrum_rows))
    methods = list(dict.fromkeys(row['method'] for row in spectrum_rows))
    layers = sorted({int(row['layer_idx']) for row in spectrum_rows})
    bins = np.logspace(-6, 0, 45)

    for problem in problems:
        fig, axes = plt.subplots(len(methods), len(layers), figsize=(3.0 * len(layers), 2.25 * len(methods)), squeeze=False)
        for i, method in enumerate(methods):
            for j, layer in enumerate(layers):
                ax = axes[i, j]
                for epoch, color, alpha, label in [
                    (early, '#1f77b4', 0.28, f'epoch {early}'),
                    (late, '#d62728', 0.28, f'epoch {late}'),
                ]:
                    vals = [
                        max(float(row['singular_value']), 1e-6)
                        for row in spectrum_rows
                        if row['problem'] == problem
                        and row['method'] == method
                        and int(row['layer_idx']) == layer
                        and int(row['epoch']) == epoch
                    ]
                    if vals:
                        ax.hist(vals, bins=bins, density=True, color=color, alpha=alpha, label=label)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.2)
                if i == 0:
                    ax.set_title(f'layer {layer}', fontsize=9)
                if j == 0:
                    ax.set_ylabel(method, fontsize=8)
                if i == len(methods) - 1:
                    ax.set_xlabel('singular value', fontsize=8)
                if i == 0 and j == len(layers) - 1:
                    ax.legend(fontsize=7)
        fig.suptitle(f'Per-layer normalized momentum spectra: {problem}', y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f'hist_{problem}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def overview_values(metric_rows: list[dict], metric: str):
    groups: dict[tuple, list[float]] = {}
    for row in metric_rows:
        key = (row['problem'], row['method'], int(row['epoch']))
        groups.setdefault(key, []).append(float(row[metric]))
    return {key: float(np.mean(vals)) for key, vals in groups.items()}


def plot_metric_overview(metric_rows: list[dict], out_dir: Path, metric: str, ylabel: str, filename: str):
    if not metric_rows:
        return
    values = overview_values(metric_rows, metric)
    problems = list(dict.fromkeys(row['problem'] for row in metric_rows))
    methods = list(dict.fromkeys(row['method'] for row in metric_rows))
    epochs = sorted({int(row['epoch']) for row in metric_rows})
    colors = {
        'Jordan Muon': '#d62728',
        'FastNS Muon': '#2ca02c',
        'Adaptive Dense Muon': '#9467bd',
        'Standard Muon': '#1f77b4',
    }
    ncols = 2
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.7 * nrows), squeeze=False)
    for idx, problem in enumerate(problems):
        ax = axes[idx // ncols, idx % ncols]
        for method in methods:
            ys = [values.get((problem, method, epoch), np.nan) for epoch in epochs]
            ax.plot(epochs, ys, marker='o', lw=1.5, ms=3, label=method, color=colors.get(method))
        ax.set_title(problem)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for idx in range(len(problems), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(methods), 4), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def train_one(problem, method_key: str, init_idx: int, problem_idx: int, method_idx: int, args, device, log_steps: set[int], spectrum_rows, metric_rows, loss_rows):
    name, slug, _coeffs = METHODS[method_key]
    seed = 10000 * problem_idx + 100 * method_idx + init_idx + 987
    torch.manual_seed(seed)
    model = MLP(hidden=args.hidden).to(device)

    def logger(**payload):
        svals = payload['singular_values'].numpy().astype(float)
        base = {
            'problem': problem.name,
            'method': name,
            'slug': slug,
            'init': init_idx,
            'seed': seed,
            'epoch': int(payload['step']),
            'layer_idx': int(payload['layer_idx']),
            'shape': 'x'.join(str(x) for x in payload['shape']),
            'a_target': float(payload['a_target']),
            'coeff_a': float(payload['coeffs'][0]),
            'coeff_b': float(payload['coeffs'][1]),
            'coeff_c': float(payload['coeffs'][2]),
        }
        metrics = spectrum_metrics(svals)
        metric_rows.append({**base, **metrics})
        for sv_idx, sv in enumerate(svals):
            spectrum_rows.append({**base, 'sv_index': sv_idx, 'singular_value': float(sv)})

    opt = make_optimizer(method_key, model, logger, log_steps, args.adaptive_a_init)
    losses = []
    for epoch in range(args.epochs):
        xb = torch.rand(args.batch_size, 1, device=device) * 2 - 1
        yb = problem.target_fn(xb)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        lv = float(loss.item())
        losses.append(lv)
        opt.step(loss_val=lv)
    loss_rows.append({
        'problem': problem.name,
        'method': name,
        'slug': slug,
        'init': init_idx,
        'seed': seed,
        'final_loss': float(losses[-1]),
        'min_loss': float(np.min(losses)),
    })


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    problems = select_problems(args.problems)
    log_epochs = args.log_epochs if args.log_epochs is not None else default_log_epochs(args.epochs)
    log_epochs = sorted({ep for ep in log_epochs if 0 <= ep < args.epochs})
    if not log_epochs:
        raise ValueError('no valid log epochs')
    log_steps = set(log_epochs)

    results_dir = ROOT / 'results' / args.run_name
    figures_dir = ROOT / 'figures' / args.run_name
    if args.clean:
        for path in [results_dir, figures_dir]:
            if path.exists():
                shutil.rmtree(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config['log_epochs'] = log_epochs
    (results_dir / 'config.json').write_text(json.dumps(config, indent=2))

    device = torch.device(args.device)
    spectrum_rows: list[dict] = []
    metric_rows: list[dict] = []
    loss_rows: list[dict] = []

    print(
        f'Spectrum diagnostics: problems={len(problems)}, methods={args.methods}, '
        f'n_inits={args.n_inits}, epochs={args.epochs}, log_epochs={log_epochs}',
        flush=True,
    )
    for problem_idx, problem in enumerate(problems):
        print(f'Problem: {problem.name}', flush=True)
        for method_idx, method_key in enumerate(args.methods):
            print(f'  {METHODS[method_key][0]}', flush=True)
            for init_idx in range(args.n_inits):
                train_one(
                    problem,
                    method_key,
                    init_idx,
                    problem_idx,
                    method_idx,
                    args,
                    device,
                    log_steps,
                    spectrum_rows,
                    metric_rows,
                    loss_rows,
                )

    spectrum_fields = [
        'problem', 'method', 'slug', 'init', 'seed', 'epoch', 'layer_idx', 'shape',
        'a_target', 'coeff_a', 'coeff_b', 'coeff_c', 'sv_index', 'singular_value',
    ]
    metric_fields = [
        'problem', 'method', 'slug', 'init', 'seed', 'epoch', 'layer_idx', 'shape',
        'a_target', 'coeff_a', 'coeff_b', 'coeff_c', 'n_sv', 'sv_min', 'sv_p10',
        'sv_median', 'sv_p90', 'sv_max', 'top_energy_frac', 'effective_rank',
        'participation_rank', 'tiny_frac',
    ]
    loss_fields = ['problem', 'method', 'slug', 'init', 'seed', 'final_loss', 'min_loss']
    write_csv(results_dir / 'spectra.csv', spectrum_rows, spectrum_fields)
    write_csv(results_dir / 'spectrum_metrics.csv', metric_rows, metric_fields)
    write_csv(results_dir / 'losses.csv', loss_rows, loss_fields)

    summary_rows = aggregate_metrics(metric_rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_rows:
        write_csv(results_dir / 'spectrum_metric_summary.csv', summary_rows, summary_fields)

    plot_scalar_maps(figures_dir)
    plot_histograms(spectrum_rows, figures_dir, log_epochs)
    plot_metric_overview(metric_rows, figures_dir, 'effective_rank', 'mean effective rank across layers', 'effective_rank_overview.png')
    plot_metric_overview(metric_rows, figures_dir, 'top_energy_frac', 'mean top singular energy fraction', 'top_energy_overview.png')

    print(f'Wrote {results_dir}')
    print(f'Wrote {figures_dir}')


if __name__ == '__main__':
    main()
