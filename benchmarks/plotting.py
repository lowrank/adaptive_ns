"""Plotting helpers for benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.problems import TargetFn

METHOD_COLORS = {
    'Standard Muon': '#1f77b4',
    'Jordan Muon': '#d62728',
    'FastNS Muon': '#2ca02c',
    'Adaptive Dense Muon': '#9467bd',
    'Adam': '#ff7f0e',
    'AdamW': '#8c564b',
}


def plot_target(problem_name: str, target_fn: TargetFn, out_path: Path, device: torch.device):
    """Plot the target function on [-1, 1]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = torch.linspace(-1, 1, 1000, device=device).view(-1, 1)
    with torch.no_grad():
        ys = target_fn(xs).detach().cpu().numpy().ravel()
    x_np = xs.detach().cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x_np, ys, color='black', lw=1.4)
    ax.set_title(problem_name)
    ax.set_xlabel('x')
    ax.set_ylabel('target')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_curves(problem_name: str, method_curves: dict[str, list[np.ndarray]], out_path: Path):
    """Plot mean loss curves with one-standard-deviation shaded bands."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for method_name, curves in method_curves.items():
        arr = np.asarray(curves, dtype=float)
        x = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        positives = arr[arr > 0]
        floor = max(float(positives.min()) * 0.5, 1e-12) if positives.size else 1e-12
        lower = np.maximum(mean - std, floor)
        upper = np.maximum(mean + std, floor)
        line, = ax.plot(x, mean, color=METHOD_COLORS.get(method_name), lw=1.9, label=method_name)
        ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.14, linewidth=0)
    ax.set_yscale('log')
    ax.set_title(f'Mean loss curves: {problem_name}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('training MSE')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_loss_bars(summary_rows: list[dict], out_path: Path):
    """Plot mean final loss by problem and method."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    problems = list(dict.fromkeys(row['problem'] for row in summary_rows))
    methods = list(dict.fromkeys(row['method'] for row in summary_rows))
    values = np.full((len(methods), len(problems)), np.nan)
    for row in summary_rows:
        i = methods.index(row['method'])
        j = problems.index(row['problem'])
        values[i, j] = row['final_mean']

    x = np.arange(len(problems))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values[i], width=width, label=method, color=METHOD_COLORS.get(method))
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=25, ha='right')
    ax.set_ylabel('final training MSE')
    ax.set_title('Mean final loss by benchmark problem')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_ratio_heatmap(ratio_rows: list[dict], out_path: Path):
    """Plot final-loss ratios to the best method for each problem."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    problems = list(dict.fromkeys(row['problem'] for row in ratio_rows))
    methods = list(dict.fromkeys(row['method'] for row in ratio_rows))
    values = np.full((len(methods), len(problems)), np.nan)
    for row in ratio_rows:
        values[methods.index(row['method']), problems.index(row['problem'])] = row['final_ratio']

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    im = ax.imshow(values, aspect='auto', cmap='viridis_r', vmin=1.0, vmax=np.nanpercentile(values, 90))
    ax.set_xticks(np.arange(len(problems)))
    ax.set_xticklabels(problems, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(problems)):
            ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='white' if values[i, j] > 1.8 else 'black', fontsize=8)
    ax.set_title('Final-loss ratio to best method per problem')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('ratio (lower is better)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
