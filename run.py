#!/usr/bin/env python3
"""Experiment driver: compare fixed vs adaptive NS coefficients."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from muon.train import generate_random_function, train_one, MLP
from muon.coeff_table import STANDARD_NS, JORDAN_NS, FAST_NS
from muon.adaptive_optimizer import AdaptiveMuon


def make_fixed_opt(ns_coeffs):
    """Create a non-adaptive Muon optimizer with fixed coefficients."""
    a, b, c = ns_coeffs

    def factory(model):
        table = {a: (a, b, c)}
        from muon.adaptive_optimizer import AdaptiveMuon
        opt = AdaptiveMuon(model, a_init=a, coeff_table=table)
        opt.a_min = a
        opt.a_max = a
        opt.step_up = 0.0
        opt.step_down = 0.0
        return opt

    return factory


# ── Configurations ──
k_max = 10
n_funcs = 8
epochs = 2000

print(f'Adaptive NS Comparison (K_max={k_max}, {n_funcs} functions)')
print(f'{"Method":<30s} {"final":>10s} {"min":>10s} {"spikes":>8s}')
print('-' * 62)

for name, opt_factory in [
    ('Standard NS (a=1.5, fixed)',
     make_fixed_opt(STANDARD_NS)),
    ('Jordan (a=3.44, fixed)',
     make_fixed_opt(JORDAN_NS)),
    ('FastNS (a=3.87, fixed)',
     make_fixed_opt(FAST_NS)),
    ('Adaptive (a_init=3.0)',
     lambda m: AdaptiveMuon(m, a_init=3.0)),
]:
    finals, mins, spikes_l = [], [], []
    for func_idx in range(n_funcs):
        target_fn = generate_random_function(func_idx * 100 + 42, k_max)
        arr = train_one(target_fn, opt_factory, epochs=epochs,
                        seed=func_idx * 10 + 42)
        rm = np.minimum.accumulate(arr)
        finals.append(arr[-1])
        mins.append(arr.min())
        spikes_l.append(int(np.sum(arr[100:] > 3 * rm[100:])))

    print(f'{name:<30s} {np.mean(finals):>10.2e} {np.mean(mins):>10.2e} '
          f'{np.mean(spikes_l):>8.1f}')

# ── Plot one example ──
target_fn = generate_random_function(42, k_max)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax1, ax2), (ax3, ax4) = axes

colors = {'Standard NS': 'blue', 'Jordan': 'red',
          'FastNS': '#2ca02c', 'Adaptive': 'purple'}

def smooth(arr, window=100):
    alpha = 2.0 / (window + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

for name, factory in [
    ('Standard NS', make_fixed_opt(STANDARD_NS)),
    ('Jordan', make_fixed_opt(JORDAN_NS)),
    ('FastNS', make_fixed_opt(FAST_NS)),
    ('Adaptive', lambda m: AdaptiveMuon(m, a_init=3.0)),
]:
    arr = train_one(target_fn, factory, epochs=epochs, seed=42)
    smoothed = smooth(arr, window=100)
    ax1.semilogy(arr, color=colors[name], lw=0.3, alpha=0.2)
    ax2.semilogy(arr[-500:], color=colors[name], lw=0.3, alpha=0.2)
    ax1.semilogy(smoothed, color=colors[name], lw=1.5, alpha=0.9, label=name)
    ax2.semilogy(smoothed[-500:], color=colors[name], lw=1.5, alpha=0.9)

ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train MSE')
ax1.set_title(f'Loss (faded: raw, bold: smoothed, K_max={k_max})')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Epoch'); ax2.set_title('Zoom: last 500 epochs')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# ── Adaptive a trajectory ──
torch.manual_seed(42)
model = MLP(hidden=128)
opt = AdaptiveMuon(model, a_init=3.0)
a_vals, loss_vals = [], []
for ep in range(epochs):
    xb = torch.rand(256, 1) * 2 - 1; yb = target_fn(xb)
    pred = model(xb); loss = ((pred - yb) ** 2).mean()
    opt.zero_grad(); loss.backward()
    opt.step(loss_val=loss.item())
    a_vals.append(opt.a_target)
    loss_vals.append(loss.item())

ax3.plot(a_vals, color='purple', lw=1.5)
ax3.axhline(3.44, color='red', ls='--', lw=1, label='Jordan a=3.44')
ax3.axhline(1.50, color='blue', ls='--', lw=1, label='Standard a=1.50')
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Coefficient a')
ax3.set_title('Adaptive a trajectory')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

ax4.semilogy(a_vals, loss_vals, color='purple', lw=0.3, alpha=0.4)
ax4.semilogy(a_vals, smooth(np.array(loss_vals), window=100), color='purple', lw=1.5)
ax4.set_xlabel('Coefficient a'); ax4.set_ylabel('Train MSE')
ax4.set_title('Loss vs a (phase plot)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaptive_ns_comparison.png', dpi=200)
plt.close()
print('\nSaved adaptive_ns_comparison.png')
