"""Training utilities for oscillatory function approximation. (Provided)"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """5-layer MLP — all 2D weights go through Muon."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def generate_random_function(seed, k_max, n_components=None):
    """Generate a random oscillatory function with bandwidth up to k_max."""
    rng = np.random.RandomState(seed)
    if n_components is None:
        n_components = k_max
    amplitudes = 1.0 / (1 + np.arange(1, n_components + 1))
    phases = rng.uniform(0, 2 * np.pi, n_components)
    freqs = np.arange(1, n_components + 1)

    def f(x):
        y = torch.zeros_like(x)
        for a, phi, k in zip(amplitudes, phases, freqs):
            y += a * torch.sin(2 * np.pi * k * x + phi)
        return y

    return f


def train_one(target_fn, opt_factory, epochs=2000, batch_size=256, seed=42):
    """Train for `epochs`, return per-epoch losses."""
    torch.manual_seed(seed)

    model = MLP(hidden=128)
    opt = opt_factory(model)

    losses = []
    for ep in range(epochs):
        xb = torch.rand(batch_size, 1) * 2 - 1
        yb = target_fn(xb)

        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()

        opt.zero_grad()
        loss.backward()

        lv = loss.item()
        losses.append(lv)

        # Pass loss to optimizer for adaptive methods
        try:
            opt.step(loss_val=lv)
        except TypeError:
            opt.step()

    return np.array(losses)


def train_many(target_fn, opt_factory, n_runs=5, epochs=2000):
    """Average over `n_runs` independent training runs."""
    all_losses = []
    for run in range(n_runs):
        arr = train_one(target_fn, opt_factory, epochs=epochs,
                        seed=run * 100 + 42)
        all_losses.append(arr)
    arr = np.array(all_losses)
    return arr.mean(axis=0), arr.std(axis=0)
