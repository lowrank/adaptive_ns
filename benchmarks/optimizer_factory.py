"""Optimizer factories used by the benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from muon.adaptive_optimizer import AdaptiveMuon, ScheduledFamilyMuon
from muon.coeff_table import (
    FAST_NS, FAST_NS_TABLE_KEY, JORDAN_NS, STANDARD_NS,
    STRICT_FALLBACK_TABLE_KEY,
)

OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]


@dataclass(frozen=True)
class OptimizerSpec:
    """Named optimizer configuration for experiments."""

    name: str
    slug: str
    factory: OptimizerFactory
    description: str


def fixed_muon_factory(coeffs: tuple[float, float, float]) -> OptimizerFactory:
    """Create a Muon optimizer with coefficients pinned to one triple."""
    a, b, c = coeffs

    def factory(model: torch.nn.Module) -> AdaptiveMuon:
        opt = AdaptiveMuon(model, a_init=a, coeff_table={a: (a, b, c)})
        opt.a_min = a
        opt.a_max = a
        opt.step_up = 0.0
        opt.step_down = 0.0
        return opt

    return factory


def adaptive_muon_factory(
    a_init: float = 3.87,
    transition_start: int | None = None,
    transition_steps: int = 100,
    spike_window: int = 40,
    spike_threshold: float = 1.25,
    spike_ema_beta: float = 0.98,
    min_spikes: int = 2,
    spike_count_margin: int = 1,
) -> OptimizerFactory:
    # Create the current adaptive Muon baseline. The a_init argument is kept
    # for CLI compatibility; the family starts at FastNS by construction.
    _ = a_init
    return scheduled_family_muon_factory(
        transition_start,
        transition_steps,
        spike_window,
        spike_threshold,
        spike_ema_beta,
        min_spikes,
        spike_count_margin,
    )


def scheduled_family_muon_factory(
    transition_start: int | None = None,
    transition_steps: int = 100,
    spike_window: int = 40,
    spike_threshold: float = 1.25,
    spike_ema_beta: float = 0.98,
    min_spikes: int = 2,
    spike_count_margin: int = 1,
) -> OptimizerFactory:
    # Create Muon on a FastNS-to-strict-fallback one-parameter table-key path.

    def factory(model: torch.nn.Module) -> ScheduledFamilyMuon:
        return ScheduledFamilyMuon(
            model,
            start_a=FAST_NS_TABLE_KEY,
            end_a=STRICT_FALLBACK_TABLE_KEY,
            transition_start=transition_start,
            transition_steps=transition_steps,
            spike_window=spike_window,
            spike_threshold=spike_threshold,
            spike_ema_beta=spike_ema_beta,
            min_spikes=min_spikes,
            spike_count_margin=spike_count_margin,
        )

    return factory


def adam_factory(lr: float = 1e-3) -> OptimizerFactory:
    def factory(model: torch.nn.Module) -> torch.optim.Adam:
        return torch.optim.Adam(model.parameters(), lr=lr)

    return factory


def adamw_factory(lr: float = 1e-3, weight_decay: float = 1e-2) -> OptimizerFactory:
    def factory(model: torch.nn.Module) -> torch.optim.AdamW:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return factory


def default_optimizer_specs(
    *,
    adaptive_a_init: float = 3.87,
    adam_lr: float = 1e-3,
    adamw_lr: float = 1e-3,
    adamw_weight_decay: float = 1e-2,
    adaptive_transition_start: int | None = None,
    adaptive_transition_steps: int = 100,
    adaptive_spike_window: int = 40,
    adaptive_spike_threshold: float = 1.25,
    adaptive_spike_ema_beta: float = 0.98,
    adaptive_min_spikes: int = 2,
    adaptive_spike_count_margin: int = 1,
) -> list[OptimizerSpec]:
    """Return the default set of optimizers for comparison."""
    return [
        OptimizerSpec(
            name='Standard Muon',
            slug='standard_muon',
            factory=fixed_muon_factory(STANDARD_NS),
            description='Muon with classical Newton-Schulz coefficients (1.5, -0.5, 0).',
        ),
        OptimizerSpec(
            name='Jordan Muon',
            slug='jordan_muon',
            factory=fixed_muon_factory(JORDAN_NS),
            description='Muon with Keller Jordan coefficients.',
        ),
        OptimizerSpec(
            name='FastNS Muon',
            slug='fastns_muon',
            factory=fixed_muon_factory(FAST_NS),
            description='Muon with the fastest strict-band coefficient triple from the PDF.',
        ),
        OptimizerSpec(
            name='Adaptive Dense Muon',
            slug='adaptive_dense_muon',
            factory=adaptive_muon_factory(
                adaptive_a_init,
                adaptive_transition_start,
                adaptive_transition_steps,
                adaptive_spike_window,
                adaptive_spike_threshold,
                adaptive_spike_ema_beta,
                adaptive_min_spikes,
                adaptive_spike_count_margin,
            ),
            description='Current adaptive Muon baseline: FastNS-to-fallback a schedule triggered by rising loss-spike frequency.',
        ),
        OptimizerSpec(
            name='Adam',
            slug='adam',
            factory=adam_factory(adam_lr),
            description='Torch Adam baseline.',
        ),
        OptimizerSpec(
            name='AdamW',
            slug='adamw',
            factory=adamw_factory(adamw_lr, adamw_weight_decay),
            description='Torch AdamW baseline with decoupled weight decay.',
        ),
    ]


def select_optimizers(specs: list[OptimizerSpec], names_or_slugs: list[str] | None) -> list[OptimizerSpec]:
    """Select optimizers by display name or slug."""
    if not names_or_slugs:
        return specs
    wanted = set(names_or_slugs)
    selected = [spec for spec in specs if spec.slug in wanted or spec.name in wanted]
    found = {spec.slug for spec in selected}.union(spec.name for spec in selected)
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f'unknown optimizer(s): {", ".join(missing)}')
    return selected
