"""Concrete benchmark problem definitions for 1D function approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

TargetFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ProblemSpec:
    """A deterministic supervised regression target."""

    name: str
    target_fn: TargetFn
    description: str
    formula: str


def low_frequency() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * x) + 0.30 * torch.cos(6 * torch.pi * x)

    return f


def medium_fourier() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        return (
            0.70 * torch.sin(4 * torch.pi * x)
            + 0.40 * torch.sin(10 * torch.pi * x + 0.30)
            + 0.20 * torch.cos(18 * torch.pi * x - 0.20)
        )

    return f


def high_frequency() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        return (
            0.55 * torch.sin(8 * torch.pi * x)
            + 0.35 * torch.sin(18 * torch.pi * x + 0.40)
            + 0.25 * torch.cos(34 * torch.pi * x - 0.10)
            + 0.15 * torch.sin(58 * torch.pi * x)
        )

    return f


def chirp_quadratic() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        phase = 2 * torch.pi * (2.0 * x + 7.0 * x * x)
        return torch.sin(phase) + 0.25 * torch.sin(34 * torch.pi * x)

    return f


def gaussian_bumps() -> TargetFn:
    centers = torch.tensor([-0.75, -0.35, -0.05, 0.38, 0.78], dtype=torch.float32)
    widths = torch.tensor([0.07, 0.10, 0.04, 0.08, 0.05], dtype=torch.float32)
    amps = torch.tensor([0.80, -0.55, 1.10, -0.70, 0.45], dtype=torch.float32)

    def f(x: torch.Tensor) -> torch.Tensor:
        c = centers.to(x.device).view(1, -1)
        w = widths.to(x.device).view(1, -1)
        a = amps.to(x.device).view(1, -1)
        z = (x - c) / w
        return (a * torch.exp(-0.5 * z * z)).sum(dim=1, keepdim=True)

    return f


def runge_rational() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + 25.0 * x * x) + 0.15 * torch.sin(12 * torch.pi * x)

    return f


def cusp_abs() -> TargetFn:
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) - 0.50 + 0.25 * torch.sin(8 * torch.pi * x)

    return f


def multiscale_mix() -> TargetFn:
    bumps = gaussian_bumps()

    def f(x: torch.Tensor) -> torch.Tensor:
        return (
            0.45 * torch.sin(2 * torch.pi * x)
            + 0.25 * torch.sin(14 * torch.pi * x + 0.20)
            + 0.12 * torch.cos(46 * torch.pi * x)
            + 0.35 * bumps(x)
        )

    return f


def get_problem_specs() -> list[ProblemSpec]:
    """Return the default concrete benchmark suite."""
    return [
        ProblemSpec(
            name='low_frequency',
            target_fn=low_frequency(),
            description='Smooth low-frequency trigonometric target.',
            formula=r'f(x)=sin(2 pi x)+0.30 cos(6 pi x)',
        ),
        ProblemSpec(
            name='medium_fourier',
            target_fn=medium_fourier(),
            description='Moderate Fourier mixture with three frequencies.',
            formula=r'f(x)=0.70 sin(4 pi x)+0.40 sin(10 pi x+0.30)+0.20 cos(18 pi x-0.20)',
        ),
        ProblemSpec(
            name='high_frequency',
            target_fn=high_frequency(),
            description='Higher-frequency Fourier mixture with four modes.',
            formula=r'f(x)=0.55 sin(8 pi x)+0.35 sin(18 pi x+0.40)+0.25 cos(34 pi x-0.10)+0.15 sin(58 pi x)',
        ),
        ProblemSpec(
            name='chirp_quadratic',
            target_fn=chirp_quadratic(),
            description='Frequency-increasing chirp plus a high-frequency perturbation.',
            formula=r'f(x)=sin(2 pi (2x+7x^2))+0.25 sin(34 pi x)',
        ),
        ProblemSpec(
            name='gaussian_bumps',
            target_fn=gaussian_bumps(),
            description='Fixed mixture of narrow Gaussian bumps.',
            formula=r'f(x)=sum_i A_i exp(-(x-c_i)^2/(2 sigma_i^2)) with fixed (A,c,sigma)',
        ),
        ProblemSpec(
            name='runge_rational',
            target_fn=runge_rational(),
            description='Runge-style rational peak with sinusoidal ripple.',
            formula=r'f(x)=1/(1+25x^2)+0.15 sin(12 pi x)',
        ),
        ProblemSpec(
            name='cusp_abs',
            target_fn=cusp_abs(),
            description='Nonsmooth absolute-value cusp with oscillatory ripple.',
            formula=r'f(x)=|x|-0.50+0.25 sin(8 pi x)',
        ),
        ProblemSpec(
            name='multiscale_mix',
            target_fn=multiscale_mix(),
            description='Low, medium, and high-frequency Fourier components plus localized bumps.',
            formula=r'f(x)=0.45 sin(2 pi x)+0.25 sin(14 pi x+0.20)+0.12 cos(46 pi x)+0.35 g_bumps(x)',
        ),
    ]


def select_problems(names: list[str] | None = None) -> list[ProblemSpec]:
    """Select benchmark problems by name, preserving default order."""
    problems = get_problem_specs()
    if not names:
        return problems
    wanted = set(names)
    selected = [problem for problem in problems if problem.name in wanted]
    missing = sorted(wanted.difference(problem.name for problem in selected))
    if missing:
        raise ValueError(f'unknown benchmark problem(s): {", ".join(missing)}')
    return selected
