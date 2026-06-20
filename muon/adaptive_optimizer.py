"""Muon optimizers with adaptive or scheduled Newton-Schulz coefficients.

``AdaptiveMuon`` is the legacy loss-feedback dense-table optimizer.
``ScheduledFamilyMuon`` is the current benchmark default: it schedules only
the target a-value and obtains (a,b,c) from the prepared coefficient table. Its
default transition trigger is conservative spike-frequency feedback: adapt only
when local loss spikes appear more often than in the previous period.
"""

from bisect import bisect_right

import torch
from torch.optim import Optimizer

from .coeff_table import FAST_NS_TABLE_KEY, STRICT_FALLBACK_TABLE_KEY, get_muon_coeff_lookup, get_muon_coeff_table


def _apply_ns(M, a, b, c, steps=5):
    """Apply Newton-Schulz iteration with given coefficients."""
    X = M / (M.norm() + 1e-7)
    transposed = M.size(0) > M.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X


class AdaptiveMuon(Optimizer):
    """Muon with self-tuning Newton-Schulz coefficients.

    The coefficient 'a' adapts based on training stability:
      - Spike detected (loss > spike_threshold * EMA): decrease a.
      - Stable for stable_threshold steps: increase a.

    Parameters
    ----------
    model : nn.Module
    lr_muon : float
        Learning rate for matrix parameters.
    lr_adam : float
        Learning rate for vector parameters.
    momentum : float
        Momentum coefficient.
    a_init : float
        Starting target coefficient.
    coeff_table : dict
        Lookup table {a_target: (a, b, c)}.
    ns_steps : int
        NS iterations per step.
    spectrum_logger : callable or None
        Optional callback for diagnostics. When enabled, it receives singular
        values of the Frobenius-normalized Muon momentum matrix at selected
        optimizer steps.
    spectrum_log_steps : iterable of int or None
        Optimizer step indices to log. If None and spectrum_logger is set, log
        every step.
    """

    def __init__(self, model, lr_muon=0.02, lr_adam=0.001, momentum=0.95,
                 a_init=3.0, coeff_table=None, ns_steps=5,
                 spectrum_logger=None, spectrum_log_steps=None):
        matrix_params, vector_params = [], []
        for p in model.parameters():
            (matrix_params if p.dim() == 2 else vector_params).append(p)

        defaults = dict(lr_m=lr_muon, lr_a=lr_adam, mom=momentum)
        super().__init__([
            {'params': matrix_params, 'mu': True},
            {'params': vector_params, 'mu': False},
        ], defaults)

        self.a_target = a_init
        self.ns_steps = ns_steps
        self.coeff_table = get_muon_coeff_table() if coeff_table is None else dict(coeff_table)
        if coeff_table is None:
            self.coeff_keys, self.coeff_values = get_muon_coeff_lookup()
        else:
            self.coeff_keys = tuple(sorted(self.coeff_table))
            self.coeff_values = tuple(self.coeff_table[key] for key in self.coeff_keys)

        self.loss_ema = None
        self.stable_count = 0
        self.spike_threshold = 3.0
        self.step_up = 0.05
        self.step_down = 0.2
        self.stable_threshold = 100
        self.a_min = min(self.coeff_table)
        self.a_max = 4.0
        self.spectrum_logger = spectrum_logger
        self.spectrum_log_steps = None if spectrum_log_steps is None else set(spectrum_log_steps)
        self._step_idx = 0

    def _get_coeffs(self, a_target):
        """Return (a,b,c) for the given target a using nearest-lower lookup."""
        idx = bisect_right(self.coeff_keys, a_target) - 1
        if idx < 0:
            idx = 0
        return self.coeff_values[idx]

    def _adapt_a(self, loss_val):
        """Update target a from loss feedback."""
        if self.loss_ema is None:
            self.loss_ema = loss_val
            return

        prev_ema = self.loss_ema
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss_val

        if prev_ema > 0 and loss_val > self.spike_threshold * prev_ema:
            self.a_target = max(self.a_target - self.step_down, self.a_min)
            self.stable_count = 0
            return

        self.stable_count += 1
        if self.stable_count >= self.stable_threshold:
            self.a_target = min(self.a_target + self.step_up, self.a_max)
            self.stable_count = 0

    @torch.no_grad()
    def step(self, loss_val=None):
        """Perform one optimizer step.

        Pass loss_val to enable coefficient adaptivity.
        """
        if loss_val is not None:
            self._adapt_a(float(loss_val))

        a, b, c = self._get_coeffs(self.a_target)
        should_log_spectrum = (
            self.spectrum_logger is not None
            and (self.spectrum_log_steps is None or self._step_idx in self.spectrum_log_steps)
        )

        for g in self.param_groups:
            if g.get('mu'):
                matrix_idx = 0
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if 'b' not in st:
                        st['b'] = torch.zeros_like(p)

                    st['b'].mul_(g['mom']).add_(p.grad, alpha=1 - g['mom'])
                    M = p.grad.add(st['b'], alpha=g['mom'])
                    if should_log_spectrum:
                        X = M / (M.norm() + 1e-7)
                        svals = torch.linalg.svdvals(X.float()).detach().cpu()
                        self.spectrum_logger(
                            step=self._step_idx,
                            layer_idx=matrix_idx,
                            shape=tuple(p.shape),
                            a_target=float(self.a_target),
                            coeffs=(float(a), float(b), float(c)),
                            singular_values=svals,
                        )
                    O = _apply_ns(M, a, b, c, steps=self.ns_steps)
                    p.sub_(O, alpha=g['lr_m'])
                    matrix_idx += 1
            else:
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if 's' not in st:
                        st['s'] = 0
                        st['m'] = torch.zeros_like(p)
                        st['v'] = torch.zeros_like(p)
                    st['s'] += 1
                    st['m'].mul_(0.9).add_(p.grad, alpha=0.1)
                    st['v'].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                    m_hat = st['m'] / (1 - 0.9 ** st['s'])
                    v_hat = st['v'] / (1 - 0.999 ** st['s'])
                    p.sub_(m_hat / (v_hat.sqrt() + 1e-8), alpha=g['lr_a'])

        self._step_idx += 1


class ScheduledFamilyMuon(AdaptiveMuon):
    """Muon with a spike-frequency-triggered scalar target and table lookup.

    The family is one-dimensional in the target key ``a``. Before the trigger,
    the optimizer uses the FastNS table key. A dynamic transition starts only
    when relative loss spikes become more frequent in the current period than
    in the previous period. This avoids using absolute loss levels or loss
    progress, which are unreliable when the attainable loss floor is nonzero.
    The actual ``(a,b,c)`` triple is always selected from the prepared table by
    nearest-lower lookup.
    """

    def __init__(self, model, lr_muon=0.02, lr_adam=0.001, momentum=0.95,
                 start_a=FAST_NS_TABLE_KEY, end_a=STRICT_FALLBACK_TABLE_KEY,
                 transition_start=None, transition_steps=100, ns_steps=5,
                 spike_window=40, spike_threshold=1.25, spike_ema_beta=0.98,
                 min_spikes=2, spike_count_margin=1,
                 spectrum_logger=None, spectrum_log_steps=None):
        self.start_a = float(start_a)
        self.end_a = float(end_a)
        if transition_start is not None and int(transition_start) < 0:
            transition_start = None
        self.fixed_transition_start = None if transition_start is None else int(transition_start)
        self.transition_start = self.fixed_transition_start
        self.transition_steps = max(1, int(transition_steps))
        self.spike_window = max(1, int(spike_window))
        self.spike_threshold = float(spike_threshold)
        self.spike_ema_beta = float(spike_ema_beta)
        self.min_spikes = max(1, int(min_spikes))
        self.spike_count_margin = max(0, int(spike_count_margin))
        self.family_lambda = 0.0
        self.trigger_step = self.fixed_transition_start
        self.loss_spike = 0.0
        self.current_spike_count = 0.0
        self.previous_spike_count = 0.0
        self.spike_count_delta = 0.0
        self._spike_history = []
        super().__init__(
            model,
            lr_muon=lr_muon,
            lr_adam=lr_adam,
            momentum=momentum,
            a_init=self.start_a,
            coeff_table=None,
            ns_steps=ns_steps,
            spectrum_logger=spectrum_logger,
            spectrum_log_steps=spectrum_log_steps,
        )
        self.step_up = 0.0
        self.step_down = 0.0
        self.a_min = min(self.start_a, self.end_a)
        self.a_max = max(self.start_a, self.end_a)
        # AdaptiveMuon initializes legacy spike settings; restore the scheduled
        # family's conservative spike-frequency settings after super().__init__.
        self.spike_threshold = float(spike_threshold)
        self.spike_ema_beta = float(spike_ema_beta)
        self.min_spikes = max(1, int(min_spikes))
        self.spike_count_margin = max(0, int(spike_count_margin))

    def _adapt_a(self, loss_val):
        # A spike is local and relative to the recent EMA, so it does not assume
        # that the optimal loss is zero or even known.
        loss_val = float(loss_val)
        if self.loss_ema is None:
            self.loss_ema = loss_val
            self._spike_history.append(0)
            return

        prev_ema = self.loss_ema
        self.loss_spike = float(prev_ema > 0.0 and loss_val > self.spike_threshold * prev_ema)
        self._spike_history.append(int(self.loss_spike))
        beta = min(0.999, max(0.0, self.spike_ema_beta))
        self.loss_ema = beta * self.loss_ema + (1.0 - beta) * loss_val

        if self.fixed_transition_start is not None or self.trigger_step is not None:
            return

        window = self.spike_window
        if len(self._spike_history) < 2 * window:
            return

        previous = self._spike_history[-2 * window:-window]
        current = self._spike_history[-window:]
        self.previous_spike_count = float(sum(previous))
        self.current_spike_count = float(sum(current))
        self.spike_count_delta = self.current_spike_count - self.previous_spike_count

        if self.current_spike_count < self.min_spikes:
            return
        if self.spike_count_delta < self.spike_count_margin:
            return
        self.trigger_step = self._step_idx
        self.transition_start = self._step_idx

    def _scheduled_lambda(self) -> float:
        start = self.fixed_transition_start if self.fixed_transition_start is not None else self.trigger_step
        if start is None:
            return 0.0
        raw = (self._step_idx - start) / self.transition_steps
        return float(min(1.0, max(0.0, raw)))

    def _get_coeffs(self, _a_target):
        self.family_lambda = self._scheduled_lambda()
        if self.family_lambda <= 0.0:
            self.a_target = self.start_a
        elif self.family_lambda >= 1.0:
            self.a_target = self.end_a
        else:
            self.a_target = (1.0 - self.family_lambda) * self.start_a + self.family_lambda * self.end_a
        return super()._get_coeffs(self.a_target)
