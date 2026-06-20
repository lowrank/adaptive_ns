"""Tests for the adaptive optimizer."""

import torch
import torch.nn as nn
from benchmarks.optimizer_factory import OptimizerSpec, default_optimizer_specs, fixed_muon_factory, scheduled_family_muon_factory
from benchmarks.problems import select_problems
from benchmarks.runner import BenchmarkConfig, train_one
from muon.adaptive_optimizer import AdaptiveMuon, ScheduledFamilyMuon
from muon.coeff_table import (
    FAST_NS, FAST_NS_TABLE_KEY, JORDAN_NS, MUON_COEFF_TABLE,
    STRICT_FALLBACK_NS, STRICT_FALLBACK_TABLE_KEY,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


class TestAdaptiveMuon:
    """Tests for the adaptive optimizer."""

    def test_initializes_with_default_target(self):
        """Default init should use the conservative a=3.0 table entry."""
        model = TinyModel()
        opt = AdaptiveMuon(model)
        assert opt.a_target == 3.0
        a, b, c = opt._get_coeffs(opt.a_target)
        assert abs(a - 3.0) < 0.01

    def test_fastns_table_key_selects_fastns_coefficients(self):
        """Adaptive start at 3.87 should select the FastNS table entry."""
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.87)
        assert opt._get_coeffs(opt.a_target) == FAST_NS

    def test_spectrum_logger_records_requested_step(self):
        """Spectrum logger should record only requested optimizer steps."""
        torch.manual_seed(0)
        records = []

        def logger(**payload):
            records.append(payload)

        model = TinyModel()
        opt = AdaptiveMuon(model, spectrum_logger=logger, spectrum_log_steps={0})
        x = torch.randn(8, 16)
        y = torch.randn(8, 4)
        for _ in range(2):
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            opt.step(loss_val=1.0)

        assert len(records) == 1
        assert records[0]['step'] == 0
        assert records[0]['layer_idx'] == 0
        assert records[0]['singular_values'].ndim == 1

    def test_loss_decreases_after_step(self):
        """Loss should decrease after one step."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.44,
                           coeff_table={3.44: JORDAN_NS})
        x = torch.randn(8, 16)
        y = torch.randn(8, 4)

        loss_before = nn.MSELoss()(model(x), y).item()
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()
        loss_after = nn.MSELoss()(model(x), y).item()

        assert loss_after < loss_before, "Loss did not decrease"

    def test_a_stays_in_bounds(self):
        """a should stay in [1.5, 4.0]."""
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.0)
        for _ in range(100):
            opt.step(loss_val=0.1)
        assert 1.5 <= opt.a_target <= 4.0

    def test_a_increases_when_stable(self):
        """After many stable steps, a should increase above initial value."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.0)

        # Feed decreasing losses (simulating stable training)
        for i in range(500):
            opt.step(loss_val=1.0 / (1 + i * 0.01))

        assert opt.a_target > 3.0, (
            f"a should increase during stable training. "
            f"Got a={opt.a_target:.2f}.  Check your loss_ema and stable_count logic."
        )

    def test_a_decreases_on_spike(self):
        """A large loss spike should decrease a."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.0)

        # Build up stable history
        for _ in range(200):
            opt.step(loss_val=0.1)

        a_before = opt.a_target

        # Inject a spike
        opt.step(loss_val=100.0)

        assert opt.a_target < a_before, (
            f"a should decrease after a spike. "
            f"Before: {a_before:.2f}, After: {opt.a_target:.2f}"
        )

    def test_default_adaptive_factory_uses_scheduled_family(self):
        """Default benchmark adaptive method should use the scheduled family."""
        specs = default_optimizer_specs()
        adaptive = next(spec for spec in specs if spec.slug == 'adaptive_dense_muon')
        opt = adaptive.factory(TinyModel())
        assert isinstance(opt, ScheduledFamilyMuon)
        assert opt.start_a == FAST_NS_TABLE_KEY
        assert opt.end_a == STRICT_FALLBACK_TABLE_KEY
        assert opt.spike_window == 40
        assert opt.spike_threshold == 1.25


    def test_spike_frequency_sets_dynamic_trigger(self):
        model = TinyModel()
        opt = ScheduledFamilyMuon(
            model,
            transition_start=None,
            transition_steps=10,
            spike_window=2,
            spike_threshold=1.5,
            spike_ema_beta=0.0,
            min_spikes=1,
            spike_count_margin=1,
        )
        for step, loss in enumerate([1.0, 1.0, 1.0, 2.0]):
            opt._step_idx = step
            opt._adapt_a(loss)
        assert opt.trigger_step == 3
        assert opt.transition_start == 3
        assert opt.current_spike_count == 1
        assert opt.previous_spike_count == 0

    def test_spike_frequency_requires_increase_over_previous_period(self):
        model = TinyModel()
        opt = ScheduledFamilyMuon(
            model,
            transition_start=None,
            transition_steps=10,
            spike_window=2,
            spike_threshold=1.5,
            spike_ema_beta=0.0,
            min_spikes=1,
            spike_count_margin=1,
        )
        for step, loss in enumerate([1.0, 2.0, 1.0, 2.0]):
            opt._step_idx = step
            opt._adapt_a(loss)
        assert opt.trigger_step is None
        assert opt.transition_start is None
        assert opt.current_spike_count == opt.previous_spike_count


    def test_untriggered_scheduled_family_matches_fastns_trace(self):
        problem = select_problems(['high_frequency'])[0]
        config = BenchmarkConfig(epochs=8, batch_size=16, hidden=8, n_inits=1)
        seed = 20240619
        fastns = OptimizerSpec(
            name='FastNS Muon',
            slug='fastns_muon',
            factory=fixed_muon_factory(FAST_NS),
            description='test fixed FastNS',
        )
        adaptive = OptimizerSpec(
            name='Adaptive Dense Muon',
            slug='adaptive_dense_muon',
            factory=scheduled_family_muon_factory(
                transition_start=None,
                transition_steps=100,
                spike_window=40,
                spike_threshold=1e9,
                min_spikes=999,
            ),
            description='test untriggered adaptive',
        )

        fast_losses, *_ = train_one(
            problem, fastns, seed=seed, config=config, device=torch.device('cpu')
        )
        adaptive_losses, adaptive_a, adaptive_triggers, *_ = train_one(
            problem, adaptive, seed=seed, config=config, device=torch.device('cpu')
        )

        assert torch.tensor(adaptive_losses).allclose(torch.tensor(fast_losses), atol=0.0, rtol=0.0)
        assert torch.isnan(torch.tensor(adaptive_triggers)).all()
        assert torch.allclose(
            torch.tensor(adaptive_a),
            torch.full_like(torch.tensor(adaptive_a), FAST_NS_TABLE_KEY),
        )

    def test_scheduled_family_schedules_a_and_uses_table_lookup(self):
        # The family should schedule only a and select prepared table triples.
        model = TinyModel()
        opt = ScheduledFamilyMuon(model, transition_start=0, transition_steps=10)

        opt._step_idx = 0
        assert opt._get_coeffs(opt.a_target) == FAST_NS
        assert opt.a_target == FAST_NS_TABLE_KEY

        opt._step_idx = 5
        coeffs = opt._get_coeffs(opt.a_target)
        assert opt.a_target == (FAST_NS_TABLE_KEY + STRICT_FALLBACK_TABLE_KEY) / 2
        assert coeffs == MUON_COEFF_TABLE[3.74]

        opt._step_idx = 10
        coeffs = opt._get_coeffs(opt.a_target)
        assert coeffs == STRICT_FALLBACK_NS
        assert opt.a_target == STRICT_FALLBACK_TABLE_KEY
        assert opt.family_lambda == 1.0
