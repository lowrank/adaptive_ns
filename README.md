# Adaptive Newton-Schulz Coefficients for Muon

This repository studies adaptive Newton-Schulz (NS) coefficient schedules for the
Muon optimizer. The core question is whether Muon should use one fixed degree-5
NS polynomial, or a dense table of stable coefficient triples that can be selected
online from loss-stability feedback.

The project is organized as a small research codebase: coefficient search,
optimizer implementation, benchmark problems, optimizer factories, generated
figures, numeric results, and a writeup are kept separate.

## Repository Layout

```text
.
├── muon/
│   ├── adaptive_optimizer.py      # AdaptiveMuon implementation
│   ├── coeff_table.py             # Prepared in-memory Muon coefficient table
│   ├── coeff_search.py            # Offline vectorized coefficient search
│   ├── newton_schulz.py           # Scalar polynomial and stability checks
│   └── train.py                   # MLP and legacy training utilities
├── benchmarks/
│   ├── problems.py                # Benchmark target functions
│   ├── optimizer_factory.py       # Muon, Adam, AdamW factories
│   ├── plotting.py                # Loss/target plotting helpers
│   ├── runner.py                  # Benchmark execution and metric writing
│   └── run_suite.py               # CLI entrypoint
├── scripts/
│   ├── build_coeff_table.py       # Print candidate dense table entries
│   ├── generate_all_experiments.py # Rebuild benchmark outputs and report
│   ├── generate_report.py          # Rebuild LaTeX/PDF report from results
│   ├── search_family_endpoint.py   # Rank FastNS-to-fallback family endpoints
│   ├── spectrum_diagnostics.py     # Singular-value spectrum diagnostics
│   └── summarize_results.py        # Compute ratios and aggregate summaries
├── results/                       # Numeric benchmark outputs
├── figures/                       # Generated plots
├── docs/                          # Report source/PDF artifacts
├── tests/                         # Regression tests
└── run.py                         # Original experiment driver
```

## Coefficient Search

The dense coefficient table follows the 5-iteration band criterion from
`../ns_coeff_search_algorithm.pdf`:

- sample triples `(a, b, c)`;
- evaluate `f(x) = ax + bx^3 + cx^5` for 5 iterations on a log grid in
  `[1e-3, 1]`;
- keep triples with final values in `[0.5, 1.5]` and bounded intermediate orbits;
- bucket stable high-`a` candidates into a dense lookup table.

The runtime table lives in `muon/coeff_table.py` as `MUON_COEFF_TABLE`. It is
prepared before experiments, imported into memory, and pre-indexed as sorted
key/value tuples for fast nearest-lower lookup. Benchmark runs do not create the
table on the fly and do not load it from a JSON file.

Regenerate candidate high-`a` entries for review with:

```bash
python3 scripts/build_coeff_table.py
```

When entries are accepted, copy them into `muon/coeff_table.py` and rerun the
tests before launching benchmarks.

The prepared high-`a` table currently covers `a = 3.50, 3.51, ..., 3.89`.
Lower entries (`1.50`, `2.00`, `2.50`, `3.00`, `3.44`) are retained as
conservative warm-up and comparison points; they are not all strict-band
admissible from `x = 1e-3`.

## Benchmark Suite

The benchmark suite compares:

- Standard Muon fixed at classical NS coefficients;
- Jordan Muon fixed at Keller Jordan's coefficients;
- FastNS Muon fixed at the fastest strict-band triple;
- Adaptive Dense Muon, now the one-parameter path from FastNS to a searched strict fallback endpoint;
- Adam;
- AdamW.

Default benchmark target problems:

- `low_frequency`: `sin(2*pi*x) + 0.30*cos(6*pi*x)`;
- `medium_fourier`: three-mode Fourier mixture;
- `high_frequency`: four-mode higher-frequency Fourier mixture;
- `chirp_quadratic`: quadratic chirp plus ripple;
- `gaussian_bumps`: fixed five-bump Gaussian mixture;
- `runge_rational`: Runge-style rational peak plus sinusoidal ripple;
- `cusp_abs`: nonsmooth absolute-value cusp plus ripple;
- `multiscale_mix`: Fourier components plus localized bumps.

Run a smoke test:

```bash
python3 -m benchmarks.run_suite --run-name smoke --epochs 5 --n-inits 1 --hidden 16
```

Regenerate the full current experiment suite, summaries, figures, optional spectrum diagnostics, and PDF report. The default uses 20 initializations per problem/method pair, and Adaptive Dense Muon starts at FastNS before transitioning to the searched fallback endpoint. Pass `--skip-spectrum-diagnostics` when only the benchmark results and report need refreshing:

```bash
python3 scripts/generate_all_experiments.py
```

For a fast smoke run:

```bash
python3 scripts/generate_all_experiments.py \
  --run-name smoke \
  --epochs 5 \
  --n-inits 1 \
  --hidden 16 \
  --skip-report-compile
```

Skip the spectrum diagnostic when only the final-loss benchmark is needed:

```bash
python3 scripts/generate_all_experiments.py --skip-spectrum-diagnostics
```

Run only the benchmark suite manually:

```bash
python3 -m benchmarks.run_suite \
  --run-name adaptive_ns_suite \
  --epochs 500 \
  --n-inits 20 \
  --hidden 32 \
  --threads 1 \
  --adaptive-a-init 3.87
```

Outputs are written to:

```text
results/<run-name>/summary.csv
results/<run-name>/summary.json
results/<run-name>/final_loss_ratios.csv
results/<run-name>/aggregate.csv
results/<run-name>/config.json
results/<run-name>/problems.json
results/<run-name>/optimizers.json
results/<run-name>/curves_npz/*.npz
figures/<run-name>/loss_curves/*.png      # mean curves with +/- one-sigma bands
figures/<run-name>/targets/*.png
figures/<run-name>/final_loss_by_problem.png
figures/<run-name>/final_loss_ratio_heatmap.png
```


## Current Adaptive Baseline

The legacy loss-spike dense-table adaptive method has been replaced in the default benchmark suite. The current Adaptive Dense Muon baseline starts at FastNS and then moves along a one-parameter table-key path

```text
a(lambda) = (1 - lambda) * 3.87 + lambda * 3.62
(a,b,c) = nearest_lower_lookup(MUON_COEFF_TABLE, a(lambda))
```

with `lambda = 0` until the adaptive spike-frequency trigger fires and `lambda = 1` after the transition window. The current endpoint is the prepared table entry at `a = 3.62`:

```text
STRICT_FALLBACK_NS = (3.624985, -7.285039, 4.444867)
```


By default, Adaptive Dense Muon does not use a fixed transition step. It tracks local relative loss spikes while starting from FastNS. With the current conservative default, a transition begins only when the current 40-step window has at least two spikes and at least one more spike than the previous 40-step window. A spike means `loss_t > 1.25 * ema_loss_{t-1}` with EMA beta `0.98`. Use `--adaptive-transition-start N` to force a fixed start for ablations; the default `-1` keeps the dynamic trigger.

This endpoint is not assumed to be globally safest. It is a reproducible table-mined starting point: its scalar map has lower overshoot than FastNS, and every prepared table entry visited by the a-only schedule stays inside the strict five-step band. Re-rank prepared-table endpoints with:

```bash
python3 scripts/search_family_endpoint.py
```

Run a focused 20-initialization comparison with:

```bash
python3 -m benchmarks.run_suite \
  --run-name family_transition_suite \
  --epochs 500 \
  --n-inits 20 \
  --hidden 32 \
  --threads 1 \
  --optimizers jordan_muon fastns_muon adaptive_dense_muon
python3 scripts/summarize_results.py --run-name family_transition_suite
```

Current full-suite replacement results with matched seeds: the main empirical
signal is fast early decay from FastNS and from Adaptive Dense Muon before it
triggers. FastNS is also the adaptive method's no-trigger fallback: if the spike
trigger never fires, the optimizer remains exactly FastNS. Adaptive Dense Muon
wins final loss on 3/8 problems, matches fixed FastNS exactly on the four
no-trigger problems, and is slightly worse on multiscale mix because a very late
step-491 trigger changes only the last few updates. It reduces mean FastNS spike count from `16.17` to `12.09`, and its
geometric final-loss ratio is `1.024` versus fixed Jordan (`1.099`) and fixed
FastNS (`1.152`). These final-loss gains are not uniform: on harder oscillatory,
chirp, cusp, and multiscale targets the advantage is small, absent, or method-
dependent. The current spike-frequency trigger is therefore a conservative
baseline, not an optimized adaptive policy. The benchmark runner uses the same
initialization/minibatch seed for all methods in a problem/initialization pair,
and a regression test checks that an untriggered adaptive run has the same loss
trace as fixed FastNS.

Future endpoint searches may still use spectral energy as an offline explanatory diagnostic, but the online adaptive trigger should stay local and conservative. The next controller should optimize when to leave FastNS, whether to move to a gentler endpoint at all, and how quickly to move. It should combine spike-frequency safety with gradient-value distribution statistics, and should explicitly detect regimes where the NS family itself is weak.

## Spectrum Diagnostics

The spectrum diagnostic tests whether later-stage Muon updates have different singular-value structure. It logs singular values of the Frobenius-normalized momentum matrix before the Newton-Schulz polynomial is applied. These are singular values, not eigenvalues, because the Muon matrix polynomial acts on singular values.

Run the optional diagnostic:

```bash
python3 scripts/spectrum_diagnostics.py
```

Outputs are written to:

```text
results/spectrum_diagnostics/spectra.csv
results/spectrum_diagnostics/spectrum_metrics.csv
results/spectrum_diagnostics/spectrum_metric_summary.csv
results/spectrum_diagnostics/losses.csv
figures/spectrum_diagnostics/ns_scalar_maps.png
figures/spectrum_diagnostics/effective_rank_overview.png
figures/spectrum_diagnostics/top_energy_overview.png
figures/spectrum_diagnostics/hist_<problem>.png
```

The current optional diagnostic supports a nuanced version of the spectral-spread hypothesis: hidden-layer spectra become less top-dominated later in training, but many singular values remain very small. This makes Jordan's gentler small-mode amplification plausible late in training while explaining why FastNS-start adaptive runs can still keep pushing weak modes.

## Tests

Run the regression tests with:

```bash
pytest -q
```

The tests cover scalar NS stability, coefficient search, the prepared coefficient
table, adaptive optimizer behavior, and the spectrum logging hook.

## Notes

The project is intentionally CPU-friendly, but Muon's matrix polynomial update is
more expensive than Adam on small MLPs. Increase `--hidden` and `--epochs` for stronger evidence once the local smoke and standard benchmark suites pass; the standard suite already uses 20 initializations by default.
