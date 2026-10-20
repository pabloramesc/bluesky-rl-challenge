# Validation

Run the required repository checks from the project root with the development
virtual environment activated:

```bash
python -m pytest
ruff check src experiments tests
ruff format --check src experiments tests
```

Temporal validation uses synthetic environments so it does not create additional
BlueSky simulators or require trained policies.

| Suite | Coverage |
|---|---|
| `test_temporal.py` | Basic history lifecycle, temporal order, causality, training and persistence |
| `test_temporal_matrix.py` | Every spatial option at the encoder level; temporal/history boundaries; all six builders × four temporal methods; both research winners with PPO/SAC; observation layouts and binary masks |
| `test_temporal_boundaries.py` | Actual SuperSuit ten-slot departures/reset, timeout bootstrapping, replay contents, recurrent references, TCN receptive field, normalization counts and invalid inputs |
| `test_temporal_checkpoints.py` | Every saved spatial option; all temporal × CLI mode dispatches; eight full workflows covering PPO/SAC × temporal methods, distributed across entity MLP, Transformer, GCN and GAT |
| `test_architecture_comparison.py` | Conditional tree expansion; invalid/duplicate settings; seed-weighted metrics and spatially matched temporal baselines; real subprocess bounds, failures, timeout/interrupt cleanup, resume/retry; full synthetic training/aggregation workflows |

Tests separate independent concerns instead of crossing every option through
expensive training workflows:

- Encoder contracts cover every pooling/context and graph choice once, then cross
  each temporal method with histories 1/2/17 on representative spatial encoders.
  History 8 covers the spatial options. Transformer depths 1/2/3 and multilayer
  GCN/GAT remain covered. Assertions retain finite gradients, NaN padding masks,
  empty/partial/full histories and independent per-frame aircraft permutations.
- Real optimizer updates and policy archive round trips cover every builder with
  every temporal method. Additional cases exercise both shortlisted spatial
  encoders with PPO/SAC and all temporal methods. Shared-policy builders still
  use ten slots; actor/critic separation and raw replay windows are checked.
- All observation encodings and both profiles run against flat, entity MLP,
  Transformer, distance-GCN and kNN-GAT layouts. One temporal method suffices here:
  these tests target layout/normalization, while temporal math is checked above.
- CLI restoration covers each saved spatial option and every temporal/mode pair.
  Full workflows retain normalization sidecars, evaluation, replay, resumed
  training and SAC replay-buffer restoration without repeating every combination.

Focused numerical reference tests, episode/agent boundary tests, non-temporal
baselines, and all CUDA parity tests are retained. CUDA tests skip when no device
is available. Spatial-only tests in `test_entities.py` and `test_gnns.py` cover
pooling and graph math independently. Cases are deterministic; no random subset
or optional exhaustive suite is required for normal validation.

Measured on the same development machine on 2026-10-20: **5,519 tests / 521.67 s**
before, **980 tests / 118.11 s** after (about 77% less runtime). Timings vary with
hardware and load. The slowest retained checks are actual training subprocess
aggregation/resume (12.92 s) and SIGINT checkpoint recovery (5.90 s); these protect
process isolation and saved training state.

Profile the complete suite before adding more integration combinations:

```bash
python -m pytest --durations=20
```

For focused checks:

```bash
python -m pytest tests/test_temporal.py tests/test_temporal_boundaries.py
python -m pytest tests/test_temporal_matrix.py tests/test_temporal_checkpoints.py
```

These are functional and numerical tests, not policy-performance ablations.
Positive history lengths and layer counts are configurable without an upper
bound; representative boundary values are tested rather than every integer.
Continuous radii, seeds, and simulator scenarios are not exhaustively enumerated.
