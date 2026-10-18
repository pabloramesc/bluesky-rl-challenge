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
| `test_temporal_matrix.py` | All valid encoder/pooling/context combinations; GCN/GAT and all graph types; encoder depths 1/2/3; temporal histories 1/2/8/17; six builders; all six observation encodings and both profiles |
| `test_temporal_boundaries.py` | Actual SuperSuit ten-slot departures/reset, timeout bootstrapping, replay contents, recurrent references, TCN receptive field, normalization counts and invalid inputs |
| `test_temporal_checkpoints.py` | CLI resume/evaluate/replay restoration for every spatial/pooling/temporal combination; full PPO/SAC workflows for entity MLP, Transformer, GCN and GAT |
| `test_architecture_comparison.py` | Conditional tree expansion; invalid/duplicate settings; seed-weighted metrics; real subprocess bounds, failures, timeout/interrupt cleanup, resume/retry; full synthetic training/aggregation workflows |

The encoder matrix checks outputs and gradients for empty, partially padded, and
populated windows, masks NaN padding, and independently permutes aircraft within
each frame. The training matrix crosses every pooling/context and graph option
with all four temporal methods and all six builders, including ten slots for
shared-policy PPO/SAC. Models undergo real optimizer updates and policy archive
round trips. Only collection, minibatch, and update budgets are shortened;
production encoder and policy-head settings are retained.

`none` is checked as the non-temporal spatial baseline. CUDA forward/backward and
evaluation parity checks run when a CUDA device is available; otherwise those
cases are skipped. PPO/SAC workflow tests additionally exercise normalization
sidecars, evaluation, saved replay buffers and resumed training.

For focused checks:

```bash
python -m pytest tests/test_temporal.py tests/test_temporal_boundaries.py
python -m pytest tests/test_temporal_matrix.py tests/test_temporal_checkpoints.py
```

These are functional and numerical tests, not policy-performance ablations.
Positive history lengths and layer counts are configurable without an upper
bound; representative boundary values are tested rather than every integer.
Continuous radii, seeds, and simulator scenarios are not exhaustively enumerated.
