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
| `test_mappo.py` | Masked GAE and clipped-loss references; decentralized actor/centralized critic; joint departures, final-state timeout bootstrap, real updates, replaceable modules, checkpoints, normalization, resume, cancellation, and CLI dispatch |
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

`test_ego_attention.py` verifies a single context-conditioned four-head query,
masked neighbours, context selection, empty traffic, all policy builders, and
training/checkpoint/evaluation/resume round trips. Existing entity/GNN tests also
check ego readout permutation invariance, padding, and finite gradients. The
architecture comparison tests cover the three-variant attention preset, matched
protocol, reporting, and reuse on resume.

`test_rewards.py` checks the reward formulas, finite CPA edge cases, signed polygon
clearances, command wrapping, interval scaling, reset, and individual completion
using small synthetic environments. `test_reward_checkpoints.py` checks reward CLI
validation, component reporting, legacy defaults, archive/sidecar consistency,
and training/resume/evaluation configuration restoration.

`test_reward_ablation.py` checks one-term variant isolation, fixed attention/PPO
commands, matched scenarios, objective pairing, per-seed aggregation, report
validation, and process scheduling/resume without simulator training.

`test_ippo.py` verifies separate actors/critics/optimizers, peer-independent local
inference and updates, independent KL stopping, departed-agent masking, timeout
bootstrapping, every optimizer's checkpoint state, frozen evaluation statistics,
and train/resume/evaluate/replay CLI workflows. It also checks SuperSuit's integer
done flags and preserves terminal metrics across auto-reset during scenario
evaluation. Synthetic traffic uses the MAPPO test
fixture so boundary behavior stays comparable.

`test_multi_agent_comparison.py` verifies matched worker commands and budgets,
strict report validation, paired summaries with safety and training costs,
read-only dry runs, and rejection of changed protocols on resume for flat MLPs
and additive attention. It rejects mismatched attention network classes and
context, and reuse of MLP manifests for attention runs.

`test_multi_agent_attention.py` checks local aircraft permutation invariance,
decentralized actors and local critics, peer-sensitive centralized values,
agent permutation equivariance, departed-agent and empty-set masking, and finite
gradients. Synthetic joint environments exercise updates to actor/critic attention,
all three algorithms' optimizer round trips, architecture sidecars, entity
normalization, frozen evaluation, resumed training, and CLI configuration.

## Context-token Transformer checks

`test_context_transformer.py` checks context/aircraft interactions before readout,
aircraft permutation invariance, padding, empty traffic, finite gradients,
separate actor/critic encoders, short policy updates, serialization, checkpoint
evaluation/resume, temporal integration, CLI dispatch and custom comparison trees.

`test_context_attention_comparison.py` checks that only the new architecture
launches, reference files remain unchanged, inherited protocols match, all four
results appear in paired reports, and resume/retry never trains a reference.
Missing, changed or mismatched evidence is rejected before reuse.
