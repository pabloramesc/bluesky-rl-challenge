# Personal agents and experiments

All code in this directory is personal work built on the upstream TU Delft
competition environments. Run commands from the repository root.

## Setup and training

Create `.venv` using the [setup guide](../README.md#setup), then run from the
repository root:

```bash
source .venv/bin/activate
uv pip install -e ./vendor/bluesky-gym -e ".[dev]"
```

### Single-agent MLP baselines

PPO, SAC, TD3, and DDPG control one aircraft in the Gymnasium `CompetitionEnv`.
All use two hidden layers of 256 units. PPO uses tanh actor/value networks;
SAC and TD3 use ReLU actors and twin Q critics; DDPG uses a ReLU actor and one
Q critic. TD3 and DDPG explore with Gaussian action noise (standard deviation
0.1 in normalized action coordinates). TD3 uses delayed actor updates and
target policy smoothing; DDPG updates its actor each step without smoothing. Edit the builders in
[`models.py`](../src/bluesky_pabloramesc/models.py) to customize them.
These baselines use Stable-Baselines3 for the learning algorithms.

```bash
python -m experiments.train_single_ppo --timesteps 100000
python -m experiments.train_single_sac --timesteps 100000
python -m experiments.train_single_td3 --timesteps 100000
python -m experiments.train_single_ddpg --timesteps 100000
python -m experiments.train_single_ppo evaluate --episodes 100 --eval-seed 10000
python -m experiments.train_single_ppo replay --episodes 3
python -m experiments.train_single_sac replay --episodes 3
python -m experiments.train_single_td3 evaluate --episodes 100 --eval-seed 10000
python -m experiments.train_single_ddpg replay --episodes 3
```

Training defaults to **100,000 steps**; `--timesteps` overrides this budget.
Install the project before running these commands. Models
default to `experiments/models/{algorithm}_single_mlp.zip` (`ppo`, `sac`,
`td3`, or `ddpg`), each with a matching `_vecnormalize.pkl` file. Keep both files for replay.
Use `--model-path`, `--seed`, and `--device` to configure training. Evaluation
uses a separate `--eval-seed` (default `2026`) in all modes. `evaluate` loads a
checkpoint without rendering; `replay` renders it. These modes also accept
`--stats-path` for an explicit normalization file. Use the same evaluation seed,
episode count, and environment settings when comparing policies; this does not
yet replace an explicit fixed scenario suite or multi-seed analysis. Training ends
with evaluation on raw rewards, with normalization statistics frozen. Run one
environment per process because BlueSky shares simulator state.

### Multi-agent baselines

```bash
python -m experiments.train_zoo_ppo 100000
python -m experiments.train_zoo_sac 100000
```

PPO and SAC share wrappers in
[`multi_agent.py`](../src/bluesky_pabloramesc/multi_agent.py). Their default models
and normalization sidecars are written to `experiments/models/`.

```bash
python -m experiments.train_zoo_ppo replay
python -m experiments.train_zoo_sac replay
```

The earlier upstream-example checkpoint is preserved as
`experiments/models/ppo_competition.zip`; it is distinct from the tuned trainers'
models and has no normalization sidecar.

Put custom agents, algorithms, and reusable utilities in
[`src/bluesky_pabloramesc/`](../src/README.md). Keep training scripts, configs,
and experiment notes here. Models, results, and virtual environments are ignored
by Git; store large outputs separately when sharing.

## Reviewing a training run

Every training command creates a unique folder under `experiments/results/`,
including the checkpoint name, UTC start date/time, and a random suffix. The
results directory is created automatically if missing. Use `--results-dir PATH`
to choose another location.

| File | Contents |
| --- | --- |
| `run.json` | UTC start/end dates, completion status, training duration and speed, algorithm, settings, architecture, device, seeds, package versions, requested/actual steps, and final evaluation scores |
| `episodes.csv` | Completed episode number, training steps, environment index, raw return, episode length, and elapsed seconds; written during training |
| `progress.csv` | SB3 metrics at its logging intervals, including available losses and learning statistics; columns vary by algorithm |
| `performance.png` | Raw training return, rolling mean over up to 20 episodes, and episode length versus completed episode |
| `policy.zip` | This run's policy checkpoint |
| `policy_vecnormalize.pkl` | This run's matching normalization statistics |
| `policy_replay_buffer.pkl` | SAC/TD3/DDPG replay experience for resuming (can be large) |
| `checkpoints/step_N/policy.*` and sidecars | Periodic snapshots at cumulative timestep N, with matching normalization and replay buffers |

The regular checkpoint under `experiments/models/` is still updated. Each run's
copy in results is preserved, even when you train the same algorithm again.
Replay a particular run with:

```bash
python -m experiments.train_single_td3 replay \
  --model-path experiments/results/RUN_FOLDER/policy
```

Training curves show exploratory behavior on training scenarios, using raw
rewards from Monitor before reward normalization. Final deterministic evaluation
runs after training on the same environment with frozen normalization and the
separate evaluation seed; `run.json` includes each evaluation return and length,
plus mean and standard deviation. These are final evaluation scores, not periodic
evaluation curves. For shared policies, each CSV episode is one agent's episode;
it is not a joint team return.

Only completed training episodes enter `episodes.csv`. Very short runs may have
no completed episodes or logged losses.

### Periodic checkpoints and Ctrl+C

By default, training saves a complete checkpoint every **10,000 additional steps**
under the run's `checkpoints/step_N/` directory. Set `--checkpoint-interval` to a
positive step count to change this interval:

```bash
python -m experiments.train_single_td3 --checkpoint-interval 5000
```

Each snapshot contains `policy.zip`, `policy_vecnormalize.pkl`, and an off-policy
`policy_replay_buffer.pkl` when applicable. Saves occur at rollout boundaries,
after collected transitions enter the buffer; PPO may pass the interval before
finishing a rollout. Shared-policy intervals count transitions across all vector
slots. For resumed training the interval starts from the loaded timestep count,
and directory names use cumulative steps. `run.json` records the latest completed
periodic checkpoint. Snapshots are retained, so replay buffers can use substantial
disk space.

Press **Ctrl+C once** to stop and save current progress. Wait for the saved-path
message: the regular checkpoint in `experiments/models/` and the run's archived
`policy` are updated, the report is marked `interrupted`, timings and step counts
are recorded, and a partial performance plot is generated. Final evaluation is
skipped. You can then use the normal `resume` command. The simulator and incomplete
PPO rollout start fresh on resume; this is not an exact simulator-state snapshot.
An interruption before the model is initialized cannot save a policy.

Unexpected failures mark the report `failed` and preserve existing CSV rows and
completed periodic checkpoints. Forced termination or power loss does not run the
Ctrl+C handler; resume from the last completed periodic checkpoint instead:

```bash
python -m experiments.train_single_td3 resume \
  --model-path experiments/results/RUN_FOLDER/checkpoints/step_10000/policy \
  --output-path experiments/models/td3_continued \
  --timesteps 100000
```

## Resuming training

Use `resume` to load an existing policy, optimizer state, and its matching
normalization statistics. `--timesteps` is the number of **additional environment
steps**, not the final total or a number of episodes. PPO may finish a complete
rollout beyond that budget. `--episodes` still controls the final evaluation.

```bash
python -m experiments.train_single_ppo resume --timesteps 100000
python -m experiments.train_single_td3 resume \
  --model-path experiments/results/RUN_FOLDER/policy \
  --output-path experiments/models/td3_continued \
  --timesteps 100000
```

Resume is supported for all four single-agent trainers and the shared PPO/SAC
trainers. The default source is the usual algorithm checkpoint. By default,
resuming updates that checkpoint; use `--output-path` to preserve the source.
`--stats-path` can specify a different source normalization file. Each continuation
also gets a new results folder, preserving its policy and recording the source,
initial/added/cumulative step counts, timings, and new episode history.

SAC, TD3, and DDPG now save a matching `_replay_buffer.pkl` beside both their
regular and archived policies. Resume restores it when present. Older checkpoints
without one can still resume, but warn and start a new buffer with a fresh
exploration warmup before learning. Buffer files can consume substantial disk
space. PPO collects fresh rollouts and needs no replay buffer.

Normalization resumes updating during training and freezes during final evaluation.
The simulator starts fresh episodes; the original simulator state and exact random
number generator state are not restored. `--seed` seeds this continuation.

## Checks

```bash
python -m pytest
ruff check src experiments tests
ruff format --check src experiments tests
```

Tests use short synthetic environments to check policy updates, checkpoint
round trips, frozen evaluation statistics, cleanup, and multi-agent padding.
They do not start BlueSky or measure training quality.

Implementation status and remaining comparisons live in [ROADMAP.md](../ROADMAP.md).
