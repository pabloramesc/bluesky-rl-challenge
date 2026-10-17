# Experiments

Run scripts from the repository root with `python -m experiments.<script>`.
Use `--help` for options.

Folders:

- `models/`: latest policies and matching normalization statistics.
- `results/`: per-run checkpoints, metrics, logs, plots, and comparison reports.
- Both folders are ignored by Git.

Scripts:

- `train_single_ppo`, `train_single_sac`, `train_single_td3`, `train_single_ddpg`:
  single-agent training, with `resume`, `evaluate`, and `replay` modes.
- `train_zoo_ppo`, `train_zoo_sac`: multi-agent shared-policy training; same modes.
- `compare_actions`: compare four relative/absolute heading and speed combinations.
- `compare_observations`: compare upstream, ego Cartesian, and goal-aligned features.

Comparisons default to PPO and one training seed. Multiple seeds require
`--seeds`; `--output-dir` overrides their default folder under `results/`.
Use `--resume` to reuse completed comparison runs.

See [RESEARCH.md](../RESEARCH.md) for findings and next experiments.
