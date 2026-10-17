# Experiments

Run from the repository root:

```bash
python -m experiments.train_single_ppo --timesteps 100000
python -m experiments.train_single_ppo replay --episodes 3
```

- `train_single_*`: single-agent PPO, SAC, TD3, or DDPG.
- `train_zoo_*`: multi-agent shared-policy PPO or SAC.
- Modes: `train`, `resume`, `evaluate` (headless), `replay` (visual).
- `models/`: latest checkpoints and matching normalization statistics.
- `results/`: preserved per-run checkpoints, scores, logs, and plots.

Training ends with headless evaluation. Use `--model-path PATH` to select a
checkpoint. Parallel runs need separate processes and model paths.
Both output folders are ignored by Git.
