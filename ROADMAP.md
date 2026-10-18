# bluesky-pabloramesc — Roadmap

Progress from reproducible baselines to advanced, safety-aware multi-agent control.

## 1. Baseline algorithms

- [x] Single-agent PPO + MLP — [builder](src/bluesky_pabloramesc/models.py), [training/checkpoint tests](tests/test_training.py)
- [x] Single-agent SAC + MLP — same builder/test modules
- [x] Single-agent TD3 / DDPG + MLP — same builder/test modules
- [ ] Hyperparameter optimization (Optuna)

## 2. Actions and observations

- [x] Implement and compare relative/absolute heading/speed actions — initial comparison complete; relative heading/absolute speed preferred
- [x] Compare 5/10/20-second decision intervals — retain default 10 seconds; initial single-seed comparison complete, validate across seeds
- [x] Compare custom ego-centric/goal-relative features against the upstream baseline — initial single-seed comparison complete; validate across seeds
- [x] Add TCPA, DCPA, closing rate, and presence mask (`conflict` observation encoding). Compare with existing relative-bearing features — initial single-seed comparison complete; no robust observation winner
- [ ] Improve obstacle/sector geometry (nearest edge, ray casting)
- [ ] Compare raw vs engineered features

## 3. Neural architectures

- [x] Shared entity MLP + mean/max pooling — `--architecture entity_mlp --pooling mean|max|mean_max`
- [x] Additive attention — `--architecture entity_mlp --pooling additive_attention --attention-context ownship_goal|all`
- [x] Transformer / Set Transformer — `--architecture entity_transformer --transformer-layers N --pooling mean|max|mean_max|additive_attention|pma`
- [x] Graph Neural Network (GCN / GAT) — `--architecture entity_gnn --gnn-type gcn|gat --graph fully_connected|distance|knn`
- [x] Fixed-window temporal modeling — `--temporal frame_stack|gru|lstm|tcn --history-length N`
- [ ] Stateful recurrent policies with sequence-aware training
- [ ] Compare architectures using the best single-agent algorithm

## 4. Reward engineering

- [ ] Add goal-progress shaping and compare completion reward weights (completion already upstream)
- [ ] Predictive separation penalties (TCPA/DCPA)
- [ ] Obstacle and sector-boundary clearance penalties
- [ ] Time and action-smoothness penalties
- [ ] Reward ablation and sensitivity analysis

## 5. Training strategies

- [x] Complete a million-step PPO run — relative heading, absolute speed, goal-aligned control observations
- [ ] Evaluate intermediate checkpoints on a validation suite to select training duration
- [ ] Traffic-density curriculum
- [ ] Conflict-complexity and obstacle curricula
- [ ] Compare controlled procedural randomization ranges (seeded generation already upstream)
- [ ] Training above nominal traffic density
- [ ] Hard-scenario mining and replay
- [ ] Prioritized experience replay (SAC)

## 6. Multi-agent RL

- [ ] Independent PPO baseline (separate policies; current shared PPO is not this)
- [x] Parameter-shared PPO — [builder](src/bluesky_pabloramesc/models.py), [ten-slot prediction test](tests/test_models.py), [wrappers](tests/test_multi_agent.py)
- [ ] Custom MAPPO with centralized MLP critic (no MARLlib dependency)
- [ ] MAPPO + shared attention/Transformer actor
- [ ] MAPPO + centralized Transformer/GAT critic
- [ ] Compare local, global, and pairwise safety rewards
- [x] Parameter-shared SAC with local critics — [builder](src/bluesky_pabloramesc/models.py), [ten-slot prediction test](tests/test_models.py)
- [ ] Centralized-critic multi-agent SAC (optional)

## 7. Safe and robust RL

- [ ] Separate reward and safety-cost objectives
- [ ] Lagrangian constrained SAC / MAPPO
- [ ] Multiple safety critics (separation, obstacles, boundaries)
- [ ] Distributional / quantile critics
- [ ] CVaR risk-sensitive optimization
- [ ] Domain randomization (noise, delays, dynamics)
- [ ] Optional runtime safety shield (subject to competition rules)

## 8. Advanced action policies

- [ ] Learned intervention priority
- [ ] Learned action duration / action stacking
- [ ] Hierarchical RL / options
- [ ] Mixture of Experts
- [ ] Diffusion actor / multimodal policies

## 9. Advanced learning techniques

- [ ] Auxiliary prediction tasks (future positions, conflict probability)
- [ ] Intent / trajectory prediction of other aircraft
- [ ] Self-supervised encoder pretraining
- [ ] Imitation learning + RL fine-tuning
- [ ] Model-based RL / world models
- [ ] Offline RL
- [ ] Adversarial scenario generation
- [ ] Automatic curriculum / population-based training

## 10. Evaluation and submission

- [x] Separate training/evaluation seeds (`--seed`, `--eval-seed`; default evaluation seed: 2026)
- [x] Headless checkpoint evaluation with frozen normalization (`evaluate` mode; [tests](tests/test_training.py))
- [ ] Persist run configuration, package/upstream revisions, and evaluation results
- [ ] Explicit fixed evaluation scenario suite across policies and training seeds
- [ ] Automated trajectory visualization and failure analysis
- [ ] Matched-budget algorithm and architecture ablations
- [ ] Multi-seed evaluation and confidence intervals
- [ ] Pareto ranking: **safety → completion → efficiency**
- [ ] Select primary and backup policies
- [ ] Final 1000-episode evaluation
- [ ] Package and validate submission

## Highest-priority path

**Validate the current PPO configuration and safety failures → paired conflict-feature
ablation → matched-budget PPO/SAC → additive attention/Transformer → predictive
rewards → curriculum → shared MAPPO → centralized graph critic → constrained RL
→ hard-scenario mining.**

Next: validate the selected actions across seeds, compare goal-aligned observations
with and without conflict features, and review checkpoint safety failures. Then
revisit PPO/SAC under matched budgets and paired evaluation scenarios.

Match simulator transitions rather than command-line step counts: one
multi-agent step contributes one transition per aircraft, and PPO may finish
its current rollout beyond the requested count.
