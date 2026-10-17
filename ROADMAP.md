# bluesky-pabloramesc — Roadmap

Progress from reproducible baselines to advanced, safety-aware multi-agent control.

## 1. Baseline algorithms

- [x] Single-agent PPO + MLP — [builder](src/bluesky_pabloramesc/models.py), [training/checkpoint tests](tests/test_training.py)
- [x] Single-agent SAC + MLP — same builder/test modules
- [ ] TD3 / DDPG (optional)
- [ ] Hyperparameter optimization (Optuna)

## 2. Actions and observations

- [ ] Compare relative vs absolute heading/speed actions
- [ ] Optimize decision frequency
- [ ] Compare custom ego-centric/goal-relative features against the upstream baseline
- [ ] Add TCPA, DCPA, and closing rate; compare with existing relative-bearing features
- [ ] Improve obstacle/sector geometry (nearest edge, ray casting)
- [ ] Compare raw vs engineered features

## 3. Neural architectures

- [ ] Shared entity MLP + mean/max pooling
- [ ] Additive attention
- [ ] Single-layer Transformer / Set Transformer
- [ ] Graph Neural Network (GCN / GAT)
- [ ] Temporal modeling (frame stacking vs GRU)
- [ ] Compare architectures using the best single-agent algorithm

## 4. Reward engineering

- [ ] Add goal-progress shaping and compare completion reward weights (completion already upstream)
- [ ] Predictive separation penalties (TCPA/DCPA)
- [ ] Obstacle and sector-boundary clearance penalties
- [ ] Time and action-smoothness penalties
- [ ] Reward ablation and sensitivity analysis

## 5. Training strategies

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

- [x] Separate training/evaluation seeds (`--seed`, `--eval-seed`; default evaluation seed: 10000)
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

**Validate PPO/MLP and SAC/MLP under matched evaluation → conflict features → additive attention/Transformer → predictive rewards → curriculum → shared MAPPO → centralized graph critic → constrained RL → hard-scenario mining.**

Next: train both single-agent baselines, evaluate on the same scenario suite and
simulation budget, and record safety/completion/efficiency metrics before adding
features. Match simulator transitions rather than command-line step counts:
one multi-agent step contributes one transition per aircraft, and PPO may finish
its current rollout beyond the requested timestep count.
