# bluesky-pabloramesc — Roadmap

## 1. Baseline algorithms

- [x] Single-agent PPO + MLP
- [x] Single-agent SAC + MLP
- [x] Single-agent TD3 / DDPG + MLP
- [ ] Hyperparameter optimization (Optuna)

## 2. Actions and observations

- [x] Implement and compare relative/absolute heading/speed actions
- [x] Compare 5/10/20-second decision intervals
- [x] Compare custom ego-centric/goal-relative features against the upstream baseline
- [x] Add TCPA, DCPA, closing rate, and presence mask (`conflict` observation encoding)
- [x] Implement and evaluate obstacle/sector geometry alternatives (nearest edge, ray casting)
- [ ] Compare raw vs engineered features

## 3. Neural architectures

- [x] Shared entity MLP + mean/max pooling
- [x] Ego multihead cross-attention
- [x] Additive attention
- [x] Entity Transformer (Set Transformer)
- [x] Context-token Transformer
- [x] Dual Transformer (traffic and obstacles)
  - [ ] Try different depths
  - [ ] Investigate PPO update stability
- [x] Graph Neural Network (GCN and GAT)
- [x] Fixed-window temporal modeling (LSTM, GRU, and TCN)

## 4. Reward engineering

- [x] Goal-progress reward
- [x] Predictive separation penalties (TCPA/DCPA)
- [x] Obstacle and sector-boundary clearance penalties
- [x] Time and action-smoothness penalties
- [x] One-term reward ablation
- [ ] Reward-weight sensitivity analysis and multi-seed validation

## 5. Training strategies

- [x] Complete a million-step PPO run
- [ ] Select training duration using intermediate checkpoints
- [ ] Traffic-density curriculum
- [ ] Conflict-complexity and obstacle curricula
- [ ] Compare procedural randomization ranges
- [ ] Training above nominal traffic density
- [ ] Hard-scenario mining and replay
- [ ] Prioritized experience replay (PER) for SAC

## 6. Multi-agent RL

- [x] Independent PPO (IPPO)
- [x] Parameter-shared PPO
- [x] MAPPO with a centralized critic
- [x] Additive-attention actors and critics for shared PPO, IPPO, and MAPPO
- [ ] Add entity pooling, Transformer, and GCN/GAT centralized critics
- [ ] Configure MAPPO actor and critic architectures independently
- [x] Compare MLP shared PPO, IPPO, and MAPPO
- [x] Compare additive-attention shared PPO, IPPO, and MAPPO
- [ ] Repeat multi-agent PPO comparisons across training seeds
- [ ] Compare local, global, and pairwise safety rewards
- [x] Parameter-shared SAC with local critics
- [x] Additive-attention ISAC and masked parameter-shared local-critic SAC
- [x] Centralized-critic multi-agent SAC (MASAC with additive attention)
- [ ] Compare additive-attention shared SAC, ISAC, and MASAC against PPO-family results

## 7. Safe and robust RL

- [ ] Separate reward and safety-cost objectives
- [ ] Lagrangian constrained SAC / MAPPO
- [ ] Multiple safety critics (separation, obstacles, boundaries)
- [ ] Distributional / quantile critics
- [ ] CVaR risk-sensitive optimization
- [ ] Domain randomization (noise, delays, dynamics)
- [ ] Runtime safety shield

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

- [x] Separate training and evaluation seeds
- [x] Headless checkpoint evaluation with frozen normalization
- [ ] Persist run configuration, package/upstream revisions, and evaluation results
- [ ] Explicit fixed evaluation scenario suite across policies and training seeds
- [ ] Automated trajectory visualization and failure analysis
- [ ] Matched-budget algorithm and architecture ablations
- [ ] Multi-seed evaluation and confidence intervals
- [ ] Pareto ranking by safety, completion, and efficiency
- [ ] Select primary and backup policies
- [ ] Final 1000-episode evaluation
- [ ] Package and validate submission
