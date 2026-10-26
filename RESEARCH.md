# Research status

Last reviewed: 2026-10-26. Run evidence: 2026-10-17–26 (Europe/Madrid).

## Current findings

- **Single-agent:** PPO has the best observed return; SAC is second.
- **Multi-agent:** shared PPO leads MAPPO/IPPO in the matched 100k-joint-step
  comparisons with flat MLP and additive attention; SAC led PPO in the earlier,
  smaller-budget baseline.
- **Rewards:** retain the base reward; none of six added terms improved the
  common base return in the seed-42 ablation.
- **Confidence:** preliminary. The initial baselines use training seed 42 and evaluation
  seed 2026; later comparisons use paired scenarios 2026–2035. More training
  seeds and fresh evaluation scenarios are needed.
- **Actions:** relative heading + absolute speed leads the initial paired comparison.
- **Observations:** conflict narrowly leads return; goal-aligned leads completion.
  No robust winner yet; comparisons use training seed 0 only.
- **Decision interval:** retain the default 10 seconds; best mean return in the
  initial 5/10/20-second comparison, with mixed safety results and one training seed.
- **Architectures:** entity MLP with all-context attention and Transformer with
  mean/max pooling nearly tie for best return; both complete 10/10 scenarios.
  Preliminary: one training seed, with safety violations remaining.
- **Temporal models:** retain no-history policies at 100k steps; all eight
  temporal variants reduced return in the seed-42 comparison.
- **Million-step single-agent configuration:** PPO with relative
  heading, absolute speed, and goal-aligned control observations; 10/10 completions,
  but safety violations remain. Budget and configuration effects are not isolated.

## Baseline results

MLP policies, approximately 100k SB3 timesteps. Higher return is better.
Compare algorithms within each setup; multi-agent timesteps count agent
transitions and do not represent the same simulator budget as single-agent runs.
± denotes evaluation standard deviation, not a confidence interval.

| Setup | Algorithm | Actual steps | Mean return ± std | Run ID |
|---|---|---:|---:|---|
| Single | **PPO** | 100,352 | **−432 ± 428** | `ppo_single_mlp_20261017T191726_2961b567` |
| Single | SAC | 100,000 | −610 ± 368 | `sac_single_mlp_20261017T201008_b0de515d` |
| Single | TD3 | 100,000 | −1,069 ± 410 | `td3_single_mlp_20261017T201107_7180d895` |
| Single | DDPG | 100,000 | −1,211 ± 781 | `ddpg_single_mlp_20261017T201058_122280ae` |
| Multi | **SAC** | 100,000 | **−422 ± 451** | `sac_marl_competition_20261017T215211_69bcc645` |
| Multi | PPO | 102,400 | −1,575 ± 1,333 | `ppo_marl_competition_20261017T215155_d345bd62` |

Single-agent evaluations contain 5 episodes; multi-agent evaluations contain
12 agent returns, which can share scenarios and are not independent trials.
Raw evidence lives in `experiments/results/<run ID>/run.json`; these files are
local and ignored by Git, so key results are preserved here.

## Action and observation comparisons

PPO/MLP, training seed 0, approximately 100k steps, speed bounds 100–500 kt.
Actions use upstream/control observations and paired scenarios 2026–2028;
observations use relative/relative actions, the legacy profile, and scenarios
2026–2035. Higher return is better.

| Comparison | Variant | Mean return | Completed |
|---|---|---:|---:|
| Actions (heading/speed) | **Relative/absolute** | **−553** | **3/3** |
| Actions | Absolute/absolute | −1,242 | 0/3 |
| Actions | Absolute/relative | −1,457 | 0/3 |
| Actions | Relative/relative | −1,813 | 0/3 |
| Observations | Conflict | **−508** | 0/10 |
| Observations | Goal-aligned | −518 | **2/10** |
| Observations | Upstream | −591 | 0/10 |
| Observations | Ego Cartesian | −1,133 | 1/10 |

Evidence: `action_modes_ppo_single`, `observation_modes_ppo_single`, and
`default_vs_conflict_now` under `experiments/results/`. Relative/relative action
metrics above use only the three shared scenarios from its ten-scenario evaluation.

## Million-step run

Fresh PPO, seed 0, relative heading/absolute speed (100–500 kt), goal-aligned
control observations: **1,001,472 steps**, **−306.54 ± 263.28 return**, **10/10
completions** on scenarios 2026–2035. Mean events per episode: **0.9 intrusions,
1.1 restricted-area entries, 0.1 sector exits**. Training took 4 h 55 min.

Run: `ppo_best_1m_20261018T040641/reports/policy_20261018T020645_170ce915`.
This combines choices from separate comparisons at a larger budget; it is not
an ablation or evidence of generalization across training seeds.

## Decision interval comparison

PPO/MLP, training seed 0, scenarios 2026–2035, relative heading/absolute speed
(100–500 kt), `goal_aligned_conflict` control observations. Matched nominal budget:
one million simulated seconds per variant (200k/100k/50k decisions), with PPO
rollout rounding and time-scaled gamma/drift penalties.

| Interval | Mean return | Completed |
|---|---:|---:|
| 5 s | −360.02 | 10/10 |
| **10 s (default)** | **−260.55** | 10/10 |
| 20 s | −266.60 | 10/10 |

**Retain 10 seconds** for its best mean return. Safety is mixed: 20 seconds had
less intrusion/restricted-area exposure but more time outside the sector
(66.6 vs 9.7 s/episode). Ranking remains preliminary with one training seed.
Evidence: `experiments/results/decision_interval_comparison/` (`comparison.json`,
`summary.csv`, `paired.csv`, `episodes.csv`).

## Architecture comparison

All 24 spatial variants completed: single-agent PPO, training seed 42,
100,352 steps each, paired scenarios 2026–2035. Relative heading/absolute speed
(100–500 kt), `goal_aligned_conflict` control observations, 10-second decisions;
no temporal processing. Transformers/GNNs use one layer; GNNs are fully connected.

| Variant | Mean return | Completed |
|---|---:|---:|
| Entity MLP, additive attention (`all`) | **−148.27** | **10/10** |
| Transformer, mean/max pooling | −148.44 | **10/10** |
| GCN, mean pooling | −161.50 | 8/10 |
| GAT, PMA pooling | −182.29 | **10/10** |
| Flat MLP baseline | −645.28 | 6/10 |

**Shortlist the top two for repeated seeds**; their 0.18 return gap does not
establish a winner. Entity MLP trained in 47.3 min versus 53.5 min for Transformer.
Safety remains unresolved: respectively 1.2/0.8 intrusions and 0.8/0.7
restricted-area entries per episode; neither had sector exits. Entity structure
helped most variants, but attention was not consistently beneficial across models.
Evidence: `experiments/results/architecture_comparison/` (`comparison.json`,
`summary.csv`, `runs.json` with per-run IDs); variant paths above map to seed 42.

## Attention comparison

Single-agent PPO, training seed 42, 100,352 steps per variant, paired scenarios
2026–2035; same actions, observations and 10-second interval as the architecture
comparison, with no temporal processing. Additive attention, ego cross-attention
and the context Transformer use all context. The context comparison reuses the
three completed reference runs and trains only the new context Transformer.

| Variant | Mean return | Completed |
|---|---:|---:|
| Entity MLP, additive attention | **−148.27** | **10/10** |
| Transformer, mean/max pooling | −148.44 | **10/10** |
| Entity MLP, ego cross-attention | −283.80 | 7/10 |
| Context Transformer, context-token pooling | −520.95 | 8/10 |

Additive attention independently encodes each aircraft, then uses context to
weight and sum the aircraft embeddings. The mean/max Transformer first applies
self-attention between aircraft, then pools their updated embeddings. The context
Transformer instead encodes the full non-aircraft context into a 64-feature token
alongside the 64-feature aircraft tokens. One four-head, pre-normalized Transformer
layer attends jointly over these tokens, masking absent aircraft. Its updated
context token is concatenated with the raw context for the policy and value
networks. This summarizes the current observation; it adds no temporal memory.

**Keep additive attention and Transformer shortlisted.** Ego cross-attention
reduced restricted-area entries (0.4 vs 0.8/0.7 per episode), but had more
intrusions (1.5 vs 1.2/0.8), 0.8 sector exits per episode and three timeouts.
The context Transformer ranked fourth: mean events per episode were 0.9 intrusions,
0.9 restricted-area entries and 0.1 sector exits; 2/10 episodes timed out.
Mean restricted-area exposure was 186.5 s versus 73.0/60.2 s for additive attention
and mean/max Transformer; mean time outside the sector was 271.2 s versus zero
for both. Context Transformer training took 69.6 min. Its lower return does not
identify the cause or establish whether a larger training budget would help.
This single-seed comparison does not establish a general attention winner.
Evidence: `experiments/results/attention_comparison/` (`summary.csv`, `paired.csv`,
`runs.json` with per-run IDs), extended by
`experiments/results/context_attention_comparison/` (`summary.csv`, `paired.csv`,
`runs.json`, `report.md`). Context Transformer run ID:
`policy_20261025T211839_db413510`.

## Temporal comparison

All ten trials completed: single-agent PPO, seed 42, **100,352 actual steps** each,
paired scenarios 2026–2035. Same spatial winners, actions, observations and
10-second interval as the architecture comparison; temporal histories contain
eight observations. Higher return is better; completions are out of ten.

| Temporal model | Entity MLP return | Completed | Transformer return | Completed |
|---|---:|---:|---:|---:|
| **None** | **−148.27** | 10/10 | **−148.44** | 10/10 |
| Frame stack | −266.88 | 10/10 | −531.31 | 8/10 |
| GRU | −788.89 | 2/10 | −241.13 | 10/10 |
| LSTM | −428.06 | 10/10 | −201.70 | 10/10 |
| TCN | −322.04 | 10/10 | −300.62 | 10/10 |

**Retain no-history policies.** Transformer LSTM is the closest temporal challenger
(−53.26 versus its own baseline), but increases intrusion/restricted-area events
from 0.8/0.7 to 1.1/1.0 per episode. Lower intrusion counts alone can mislead:
entity MLP GRU has 0.5 per episode but only 2/10 completions and 8/10 timeouts.
No overall temporal safety win is established; results use one training seed.
Training took 50–56 min per baseline and 58–95 min per temporal trial; the full
comparison took about 3 h 20 min with four CPU workers.
Evidence: `experiments/results/temporal_comparison/` (`summary.csv`, `paired.csv`,
`runs.json` with per-run IDs). Paired differences use each spatial model's own
no-history baseline. This tests eight-frame windows at 100k steps, not all temporal
settings or training budgets.

## Reward ablation

Single-agent PPO, entity MLP with all-context additive attention, seed 42,
100,352 steps each, paired scenarios 2026–2035; same control setup as the
architecture comparison. Each variant adds only one term to the base reward.

| Added term | Weight | Base return | Completed |
|---|---:|---:|---:|
| **None** | 0 | **−148.27** | **10/10** |
| Progress | 0.125 | −203.75 | 9/10 |
| Predictive separation | 0.5 | −285.72 | 10/10 |
| Obstacle clearance | 0.1 | −1,181.45 | 0/10 |
| Sector clearance | 0.1 | −210.96 | 8/10 |
| Time | 0.001 | −210.67 | 8/10 |
| Smoothness | 0.001 | −228.05 | 10/10 |

**Retain the base reward at these weights.** Base return subtracts added terms
so objectives remain comparable. Predictive shaping reduced intrusion exposure
from 75.7 to 68.5 s/episode but increased restricted-area exposure from 73.0 to
134.8 s and outside-sector time from zero to 82.6 s. Obstacle shaping reduced
restricted-area exposure to 21.0 s but completed no flights. Weight sensitivity
and repeated training seeds remain untested.
Evidence: `experiments/results/reward_ablation/` (`comparison.json`, `summary.csv`,
`paired.csv`, `runs.json` with per-run IDs, `report.md`).

## Multi-agent PPO comparison

MLP actors/critics, training seed 42, paired scenarios 2026–2035 with ten aircraft:
100,352 joint steps each (1,003,520 agent slots; valid transitions differ as
flights finish). Shared PPO uses a shared actor/local critic, IPPO separate
actors/local critics, and MAPPO a shared actor/centralized critic. All use the
same masked collector and frozen evaluation normalization.

| Method | Mean return | Completion | Intrusion s | Restricted s | Outside s | Training min |
|---|---:|---:|---:|---:|---:|---:|
| **Shared PPO** | **−217.21** | **100%** | **58.20** | **145.27** | **14.15** | 114.1 |
| MAPPO | −331.59 | 92% | 124.04 | 189.21 | 17.89 | 114.4 |
| IPPO | −407.01 | 77% | 101.84 | 244.15 | 59.63 | 130.2 |

**Retain shared PPO as the comparison baseline.** It leads return, completion
and all three exposure measures (seconds per aircraft episode). Safety violations
remain. These single-seed results do not establish a general algorithm winner;
parameter counts differ, and this collector/budget differs from the earlier
SB3 PPO/SAC baseline. Evidence:
`experiments/results/multi_agent_mlp_seed42_100k/` (`comparison.json`, `summary.csv`,
`paired.csv`, `runs.json` with per-run IDs, `report.md`).

### Additive-attention follow-up

All three methods completed with training seed 42, paired scenarios 2026–2035
and ten aircraft (100 aircraft episodes per method). Each trained for 100,352
joint steps (1,003,520 agent slots), matching the MLP comparison. Actions,
observations, base reward and no temporal history are unchanged. All methods use
separate local actor/critic entity MLP encoders with all-context additive aircraft
attention; MAPPO also uses focal-agent-conditioned additive attention over active
agents in its centralized critic. Normalization is frozen during evaluation.

| Method | Mean return | Completion | Intrusion s | Restricted s | Outside s | Training min |
|---|---:|---:|---:|---:|---:|---:|
| **Shared PPO** | **−162.28** | 99% | 58.04 | **94.02** | **10.71** | 96.6 |
| MAPPO | −179.29 | **100%** | **33.48** | 126.16 | 20.19 | 101.2 |
| IPPO | −361.97 | 65% | 72.12 | 239.87 | 48.73 | 121.1 |

**Retain shared PPO as the return baseline; keep MAPPO as a safety/completion
challenger.** MAPPO trails shared PPO by 17.01 return but completes every flight
and has less intrusion exposure. Shared PPO has less restricted-area and
outside-sector exposure. Safety violations remain for all methods.

Attention improved observed return over each method's flat MLP result, with the
same ranking. Completion changed from 100%/92%/77% to 99%/100%/65% for shared
PPO/MAPPO/IPPO. This tests attention in both actors and critics, so it does not
isolate their contributions. One training seed and shared scenarios do not
establish a general attention or algorithm winner; aircraft episodes within a
scenario are not independent trials.

Evidence: `experiments/results/multi_agent_additive_seed42_100k/`
(`comparison.json`, `summary.csv`, `paired.csv`, `runs.csv`, `report.md`). Run IDs:

- Shared PPO: `policy_20261026T072127_99b81512`.
- MAPPO: `policy_20261026T072127_ef154973`.
- IPPO: `policy_20261026T072127_998fbc17`.

## Next experiments

1. Repeat the top two architectures and flat MLP across training seeds.
2. Validate action selection across training seeds; compare `goal_aligned` with
   `goal_aligned_conflict` using the selected actions and control profile.
3. Repeat PPO/SAC with matched budgets and paired scenarios across training seeds.
4. Review safety failures and intermediate checkpoints; reserve fresh scenarios
   for final evaluation.

## Findings log

| Reviewed | Finding | Implication |
|---|---|---|
| 2026-10-18 | Resuming single-agent PPO to 200,704 steps changed return from −432 to −661 ± 427 (`ppo_single_mlp_20261017T200828_e79e5ce6`). | Preserve the earlier policy; longer training needs validation. |
| 2026-10-18 | PPO leads single-agent; SAC leads multi-agent in the baseline runs above. | Prioritize these candidates, pending repeated trials. |

Update this file after each completed comparison: record the configuration,
seeds, budget, metrics, run IDs, and the resulting decision. Keep commands in
[experiments/README.md](experiments/README.md) and implementation plans in
[ROADMAP.md](ROADMAP.md).
