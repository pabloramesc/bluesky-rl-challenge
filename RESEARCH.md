# Research status

Last reviewed: 2026-10-20. Run evidence: 2026-10-17–20.

## Current findings

- **Single-agent:** PPO has the best observed return; SAC is second.
- **Multi-agent:** shared-policy SAC leads shared-policy PPO.
- **Confidence:** preliminary. Each baseline uses training seed 42 and evaluation
  seed 2026. More training seeds and evaluation scenarios are needed.
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
