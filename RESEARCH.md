# Research status

Last reviewed: 2026-10-18. Run evidence: 2026-10-17.

## Current findings

- **Single-agent:** PPO has the best observed return; SAC is second.
- **Multi-agent:** shared-policy SAC leads shared-policy PPO.
- **Confidence:** preliminary. Each baseline uses training seed 42 and evaluation
  seed 2026. More training seeds and evaluation scenarios are needed.
- **Action modes:** comparison incomplete; no preferred configuration yet.
- **Observation encodings:** [three-variant comparison](experiments/README.md)
  implemented (upstream, ego Cartesian, goal-aligned); results pending. Predictive
  features such as TCPA/DCPA remain deferred.

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

## Next experiments

1. **Validate algorithm ranking:** repeat PPO and SAC with multiple training
   seeds, matched budgets, and the same held-out scenario suite. Track completion,
   intrusions, sector/restricted-area violations, and return.
2. **Finish action comparison:** compare all four heading/speed combinations
   using control observations and consistent speed bounds. Only relative/relative
   has a completed full run in `action_modes_ppo_single`; two-step SAC smoke tests
   do not count as research evidence.
3. **Check training duration:** evaluate intermediate PPO checkpoints on a
   validation suite before choosing a training budget.

## Findings log

| Reviewed | Finding | Implication |
|---|---|---|
| 2026-10-18 | Resuming single-agent PPO to 200,704 steps changed return from −432 to −661 ± 427 (`ppo_single_mlp_20261017T200828_e79e5ce6`). | Preserve the earlier policy; longer training needs validation. |
| 2026-10-18 | PPO leads single-agent; SAC leads multi-agent in the baseline runs above. | Prioritize these candidates, pending repeated trials. |

Update this file after each completed comparison: record the configuration,
seeds, budget, metrics, run IDs, and the resulting decision. Keep commands in
[experiments/README.md](experiments/README.md) and implementation plans in
[ROADMAP.md](ROADMAP.md).
