# Novelty assessment and proposed experiments

Research date: 2026-11-08. This is a research proposal, not a record of new
experimental results. Existing results are taken from `RESEARCH.md`; their raw
run artifacts were not independently re-audited for this review.

## Assessment

The strongest opportunity is to learn cooperative conflict resolution conditioned
on the space available for each aircraft to maneuver, while treating separation,
restricted areas, and sector boundaries as separate safety objectives.

Most existing experiments are useful baselines and engineering contributions.
PPO/SAC/MAPPO, attention, Transformers, GNNs, temporal encoders, dual encoders,
TCPA/DCPA features, and reward shaping do not individually establish a new RL
method. A focused, reproducible comparison on this competition's particular
combination of procedural geometry, heading/speed control, and cooperative
traffic can still be an original empirical contribution.

The search covered primary papers and proceedings, emphasizing 2025–2026 work
and older foundational work where necessary. Some papers were accessible only
through their abstracts or indexed manuscript excerpts. This is a targeted
literature review, not an exhaustive systematic review. Failure to find an exact
match cannot establish that an idea has never been tried. The proposals below
are hypotheses; no first-ever claim is justified yet.

## Closest prior work

| Work | Relevance and implication |
| --- | --- |
| [Groot, Ellerbroek, Hoekstra, 2025: attention versus LSTM](https://doi.org/10.1016/j.engappai.2025.111592) | Compares additive, scaled dot-product, and context-aware attention with LSTM using SAC for separation management. Additive attention performs best. This is particularly close to the existing architecture comparison. |
| [Brittain, Yang, Wei, 2020: autonomous separation assurance](https://arxiv.org/abs/2003.08353) | PPO, attention, parameter sharing, and decentralized execution in BlueSky already have direct precedent. |
| [Aziz and Wei, SESAR Innovation Days 2025](https://www.sesarju.eu/sites/default/files/documents/sid/2025/papers/SIDs_2025_paper_54-final.pdf) | Relative polar observations, an ownship-conditioned Transformer token, procedural traffic variation, and encoder-depth comparisons have been studied together. |
| [Li et al., 2025: multi-scale graph RL](https://doi.org/10.1109/JIOT.2025.3608865) | Models aircraft and obstacle interactions and constrains maneuver actions. A generic aircraft/obstacle graph or geometry-aware action mechanism is insufficient as a novelty claim. |
| [HYGMA, ICML 2025](https://proceedings.mlr.press/v267/liu25af.html) | Dynamic hypergraphs and higher-order MARL coordination already exist. |
| [Scalable constrained policy optimization, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa76985f05e0a25c66528308dda33de0-Paper-Conference.pdf) | Local interactions and constrained MAPPO-Lagrangian are established safe-MARL directions. |
| [Distributional safety critics and maximum entropy, AAAI 2026](https://doi.org/10.1609/aaai.v40i35.40198) | Distributional cost critics, CVaR, and multi-agent SAC have already been combined. |
| [Co2PO, February 2026 preprint](https://arxiv.org/abs/2602.02970) | Hazard prediction, intent/yield signals, and selective risk-aware communication have direct precedent. |
| [Safety-prioritizing curricula, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html) | Safety-aware curriculum generation is established; simple difficulty scheduling is not a new method. |
| [Diffusion-AC, September 2025 preprint](https://arxiv.org/abs/2509.03550) | Diffusion actors and a traffic-density curriculum have already been proposed specifically for aircraft conflict resolution. |
| [Decentralized diffusion policy learning, May 2026 preprint](https://arxiv.org/abs/2605.07101) | Expressive multimodal policies are an active continuous-action MARL direction. |
| [M3W, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/3a2d96d2eb2902043c2db705ca03e9a2-Abstract-Conference.html) | Modular multi-agent world models combine learning and planning. Building a world model is not sufficient novelty by itself. |

Important distinction: Groot et al.'s LSTM processes a sequence of aircraft to
encode a variable-size entity set. This project's temporal LSTM processes a
history of observations. These test different questions; the published result
does not duplicate or explain the project's temporal result.

## What the existing evidence can support

1. **Entity structure helps in this preliminary protocol.** The additive entity
   model and pooled Transformer outperform the flat MLP at the tested budget.
   Their roughly 0.18 return difference is not evidence of a winner.
2. **More complexity has not consistently helped.** The context-token and dual
   models underperform the leading simpler encoders; all tested eight-frame
   temporal variants reduce return at 100k steps. These are budget-specific
   observations, not general conclusions about memory or Transformers.
3. **Safety objectives trade off.** In the additive-attention MARL comparison,
   MAPPO reduces intrusion exposure from 58.04 to 33.48 seconds per aircraft
   episode relative to shared PPO, but increases restricted-area exposure from
   94.02 to 126.16 seconds and outside-sector exposure from 10.71 to 20.19 seconds.
   Completion is 100% versus 99%. This is the most useful research question.
4. **Adding one penalty did not solve safety.** The six reward additions failed
   to improve common base return at their tested weights. This does not show
   that reward shaping cannot work, or that constrained learning will work.

Repeated training seeds, fresh validation scenarios, and component ablations
are needed before these become strong empirical contributions. In particular,
attention was changed in both actor and critic, so its benefit is not isolated.

## Proposal 1: Geometry-conditioned conflict cost critics

**Priority: first. Implementation effort: medium. Novelty potential: plausible
method adaptation, requiring comparison with graph/safe-navigation literature.**

Question: does an aircraft need to know not only which peer threatens separation,
but also which peer has room to yield safely?

A head-on encounter in open airspace and an otherwise similar encounter beside a
restricted polygon offer different escape maneuvers. Current separate traffic
and obstacle summaries may make learning this dependency difficult. This is an
inductive-bias hypothesis, not a demonstrated limitation of the current networks.

Start with the existing additive actor. Add three separately supervised cost
critics for separation, restricted-area exposure, and sector exposure. Condition
the separation critic's pairwise interactions on geometry around both aircraft:
signed clearances, directional free-space rays, and current motion. A compact
pair interaction layer is a better first test than another large Transformer.

For MAPPO, estimate separate cost values/advantages. For MASAC, estimate separate
action-conditioned cost Q functions. Compare an unstructured critic with one
that explicitly builds pairwise conflict terms from both aircraft and their
available maneuvering space. Keep padding masks and entity permutation symmetry.
Use feasible short-horizon motion features rather than instantaneous heading
jumps if dynamics-based features are introduced.

Train against separate cost budgets with adaptive multipliers. Measure costs
from the unchanged one-second metric stream; use consistent episode units and
keep event counts distinct from exposure seconds. Define the task reward and
existing safety penalties explicitly so the experiment does not silently change
several objectives at once. Expected-cost constraints do not guarantee zero
violations, and zero budgets may be infeasible for some generated encounters.

The possible contribution is a particular geometry-conditioned safety-learning
mechanism and its measured effect on safety trade-offs. It is not the invention
of constrained MARL, graphs, or obstacle-aware navigation.

Minimum factorial comparison:

| Variant | Separate safety costs/constraints | Geometry-conditioned pair interactions |
| --- | --- | --- |
| A | No | No |
| B | Yes | No |
| C | No | Yes |
| D | Yes | Yes |

Give all variants the same raw geometric information and comparable model size.
For C, use the structured interaction encoder for ordinary value prediction.
First hold the actor fixed to isolate critic effects. If actor geometry is then
added, give the same actor features to every corresponding control. A simple
concatenation control tests whether structure adds value beyond extra inputs.

Success means lower separation exposure without transferring violations to
obstacles/boundaries or reducing completion. Diagnose results by open versus
constrained encounter geometry. This would test a mechanism beyond aggregate
benchmark performance.

## Proposal 2: Counterfactual safety responsibility

**Priority: second, after reliable cost critics. Effort: medium to high. Novelty
potential: plausible in the precise formulation, with substantial prior art.**

Question: which aircraft's maneuver actually reduces joint risk, and does it
create a different hazard?

Use an action-conditioned cost critic to compare the chosen joint action with
actions in which one aircraft's command is replaced by a defined alternative,
holding peer commands fixed. Estimate separate changes in predicted separation,
obstacle, and sector cost. Use these estimates to investigate training credit
assignment or to supervise a learned yield/responsibility head.

Two distinct counterfactuals should not be confused: sampling alternative actions
from the policy gives a COMA-style baseline; substituting a maintain-course/speed
command is an intervention diagnostic with a different interpretation. The
relative-heading/absolute-speed control needs an explicit physical no-change
command, not an all-zero policy vector.

[COMA](https://arxiv.org/abs/1705.08926) already introduced counterfactual credit
assignment. [Responsibility-associated collision avoidance](https://arxiv.org/abs/2206.09030)
already allocates pairwise avoidance responsibility. Co2PO also studies yielding
and hazard forecasts. The candidate distinction is learning responsibility from
changes in several safety costs while accounting for each aircraft's geometry.

Validate sampled counterfactual predictions with real simulator rollouts in
separate processes. BlueSky's global state prevents assuming that an independent
second simulator can safely live in the same process. Check critic calibration
on alternative actions; changing an action can put it outside replay coverage.

Compare local/global rewards, a standard counterfactual baseline, and the
geometry-conditioned version. Measure secondary conflicts, new obstacle
violations following avoidance, maneuver burden, and official metrics. Do not
claim causal identification from a learned Q function alone. Keep actors local
unless a communication channel is deliberately introduced and reported.

## Proposal 3: Curriculum over conflict structure and escape space

**Priority: practical parallel research direction, not an instruction to launch
jobs. Effort: low to medium. Novelty potential: mainly empirical unless a new
sampling objective is introduced.**

Traffic count is a weak proxy for difficulty: the same ten aircraft can create
independent pairs, a connected conflict chain, or a bottleneck beside obstacles.

Stratify training scenarios by predicted conflict-graph size/degree, simultaneous
encounters, encounter angles, and directional free space near likely conflicts.
Compare uniform sampling, density scheduling, loss-based prioritized replay, and
learning-progress sampling across these strata. Preserve ordinary random training
coverage so concentrating on failures does not erase routine competence.

Use labels derived from initial geometry or training-only rollouts. Count scenario
screening interactions and replay/evaluation cost in the compute report. Do not
mine validation or official test episodes. Original scenario distribution and
configuration remain unchanged for official evaluation.

The hypothesized contribution is showing which combinations of conflict
structure and constrained escape space predict failure, and improving transfer
to fresh official-distribution scenarios through that curriculum. Safety-aware
curricula and density curricula already have precedent in the sources above.

## Proposal 4: Coordinated multimodal maneuver policies

**Priority: exploratory. Effort: medium for a small mixture, high for diffusion.
Novelty potential: low for merely changing the distribution, higher for a new
coordination mechanism.**

Test whether Gaussian policy expressiveness is actually a failure source. A
small categorical maneuver mode plus a continuous heading/speed policy can
represent distinct alternatives. Compare it with the Gaussian baseline before
investing in diffusion. For symmetric choices, independently sampling multimodal
actions can still produce incompatible joint maneuvers; learning consistent
selection and persistence is the interesting MARL question.

Possible modes include left/right deviation and speed-led yielding, implemented
as learned policy components. Measure oscillations, mode switching, pairwise
compatibility, and completion. Temporal persistence could help, but needs its own
ablation. The existing negative history experiments do not rule out persistent
decision modes because those are different mechanisms.

Diffusion-AC already applies diffusion to aircraft conflict resolution, and DDPL
studies decentralized diffusion policies. Avoid claiming a new application of
diffusion to ATC. A full diffusion implementation is a weak deadline bet unless
simple mixture experiments establish a specific representational bottleneck.

## Proposal 5: Predictive auxiliary learning before full world models

**Priority: supporting ablation. Effort: medium. Novelty potential: modest alone.**

Train shared encoders to predict future minimum separation, obstacle/boundary
entry, and exposure at several horizons in addition to RL losses. Compare
against analytic constant-velocity TCPA/DCPA features and identical encoders
without the prediction losses. These targets add temporal supervision without
necessarily adding recurrent inference.

Targets must specify whether predictions assume current commands persist or
future actions follow the current policy; these are different learning tasks.
Recalibrate as policies change. Prediction accuracy alone does not establish
better control. M3W and other model-based MARL already study predictive models;
the modest opportunity here is score-aligned forecasting of interacting hazards
at a small compute budget, with an isolated policy benefit.

## Recommended execution and evidence standard

With the November 30 deadline, choose one core hypothesis. The recommended core
is Proposal 1, with Proposal 3 only if implementation and validation leave time.
Proposals 2 and 4 are follow-ups, not necessary parts of the first submission.

1. Reproduce additive shared PPO and MAPPO on at least three training seeds.
   Separate actor and critic attention effects if making representation claims.
   Finish the implemented SAC-family comparison only if it helps choose the
   baseline or validates the cost-critic approach; it has no recorded results yet.
2. Inspect trajectories to establish whether obstacle-constrained encounters
   actually explain the observed safety trade-off. Abandon or revise the proposed
   mechanism if they do not.
3. Run the four-cell constrained/geometry experiment at 100,352 joint decisions
   per run, initially three training seeds: 12 screening runs. Treat this as
   screening, not sufficient evidence that harder methods cannot work.
4. Allocate larger matched simulator budgets to the leading method and relevant
   controls. Report environment interactions, valid agent transitions, gradient
   updates, parameter counts, and wall time; equal joint decisions alone do not
   equalize PPO/SAC compute or update effort.
5. Use a predeclared fresh validation suite, suggested size 100 scenarios per
   policy for screening, expanding when variability requires it. Preserve paired
   scenarios between variants, freeze normalization, and separate tuning from
   final evaluation.
6. Report completion, every official event/exposure metric, and flight time.
   Add all-aircraft-safe-completion rate as a supplementary diagnostic, never a
   replacement score. Report flight-time comparisons alongside completion to
   avoid rewarding short incomplete flights. Show Pareto trade-offs rather than
   using return to choose a safety winner.
7. Treat scenario as the cluster: aircraft in one scenario are dependent. Show
   variation across training seeds and scenario-cluster paired uncertainty;
   three training seeds still provide limited precision. Tail metrics need more
   than ten scenarios; CVaR estimates from ten episodes are especially unstable.
8. Lock the final configuration before the official 1000-episode seed-42 stream.
   Independent resets at seeds 42…1041 are not the prescribed stream. Keep
   supplementary stress tests separate from official-distribution results.

The [competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
permit observation/reward/action design and RL training changes while fixing
world dynamics, scoring, and evaluation. These training and learned-policy
proposals fit that design surface when implemented accordingly. Runtime classical
controllers or shields introduce an additional interpretation question under the
RL-based-solution requirement; they are not needed for the recommended study.

An honest present contribution statement is:

> We investigate how entity representation and centralized training affect the
> safety–completion trade-off in procedural multi-aircraft control.

If Proposal 1 succeeds with the stated controls, a more specific statement is:

> We introduce a geometry-conditioned conflict-cost critic and evaluate whether
> it reduces separation violations without shifting risk to restricted areas or
> sector boundaries in decentralized multi-aircraft control.

Neither statement asserts a global first. Establishing a stronger priority claim
requires a precise algorithm specification followed by full-text comparison and
citation tracing of the closest graph, constrained-MARL, responsibility, and
navigation papers.
