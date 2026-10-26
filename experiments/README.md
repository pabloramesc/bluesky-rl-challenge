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
- `train_zoo_mappo`: custom MAPPO with a shared local actor and centralized critic.
- `compare_actions`: compare four relative/absolute heading and speed combinations.
- `compare_observations`: compare upstream, ego Cartesian, goal-aligned, and conflict features.
- `compare_architectures`: compare conditional architecture trees in bounded parallel subprocesses.
- `compare_attention`: compare additive attention, ego cross-attention, and the entity Transformer.
- `compare_context_attention`: train only the context-token Transformer and import the three completed attention results.

Most comparisons default to PPO and one training seed. Multiple seeds require
`--seeds`; `--output-dir` overrides their default folder under `results/`.
Use `--resume` to reuse completed comparison runs.
`compare_context_attention` inherits the completed reference protocol instead.

See [RESEARCH.md](../RESEARCH.md) for findings and next experiments.

### Custom MAPPO

Train the MLP baseline, add training to its checkpoint, or evaluate it:

```bash
python -m experiments.train_zoo_mappo --timesteps 100000 --device cpu
python -m experiments.train_zoo_mappo resume --timesteps 100000 --device cpu
python -m experiments.train_zoo_mappo evaluate --episodes 12 --device cpu
python -m experiments.train_zoo_mappo replay --episodes 1
```

The default checkpoint is `experiments/models/mappo_marl_competition`. Matching
normalization statistics and control/architecture sidecars are saved alongside
it. Evaluation freezes normalization; resume restores both networks and Adam
state. Periodic checkpoints, run reports, and Ctrl+C recovery use the shared
training workflow.

MAPPO learns one shared Gaussian actor from each aircraft's local observation.
The centralized critic receives all fixed-slot observations plus an active-agent
mask and predicts a separate value for each agent's own reward stream. Completed
aircraft are excluded from advantages and losses. Goal completion stops
bootstrapping; time limits bootstrap from the final joint observation and stop
the GAE trace before the automatic reset. Executed Gaussian actions are clipped
to environment bounds; the stored samples retain their original log probabilities.

Defaults live in `models.py`: separate 256×256 tanh actor/critic MLPs, 1,024 joint
steps per rollout, minibatches of 64 joint steps, ten update epochs, gamma 0.995,
GAE lambda 0.95, policy/value clipping 0.2, and learning rate 3e-4. Both `n_steps`
and `batch_size` count **joint timesteps**; `--timesteps` and checkpoint intervals
count **agent slots** (ten per BlueSky decision, including inactive padding).
Training completes full rollouts, so 100,000 requested slots produces 102,400.
The completed matched MLP benchmark is recorded in [RESEARCH.md](../RESEARCH.md#multi-agent-ppo-comparison).

The Python builder accepts overrides and replaceable actor/critic classes:

```python
from bluesky_pabloramesc.models import build_mappo

model = build_mappo(
    env,
    n_steps=512,
    batch_size=32,
    policy_kwargs={
        "actor_class": MyLocalActor,
        "critic_class": MyJointCritic,
        "actor_kwargs": {"hidden_sizes": (128, 128)},
        "critic_kwargs": {},
    },
)
```

Use `make_mappo_env()` for the joint environment; independent vector environments
do not satisfy MAPPO's joint-state contract. See the [module guide](../src/README.md)
for the network tensor interfaces. MAPPO and IPPO also support
`--architecture entity_mlp --pooling additive_attention --attention-context all`.
This selects separate additive aircraft encoders for actor and critic; MAPPO's
critic additionally attends over active agents. Transformer, GNN, and temporal
encoders are not yet supported by these custom trainers.

### Predictive conflict features

Use `--observation-encoding conflict` with any training entry point to append
TCPA, DCPA, closing rate, and an intruder presence mask to the upstream features:

```bash
python -m experiments.train_single_ppo --timesteps 100000 --observation-encoding conflict
```

`compare_observations` includes this fourth variant, paired against `upstream`
with identical scenario seeds and action settings. Existing three-variant
comparison directories cannot be resumed with the expanded variant set; use a
new output directory.

Features use horizontal constant-velocity motion from upstream relative
positions and heading/TAS velocities, retaining distance-sorted aircraft slots.
TCPA is signed (negative means CPA is in the past), divided by 600 seconds and
clipped to [-1, 1]. DCPA is the minimum distance over future time, divided by
10,000 metres; receding or effectively parallel traffic uses current distance.
The DCPA calculation uses the uncapped future CPA time. Closing rate is positive
for approaching traffic and divided by 150 m/s. Relative speeds at or below
0.001 m/s use TCPA zero. Coincident positions use closing rate zero. The presence
mask distinguishes occupied slots from padding; padded predictive features are
zero. These are horizontal estimates using the upstream motion model, without
route-turn or wind prediction. Settings are saved with policies and restored on
resume/evaluation through the existing checkpoint configuration.

To compare only upstream observations against upstream plus TCPA, DCPA, closing
rate, and presence (keeping the original geometry in both):

```bash
python -m experiments.compare_observations --encodings upstream conflict --algorithm ppo --seeds 0 1 2 --timesteps 100000 --output-dir experiments/results/tcpa_dcpa_comparison
```

This runs two variants per training seed on the same evaluation scenarios.
`--encodings` defaults to the original four variants. The first selected encoding
is the paired baseline. Keep the same selection when using `--resume`.

After selecting a geometry encoding, compare it against the same encoding with
conflict features appended. For example, if `goal_aligned` wins:

```bash
python -m experiments.compare_observations --encodings goal_aligned goal_aligned_conflict --algorithm ppo --seeds 0 1 2 --timesteps 100000 --output-dir experiments/results/goal_conflict_comparison
```

For an ego Cartesian winner use `--encodings ego_cartesian ego_cartesian_conflict`;
for upstream use `--encodings upstream conflict`. Only the four conflict fields
are added; the base geometry is identical. Both variants train fresh policies.
Use a new output directory and retain the original action settings, observation
profile, speed bounds, budget, and seed lists for a controlled comparison.

### Decision interval comparison

Compare decisions every 5, 10, and 20 simulated seconds using fresh single-agent
PPO policies, relative heading, absolute speed (100–500 kt), and
`goal_aligned_conflict` control observations:

```bash
python -m experiments.compare_decision_intervals --device cpu
```

Defaults: training seed 0, evaluation scenarios 2026–2035, and a nominal
1-million-simulated-second training budget per variant. `--timesteps` is the
number of decisions at the 10-second baseline (default 100,000); the 5/10/20
variants request 200,000/100,000/50,000 decisions. PPO completes full
2,048-step rollouts, and early goal arrivals can shorten individual decisions,
so simulated budgets are approximate. Optimizer and rollout settings remain
fixed in decision units. For repeated trials add `--seeds 0 1 2`.

Gamma is `0.995 ** (interval / 10)` for both PPO and reward normalization.
The per-decision drift penalty is scaled by actual elapsed simulated time / 10;
terminal rewards and time-integrated safety penalties retain their weights.
Relative heading limits remain fixed per decision: changing the interval also
changes how often the policy can revise its target.

Outputs default to `experiments/results/decision_interval_comparison`: raw
evaluation episodes, summary metrics, seed-paired differences against 10 seconds,
a plot, and per-variant policies with matching normalization and configuration.
Use `--resume` to reuse completed variants; interrupted training must be resumed
separately using its saved checkpoint. These runs can take many hours.

The initial comparison is complete: retain the default **10 seconds**, which had
the best mean return; all intervals completed 10/10 evaluation scenarios. See
[research results](../RESEARCH.md#decision-interval-comparison) for metrics and
the single-seed limitation.

For standalone training use `--decision-interval 5` on `train_single_ppo`.
Evaluation and resume restore the interval from the saved policy configuration
and reject conflicting overrides. Older configurations default to 10 seconds.

### Policy architectures

The default `--architecture mlp` flattens observations into the existing MLP.
Select a shared aircraft encoder using `--architecture entity_mlp`, with
`--pooling mean`, `max`, `mean_max` (the default), `additive_attention`, or
`ego_attention`.
Use experiment labels `mlp`, `entity_mlp_mean`, `entity_mlp_max`, and
`entity_mlp_mean_max` when comparing variants. For additive attention, use
`entity_mlp_additive_attention_ownship_goal` or
`entity_mlp_additive_attention_all`.

```bash
python experiments/train_single_ppo.py --architecture entity_mlp --pooling mean_max --model-path experiments/models/ppo_entity_mlp_mean_max
```

These options work with all single-agent algorithms and parameter-shared PPO/SAC.
The aircraft encoder uses two 64-unit ReLU layers with weights shared across
nearby aircraft. Masked pooling excludes absent aircraft; an empty traffic set
produces a zero summary. Ownship, goal, obstacle, and sector features are
concatenated as context before the existing policy/value heads. Actor and critic
use separate encoders. Aircraft feature normalization shares statistics across
occupied slots, while presence masks remain binary. This preserves invariance
to aircraft slot order, including during evaluation with frozen statistics.

Architecture settings are saved in the policy and an `_architecture.json`
sidecar; normalization is saved in the usual `_vecnormalize.pkl` file. Resume,
evaluate, and replay restore the architecture automatically. Keep these files
together. Use distinct `--model-path` values for ablations to avoid overwriting
other variants. Spatial comparison results are recorded in [RESEARCH.md](../RESEARCH.md).

Additive attention scores encoded aircraft with a 64-unit layer:
`score_i = vᵀ tanh(W_h h_i + W_q q + b)`. A masked softmax turns scores
into weights, then their weighted average produces a 64-feature traffic summary.
Actor and critic learn separate attention parameters.

```bash
python experiments/train_single_ppo.py --architecture entity_mlp --pooling additive_attention --attention-context ownship_goal --model-path experiments/models/ppo_attention_ownship_goal
python experiments/train_single_ppo.py --architecture entity_mlp --pooling additive_attention --attention-context all --model-path experiments/models/ppo_attention_all
```

`--attention-context ownship_goal` is the default for attention. It uses available
ownship speed/heading and goal features (drift/distance or Cartesian goal
coordinates, depending on observation encoding). The `control` observation profile
adds explicit ownship heading and CAS; the legacy profile uses its existing
features. `--attention-context all` additionally uses obstacle and sector context
when scoring aircraft. Both variants still pass the full context to the policy
heads. The context option is valid with `--pooling additive_attention` or
`ego_attention`.
Attention context is persisted with architecture settings and restored on resume,
evaluation, and replay; it cannot be overridden for an existing checkpoint.

### Ego cross-attention

Use `--architecture entity_mlp --pooling ego_attention` for a controlled-agent
query attending to all present neighbours. A two-layer MLP maps the selected
context to one 64-dimensional query; the shared aircraft MLP provides keys and
values to four-head cross-attention. A residual connection, layer normalization,
and a 128-unit feed-forward block produce a 64-feature summary. Unlike the
entity Transformer, this variant does not run neighbour-to-neighbour attention.
Attention work scales linearly with the number of neighbours at fixed width.

```bash
python -m experiments.train_single_ppo --architecture entity_mlp --pooling ego_attention --attention-context ownship_goal --model-path experiments/models/ppo_ego_attention
```

The default query uses ownship/goal features. `--attention-context all` also
includes obstacle and sector context. Full context is concatenated with the
summary in both cases. Absent slots are masked, empty traffic returns a zero
summary, and aircraft slot order does not affect the output. Actor and critic
learn separate encoders. Checkpoints save and restore the pooling and context
settings alongside normalization statistics. The same readout also works after
`entity_transformer` or `entity_gnn` encoding.

### Context-token Entity Transformer

Use `--architecture context_entity_transformer` to let ownship, goal, obstacle and
sector context participate in attention alongside aircraft. A context MLP produces
one token; shared aircraft MLPs produce the remaining tokens. Pre-norm Transformer
blocks jointly update them, then a final LayerNorm and context-token readout
produce a 64-feature summary. Raw context is also concatenated into the policy
input. Tokens use 64 dimensions, four attention heads, 128-unit feed-forward
blocks, zero dropout and no slot positional embeddings.

```bash
python experiments/train_single_ppo.py --architecture context_entity_transformer --transformer-layers 1 --heading-action relative --speed-action absolute --observation-profile control --observation-encoding goal_aligned_conflict --model-path experiments/models/ppo_context_transformer
```

The readout defaults to `--pooling context_token`; other pooling methods and
`--attention-context` do not apply. All available non-aircraft context is always
used. Absent aircraft are masked while the context token remains visible, so
empty traffic still produces a context-dependent summary. Actor and critic have
separate encoders. The architecture works with SB3 single-agent and shared-policy
builders and existing temporal windows. Checkpoints restore architecture, readout,
depth and normalization statistics for evaluation/resume. Custom comparison trees
can select `context_entity_transformer` and sweep `transformer_layers`; existing
default comparison trees are unchanged. No performance advantage is established
until matched training comparisons are run.

### Transformer and Set Transformer variants

Use `--architecture entity_transformer` to add aircraft self-attention between
the shared MLP and pooling. The encoder uses 64-dimensional embeddings, four
attention heads, 128-unit feed-forward layers, residual connections, layer
normalization, and zero dropout. No positional encoding is used. The default
`--transformer-layers 1` gives a single-layer encoder; use `2` or `3` to compare
depths (any positive depth is accepted). Each layer has independent parameters.
Actor and critic use separate encoders, as with `entity_mlp`.

All existing pooling methods are supported: `mean`, `max`, `mean_max` (default),
`additive_attention`, and `ego_attention`, including both `--attention-context` options. The
Transformer also supports `--pooling pma`, our Set Transformer variant. PMA
uses one learned seed query and four-head attention, followed by residual
connections, normalization, and a feed-forward network, to produce one
64-feature summary. The learned seed is independent of ownship context; full
context is concatenated after pooling. `--attention-context` is only applicable
to additive attention, and PMA is available with `entity_transformer` and
`entity_gnn`.

```bash
python experiments/train_single_ppo.py --architecture entity_transformer --transformer-layers 1 --pooling mean --model-path experiments/models/ppo_transformer_l1_mean
python experiments/train_single_ppo.py --architecture entity_transformer --transformer-layers 2 --pooling additive_attention --attention-context all --model-path experiments/models/ppo_transformer_l2_attention_all
python experiments/train_single_ppo.py --architecture entity_transformer --transformer-layers 3 --pooling pma --model-path experiments/models/ppo_transformer_l3_pma
```

The same options work with single-agent PPO/SAC/TD3/DDPG and parameter-shared
PPO/SAC. Padding is masked during self-attention and pooling, and empty traffic
produces a zero summary. Normalization continues to share occupied-aircraft
statistics across slots. Checkpoints persist encoder depth along with pooling
and attention context; resume/evaluation/replay restore these automatically
and reject conflicting CLI overrides. Existing MLP checkpoints remain supported.

Use labels such as `entity_transformer_l1_mean`,
`entity_transformer_l2_additive_attention_all`, and `entity_transformer_l3_pma`.
Compare MLP versus Transformer with the same pooling to measure the effect of
aircraft interactions, then compare Transformer mean versus PMA to measure
learned multi-head pooling. Multi-seed architecture ablations remain pending.

### Graph neural networks

`--architecture entity_gnn` adds message passing after the shared 64-feature
aircraft MLP. Use `--gnn-type gat` (default) for four concatenated 16-feature
attention heads, or `--gnn-type gcn` for symmetric degree-normalized graph
convolution. `--gnn-layers N` selects any positive depth (default 1). GCN uses
ReLU activations; GAT uses LeakyReLU attention scores and ELU node updates.
These are custom batched PyTorch layers, without a graph-library dependency.
Actor and critic learn separate graph encoders.

`--graph fully_connected` is the default: every occupied aircraft pair is
connected, including self-connections, without a distance threshold. Standard
GCN on a complete graph gives every node the same aggregated representation;
retain this as a baseline and use sparse distance graphs to test GCN locality.
GAT learns attention weights over the connected neighbors.

For later graph ablations, select `--graph distance --graph-radius-nm 25`.
Distance graphs connect occupied pairs within the selected horizontal radius
(default 25 NM for this graph option). They retain self-connections for isolated
aircraft. Edges are calculated using Cartesian geometry before running
normalization, are passed as binary adjacency observations, and are excluded
from policy context and observation normalization. This preserves the radius
semantics during training, evaluation, and off-policy replay. No aircraft are
removed from the observation or pooling because of distance.

```bash
python experiments/train_single_ppo.py --architecture entity_gnn --gnn-type gat --model-path experiments/models/ppo_gat_full
python experiments/train_single_ppo.py --architecture entity_gnn --gnn-type gcn --gnn-layers 2 --pooling mean --model-path experiments/models/ppo_gcn_l2_full_mean
python experiments/train_single_ppo.py --architecture entity_gnn --gnn-type gat --graph distance --graph-radius-nm 25 --model-path experiments/models/ppo_gat_distance25
```

GNNs support all existing pooling methods (`mean`, `max`, `mean_max`,
`additive_attention`, `ego_attention`, `pma`) and both attention contexts. The default
pooling remains `mean_max`. Padding is excluded from graph edges and pooling;
empty traffic produces zeros. Graph type, layer count, GCN/GAT choice, and radius
are saved with every training checkpoint and restored automatically. GNN flags
require `entity_gnn`; an explicit radius requires `--graph distance`. All existing
single-agent algorithms and parameter-shared PPO/SAC support these options.
Existing MLP/Transformer checkpoints remain compatible. Graph ablation runs have
not been performed yet.

For radius-free sparse graphs, select `--graph knn --graph-k 3` (`k=3` is the
default for this option). Each occupied aircraft selects its k nearest other
aircraft, and an undirected edge is kept if either endpoint selects the other.
All aircraft tied at the kth distance are included; no aircraft-slot order is
used to break ties. Consequently, ties and bidirectional edges can give a node
more than k neighbors. Self-connections are added separately and do not count
toward k. With fewer than k other aircraft, all available neighbors are used.
Padding is excluded, and empty traffic remains supported.

```bash
python experiments/train_single_ppo.py --architecture entity_gnn --gnn-type gat --graph knn --graph-k 3 --model-path experiments/models/ppo_gat_knn3
```

K-nearest edges use geometry before running normalization and share the binary
adjacency path used by distance graphs. The selected k is saved in checkpoints
and automatically restored for resume, evaluation, and replay. `--graph-k` is
only valid with `--graph knn`; a radius cannot be combined with this graph option.
Fully connected remains the default graph, and older checkpoints remain supported.

### Temporal modeling

Temporal processing is independent of the spatial architecture:
`--temporal none` (default), `frame_stack`, `gru`, `lstm`, or `tcn`.
`--history-length N` selects a positive number of observations (default 8 when
using a temporal encoder). Each frame passes through the spatial encoder first:
flat features for `mlp`, or the pooled aircraft/context representation for entity
MLP, Transformer, or GNN encoders. Entity pooling therefore preserves aircraft
permutation invariance independently at each time step.

```bash
python experiments/train_single_ppo.py --temporal frame_stack --history-length 8 --model-path experiments/models/ppo_stack8
python experiments/train_single_ppo.py --architecture entity_mlp --pooling mean --temporal gru --history-length 8 --model-path experiments/models/ppo_entity_gru8
python experiments/train_single_sac.py --architecture entity_transformer --pooling pma --temporal lstm --history-length 8 --model-path experiments/models/sac_transformer_lstm8
python experiments/train_zoo_ppo.py --architecture entity_gnn --graph knn --graph-k 3 --temporal tcn --history-length 8 --model-path experiments/models/shared_gat_tcn8
```

Frame stacking concatenates encoded frames and validity flags. GRU/LSTM use one
64-unit recurrent layer, starting from zero hidden state for every window. They
are fixed-window sequence encoders, with backpropagation through that window;
no hidden state is carried between prediction calls. TCN uses 64 channels and
causal, kernel-3 convolutions with doubling dilations until the receptive field
covers the window (dilations 1/2/4 for length 8), returning the latest output.
Actor and critic have separate spatial and temporal encoders. All existing
algorithms support these modes; stateful recurrent RL is a separate future task.

Windows contain oldest-to-newest raw observations and a binary validity column.
Before sufficient history is available, missing frames are masked zero padding.
GRU/LSTM skip invalid frames; TCN masks invalid positions between layers.
History clears on episode reset and for departed aircraft in shared-policy runs.
Terminal observations retain the final window for replay and time-limit
bootstrapping. Past observations never cross episode or agent boundaries.

Normalization statistics are shared across time, updated once per latest valid
frame, and saved alongside the policy. For entity encoders, aircraft statistics
remain shared across occupied slots and presence/graph-edge masks bypass
normalization. Frame-validity flags bypass normalization for every spatial
architecture. Replay buffers store complete raw windows, so off-policy training
reconstructs temporal inputs with current normalization rather than cached latent
vectors. Evaluation freezes statistics. Checkpoints restore temporal mode and
history length automatically and reject conflicting CLI overrides. Resuming
training starts new simulator episodes with fresh histories and restores saved
replay windows when available. Older non-temporal checkpoints remain supported.
Temporal ablation runs have not been performed yet.

Compare temporal methods on the two spatial winners:

```bash
python -m experiments.compare_temporal --dry-run
python -m experiments.compare_temporal --jobs 4
```

Defaults: entity MLP with additive attention/context `all`, and one-layer
Transformer with `mean_max` pooling; each tests `none`, `frame_stack`, `gru`,
`lstm`, and `tcn`. Temporal windows contain eight observations. All ten trials
use fresh single-agent PPO policies, seed 42, 100k requested steps (100,352 after
rollout rounding), relative heading/absolute speed (100–500 kt),
`goal_aligned_conflict` control observations, 10-second decisions, and evaluation
scenarios 2026–2035. These observations match the spatial comparison; observation
selection remains preliminary. Output: `experiments/results/temporal_comparison/`.
In `paired.csv`, each temporal variant uses its matching spatial no-history
control, isolating its temporal effect. If no matching control is in the plan,
pairing falls back to the first variant; each row records its actual baseline.

Use `--history-lengths 4 8` for 18 variants, `--seeds 0 1 2` for repeated trials,
and `--resume` to reuse completed trials. Resume restores saved temporal choices
and windows; repeat any non-default seeds or training/control options. Use
`--temporal none` to run only the two spatial baselines. Other architecture
comparison controls apply; temporal choices and history lengths remain overridable.

Runtime estimate on the previous CPU setup: the no-history winners took
47–54 minutes of training per 100,352 steps, plus about 40 seconds of startup and
evaluation. Eight-frame temporal trials provisionally budget **1–3 hours each**;
this is an unmeasured planning estimate because every frame repeats spatial
encoding and sequence processing adds work. Allow roughly **10–25 hours serial**
or **3–8 hours with four jobs** for the ten-trial default, subject to CPU
contention. Measure the first temporal trials before committing to more seeds;
three training seeds triple the work. These are fixed-window encoders, not
stateful recurrent PPO.

The temporal option matrix, numerical boundary checks, and persistence tests are
described in [tests/README.md](../tests/README.md).

### Attention comparison

Compare the two shortlisted spatial encoders with the new ego cross-attention
variant:

```bash
python -m experiments.compare_attention --dry-run
python -m experiments.compare_attention --seeds 0 1 2 --jobs 3
```

The three variants are:

- Entity MLP + additive attention with `attention_context=all`.
- Entity MLP + ego cross-attention with `attention_context=all`.
- One-layer entity Transformer + `mean_max` pooling.

Both attention readouts receive the same query context; every variant receives
full non-aircraft context in its policy/value heads. This compares complete
architectures, rather than equal parameter counts. Defaults use PPO, 100,000
requested transitions, training seed 0, evaluation seeds 2026–2035, and no temporal
history. Action, observation, and decision-interval defaults match the architecture
comparison below. Use `--algorithm`, `--timesteps`, `--seeds`, and `--eval-seeds`
to change the shared protocol. PPO completes full rollouts, so inspect actual
transition counts in the report.

Each training run uses its own subprocess and normalization statistics;
evaluation freezes those statistics. Results, paired scenario comparisons,
plots, and checkpoints go under `experiments/results/attention_comparison/`.
Use `--resume` to reuse completed runs and `--resume --retry-failed` to retry
failed runs in fresh attempt directories. `--temporal` and `--history-lengths`
can cross all three variants with temporal encoders. Custom `--tree` and
`--winning-architectures` options belong to `compare_architectures` instead.

### Context-token attention comparison

Train the context-token Entity Transformer against the three completed attention
variants, without training or reevaluating those reference policies:

```bash
python -m experiments.compare_context_attention --dry-run
python -m experiments.compare_context_attention
```

The source defaults to `experiments/results/attention_comparison/`; use
`--reference-dir PATH` to select another completed three-variant attention
comparison. Training and evaluation seeds, nominal budget, controls, device and
CPU thread count are inherited from its saved evidence. With the recorded seed-42
comparison this schedules exactly one new run: PPO, 100,000 requested steps
(100,352 actual), paired scenarios 2026–2035, relative heading/absolute speed
(100–500 kt), `goal_aligned_conflict` control observations, 10-second decisions,
one Transformer layer and no temporal history. The new policy has its own
normalization statistics, frozen during evaluation.

All three references must have completed, finite, matched scenario reports with
the same PPO settings and actual training budget. Missing or invalid reference
evidence stops the command before training; reference variants never enter the
training or retry queue. Source files are read only. Protocol, PPO settings,
actual steps and unshaped rewards are also checked on the new result.

The separate output directory defaults to
`experiments/results/context_attention_comparison/` (`--output-dir PATH` to
change it). It contains the new checkpoint/logs plus combined four-variant
`report.md`, `comparison.png`, `summary.csv`, `episodes.csv`, `paired.csv` and
`runs.json`/`runs.csv`. Paired return differences use the original additive-attention
baseline. The manifest records hashes of the reference evidence; changed
references are rejected on resume.

```bash
python -m experiments.compare_context_attention --resume
python -m experiments.compare_context_attention --resume --retry-failed
```

Resume reuses a completed new run. `--retry-failed` creates a fresh attempt only
for the new architecture; prior attempts remain available. `--timeout SECONDS`
optionally limits each new training/evaluation subprocess. Seeds, budgets and
control settings cannot be overridden through this entry point.

### Architecture comparisons

Preview the complete experiment plan before training:

```bash
python -m experiments.compare_architectures --dry-run
```

The default tree contains 32 spatial variants: flat MLP (the paired baseline),
entity MLP with seven pooling/context combinations, Transformer with eight, and
GCN/GAT with eight each. Both additive and ego attention include ownship/goal
and full-context queries. Depth is 1 and GNN graphs are fully connected. Temporal
modeling is opt-in. Defaults retain the selected relative-heading/absolute-speed
actions, 100–500 kt bounds, `goal_aligned_conflict` control observations, 10-second
decisions, PPO, 100,000 requested transitions per run, training seed 0, and the
same evaluation scenarios 2026–2035 for every variant.

```bash
python -m experiments.compare_architectures --jobs 4 --seeds 0 1 2 --devices cpu --output-dir experiments/results/spatial_architectures
python -m experiments.compare_architectures --temporal none frame_stack gru lstm tcn --history-lengths 4 8 --jobs 2 --output-dir experiments/results/temporal_architectures
```

The second command produces 288 variants (32 spatial × nine temporal/history
choices). All variants train fresh policies. Use `--dry-run` to inspect their
commands. `--algorithm` supports single-agent PPO/SAC/TD3/DDPG; `--multi-agent`
supports shared PPO/SAC. PPO completes full rollouts, so actual budgets can
exceed requested transitions; reports retain both. Compare algorithms and agent
modes in separate output directories, and inspect actual budgets before drawing
matched-budget conclusions.

Execution uses one coordinator and at most `--jobs` leaf training subprocesses.
The tree describes choices; it does not create waiting branch processes or
nested process pools. Each subprocess owns one simulator, trains and evaluates
through the existing workflow, and has separate policy/normalization/configuration
files. A free slot immediately receives the next run, avoiding static branch
allocation when training times differ. No additional Python worker processes or
interprocess transfers of models, tensors, or simulator state are needed.

Start with `--jobs 1` (the default) and increase for available CPU/RAM capacity.
`--threads-per-job 1` limits numerical-library CPU threads per subprocess;
raise it if a measured workload benefits. The fastest job/thread combination
depends on simulator cost, model size, memory and hardware. Compare recorded
training throughput and wall times when tuning it. For two GPUs use
`--jobs 2 --devices cuda:0 cuda:1`. Devices map cyclically to scheduler slots;
more jobs than devices intentionally allows concurrent runs on a device.
The default device is CPU. `--timeout SECONDS` optionally bounds an entire run,
including evaluation. This runner uses POSIX process groups and file locks.

To define conditional options and suboptions, edit
[configs/architecture_options.json](configs/architecture_options.json):

```bash
python -m experiments.compare_architectures --tree experiments/configs/architecture_options.json --dry-run
python -m experiments.compare_architectures --tree experiments/configs/architecture_options.json --jobs 4 --output-dir experiments/results/custom_architectures
```

This example has 13 leaves. Each node has `name`, optional `settings`, optional
`sweep`, and optional `children`. Settings inherit down the tree; children may
override them. Sweeps form a Cartesian product at that node, before selecting
child branches. For example, a `knn` child can sweep `graph_k` while a `distance`
child sweeps `graph_radius_nm`. Add additive-attention context branches or
temporal/history sweeps in the same way. With `--tree`, put temporal choices in
the tree instead of passing `--temporal`/`--history-lengths`. Temporal comparisons
pair against matching spatial no-history controls where present; otherwise the
first leaf is the baseline. If a matched control fails, its paired differences
are omitted rather than switched to another architecture. Duplicate effective
variants and invalid option combinations fail validation before training.

The output directory contains:

| File | Contents |
|---|---|
| `report.md` | Consolidated ranked table with the return bar plot |
| `summary.csv` | Returns, recorded safety/completion metrics, seed SD, training timing and actual budgets |
| `episodes.csv` | Every evaluation scenario/agent, with variant and training seed |
| `paired.csv` | Within-training-seed/scenario return differences and the selected baseline |
| `comparison.png` | Mean return with SD across training seed means |
| `runs.csv`, `runs.json` | Every job's status, command, device, timing and log/report paths |
| `comparison.json`, `execution.json` | The tree, matched protocol, resolved variants and execution resources |
| `runs/<tree path>/seed_N/attempt_N/` | Per-run log, policy, normalization, configuration sidecars and training reports |

Aggregation first averages scenarios and agents within a training seed, then
weights training seeds equally. Agents are not counted as independent training
replicates. Partial variants remain visible without receiving a rank. Rank uses
raw return; inspect safety/completion columns separately. A one-seed SD of zero
does not estimate uncertainty. Only complete, finite evaluations covering every
requested scenario/agent are accepted.

Failures do not prevent independent jobs from finishing; the command returns a
nonzero exit status if any job fails. Ctrl-C or SIGTERM stops active process
groups, allowing checkpoint saving before escalating termination, and preserves
partial reports. Reuse completed runs with:

```bash
python -m experiments.compare_architectures --output-dir experiments/results/custom_architectures --resume --jobs 4
```

Keep the same algorithm, seeds, evaluation scenarios and control/training
settings. Resume restores the original tree automatically unless `--tree` or
temporal overrides are supplied. Job/device/thread limits may change. Unfinished
attempts are preserved; add `--retry-failed` to train fresh attempts in new
directories. This retries a comparison leaf from scratch; use the existing
training `resume` mode separately to continue a preserved policy checkpoint.
Concurrent coordinators cannot write the same comparison directory.

## Reward engineering

Single-agent, shared-policy, and MAPPO entry points support the same optional
reward components. All six new weights default to **zero**, preserving the
existing destination, heading-drift, intrusion, restricted-area, and sector-exit
reward. For example:

```bash
python -m experiments.train_single_ppo train --timesteps 100000 \
  --reward-progress-weight 1 \
  --reward-predictive-weight 0.1 \
  --reward-obstacle-weight 0.1 \
  --reward-sector-weight 0.1 \
  --reward-time-weight 0.01 \
  --reward-smoothness-weight 0.1
```

These example weights demonstrate configuration; they have not been tuned.
Weight sensitivity studies remain future work.

The added reward is the sum of:

| Component | Weighted contribution |
| --- | --- |
| Goal progress | `w_progress * (distance_before - distance_after) / progress_scale_m` |
| Predictive separation | `-w_predictive * sum(dt * risk)` over live neighbours and simulation samples |
| Obstacle clearance | `-w_obstacle * sum(dt * max(0, 1 - clearance / obstacle_buffer_m)^2)` |
| Sector clearance | `-w_sector * sum(dt * max(0, 1 - clearance / sector_buffer_m)^2)` |
| Time | `-w_time * actual_elapsed_seconds` |
| Command smoothness | `-w_smoothness / command_interval_seconds * ((wrapped_heading_change / heading_scale_deg)^2 + (CAS_target_change / speed_scale_knots)^2)` |

Predictive risk is zero unless `0 < TCPA < prediction_horizon_s`. Otherwise it is
`(1 - TCPA / prediction_horizon_s) * max(0, 1 - DCPA / separation_distance_m)^2`.
CPA uses constant relative ground velocity and raw simulator positions, sharing
the observation CPA calculation without relying on observation clipping, sorting,
or normalization. Stationary relative traffic has zero predictive risk; existing
intrusion penalties still apply to present violations. Pairwise risks are summed
for each aircraft's own reward.

Clearance measures distance to actual polygon edges, positive outside obstacles
and inside the sector, negative on the unsafe side. With multiple obstacles, use
the minimum signed clearance; with no obstacles, the obstacle term is zero.
Geometry uses a fixed local east/north metre projection about the scenario centre.
Predictive and clearance costs are sampled every simulator second, including
short terminal steps. Progress is signed, so flying away incurs a penalty.
Smoothness compares physical command targets across updates, uses the elapsed
time between commands, and is zero for the first command after each reset.

All scales are configurable, with defaults:

| CLI option | Default |
| --- | --- |
| `--reward-progress-scale-m` | 1852 m (1 NM) |
| `--reward-prediction-horizon-s` | 120 s |
| `--reward-separation-distance-m` | 11112 m (6 NM warning distance; scored intrusion threshold stays 5 NM) |
| `--reward-obstacle-buffer-m` | 1852 m |
| `--reward-sector-buffer-m` | 1852 m |
| `--reward-heading-scale-deg` | 45 degrees |
| `--reward-speed-scale-knots` | 20/3 knots |

Weights must be finite and nonnegative; scales must be finite and positive.
The Python API accepts `RewardConfig` via `reward_config=` on the environment
factories and on `train`/`replay`. When using `train`, supply the configuration
there so it can configure the factory and persist matching metadata.

Every new training checkpoint stores rewards in both the policy archive and a
`*_reward.json` sidecar, including periodic and interrupted checkpoints. Resume
and evaluation restore the saved settings and reject conflicting overrides.
Checkpoints predating this feature use zero shaping weights. Training still
normalizes/clips the combined reward; evaluation reports its raw value.

Environment infos expose weighted step contributions in `reward_components` and
episode totals in `reward_component_totals` whenever shaping is enabled. Training
`episodes.csv` and explicit scenario evaluation reports include `reward_progress`,
`reward_predictive`, `reward_obstacle`, `reward_sector`, `reward_time`, and
`reward_smoothness`. The original scored safety/completion metrics are preserved.

## One-term reward ablation

Run a baseline plus six variants, each enabling exactly one new reward term:

```bash
python -m experiments.compare_rewards --timesteps 100000 --seeds 42 --jobs 3
```

Every trial uses single-agent PPO, entity MLP with additive attention over all
context, relative heading, absolute speed in 100–500 knots, goal-aligned conflict
observations, and 10-second decisions. Default evaluation seeds are 2026–2035.
Each trial starts fresh; the nominal weights are progress 0.125, predictive 0.5,
obstacle 0.1, sector 0.1, time 0.001, and smoothness 0.001. These are untuned
screening values, not weights selected by a sensitivity study. All other new
weights are zero within each variant, and physical scales retain RewardConfig
defaults. Change a nominal weight explicitly with e.g. `--progress-weight 0.2`.

The runner uses isolated processes with one CPU thread each by default. Outputs
are under `experiments/results/reward_ablation/`: full protocol, per-run logs and
checkpoints, scenario rows, objective paired differences, summary CSV, and a
behavior-metric report and figure. Compare safety, completion, and flight time;
different shaped returns are diagnostics and are not ranked. `base_return`
subtracts all new weighted components from the original scalar total to recover
the shared base reward. Successful flight time excludes failures and must be
interpreted together with completion. One training seed is exploratory and does
not establish training uncertainty.

Use `--dry-run` to inspect commands without training. `--resume` reuses validated
completed runs, and `--resume --retry-failed` preserves unsuccessful attempts and
starts fresh retries. The protocol and weights cannot change on resume. More
training seeds can be supplied for a new study; `--devices`, `--threads-per-job`,
and `--output-dir` are also configurable. No full-reward or sensitivity variants
are launched by this script.

## Independent PPO

IPPO assigns each fixed aircraft slot its own MLP actor, local MLP critic,
Gaussian action variance, and Adam optimizer. Every policy stays fixed while the
single BlueSky world collects a joint rollout. Each learner then updates using
only its own valid transitions, with separate advantage normalization and KL
stopping. Departed aircraft are excluded from training; time limits bootstrap
from final observations and true terminations use zero bootstrap.

```bash
python -m experiments.train_zoo_ippo --timesteps 100000 --seed 42
python -m experiments.train_zoo_ippo resume --timesteps 100000
python -m experiments.train_zoo_ippo evaluate --episodes 20
python -m experiments.train_zoo_ippo replay
```

These commands use the existing model/report paths, periodic checkpoints, and
normalization workflow. One policy archive contains every learner and its
optimizer state. Saved normalization statistics are pooled across aircraft within
the run, as for the other baselines, and frozen during evaluation. Networks and
optimizer state are never shared between IPPO learners. Checkpoints require the
same aircraft count and fixed slot ordering. Joint `model.predict` takes every
slot; for individual inference use `model.policy.learners[index].predict` with
that aircraft's normalized local observation. Both flat MLPs and entity MLPs with
additive attention are supported, without temporal history.

## Multi-agent PPO comparison

```bash
python -m experiments.compare_multi_agent \
  --joint-steps 102400 --seeds 42 43 44 \
  --eval-seeds 2026 2027 2028 2029 2030 \
  --jobs 3 --threads-per-job 1 --devices cpu \
  --output-dir experiments/results/multi_agent_mlp
```

To repeat the completed seed-42, 100k-joint-step comparison with additive
attention in **both actors and critics**:

```bash
python -m experiments.compare_multi_agent \
  --architecture entity_mlp --attention-context all \
  --joint-steps 100000 --seeds 42 \
  --eval-seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
  --jobs 3 --threads-per-job 1 --devices cpu \
  --output-dir experiments/results/multi_agent_additive_seed42_100k
```

`entity_mlp` selects masked additive attention, reusing the single-agent encoder
shortlisted in RESEARCH. Each local encoder independently embeds visible aircraft
into 64 features, weights them using all non-aircraft context by default, then
concatenates the weighted sum with that raw context. Actor and critic encoders
have separate parameters. IPPO also keeps each learner's encoders independent.
MAPPO's critic encodes all agents' local observations, projects them to 64-feature
agent embeddings, and uses a second additive pool over active agents, including
the focal agent. Each focal agent's local features query this pool; those features
and the pooled summary feed its value head. Departed agents are masked throughout.
Actor inference remains local. `--attention-context ownship_goal` changes the
local aircraft pool's query. No temporal history is added.

The default `--architecture mlp` retains the original benchmark and can resume
its existing manifests. Attention runs require a new output directory; resume
rejects changes to architecture or attention context. Add `--dry-run` to preview
the protocol and worker commands. Use `--seeds 42 43 44` for repeated training
seeds once the initial attention comparison is complete.

The three methods are:

| Method | Actor | Critic |
|---|---|---|
| `ippo` | Separate per aircraft | Separate local critic per aircraft |
| `shared_ppo` | Shared across aircraft | Shared local critic |
| `mappo` | Shared across aircraft | Centralized critic with one value per aircraft |

All actors and critics use two 256-unit tanh hidden layers by default. Change
both with `--hidden-sizes 128 128`. The runner fixes relative heading, absolute
speed (100–500 knots), control observations with `goal_aligned_conflict`
encoding, the base reward, and a ten-second decision interval. The centralized
MLP critic sees the masked concatenated aircraft observations and activity flags;
the attention critic uses the masked agent pool described above. Local critics
see only their aircraft's observation.

The comparison uses `SharedPPO`, a local-critic variant of the custom masked PPO
trainer. This gives all methods the same collection, GAE, clipping, and individual
boundary handling. The existing SB3 `train_zoo_ppo` command keeps its original
implementation and defaults. The comparison is a controlled algorithm ablation,
not a replay of those original SB3 defaults.

Matched settings: learning rate 3e-4, gamma 0.995, GAE lambda 0.95, policy/value
clip 0.2, entropy coefficient 0.01, value coefficient 0.5, gradient norm 0.5,
and initial log standard deviation -0.5. Defaults are `--rollout-steps 1024`,
`--batch-size 64`, and `--epochs 10`. Minibatches contain up to 64 joint timesteps
for shared PPO/MAPPO, and up to 64 valid own transitions for each IPPO learner.
IPPO normalizes advantages per learner; the shared methods normalize across
active transitions. Hidden widths match; total parameters and compute costs differ.

`--joint-steps` counts simulator decisions, rounded up to a complete rollout for
all three methods. With ten slots, 102400 joint steps means 1024000 slot
transitions. Standalone trainers' `--timesteps` counts slot transitions instead.
Reports retain actual joint steps, slot transitions, valid active transitions,
training seconds, throughput, and parameter counts. Valid transition counts can
differ when policies complete flights at different times. Each run uses one
simulator in its own subprocess; `--jobs` limits concurrent runs.

Evaluation freezes observation normalization and disables reward normalization.
The same training seeds and explicit evaluation scenario seeds are used for all
methods. Reports average agents/scenarios within each training seed, then weight
training seeds equally. The paired reference is `shared_ppo`; a single seed does
not estimate training uncertainty.

The output contains `comparison.json` (protocol), `runs.json`/`runs.csv`
(status, budgets, costs, and log paths), `episodes.csv` (per-aircraft metrics),
`summary.csv`, `paired.csv`, `comparison.png`, and `report.md`. Inspect safety and
completion alongside return. Each run keeps policy/normalization checkpoints and
training history under its own attempt directory.

Preview without creating files with `--dry-run`. Repeat the same command with
`--resume` to reuse validated completed runs; add `--retry-failed` to retry failed
runs in new attempt directories. Resume requires the same experimental settings.
`--timeout SECONDS` limits each run's wall time. Ctrl+C preserves checkpoints and
partial reports; retrying a failed comparison run starts a fresh attempt.

A small execution smoke test (not a performance benchmark):

```bash
python -m experiments.compare_multi_agent \
  --joint-steps 2 --rollout-steps 2 --batch-size 2 --epochs 1 \
  --hidden-sizes 8 --seeds 42 --eval-seeds 2026 \
  --output-dir experiments/results/multi_agent_smoke
```
