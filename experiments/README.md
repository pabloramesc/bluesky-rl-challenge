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
- `compare_observations`: compare upstream, ego Cartesian, goal-aligned, and conflict features.

Comparisons default to PPO and one training seed. Multiple seeds require
`--seeds`; `--output-dir` overrides their default folder under `results/`.
Use `--resume` to reuse completed comparison runs.

See [RESEARCH.md](../RESEARCH.md) for findings and next experiments.

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
`--pooling mean`, `max`, `mean_max` (the default), or `additive_attention`.
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
other variants. No architecture comparison runs have been performed yet.

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
heads. The context option is only valid with `--pooling additive_attention`.
Attention context is persisted with architecture settings and restored on resume,
evaluation, and replay; it cannot be overridden for an existing checkpoint.

### Transformer and Set Transformer variants

Use `--architecture entity_transformer` to add aircraft self-attention between
the shared MLP and pooling. The encoder uses 64-dimensional embeddings, four
attention heads, 128-unit feed-forward layers, residual connections, layer
normalization, and zero dropout. No positional encoding is used. The default
`--transformer-layers 1` gives a single-layer encoder; use `2` or `3` to compare
depths (any positive depth is accepted). Each layer has independent parameters.
Actor and critic use separate encoders, as with `entity_mlp`.

All existing pooling methods are supported: `mean`, `max`, `mean_max` (default),
and `additive_attention`, including both `--attention-context` options. The
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
`additive_attention`, `pma`) and both additive-attention contexts. The default
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
