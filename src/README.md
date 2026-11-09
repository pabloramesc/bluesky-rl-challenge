# Custom implementations

Put reusable code in `bluesky_pabloramesc/` and experiment entry points in
`experiments/`.

| Module | Purpose |
| --- | --- |
| `models.py` | PPO/SAC/TD3/DDPG/MAPPO builders and hyperparameters |
| `mappo.py` | Custom MAPPO networks, joint rollouts, masked GAE, and PPO updates |
| `multi_agent_attention.py` | Local additive-attention actors/critics and an active-agent attention critic |
| `single_agent.py` | Single-aircraft Gymnasium environment |
| `multi_agent.py` | PettingZoo wrappers and vector environment |
| `training.py` | Training, checkpoints, evaluation, and replay |
| `reporting.py` | Run metadata, raw episode histories, and performance plots |
| `cli.py` | Shared command-line options |
| `actions.py` | Validated action configuration and heading/CAS target mappings |
| `rewards.py` | Configurable progress, predictive separation, clearance, time, and command-smoothness rewards |
| `reward_ablation.py` | Seven fixed-architecture baseline/add-one reward trials and paired behavior reports |
| `geometry.py` | Closest approach, local metre projection, cached polygons, signed clearance, nearest boundary points, and finite ray intersections |
| `boundary_observations.py` | Selectable nearest-edge, ray-cast, and combined obstacle/sector observations |
| `attention_geometry_comparison.py` | Nearest-edge additive-attention PPO against the read-only completed default-geometry baseline |
| `observations.py` | Geometry encodings, optional conflict features, ownship state, and shared environment hooks |
| `comparison.py` | Isolated action/observation ablations and paired reports |
| `architecture_tree.py` | Conditional architecture options, inherited settings, and Cartesian sweeps |
| `architecture_reporting.py` | Architecture CSVs, seed-paired differences, ranked tables, and plots |
| `architecture_comparison.py` | Bounded subprocess scheduling, resume/retry, logs, and report collection |
| `context_attention_comparison.py` | New-only context-token training, read-only attention references, and matched combined reports |
| `dual_encoder_comparison.py` | New-only traffic/obstacle Transformer comparison against completed architecture runs |
| `reference_comparison.py` | Shared validation, provenance, scheduling and reporting for new-only comparisons |

Add your own modules alongside these as needed. For example, after creating
`agents.py`, import it with `from bluesky_pabloramesc.agents import MyAgent`.

Create `.venv` using the [setup guide](../README.md#setup), then install the
package and development tools from the repository root:

```bash
source .venv/bin/activate
uv pip install -e ./vendor/bluesky-gym -e ".[dev]"
python -m pytest
```

Editable installation makes source changes available without reinstalling.

Custom MAPPO uses `build_mappo` from `models.py` and `make_mappo_env` from
`multi_agent.py`. `mappo.py` implements its own collection, joint rollout buffer,
GAE, and clipped actor/value losses; SB3's on-policy base supplies the existing
callback, logging, prediction, and archive interfaces. Training and normalization
sidecars still use `training.py`.

The actor contract is `Actor(observation_dim, action_dim, **kwargs)` with
`forward(observations)` returning action means of shape `[..., action_dim]`.
The critic contract is `Critic(observation_dim, n_agents, **kwargs)` with
`forward(observations, active)` mapping `[batch, agents, features]` and a boolean
`[batch, agents]` mask to `[batch, agents]` values. Supply replacements through
`policy_kwargs` (`actor_class`, `critic_class`, `actor_kwargs`, `critic_kwargs`).
Joint minibatches preserve the agent axis so attention, Transformer, and GNN
critics can use peer information without changing the update. Actors retain
local observations at inference. MAPPO/IPPO support flat MLPs and
`--architecture entity_mlp --pooling additive_attention --attention-context all`;
other entity and temporal encoders are not yet wired to these trainers.

`multi_agent_attention.py` reuses `EntityMLP` with additive pooling for separate
local actor/critic encoders. It preserves arbitrary batch/agent dimensions around
the extractor. The centralized critic also projects local features into agent
embeddings and uses focal-agent-conditioned additive attention over active agents,
then combines the summary with the focal features to predict each value. Padding
is masked before encoding and pooling; empty active sets produce zero values.

`entities.py` implements shared aircraft MLP encoding, masked mean/max pooling,
additive attention and ego multihead cross-attention with ownship/goal or full
context, and slot-shared
normalization. Select it with `--architecture entity_mlp`;
`--architecture mlp` remains the default. Use `--pooling ego_attention` for a
context-conditioned query attending to neighbour embeddings through four heads;
`--attention-context ownship_goal|all` selects its query inputs.

`--architecture entity_transformer` adds configurable aircraft self-attention
(`--transformer-layers`, default 1). It reuses all MLP pooling methods and adds
`--pooling pma` for a Set Transformer variant with one learned pooling query.

`--architecture context_entity_transformer` instead jointly attends over aircraft
and a token encoding all non-aircraft context (ownship, goal, obstacles and sector).
It uses pre-norm blocks and a final LayerNorm, with the same 64-dimensional tokens,
four heads and configurable `--transformer-layers` (default 1). The updated context
token is concatenated with raw context for the policy/value heads. Its fixed
`context_token` readout is selected automatically; empty traffic keeps the context
token active. No aircraft slot positional embeddings are used.

`--architecture dual_entity_transformer` uses independent traffic and obstacle
Transformers. Each branch encodes entities into 64-dimensional tokens and prepends
a token encoding ownship/goal features. Pre-norm self-attention with four heads
and 128-unit feed-forward layers produces two 64-feature context-token latents.
These latents and raw non-entity context (including sector features) feed separate
actor/critic MLPs. Raw obstacle slots are excluded from that context. The fixed
`context_token` readout is automatic; `--transformer-layers` controls both branches
(default 1; the focused comparison uses 2). There are no slot positional embeddings,
causal masks or temporal memory unless a temporal encoder is selected.

This architecture enables `obstacle_observations=True` in the environment factory.
`ControlHooks` creates binary obstacle presence before geometry conversion;
`entity_layout` identifies polar or Cartesian obstacle tokens. Obstacle statistics
are shared across occupied slots and saved/frozen with existing normalization.
Existing architectures retain their original observation schemas and normalization.

`--architecture entity_gnn` adds GCN/GAT message passing with configurable depth
and graph connectivity. Fully connected graphs are the default; distance and
k-nearest (`--graph knn --graph-k 3`) graphs compute binary adjacency before
running normalization. All existing pooling
methods are supported.

`temporal.py` provides per-agent raw history windows, shared temporal normalization,
and frame-stack/GRU/LSTM/causal-TCN encoders. Use `--temporal` independently of
`--architecture`; `--history-length` defaults to 8 for temporal modes.

Architecture comparison trees use JSON objects with `name`, optional `settings`,
optional `sweep` (setting names mapped to lists), and optional `children`.
Children inherit ancestor settings. Sweeps form a Cartesian product at their
node; child branches select conditional options. Every leaf is validated using
the training CLI before execution, and duplicate effective configurations are
rejected. For example, a conditional graph branch can be written as:

```json
{
  "name": "graphs",
  "settings": {"architecture": "entity_gnn", "pooling": "mean"},
  "sweep": {"gnn_type": ["gcn", "gat"]},
  "children": [
    {"name": "full", "settings": {"graph": "fully_connected"}},
    {
      "name": "knn",
      "settings": {"graph": "knn"},
      "sweep": {"graph_k": [1, 3, 5]}
    }
  ]
}
```

`expand_tree(tree)` returns eight named variants for this example. The reporting
module averages evaluation scenarios and agents within each training seed before
measuring variation across training seeds. Temporal variants pair against a
matching spatial no-history control where available, falling back to the first
variant otherwise. Incomplete variants remain visible without receiving a rank.
Run these through `python -m experiments.compare_architectures`; see the
[comparison guide](../experiments/README.md#architecture-comparisons) for resource
limits, conditional tree examples, temporal sweeps and result reuse.

`ippo.py` implements separate PPO learners per fixed aircraft slot and the shared
local-critic PPO control. It reuses `mappo.py`'s masked collection, individual
termination/truncation handling, GAE, and clipped losses. IPPO keeps networks,
optimizers, advantage normalization, and KL stopping independent. Its checkpoint
stores every optimizer; batched prediction preserves fixed slot order.

`multi_agent_comparison.py` compares IPPO, shared PPO, and MAPPO with matched head
widths, PPO settings, joint simulator budgets, and evaluation scenarios. Select
flat MLPs (default) or additive attention with `--architecture entity_mlp` and
optional `--attention-context all|ownship_goal` (default `all`). It reuses
the bounded subprocess scheduler and paired reporting. Training/checkpoints remain
in `training.py`, builders in `models.py`, argument parsing in `cli.py`, and the
two experiment entry points only dispatch to those modules. All three methods
pool normalization statistics within a run and freeze them during evaluation.

`multi_agent_sac.py` implements masked off-policy `SharedSAC`, `ISAC`, and
`MASAC`. It uses SB3's base algorithm for callbacks, prediction and archives;
collection, raw joint replay, squashed Gaussian actors, twin Q targets and SAC
updates are custom. ISAC owns independent networks/optimizers/temperatures per
fixed aircraft slot; its batched prediction requires all slots in fixed order.
For individual inference use `model.policy.learners[index].predict(local_obs)`.
Shared SAC uses local critics; MASAC's Q critics additionally attend over active
agents' observation/action embeddings, with focal-only actor gradients.

Builders `build_shared_sac`, `build_isac`, and `build_masac` live in `models.py`
and require entity observations with additive attention. `make_mappo_env()`
provides the common activity/boundary contract; the environment adapter preserves
raw final observations before normalization. Replay capacity, collection frequency
and batches count joint decisions; algorithm `learning_starts` and `num_timesteps`
count agent slots. Replay is serialized separately alongside the policy and
normalization by `training.py`. Sampling applies current normalization to raw
transitions. All target networks and actor/critic/temperature optimizers survive
policy loading. `multi_agent_sac_comparison.py` reuses the joint comparison
scheduler and report formats with strict SAC-specific settings validation.
