# GNN Traffic Light Optimization — Project Summary

## Repository Layout

```
GNN-Traffic-Light-Optimization/
├── configs/
│   └── grid_4x4/
│       ├── grid.net.xml          # irregular 4×4 grid network
│       ├── grid.rou.xml          # vehicle demand / routes
│       ├── grid.add.xml          # lane-area detectors
│       └── grid.sumocfg          # SUMO configuration entry point
├── scripts/
│   ├── train_il.py               # CLI entry point for IL training
│   └── run_expert_grid.py        # headless/GUI expert demo runner
├── src/
│   ├── environment/
│   │   ├── __init__.py           # exports TrafficEnv, GreedyExpert
│   │   ├── sumo_env.py           # TrafficEnv (SUMO/TraCI wrapper)
│   │   ├── junction_info.py      # JunctionInfo dataclass + builder
│   │   └── expert.py             # GreedyExpert controller
│   ├── model/
│   │   └── gat_policy.py         # GATPolicy (GATv2 + residuals)
│   ├── training/
│   │   ├── eval_episode.py       # EvalMetrics, run_eval_episode, policy factories
│   │   └── imitation.py          # train_il() loop + checkpoint helpers
│   └── utils/
│       └── graph_builder.py      # GraphBuilder + RunningNormalizer
├── PLAN.md                       # full implementation plan
└── SUMMARY.md                    # this file
```

---

## SUMO Network and Configuration

- **Network**: irregular 4×4 grid, 16 traffic-light junctions (mix of 3-way and 4-way), roughly 200 m spacing
- **Detectors**: lane-area detectors on each approach lane, defined in `grid.add.xml`; used by TraCI to read halting vehicle counts, vehicle counts, and accumulated waiting time
- **Demand**: fixed route file (`grid.rou.xml`); flows vary by direction; future RL stages will randomize per-episode demand
- **SUMO flags used at runtime**:
  - `--no-step-log true`, `--no-warnings true` — suppress console spam
  - `--gui-settings-file` — optional, only in GUI mode
  - `--tripinfo-output <path>` — written during eval episodes; flushed on `traci.close()`

---

## TrafficEnv (`src/environment/sumo_env.py`)

### Interface

```python
env = TrafficEnv(cfg_path, gui=False, episode_length=3600)
obs          = env.reset()                     # dict[jid → np.ndarray(41,)]
obs, rew, done, info = env.step(actions)       # actions: dict[jid → int 0–3]
env.close()
```

### Observation Vector (41 dims per junction)

```
[0..35]   12 movements × 3 features (queue_density, approach_density, wait_density)
          slots 0–3 clockwise, each slot: left / through / right
          slot 3 is zero-padded for 3-way junctions
[36..39]  current phase one-hot (4 dims)
[40]      elapsed time in phase, normalised to [0, 1] (clamp at 45 s)
```

All density features are per-unit-length (vehicles/m or s/m), making them comparable across lanes of different lengths.

### Phase Scheme

4 canonical phases (NEMA-style, indexed against geometric slots):
- Phase 0: slots 0 & 2 green (N–S or equivalent)
- Phase 1: slots 0 & 2 left-turns green
- Phase 2: slots 1 & 3 green (E–W or equivalent)
- Phase 3: slots 1 & 3 left-turns green

Transitions: 3 s yellow → 2 s all-red → 10 s new green (within the 15 s decision interval).

### Step Timing

Decision interval = 15 s. All junctions receive a new target phase simultaneously. Switching junctions play out the yellow/all-red sequence within the same interval; holding junctions stay green for the full 15 s.

### Reward (per junction per step)

```
r_i = -α·Δlocal_wait_i - β·Δglobal_wait - γ·switch_i
α = 1.0,  β = 0.1,  γ = 0.5
```

`local_wait` and `global_wait` are wait densities (s/m): sum of lane waiting times divided by sum of detector lengths. This is length-invariant and the primary signal for the RL stage.

### `info` dict keys

- `sim_time`: current simulation time (s)
- `global_wait`: network-wide wait density (s/m)
- `local_waits`: dict[jid → float] per-junction wait density
- `switches`: dict[jid → bool] whether each junction switched this step

### `tripinfo_output` attribute

Set `env.tripinfo_output = path` before `env.reset()` to make SUMO write per-vehicle trip statistics (waiting time, travel time) to an XML file. `eval_episode.run_eval_episode()` manages this automatically via `tempfile.mkstemp`.

---

## GreedyExpert (`src/environment/expert.py`)

Scores each candidate phase by the per-vehicle waiting time attributable to vehicles whose routes use the lanes served by that phase. Avoids shared-lane over-counting by attributing each vehicle's wait to the route it is actually following.

### Hold-interval logic

- `MIN_HOLD_INTERVALS = 3` (45 s): never switches before 3 intervals on the same phase
- `MAX_HOLD_INTERVALS = 9` (135 s): forces a switch if any other phase has been starved this long

### Key methods

```python
expert = GreedyExpert(junction_infos)
expert.reset()                         # call before each episode
actions = expert.act()                 # dict[jid → int]; reads TraCI directly
expert.notify_applied(jid, phase)      # update hold-interval counter
```

---

## GraphBuilder + RunningNormalizer (`src/utils/graph_builder.py`)

### RunningNormalizer

Welford online algorithm for computing running mean and variance over 41-dim node feature vectors.

```python
norm = RunningNormalizer(dim=41)
norm.update(x)          # np.ndarray(41,) or (N,41)
x_norm = norm.normalize(x)
norm.freeze()           # call after IL training; disables further updates
d = norm.state_dict()   # {'n': int, 'mean': array, 'M2': array}
norm.load_state_dict(d)
```

### GraphBuilder

Builds static edge topology at construction time from the sumolib network object; only node features are updated each step.

```python
builder = GraphBuilder(net, junction_infos)
graph = builder.build(obs, update_normalizer=True)  # returns torch_geometric Data
```

- `junction_ids`: sorted list of all TL junction IDs — canonical order used for label alignment
- `edge_index`: shape (2, E), directed edges between TL junctions connected by SUMO edges
- `edge_attr`: shape (E, 3) — [length, n_lanes, speed_limit], z-score normalized once at construction; parallel edges are averaged
- `x`: shape (N, 41) — z-score normalized node features (or raw if normalizer not yet populated)

---

## GATPolicy (`src/model/gat_policy.py`)

```
Input: x (N, 41), edge_index (2, E), edge_attr (E, 3)

Encoder:
  Linear(41 → 128) → ReLU → LayerNorm
  Linear(128 → 128) → ReLU → LayerNorm

3 × GATv2Conv block:
  GATv2Conv(128 → 128, heads=4, head_dim=32, edge_dim=3, add_self_loops=False)
  ReLU → LayerNorm(x + residual)

Classifier:
  Linear(128 → 64) → ReLU → Linear(64 → 4)

Output: logits (N, 4)
```

- `forward(data) → (N, 4)` logits
- `predict(data) → (N,)` LongTensor of predicted phases
- `n_parameters() → int`

~200 K parameters. Parameter sharing is implicit: all junctions share the same weights; the graph structure provides context.

---

## Imitation Learning (`src/training/imitation.py`)

### `train_il()` loop

1. Expert generates actions via `expert.act()` (reads TraCI)
2. `GraphBuilder.build(obs, update_normalizer=True)` builds graph and updates normalizer
3. Labels aligned with `builder.junction_ids`
4. Cross-entropy loss on `model(graph)` vs expert labels
5. Adam optimizer step with gradient clipping (`grad_clip=0.5`)
6. Environment stepped with expert actions; `expert.notify_applied()` keeps hold-interval counts in sync
7. Every `eval_every` episodes: full model-vs-expert evaluation (see below)

### Default hyperparameters

| Parameter | Default |
|-----------|---------|
| `n_episodes` | 50 (CLI default: 20) |
| `episode_length` | 1200 s |
| `lr` | 3e-4 |
| `grad_clip` | 0.5 |
| `eval_every` | 5 |

### TensorBoard logging

**Per training step** (x-axis: global step):
- `loss/cross_entropy`
- `accuracy/phase_match`

**Per training episode** (x-axis: episode):
- `episode/avg_loss`
- `episode/avg_match_rate`
- `wait_density` → series `training`
- `policy/expert/switch_rate`
- `policy/expert/phase_dist` (histogram)

**Every `eval_every` episodes** (x-axis: episode, model and expert on same chart):
- `eval/avg_waiting_time` → `{model, expert}`
- `eval/avg_travel_time` → `{model, expert}`
- `eval/throughput` → `{model, expert}`
- `eval/max_queue_length` → `{model, expert}`
- `eval/phase_switch_freq` → `{model, expert}`
- `wait_density` → `{eval_model, eval_expert}` (shares chart with training series)

**Per-junction eval** (x-axis: episode):
- `junctions/{id}/wait_density` → `{model, expert}`
- `junctions/{id}/max_queue` → `{model, expert}`
- `junctions/{id}/model/phase_hist` (histogram)
- `junctions/{id}/expert/phase_hist` (histogram)

All multi-series charts use `writer.add_scalars(tag, {"model": v, "expert": v}, step)`, which puts both series on the same chart in TensorBoard's standard Scalars tab.

---

## Eval Episode (`src/training/eval_episode.py`)

### `run_eval_episode(env, policy_fn, on_step=None) → EvalMetrics`

- Creates a temp file, sets `env.tripinfo_output`, calls `env.reset()`
- Runs episode to completion using `policy_fn(obs)`
- Calls `env.close()` to flush the tripinfo XML
- Parses tripinfo XML for per-vehicle metrics
- Deletes temp file; clears `env.tripinfo_output`

### `EvalMetrics` fields

| Field | Source | Unit |
|-------|--------|------|
| `avg_waiting_time` | tripinfo XML | s/vehicle |
| `avg_travel_time` | tripinfo XML | s/vehicle |
| `throughput_per_hour` | tripinfo XML | vehicles/hour |
| `max_queue_length` | in-sim TraCI | vehicles |
| `phase_switch_freq` | in-sim TraCI | switches/junction/minute |
| `avg_wait_density` | in-sim TraCI | s/m |
| `per_junction_wait_density` | in-sim TraCI | s/m |
| `per_junction_max_queue` | in-sim TraCI | vehicles |
| `per_junction_phase_counts` | in-sim TraCI | counts [ph0..ph3] |

### Policy factories

```python
policy_fn = make_model_policy(model, builder, device)
policy_fn = make_expert_policy(expert)   # requires on_step for notify_applied
```

---

## Checkpoints

After IL training:

```
checkpoints/il/<timestamp>/
├── il_policy.pt      # GATPolicy state_dict
└── normalizer.npz    # RunningNormalizer state (n, mean, M2)
```

Loading:

```python
from src.training.imitation import load_checkpoint
model, norm_state = load_checkpoint('checkpoints/il/<timestamp>')
builder.normalizer.load_state_dict(norm_state)
```

---

## CLI

```bash
# Default: 20 episodes × 1200 s on grid_4x4
python scripts/train_il.py

# Custom
python scripts/train_il.py \
    --cfg      configs/grid_4x4/grid.sumocfg \
    --episodes 50 \
    --ep-len   3600 \
    --lr       3e-4 \
    --device   cuda \
    --eval-every 5 \
    --print-every 1

# TensorBoard (separate terminal)
tensorboard --logdir runs/
```

Run directories are timestamped (`runs/il/<YYYY-MM-DD_HH-MM-SS>/`) unless overridden. Log and checkpoint dirs share the same timestamp so they are always paired.

---

## What RL Needs (Not Yet Implemented)

The next stage is PPO reinforcement learning initialized from the IL checkpoint.

### Model changes
- Add a **value head** to `GATPolicy`: a second MLP head on top of the shared encoder/GAT backbone producing per-junction V(s) scalars
- The policy head (logits) is unchanged; value head is a new `Linear(128 → 64) → ReLU → Linear(64 → 1)` branch

### New components needed

| Component | File | Notes |
|-----------|------|-------|
| Rollout buffer | `src/training/rollout.py` | stores (graph, joint_action, rewards, values, log_probs, dones) per step |
| GAE computation | inside rollout | γ=0.99, λ=0.95, per junction |
| PPO update loop | `src/training/ppo.py` | K=10 epochs, minibatch=256, clip ε=0.2, entropy coeff=0.01 |
| RL training script | `scripts/train_rl.py` | CLI entry point, loads IL checkpoint |

### Episode-level changes for RL
- **Burn-in period**: first 300 s of each episode uses fixed-time control so the network reaches steady state before the GNN starts acting
- **Demand randomization**: already implemented in `TrafficEnv` — on by default, each `reset()` generates a fresh temp routes XML via `DemandRandomizer`; pass `demand_seed=<int>` for reproducible runs

### TensorBoard additions for RL
- Per-step: `loss/policy`, `loss/value`, `loss/entropy`, `loss/total`
- Per-episode: `episode/mean_reward`, `episode/value_loss`, `episode/policy_loss`
- Eval charts: same schema as IL eval, with RL model replacing the IL model
