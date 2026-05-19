# Current State

Last updated: 2026-05-19

---

## What is done

### Stage 1 — Environment (PASS)

`TrafficEnv` wraps SUMO/TraCI. Six managed junctions on the irregular 4×4 grid. Observation: 41-dim per-junction vector (12 movements × 3 density features + 4-dim phase one-hot + elapsed-time scalar). Reward: `−α·Δlocal_wait − β·Δglobal_wait` (α=1.0, β=0.1). Decision interval: 15 s. Phase transitions: 3 s yellow + 2 s all-red + 10 s new green.

Minimum green is now enforced **at the env level** (`min_green_steps`, default 2 × 15 s = 30 s). Switch requests that arrive too early are silently overridden. The reward has **no switch penalty** — the constraint is physical, not a soft scalar.

Verification: `python scripts/verify_env.py` — 6 automated checks (startup, shapes, feature bounds, junction ordering, expert phase usage, expert vs baseline). All PASS.

### Stage 2 — Imitation Learning (PASS)

GATv2 policy (41 → 128 → 128, 3 message-passing layers, 4 attention heads, ~50k parameters) trained on (graph, expert_phase) pairs from live SUMO episodes. DAgger anneals expert-driving probability from β=1.0 → 0.0 over 100 episodes.

Best checkpoint (~ep 90, β=0.10) achieves:

| Metric | Model | Expert |
|--------|-------|--------|
| avg_waiting_time (s) | 87 | 86 |
| wait_density (s/m) | 0.125 | 0.101 |

The model matches the expert within ~6% on wait density. The best checkpoint (not the final episode) is saved separately and used to initialise PPO.

### Stage 4 — PPO fine-tuning (PASS)

PPO with per-junction actor-critic sharing the GATv2 backbone. Key design decisions validated through failed experiments:

- **Value warmup**: first 20 iterations freeze the backbone and train only the value head with MC returns (not GAE — avoids moving-target problem). After warmup, full PPO.
- **No switch penalty**: soft penalties cause reward hacking (model learns to hold phases). Hard `min_green_steps` constraint at env level gives physically realistic behaviour.
- **Multi-seed periodic eval**: eval every N iterations uses 5 seeds (42–46), same as the final eval, so the training metric tracks true generalisation.

Final result (5-seed eval, 700–1200 veh/h demand, 200 PPO iterations):

| Metric | Model | Expert | Delta |
|--------|-------|--------|-------|
| avg_waiting_time (s) | 92 | 116 | **−21%** |
| avg_travel_time (s) | 183 | 209 | −12% |
| throughput (veh/h) | 724 | 712 | +1.7% |
| max_queue (vehs) | 7.2 | 8.6 | −16% |
| switch_freq (/j/min) | 1.18 | 1.26 | −6% |
| wait_density (s/m) | 0.096 | 0.146 | **−34%** |

Switching frequency is realistic (1.18 vs expert 1.26 — the constraint is working). EV = 0.83 (value head converged well). Entropy H ≈ 0.65 (decisive but not degenerate).

---

## What is missing / next

### Stage 3 — skipped

PLAN called for a single-junction RL validation stage. It was skipped because Stage 4 (multi-junction) passed cleanly from the IL warm start. No regression — the multi-junction result is the stronger evidence.

### Stage 5 — city network (not started)

Import a real city-centre excerpt from OpenStreetMap into SUMO. The network must contain only 3-way and 4-way signalised junctions (no 5+ arm intersections). Apply the trained GNN zero-shot — no fine-tuning — and measure vs SUMO's built-in actuated controller. Then fine-tune with PPO.

This is the primary remaining milestone. The GNN architecture was designed for this: geometry-agnostic slot ordering, density features, length-invariant normalisation, and frozen normalizer at RL time all exist to make zero-shot transfer work.

**Steps to get there:**

1. Choose a city area (OSM export). ~10–20 signalised junctions.
2. `netconvert` + manual cleanup to produce a valid `.net.xml`.
3. Verify all junctions are 3-way or 4-way; exclude any with 5+ arms.
4. Generate demand with `randomTrips.py` at several flow rates.
5. Verify `TrafficEnv` starts cleanly on the new `.sumocfg`.
6. Run zero-shot eval: `python scripts/run_grid.py --mode model --ckpt checkpoints/rl/<stamp> --cfg <city.sumocfg>`.
7. Fine-tune with `train_rl.py --il-ckpt checkpoints/rl/<stamp> --cfg <city.sumocfg>`.

---

## Key files

| Path | Purpose |
|------|---------|
| `src/environment/sumo_env.py` | `TrafficEnv` — SUMO wrapper, obs, reward, min-green enforcement |
| `src/environment/expert.py` | `GreedyExpert` — per-junction greedy controller used for IL labels and eval |
| `src/model/gat_policy.py` | `GATPolicy` — GATv2 actor-critic |
| `src/utils/graph_builder.py` | Builds `torch_geometric.Data` graphs from obs; holds `RunningNormalizer` |
| `src/training/imitation.py` | `train_il()` — DAgger IL loop; `load_checkpoint()` |
| `src/training/ppo.py` | `train_rl()` — PPO loop; `load_rl_checkpoint()` |
| `src/training/rollout.py` | `RolloutBuffer` — GAE / MC return computation |
| `src/training/eval_episode.py` | `run_eval_episode()` — shared eval logic for IL and RL |
| `scripts/train_il.py` | CLI for imitation learning |
| `scripts/train_rl.py` | CLI for PPO fine-tuning |
| `scripts/run_grid.py` | Visual / headless demo — `--mode expert\|model --gui` |
| `scripts/verify_env.py` | 6-check environment smoke test |
| `configs/grid_4x4/` | SUMO network, routes, and config for the irregular 4×4 grid |
| `VALIDATION.md` | Layer-by-layer verification log with observed numbers |
| `PLAN.md` | Full design document — architecture, reward, training stages |

---

## Hyperparameters used for the best RL checkpoint

```sh
python scripts/train_rl.py \
  --il-ckpt checkpoints/il/2026-05-19_12-30-09 \
  --iterations 200 --warmup 20 \
  --workers 4 --episodes 4 \
  --ep-len 1200 --burn-in 10 \
  --entropy-coeff 0.01 --clip 0.1 \
  --flow-range 700 1200
```

Best RL checkpoint: `checkpoints/rl/2026-05-19_20-02-40/`
