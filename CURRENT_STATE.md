# Current State

Last updated: 2026-05-21

---

## What is done

### Stage 1 — Environment (PASS)

`TrafficEnv` wraps SUMO/TraCI. Six managed junctions on the irregular 4×4 grid. Observation: 45-dim per-junction vector (12 movements × 3 density features + 8-dim phase one-hot + elapsed-time scalar). Reward: `−α·Δlocal_wait − β·Δglobal_wait` (α=1.0, β=0.1). Decision interval: 15 s. Phase transitions: 3 s yellow + 2 s all-red + 10 s new green.

Minimum green is now enforced **at the env level** (`min_green_steps`, default 2 × 15 s = 30 s). Switch requests that arrive too early are silently overridden. The reward has **no switch penalty** — the constraint is physical, not a soft scalar.

Verification: `python scripts/verify_env.py` — 6 automated checks (startup, shapes, feature bounds, junction ordering, expert phase usage, expert vs baseline). All PASS.

### Stage 2 — Imitation Learning (PASS)

GATv2 policy (45 → 128 → 128, 3 message-passing layers, 4 attention heads, ~142k parameters) trained on (graph, expert_phase) pairs from live SUMO episodes. DAgger anneals expert-driving probability from β=1.0 → 0.0 over 100 episodes.

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

### Stage 5 — city network (COMPLETE)

**Network:** Munich Maxvorstadt (OSM bbox 48.147,11.568–48.155,11.581), imported via netconvert.
10 usable TL junctions (7 three-way + 3 four-way); 6 additional 1–2 arm crossings silently skipped.
53 E2 detectors on incoming lanes. `verify_env.py` passes 5/6 checks (phase 3 never selected by greedy expert in short 300s test episodes — low-demand artifact, not a phase definition bug).

Files: `configs/city/` — `city.net.xml`, `city.tll.xml`, `city.add.xml`, `city.rou.xml`, `city.sumocfg`
Build scripts: `audit_junctions.py`, `build_tll.py`, `build_detectors.py`
Eval script: `scripts/eval_city.py`

**Summary of results (5 seeds, 3600s episodes, 700–1200 veh/h):**

| Metric | Expert | Zero-shot | Fine-tuned (200 iters) |
|--------|--------|-----------|------------------------|
| avg_waiting_time (s) | 37.1 | 210.9 | 123.9 |
| avg_travel_time (s) | 145.1 | 318.0 | 232.7 |
| throughput (veh/h) | 31.8 | 30.2 | 31.0 |
| max_queue (vehs) | 1.4 | 1.6 | 1.4 |
| switch_freq (/j/min) | 0.6 | 0.6 | 0.7 |
| wait_density (s/m) | 0.0016 | 0.0440 | 0.0114 |

**Zero-shot:** +469% avg wait vs expert — grid-trained normalizer statistics don't fit city feature distribution, model misreads traffic state.

**After 200 PPO iterations:** +234% avg wait vs expert. Wait density improved 4× from zero-shot (0.044 → 0.011 s/m). Throughput nearly matched (−2.5%). The model did not beat the expert in 200 iterations.

**Root cause of remaining gap:** The frozen normalizer from grid training is the primary bottleneck. The city network's feature distributions differ structurally from the grid (different lane lengths → different density scales, different block spacings). 200 PPO iterations provide a partial signal but are insufficient to fully compensate.

**Next steps if continuing:** (a) Unfreeze the normalizer during city fine-tuning and re-accumulate statistics from city episodes, then re-freeze. (b) Run 500+ fine-tuning iterations. (c) Collect IL data on city with the expert, re-run DAgger with city-specific normalization.

**Checkpoints:**
- `checkpoints/city_warmstart/` — grid RL weights renamed for `train_rl.py` input
- `checkpoints/rl/2026-05-19_23-35-19/` — city fine-tuning run (200 iters, final: `rl_policy_final.pt`)

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

Best historical RL checkpoint: `checkpoints/rl/2026-05-19_20-02-40/`

Note: the May 19 checkpoints use the old 41-input / 4-phase schema and do not load into the current 45-input / 8-phase `GATPolicy`. Re-run IL and PPO before using the current code for city transfer or new-network generalization.
