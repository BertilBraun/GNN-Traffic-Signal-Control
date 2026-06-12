# GNN-Based Traffic Signal Control

## Project Goal

Train a Graph Neural Network to control traffic signals across a city-wide road network. The GNN ingests per-junction sensor data, propagates information across the junction graph via message passing, and outputs the next signal phase for every junction simultaneously. The system is trained first via imitation learning, then reinforced with RL to minimize waiting times.

---

## 1. Graph Representation

### Nodes — Signalized Junctions

Every signalized intersection is a node. Each node carries a feature vector derived from per-movement sensor data plus junction-level state.

### Edges — Road Segments

Every road segment connecting two signalized junctions is a directed edge (one per direction if the road is two-way). Both directions are included in the message-passing graph: the edge from upstream junction A to downstream junction B carries "traffic heading toward B", and the reverse edge B → A carries "B's downstream state, which constrains A's outflow". The GNN learns to weight both via attention. Edges carry static features that inform the message-passing step.

### Edge Features (static)

| Feature         | Description                    | Source        |
| --------------- | ------------------------------ | ------------- |
| Road length     | Distance between junctions (m) | SUMO net file |
| Number of lanes | Lane count in this direction   | SUMO net file |
| Speed limit     | Max allowed speed (m/s)        | SUMO net file |

These are concatenated onto messages during GNN propagation.

---

## 2. Junction Model

### Supported Junction Types

- **4-way intersections** — 4 approaches × 3 movements = 12 movements
- **3-way intersections** — 3 approaches × 3 movements = 9 active movements, 3 zero-padded

Any junction with 5+ arms is excluded for now.

### Canonical Approach Ordering

Real junctions aren't compass-aligned, so the labels "N/S/E/W" are replaced with deterministic slot indices 0–3 assigned per junction by a purely geometric rule:

1. Take each incoming edge's approach bearing into the junction from the SUMO net geometry.
2. **4-way junctions**: sort edges clockwise by bearing and assign slots 0, 1, 2, 3 in that order. Slot 0 and slot 2 are always the opposing pair; same for slots 1 and 3.
3. **3-way junctions**: identify the two arms whose bearings are closest to 180° apart (the "straight-through" pair) and assign them slots 0 and 2. The remaining arm (the "stem") goes to slot 1. Slot 3 is empty and zero-padded.

Because the rule is purely geometric, it produces the same slot assignment for any junction of the same shape regardless of SUMO's internal link ordering. This is critical for transfer: once the GNN learns "slot 0 has heavy queue + slot 2 has heavy queue → pick the phase that serves slots 0/2 through", that knowledge applies to every junction, anywhere.

### Movement Decomposition

For each of the 4 slots, three movements are defined:

- **Left turn**
- **Through**
- **Right turn**

This gives a fixed 12-slot movement vector per junction, regardless of geometry. 3-way junctions zero-pad the empty slot's 3 movement features.

---

## 3. Input Features

### Per-Movement Features (12 movements × F features)

| Feature             | Description                                             | SUMO Source                                                 |
| ------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| Queue density       | Stopped vehicles per meter of detector (veh/m)          | `traci.lane.getLastStepHaltingNumber` / `detector_length_m` |
| Approaching density | Vehicles heading toward this movement per meter (veh/m) | Lane-area detector count / `detector_length_m`              |
| Wait density        | Accumulated waiting time per meter of detector (s/m)    | `traci.lane.getWaitingTime` / `detector_length_m`           |

With F=3 features per movement and 12 movements, this gives a 36-dimensional per-movement block.

**Why density instead of raw counts?** Detectors are nominally 200m but are clipped to the actual road segment length on short blocks (see §9). A 200m detector sees ~10× more "approaching vehicles" than an 80m detector even under identical traffic density — raw counts would create systematic distribution shift between long-block and short-block junctions, which global z-score normalization can't fix (the distributions are genuinely different, not just offset). Dividing by detector length up front makes every feature length-invariant by construction; the GNN sees the same density regardless of how much road each detector actually covers. Z-score normalization on top of the density features is then clean and the model transfers across networks with different block spacings.

### Junction-Level Features

| Feature                 | Dim | Description                               |
| ----------------------- | --- | ----------------------------------------- |
| Current phase (one-hot) | 4   | Which phase is currently active           |
| Elapsed time in phase   | 1   | Normalized (elapsed / max_phase_duration) |

### Total Node Feature Vector

36 (movements) + 4 (phase one-hot) + 1 (elapsed time) = **41 dimensions**

### Normalization

All features are z-score normalized using running statistics accumulated during IL training stages, then **frozen** for all RL stages. Freezing prevents the effective feature scale from drifting during PPO updates, which would silently alter the reward magnitude and destabilize training. Using a single globally shared set of stats (rather than per-junction stats) is what makes transfer across networks of different sizes and traffic volumes work — every junction sees the same normalization regardless of where it is.

---

## 4. Phase Definitions

All four phases are defined as sets of compatible (non-conflicting) movements, indexed against the canonical slot ordering from Section 2:

| Phase | Green Movements                                        | Description                      |
| ----- | ------------------------------------------------------ | -------------------------------- |
| 0     | slot0-through, slot0-right, slot2-through, slot2-right | Slots 0/2 through + right        |
| 1     | slot0-left, slot2-left, slot1-right, slot3-right       | Slots 0/2 left + slots 1/3 right |
| 2     | slot1-through, slot1-right, slot3-through, slot3-right | Slots 1/3 through + right        |
| 3     | slot1-left, slot3-left, slot0-right, slot2-right       | Slots 1/3 left + slots 0/2 right |

This is the same standard NEMA-style 4-phase scheme, just expressed against geometric slot indices instead of compass directions.

### 3-Way Junction Handling

Slot 3 is empty (zero-padded). All 4 phases remain selectable: phases that would have served slot 3 simply contribute fewer green movements but are still meaningful options. For example, phase 2 (which normally serves slots 1 and 3 through+right) becomes "slot 1 through+right only" on a 3-way, but it's still the right choice when slot 1 has waiting traffic.

Masking is therefore only needed if a phase has zero green movements after pruning, which doesn't happen for 3-way junctions in this scheme. The output is always a 4-class softmax for both 3-way and 4-way junctions.

### Transition Logic

The GNN outputs a **target phase** every decision interval. The actual signal controller handles the interphase transition:

1. If target phase ≠ current phase → transition via yellow (3s) → all-red (2s) → new green
2. If target phase = current phase → extend current green
3. Minimum green time: 10 seconds (enforced by the controller, not the GNN)

This keeps the GNN output simple — just a phase classification — while the transition controller ensures safety and realism.

---

## 5. Output Representation

The GNN produces per-node output: a **4-class softmax** over the 4 canonical phases. Both 3-way and 4-way junctions use the same 4-class output (see Section 4 for why 3-way doesn't need masking under this scheme).

The predicted class is the target phase for the next decision interval.

---

## 6. Decision Interval

**15 seconds** between GNN decisions.

Timing arithmetic for a phase switch:

- Yellow phase: 3s
- All-red: 2s
- New green: at least 10s (minimum green)
- Total: 5s transition + 10s green = exactly 15s, one decision interval

So if the GNN switches at decision time t, the new phase is established by t+15s, and the next decision is also at t+15s. Minimum green is satisfied by construction — the GNN never sees a state where it could violate it. If the GNN does not switch (extends current green), the decision interval simply elapses and the next decision happens 15s later.

The fastest possible full 4-phase cycle is 4 × 15s = 60s, which is realistic. SUMO simulation steps can be finer-grained (e.g., 1s); the GNN is only queried every 15 simulation seconds.

---

## 7. GNN Architecture

### Overview

```
Per-node MLP encoder → 3 × GATv2 layers → Per-node MLP classifier
```

### Encoder

- Input: 41-dim node feature vector
- MLP: 41 → 128 → 128 (ReLU, LayerNorm)

### Message Passing: 3 × GATv2 Layers

- Hidden dim: 128
- Heads: 4 (multi-head attention)
- Edge features (3-dim) projected and added to attention coefficients
- Residual connections between layers
- LayerNorm after each layer

3 hops means each junction's decision is informed by junctions up to 3 edges away.

### Classifier Head

- MLP: 128 → 64 → 4 (per-node)
- Masked softmax for junctions with fewer valid phases

### Framework

- **PyTorch Geometric** for graph neural network layers
- `torch_geometric.data.Data` for graph construction
- `GATv2Conv` for attention-based message passing

---

## 8. Training

### Expert Controller for Imitation Learning

Standard SUMO actuated controllers use their own phase programs, which generally don't align with the GNN's fixed 4-phase scheme. To produce labels that map cleanly to the GNN's output space, IL stages use a custom expert controller written against the same canonical phase set defined in Section 4:

1. At each decision interval (15s), score each of the 4 phases by the total accumulated waiting time on the lanes that phase would serve (using `traci.lane.getWaitingTime`).
2. Pick the highest-scoring phase as the target.
3. If the picked phase is the same as the current phase, hold (extend green).
4. Otherwise, switch via the standard yellow → all-red → new green transition.
5. Cap the maximum hold time at 3 consecutive decision intervals (45s) to prevent starvation — after that, force a switch to the next-highest-scoring phase.
6. Optional refinement: switch early (mid-interval) if the served lanes' total queue drops to zero AND another phase has waiting traffic. Skipped initially for simplicity; the controller just re-decides every 15s.

This controller is greedy and purely local. It's not optimal, but it (a) produces well-defined (state, phase) labels in the GNN's output space, and (b) gives the GNN a sensible warm start before RL.

### Imitation Learning — Irregular Multi-Junction Network

**Goal:** Validate that the full GATv2 pipeline runs end-to-end on a multi-junction network without producing pathologies (gridlock, starved approaches), before introducing RL. This is *not* a coordination test — the per-junction expert is purely local, so coordination only emerges in the RL stages.

- Start from a 4×4 grid of signalized junctions, then remove a few nodes and edges to break the regularity. This exposes the GNN to varied local topologies and exercises the canonical slot ordering on more than perfect grids.
- Use the same custom expert controller defined above, applied independently per junction.
- Train the full GATv2-based GNN with parameter sharing across all junctions.
- Metric: per-junction match rate, global average waiting time vs expert, and a sanity check that no junction develops unbounded queues.

### Episode Initialization — Burn-In Period

Each RL episode starts with an empty network, which is unrepresentative of steady-state traffic. To ensure the GNN always begins from a realistic state, each episode runs a burn-in period with the fixed-time baseline controller before handing control to the GNN. Transitions collected during burn-in are discarded.

- Stage 3 (1200s episode): **300s burn-in**
- Stages 4–5 (3600s episode): **600s burn-in**

Evaluation episodes use the same burn-in so metrics are comparable to training conditions.

### PPO Hyperparameters

| Parameter           | Value          | Notes                                        |
| ------------------- | -------------- | -------------------------------------------- |
| Discount γ          | 0.99           | ~100-step effective horizon ≈ 1500s          |
| GAE λ               | 0.95           | Advantage estimation smoothing               |
| Clip ε              | 0.2            | Maximum policy update ratio per step         |
| Epochs per update K | 10             | Passes over the rollout buffer per iteration |
| Minibatch size      | 256            | Stage 4: ~15 minibatches/epoch               |
| Learning rate       | 3e-4           | Adam, shared for actor and critic            |
| Value loss coeff    | 0.5            | Relative weight of critic loss term          |
| Entropy coeff       | 0.01           | Exploration bonus                            |
| Max gradient norm   | 0.5            | Gradient clipping                            |
| Episodes per update | 1 (Stage 3: 4) | Stage 3 has only ~80 transitions/ep at 1200s |

### Training Loop

PPO alternates between two phases:

1. **Collect** — run N episodes with the current policy. Store `(graph, joint_action, per-junction reward, per-junction value, log_prob)` per decision step. No weight updates during collection.
2. **Update** — compute GAE advantages over the full buffer, then run K=10 epochs of PPO gradient updates with minibatch sampling. Discard the buffer.
3. **Evaluate** — every 50 iterations, run 5 evaluation episodes (burn-in applied, no weight updates) and log all metrics to TensorBoard.
4. Repeat.

The buffer is thrown away after each update because PPO is **on-policy** — training on data collected under an old policy would produce incorrect advantage estimates. This is unlike DQN's replay buffer.

**Expected updates to convergence (from IL warm start):**

| Stage                              | Updates | Estimated wall-clock |
| ---------------------------------- | ------- | -------------------- |
| Stage 3 — single junction          | 100–300 | 15–50 min            |
| Stage 4 — 16 junctions             | 300–800 | 1–2 hrs              |
| Stage 5 — city network (fine-tune) | 200–600 | 30 min–2 hrs         |

"Convergence" = consistently outperforming the greedy IL expert on held-out demand levels. Beating SUMO's actuated controller is the harder target and may take 2–3× more updates.

From random initialization, Stage 4 would need 2000–5000+ updates (~6–15 hrs) and is prone to multi-agent instability due to all agents simultaneously changing their policies. The IL warm start is not optional.

### Stage 3: RL — Single 4-Way Junction

**Goal:** Validate RL training loop.

- PPO (Proximal Policy Optimization)
- Reward per junction per decision step: `r_i = −α·Δlocal_wait_i − β·Δglobal_wait − γ·switch_i`
  - `α = 1.0` — local waiting time change at junction i (primary signal)
  - `β = 0.1` — global waiting time change across the whole network (near-zero in Stage 3 since there's only one junction; matters from Stage 4 onward)
  - `γ = 0.5` — penalty applied when junction i's phase switches (anti-flickering)
  - `local_wait_i(t)` = sum of `traci.lane.getWaitingTime` over all incoming lanes at junction i, **divided by the sum of those lanes' detector lengths in meters** (i.e. wait-seconds per meter of approach). This is the *currently-on-lane accumulated wait density*, so it grows while cars sit at red and drops sharply when long-waiting cars finally clear the lane. `Δlocal_wait_i = local_wait_i(t) − local_wait_i(t − 15s)` over one decision interval. A phase that clears a built-up queue produces a strongly negative Δ → strongly positive reward, which is the desired training signal. Length-normalizing keeps the reward scale comparable across junctions with different detector lengths, which matters for parameter sharing in Stages 4–5 (without it, long-block junctions would produce larger raw rewards and dominate the shared gradient).
  - `global_wait(t)` = same metric averaged across all incoming lanes in the network (sum of waiting times divided by sum of detector lengths).
- Initialize policy from Stage 1 weights (warm start)
- Compare against fixed-time and actuated baselines

### Multi-Agent Setup (Stages 4 and 5)

With multiple junctions, the rollout and PPO update structure needs to be explicit.

**Setup:** Independent PPO with parameter sharing and graph-encoded observations.

- **Observation**: the full graph (all node features + edge features) is encoded once per decision step by the shared GNN. Each junction's per-node embedding is its observation; message passing means each embedding implicitly carries neighbor information up to 3 hops away.
- **Action space per junction**: categorical over 4 phases. The joint action across N junctions factorizes as the product of independent per-junction categoricals; the joint log-probability is the sum of per-junction log-probabilities.
- **Reward per junction**: `r_i = −α·Δlocal_wait_i − β·Δglobal_wait − γ·switch_i` as defined in Stage 3. The local term gives each junction direct credit for its own queue dynamics; the global term provides shared coordination pressure.
- **Value function**: a per-junction scalar `V_i(state)`, produced by a small head on top of the shared GNN encoder. Each junction has its own value estimate, but encoder weights are shared with the actor.
- **Advantage estimation**: GAE computed per junction over its own reward stream.
- **PPO loss**: standard clipped objective, averaged across all junctions and timesteps in the rollout buffer. With N junctions and T decision steps per episode, one episode contributes N·T transitions to the policy update.
- **Rollout buffer**: stores (graph, joint action, per-junction rewards, per-junction values) tuples.

This is the standard "independent learners with parameter sharing" multi-agent setup, with the twist that the GNN encoder lets each agent observe its neighbors before acting. It's simpler than CTDE (no separate centralized critic) and works well precisely because the policy network already has access to neighbor state via message passing.

### Stage 4: RL — Irregular Multi-Junction Network

**Goal:** Multi-agent coordination via shared GNN policy.

- Same network used in Stage 2 (irregular grid)
- Same reward structure as Stage 3, but now β matters because global coordination is relevant
- All junctions share the same GNN parameters (parameter sharing); see Multi-Agent Setup above
- Initialize from Stage 2 weights
- Potentially adjust α/β ratio to emphasize coordination

### Stage 5: RL — City Network

**Goal:** Transfer and scale.

- Import a real city-center neighborhood from OpenStreetMap into SUMO. Hand-pick the area so all signalized intersections are 3-way or 4-way (no 5+ arm junctions). Slip lanes are unsignalized and don't enter the graph.
- Apply the trained GNN directly (zero-shot transfer test)
- Fine-tune with RL on the real network
- Compare against SUMO's built-in actuated controller

---

## 9. Simulation Setup — SUMO

### Tools

- **SUMO** (Simulation of Urban MObility) — traffic simulation engine
- **traci** — Python API to step simulation, read state, set signals
- **sumolib** — parse/build network files
- **netedit** or **netgenerate** — create synthetic networks
- **osmWebWizard** or **netconvert** — import OpenStreetMap data

### Network Files

- `.net.xml` — road network (junctions, edges, lanes, signal plans)
- `.rou.xml` — traffic demand (vehicle routes, departure times)
- `.sumocfg` — simulation config (references net + routes)
- `.add.xml` — additional files (detectors, output definitions)

### Detectors

Lane-area detectors (E2 detectors) placed on every incoming lane, nominally covering 200m upstream of each junction. On dense city blocks where two junctions are <200m apart, detectors are clipped to the actual road segment length (the full lane minus a small buffer at the upstream junction) to avoid double-counting the same vehicle in two junctions' detectors.

Because clipped and full-length detectors see different raw counts under identical traffic density, all count-based features and the reward's wait signal are converted to **per-meter densities** before reaching the GNN (see §3 Per-Movement Features and §8 Stage 3 reward). This keeps feature and reward scales length-invariant and preserves transfer across networks with different block spacings.

### Demand Generation

**Synthetic networks (Stages 1–4):** routes generated via SUMO's `randomTrips.py` with varied flow rates. The anchor is the saturation flow rate for a single lane under a green phase (~1800 veh/h/lane theoretical max). Practical bands:

- **Light** — 200–400 veh/h/lane — queues form rarely, controller choice barely matters
- **Medium** — 500–800 veh/h/lane — regular queueing, controller differences clearly visible
- **Heavy** — 900–1200 veh/h/lane — sustained queues, coordination matters, still clearable by a good controller
- **Above ~1300 veh/h/lane** — gridlock territory where even an optimal controller can't clear the queue; avoided during training

**Default training mix:** per input edge, sample flow uniformly from **300–1000 veh/h/lane** across episodes. This covers light through heavy-but-solvable conditions and prevents overfitting to any one traffic regime.

**City network (Stage 5):** demand is sourced from SUMO's OSM tooling directly. `randomTrips.py` generates vehicles from edge distributions; `activitygen` builds realistic population-based OD matrices from OSM population data. Episode-to-episode variation is then just a scaling multiplier (0.5×–1.5×) applied to the base demand — no custom demand generation code is needed for the real-network stage.

### Episode Length

One training episode = **3600 simulation seconds** (1 hour of simulated traffic) = 240 GNN decisions per junction at 15s/decision. This isn't an arbitrary round number — it sits at the intersection of several constraints:

1. **Transit time.** A vehicle must be able to enter, traverse, and exit the network within the episode, otherwise vehicles trapped in end-of-episode queues bias the waiting-time metric toward queue build-up (the policy never gets credit for clearing them). On a 4×4 grid (~600m across) at ~10 m/s that's ~60s of transit; on a city network closer to ~200s. 3600s is >> both.
2. **Cycle count.** Queues build and clear on the order of one signal cycle (60s minimum). To observe steady-state policy behavior you want tens of cycles per episode — 3600s gives 60 minimum-length cycles.
3. **PPO effective horizon.** With γ=0.99 the discounted horizon is ~100 steps. At 15s/decision that's 1500 sim seconds. The episode should be at least as long as the horizon, otherwise bootstrapped values at episode boundaries absorb a large fraction of the return and credit assignment degrades. 3600s > 1500s.
4. **Warm-up fraction.** The network starts empty, so the first few minutes of each episode are unrepresentative. For 3600s, warm-up is ~10–15% of the episode and steady-state is 85%+.
5. **Wall-clock cost.** Longer episodes mean more transitions per rollout but slower iteration. SUMO headless runs 1h of sim time in a few seconds of wall-clock, so 3600s is cheap.

**Per-stage tuning:** Stage 1 (single junction, no transit) can use 1200s for faster dev iteration. Stage 2 onward defaults to 3600s for consistency. Stage 5 (city network) may need 5400–7200s if the network is large enough that 1h of sim doesn't cover a full rush-hour pattern.

---

## 10. Evaluation Metrics

| Metric                 | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| Average waiting time   | Mean waiting time per vehicle across all vehicles (primary) |
| Average travel time    | Mean trip duration across all completed trips               |
| Throughput             | Number of vehicles completing their trip per hour           |
| Max queue length       | Worst-case queue on any approach (safety/fairness metric)   |
| Phase switch frequency | How often the GNN changes phases (stability metric)         |

### Baselines

1. **Fixed-time controller** — pre-set cycle with equal phase splits
2. **Webster's optimal fixed-time** — phase splits optimized by traffic volume
3. **SUMO actuated controller** — detector-triggered phase extensions
4. **Independent agent** — per-junction MLP without message passing (ablation)

Baseline 4 isolates the value of the GNN's spatial message passing.

---

## 11. Transfer Learning Hypothesis

Because all junctions share the same GNN weights, the model learns a general function: *given my local state and my neighbors' messages, which phase should I pick?* This function is:

- **Inductive over graph structure** — works on unseen topologies
- **Invariant to network size** — no fixed-size graph assumption
- **Robust to junction type** — 3-way and 4-way handled via masking

Expected transfer path: train on synthetic grids with varied topology → apply zero-shot to real city network → fine-tune if needed.

To maximize transferability:

- Normalize all features (z-score with running stats)
- Train on diverse junction types and traffic patterns
- Include irregular topologies during training (not just perfect grids)

---

## 12. Visualization

### Training & Evaluation Dashboards (TensorBoard)

All metrics are logged to TensorBoard, organized by training stage.

**RL-specific logs (per training step / episode):**

| Log                         | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `reward/mean_episode`       | Mean episode reward (primary training signal)               |
| `reward/local_wait_term`    | Decomposed local waiting time component                     |
| `reward/global_wait_term`   | Decomposed global waiting time component                    |
| `reward/switch_penalty`     | Decomposed switching penalty component                      |
| `loss/policy`               | PPO policy loss                                             |
| `loss/value`                | PPO value function loss                                     |
| `loss/entropy`              | Entropy bonus (exploration health indicator)                |
| `policy/phase_distribution` | Histogram of selected phases across all junctions           |
| `policy/switch_rate`        | Fraction of decision steps where the phase actually changes |
| `policy/mean_green_time`    | Average duration a phase is held before switching           |

**IL-specific logs:**

| Log                    | Description                                |
| ---------------------- | ------------------------------------------ |
| `loss/cross_entropy`   | Classification loss against expert actions |
| `accuracy/phase_match` | Per-step match rate with expert controller |

**Evaluation logs (logged every N episodes or on explicit eval runs):**

| Log                      | Description                              |
| ------------------------ | ---------------------------------------- |
| `eval/avg_waiting_time`  | Average waiting time per vehicle         |
| `eval/avg_travel_time`   | Average trip duration                    |
| `eval/throughput`        | Vehicles completing trips per hour       |
| `eval/max_queue_length`  | Worst-case queue across all approaches   |
| `eval/phase_switch_freq` | Average switches per junction per minute |

**Per-junction breakdown (logged less frequently, e.g., every 50 episodes):**

| Log                         | Description                            |
| --------------------------- | -------------------------------------- |
| `junctions/{id}/avg_wait`   | Per-junction average waiting time      |
| `junctions/{id}/max_queue`  | Per-junction worst queue               |
| `junctions/{id}/phase_hist` | Per-junction phase selection histogram |

The per-junction logs are critical for spotting degenerate behavior — e.g., a single junction stuck in one phase while neighbors compensate, which might look fine in aggregate metrics but is clearly wrong.

### Traffic Simulation Visualization (SUMO-GUI)

SUMO ships with `sumo-gui`, a graphical frontend to the simulation engine. It renders the full network in real time: vehicles, signal states, queues, and lane utilization, all controllable via the same traci interface used for headless training.

**Usage during development and evaluation:**

- Run evaluation episodes with `sumo-gui` instead of `sumo` (single flag change)
- Vehicles can be color-coded by waiting time, speed, or route
- Signal heads show current phase with standard red/yellow/green rendering
- Zoom into individual junctions or view the full network
- Adjustable simulation speed (slow down to inspect, speed up to scan)

**Recording for portfolio / debugging:**

- SUMO-GUI supports screenshot export per simulation step
- Combine frames into video using ffmpeg: `ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 out.mp4`
- Alternatively, use screen recording during a live evaluation run
- Record side-by-side comparisons: fixed-time vs actuated vs GNN controller on the same demand scenario

**What to look for visually:**

- Queues growing unboundedly on one approach (starvation — reward or phase issue)
- Rapid signal flickering (missing or too-low switch penalty)
- All junctions synchronized in lockstep (over-coordination, lack of local adaptation)
- Green waves forming on arterial roads (desirable emergent behavior)
- Vehicles clustering at network boundaries (demand/routing issue, not a model problem)

---

## 13. Known Simplifications

- **No pedestrian signals** — pedestrian phases are ignored
- **No dedicated bus/tram lanes** — only car traffic
- **No right-on-red** — right turns only proceed during green phases
- **Max 4-way junctions** — 5+ arm intersections excluded; for the city stage, a sub-network is hand-picked to contain only 3-way and 4-way signalized intersections
- **No lane-change modeling in features** — movements aggregated per approach direction
- **Deterministic transitions** — yellow/all-red timing is fixed, not learned
- **No communication delay** — GNN decisions are applied instantly
- **Ground-truth turn intent** — "approaching vehicles per movement" is computed from `vehicle.getRoute` rather than from realistic upstream detectors, which can't observe intended turns directly. This is a sim-only signal; real-world deployment would need a different proxy (e.g., turn-lane occupancy)

---

## 14. Project Structure (Planned)

```
traffic-gnn/
├── configs/               # SUMO network + route configs
│   ├── single_junction/
│   ├── grid_4x4/
│   └── city/
├── src/
│   ├── environment/       # SUMO wrapper, step logic, reward
│   │   ├── sumo_env.py
│   │   ├── signal_controller.py
│   │   └── feature_extractor.py
│   ├── model/             # GNN architecture
│   │   ├── gat_policy.py
│   │   └── modules.py
│   ├── training/          # IL and RL training loops
│   │   ├── imitation.py
│   │   └── ppo.py
│   └── utils/             # Graph construction, normalization, logging
│       ├── graph_builder.py
│       ├── metrics.py
│       └── tb_logger.py   # TensorBoard logging helpers
├── scripts/               # Entry points
│   ├── train_il.py
│   ├── train_rl.py
│   ├── evaluate.py        # Headless evaluation with metric export
│   └── evaluate_gui.py    # Evaluation with SUMO-GUI visualization
├── notebooks/             # Exploration, visualization
└── README.md
```

---

## 15. Dependencies

- Python 3.10+
- SUMO (+ traci, sumolib)
- PyTorch
- PyTorch Geometric
- CleanRL (for PPO) — standard SB3 does not support the multi-agent GNN setup; CleanRL's ~200-line PPO implementation is the recommended starting point for customization, or write a custom training loop directly
- TensorBoard or Weights & Biases (logging)
- ffmpeg (for recording SUMO-GUI evaluation videos)
- NetworkX (optional, for graph visualization)
- OSMnx (optional, for OpenStreetMap import)