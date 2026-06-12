# Movement-Based GNN Traffic Signal Control Roadmap

This document records the current state of the movement-based traffic signal controller and defines the updated development path toward movement-score imitation learning, graph neural network control, PPO fine-tuning, generated-grid transfer, OSM/city-network integration, and transfer evaluation.

The central design decision is now fixed:

```text
The learned model scores movements.
Valid phases are generated deterministically.
Phase scores are computed by aggregating movement scores.
The controller enforces all signal legality and transition constraints.
```

The new learning architecture uses a directed bipartite graph:

```text
LaneGroup ↔ Movement ↔ LaneGroup
```

A `LaneGroup` represents a directed road segment between signalized junctions.
A `Movement` represents one legal signal-controlled flow from an incoming `LaneGroup` to an outgoing `LaneGroup`.

The old fixed canonical 8-phase architecture is retained only as legacy reference material.

---

## 1. Project Goal

The project aims to train a neural traffic signal controller for signalized city networks.

The learned controller should improve over deterministic baselines such as max pressure and queue-based control with respect to:

* vehicle throughput;
* average waiting time;
* average travel time;
* queue length;
* spillback;
* stop frequency;
* network stability under high demand;
* coordination effects such as green-wave-like behavior.

The controller must support:

* irregular intersections;
* varying numbers of controlled movements;
* varying numbers of selectable phases;
* generated grid networks;
* later OSM/city networks;
* transfer between networks.

The model does not directly predict a fixed phase class. Instead:

```text
network state
    → LaneGroup/Movement graph
    → GNN
    → movement scores
    → deterministic phase aggregation
    → selected valid phase
    → runtime transition controller
```

This avoids the old fixed action-space problem and keeps the same movement/phase abstraction usable across different junction geometries.

---

## 2. Current Implementation State

The active implementation is under:

```text
src/movement/
```

The legacy fixed canonical phase stack is under:

```text
src/legacy/
```

The legacy stack should be treated as a reference, not as active code.

---

## 3. Current Movement and Phase Representation

The core movement schema is in:

```text
src/movement/schema.py
```

The current control unit is `TrafficLightProgram`.

It contains:

* `ControlledMovement`;
* `SelectablePhase`;
* `TrafficLightProgram`.

### ControlledMovement

A `ControlledMovement` represents one SUMO controlled-link movement.

It is identified by:

* movement index;
* SUMO signal index;
* incoming lane;
* outgoing lane.

This is currently a SUMO-controlled-link-level object.

The new GNN architecture will introduce a higher-level graph abstraction where movements are still the scored objects, but they are associated with directed `LaneGroup` nodes.

### SelectablePhase

A `SelectablePhase` represents one selectable green phase.

It contains:

* SUMO signal state;
* the movement indices enabled by that phase.

A phase is selected by scoring its enabled movements.

### TrafficLightProgram

A `TrafficLightProgram` contains:

* all controlled movements for one traffic light;
* all selectable valid phases for that traffic light.

The controller no longer chooses a fixed action `0..7`.

Instead:

1. A policy scores movements.
2. Each phase score is computed by summing the scores of its enabled movements.
3. The selected phase is the maximum-scoring phase.

The phase scoring logic is in:

```text
src/movement/phase_selection.py
```

This design is the correct foundation for irregular intersections because the number of phases can vary per junction while a global policy can still share parameters by scoring movements.

---

## 4. Phase Synthesis

Phase synthesis is implemented in:

```text
src/movement/phase_synthesis.py
```

The current synthesis algorithm:

* reads SUMO controlled-link indices;
* uses SUMO `areFoes(request_a, request_b)` to prevent conflicting movements;
* adds an extra same-outgoing-edge conflict when movements from different incoming lanes enter the same outgoing edge;
* enforces shared incoming-lane completeness;
* enumerates maximal non-conflicting movement sets;
* exposes the result as `TrafficLightProgram`.

The shared incoming-lane rule is important.

If one lane serves both straight and left, opening only straight can be useless because a left-turning vehicle at the front may block the lane. Therefore, all movements from the same incoming lane are kept in sync inside a phase.

The older protected/manual phase-generation mode has been removed. The only supported path is conflict-driven synthesis.

This remains unchanged.

The neural model does not need to learn legality. It only scores movements. Phase validity is solved before the model is called.

---

## 5. Runtime and Signal Transitions

The runtime is implemented in:

```text
src/movement/runtime.py
```

`MovementControlRuntime` owns:

* SUMO lifecycle;
* extraction of movement-aware programs from TraCI;
* current signal-state application;
* minimum-green filtering;
* yellow transitions.

The runtime exposes `lane_api` through the `LaneQueueApi` protocol.

The current policy surface only requires:

```python
getLastStepHaltingNumber(lane_id: LaneId) -> int
```

Transition logic is implemented in:

```text
src/movement/transition.py
```

The transition system preserves greens that remain green in the next phase. Only signal links that lose green become yellow. This avoids unnecessary yellow/red cycles for movements that continue to be served.

Minimum-green logic is implemented in:

```text
src/movement/min_green.py
```

The default runtime path holds accepted green targets for at least two decision intervals. With a 15-second decision interval, this gives a 30-second minimum accepted green target.

Important runtime assumptions:

```text
model query interval: 15 seconds
accepted phase switch interval: 30 seconds
yellow/all-red handled externally
illegal signal states impossible by construction
```

The model is therefore not responsible for transition legality.

---

## 6. Baseline Policies

Baseline movement-scoring policies are implemented in:

```text
src/movement/policies/
```

Current baselines:

* `max_pressure.py`;
* `queue.py`;
* `__init__.py` dispatch.

The current runner:

```text
scripts/run.py
```

uses one of these movement-scoring policies and then calls:

```text
select_highest_scoring_phase
```

These baselines remain important for:

* visual debugging;
* sanity checking movement scores;
* collecting imitation targets;
* baseline evaluation;
* checking whether learned policies reproduce or improve over deterministic policies.

The max-pressure teacher should remain as currently implemented for the first imitation-learning baseline.

Do not redefine the teacher prematurely. The first goal is to reproduce the current deterministic max-pressure policy through the learned movement-score interface.

Later teacher variants may be added, but the initial IL target remains the current max-pressure movement scores.

---

## 7. Generated Grid Networks

Generated grids are created by:

```text
scripts/generate_grid_network.py
```

The generated grid path supports rectangular grids and writes:

* nodes;
* edges;
* lane connections;
* route flows;
* SUMO config;
* synthesized traffic-light programs.

Generated traffic-light programs use the same movement conflict synthesis that should later be used for OSM/city networks.

Usage examples are documented in:

```text
docs/grid_generation.md
```

Generated grids remain the first development target because they are controllable, reproducible, and easier to debug than OSM networks.

Recommended generated-grid usage:

* `3x3` for fast debugging;
* `4x4` for initial training;
* `6x6` for scale and transfer checks;
* fixed seeds for deterministic tests;
* randomized seeds for dataset diversity.

---

## 8. Current Limitations

The active movement stack does not yet have:

* global LaneGroup/Movement graph construction;
* neural-policy feature extraction;
* observation normalization;
* dataset collection;
* movement-score imitation training;
* GNN model implementation;
* model evaluation harness;
* PPO training with variable phase counts;
* OSM build integration using the new movement synthesis;
* transfer evaluation across grids/cities.

The legacy stack implemented many analogous concepts, but against fixed canonical phases. The useful pieces should be ported conceptually while replacing fixed phase labels with movement-score learning.

---

# 9. Updated Learning Architecture

## 9.1 Core Abstraction

The learning graph has two node types:

```text
LaneGroup
Movement
```

There are no signal/intersection nodes in the GNN.

Signal and intersection information still exists in runtime metadata, phase synthesis, and phase aggregation, but it is not represented as a graph node for message passing.

---

## 9.2 Directed LaneGroup

A `LaneGroup` represents a directed road segment between two signalized junctions.

For signalized junctions `A` and `B`:

```text
L_AB = traffic from A to B
L_BA = traffic from B to A
```

`L_AB` and `L_BA` are separate nodes.

They may share static geometry, but they must not share dynamic state.

Separate dynamic state:

```text
queue_AB      != queue_BA
occupancy_AB  != occupancy_BA
speed_AB      != speed_BA
arrival_AB    != arrival_BA
```

This is essential for directional traffic patterns such as morning and afternoon rush-hour asymmetry.

A `LaneGroup` collapses all physical lanes of a directed road segment into one graph node for v1.

This is a deliberate simplification. Splitting into separate left/through/right lane-use groups is deferred because it would substantially increase graph complexity.

Lane-specific information should initially enter the model through aggregated features and movement-specific demand.

---

## 9.3 Movement

A `Movement` represents one legal flow through a signalized junction.

If:

```text
L_AB = incoming directed lane group into junction B
L_BC = outgoing directed lane group from junction B
```

then the movement is:

```text
M_ABC = movement at B from L_AB to L_BC
```

A movement may correspond to one or more SUMO controlled links.

The model outputs one scalar score per movement:

```text
score(M_ABC)
```

The phase-selection layer aggregates these movement scores over valid phases.

---

## 9.4 Graph Topology

The graph alternates:

```text
LaneGroup ↔ Movement ↔ LaneGroup
```

For every movement `M_ABC`, the graph contains:

```text
L_AB ↔ M_ABC ↔ L_BC
```

Relation types should distinguish input and output semantics.

Recommended edge relations:

```text
input_lane_to_movement:
    L_AB → M_ABC

output_lane_to_movement:
    L_BC → M_ABC

movement_to_input_lane:
    M_ABC → L_AB

movement_to_output_lane:
    M_ABC → L_BC
```

The physical traffic direction is:

```text
L_AB → M_ABC → L_BC
```

But message passing is not identical to traffic flow.

For movement scoring, downstream supply information from `L_BC` must flow back into `M_ABC`, so `L_BC → M_ABC` is a valid information edge even though vehicles do not move in that direction.

No explicit conflict edges are used in v1.

Reason:

* conflicts are already handled by phase synthesis;
* the phase generator only exposes valid movement sets;
* the model does not need conflict edges for safety;
* explicit conflict or phase-competition edges can be tested later as an ablation.

No direct road-continuation edges are used in v1.

Continuation is already represented by:

```text
L_AB → M_ABC → L_BC → M_BCD → L_CD
```

---

## 9.5 Hop Definition

A macro-hop is defined as movement-to-movement information transfer through an intermediate lane group.

For a target movement:

```text
M_ABC
```

zero-hop context is:

```text
L_AB → M_ABC ← L_BC
```

That means the movement sees:

* its own movement features;
* its incoming lane group;
* its outgoing lane group.

One-hop context includes adjacent upstream and downstream movements:

```text
upstream:
    M_XAB, M_YAB, ...

target:
    M_ABC

downstream:
    M_BCD, M_BCE, ...
```

Two-hop context extends one more continuation step:

```text
M_WXA → L_XA → M_XAB → L_AB → M_ABC → L_BC → M_BCD → L_CD → M_CDE
```

Planned progression:

```text
0-hop IL:
    validate local feature extraction and baseline imitation

1-hop IL:
    validate graph construction and message passing

1-hop RL:
    learn immediate upstream/downstream coordination

2-hop RL:
    learn corridor coordination and green-wave-like behavior
```

Three-hop propagation is out of scope for the initial design.

---

# 10. Model Architecture

## 10.1 Input

Each training/inference sample contains:

* `LaneGroup` feature matrix;
* `Movement` feature matrix;
* edge indices for the typed bipartite graph;
* movement-to-phase incidence metadata;
* runtime/control metadata needed outside the model.

The model does not receive a fixed-size phase action space.

---

## 10.2 Encoders

Use separate encoders for lane groups and movements.

```python
h_lane = lane_encoder(x_lane)
```

Movement initialization should include local input/output lane context:

```python
h_move = movement_encoder(
    concat(
        x_move,
        h_lane[input_lane_id],
        h_lane[output_lane_id],
    )
)
```

This gives the zero-hop model enough information to imitate local movement scoring.

---

## 10.3 Message Passing

Each macro-hop consists of two stages.

### Stage 1: Movement → LaneGroup

Movements send messages to their adjacent lane groups.

A lane group receives messages from:

* movements feeding into it;
* movements leaving from it.

These relation types should be distinguished.

### Stage 2: LaneGroup → Movement

Lane groups send messages to their adjacent movements.

A movement receives:

* incoming-demand context from its input lane group;
* downstream-supply context from its output lane group.

After one macro-hop, a movement has information about adjacent movements through shared lane groups.

After two macro-hops, a movement has corridor-level context one continuation further away.

---

## 10.4 Score Head

The final model output is:

```text
movement_scores: [num_movements]
```

Each movement receives one scalar score.

The score head can be a simple MLP:

```python
score = score_head(h_move).squeeze(-1)
```

A possible later variant is residual max-pressure scoring:

```text
score = max_pressure_score + neural_residual
```

This should not be the first implementation unless useful for stabilization. The first IL model should directly learn to reproduce the current teacher scores.

---

## 10.5 Phase Aggregation

Phase aggregation remains outside the model.

For a selectable phase `P`:

```text
phase_score(P) = sum(score(m) for m in P.enabled_movements)
```

A capacity-weighted variant may be added later:

```text
phase_score(P) = sum(weight(m) * score(m) for m in P.enabled_movements)
```

But the initial implementation should preserve the current phase aggregation behavior unless a separate ablation explicitly changes it.

The selected phase is:

```text
selected_phase = argmax_P phase_score(P)
```

---

## 10.6 PPO Action Distribution

For PPO, the model still outputs movement scores.

Phase logits are computed after aggregation:

```text
phase_logits = aggregate_movement_scores_to_phases(movement_scores)
```

Then the action distribution is a categorical distribution over the local selectable phases.

This handles variable phase counts by using the local phase list for each traffic light.

During batching, store:

* selected local phase index;
* phase incidence matrix;
* number of selectable phases;
* movement scores/logits;
* log probability;
* value estimate;
* reward;
* done mask;
* graph sample metadata.

---

# 11. Feature and Data Architecture

## 11.1 Static LaneGroup Features

Static `LaneGroup` features come from the network topology.

Recommended features:

```text
lane_group_length
detector_length
number_of_lanes
speed_limit
freeflow_travel_time
estimated_storage_capacity
is_short_link
```

`L_AB` and `L_BA` should have separate static features because opposite directions can differ in lane count, speed, or geometry.

---

## 11.2 Dynamic LaneGroup Features

Dynamic lane-group features describe the observed traffic state of a directed road segment.

Recommended initial features:

```text
vehicle_count_detector
halting_count_detector
queue_length_m_detector
queue_length_vehicles_detector
occupancy_detector
mean_speed_detector
density_detector
available_storage_detector_ratio
arrival_rate_15s
departure_rate_15s
arrival_rate_60s
departure_rate_60s
detector_saturation
```

These features are attached to `LaneGroup` nodes.

The detector interpretation is:

```text
one detector region per directed LaneGroup,
placed near the downstream signalized junction
```

Default detector region:

```text
last min(300 m, lane_group_length) before the downstream junction
```

For short road segments, the detector covers the whole lane group.

Arrival/departure rates are valuable because raw vehicle count is ambiguous. A queue that is growing rapidly and a queue that is clearing rapidly can have the same current count but require different decisions.

Use both short and longer windows initially:

```text
15 s window:
    matches model query interval

60 s window:
    smoother recent-flow context
```

If 60 s proves too stale, it can be ablated.

---

## 11.3 Static Movement Features

Static movement features describe the movement geometry and capacity.

Recommended features:

```text
turn_type
number_of_underlying_controlled_links
saturation_flow_estimate
input LaneGroup id
output LaneGroup id
```

Optional later features:

```text
protected_or_permissive
movement_length
number_of_selectable_phases_serving_this_movement
number_of_conflicting_movements
```

Conflict-derived metadata is not needed in v1, but may be useful later.

---

## 11.4 Dynamic Movement Features

Dynamic movement features describe movement-specific demand and control state.

Recommended initial features:

```text
oracle_movement_demand
oracle_movement_demand_norm
is_currently_enabled
was_enabled_last_decision
time_since_enabled
```

For v1, movement demand is oracle-based.

That means:

```text
oracle_movement_demand(M_ABC)
    = number of vehicles on the relevant observed region of L_AB
      whose route/next link uses movement M_ABC
```

This is intentionally not fully realistic.

The purpose of the first version is to validate:

* graph abstraction;
* feature extraction;
* movement indexing;
* dataset collection;
* imitation learning;
* RL warm-start;
* evaluation pipeline.

Realistic movement-demand estimation is deferred.

Possible future non-oracle demand sources:

* dedicated turning lanes;
* dedicated turn-lane detectors;
* historical turn probabilities;
* recent observed outflow ratios;
* route-choice estimation;
* camera/tracking assumptions;
* connected-vehicle data assumptions.

---

## 11.5 Max-Pressure Teacher Target

The first imitation target is the current implemented max-pressure movement score.

Do not redefine the teacher in the first implementation.

At each decision time:

```text
target[M] = current_max_pressure_policy_score[M]
```

The model is trained to regress these movement scores.

The purpose of imitation is not to learn better-than-baseline coordination. It is to validate that the learned policy can reproduce the current deterministic movement-scoring path.

The learned model should then be usable as an initialization for RL.

---

# 12. Normalization Strategy

Normalization is critical because road lengths, detector lengths, lane counts, and queue scales vary.

The main rule:

```text
Normalize detector-local observations by detector capacity.
Normalize full-link/static quantities by lane-group scale.
Do not mix the two meanings.
```

---

## 12.1 Detector-Length Problem

A 100 m detector and a 300 m detector observe different maximum queue lengths.

A 100 m detector cannot distinguish:

```text
queue = 100 m
queue = 200 m
queue = 300 m
```

if the detector is fully saturated.

Therefore, detector features should be interpreted as local observations, not full-link truth.

Use detector-local normalization:

```text
queue_norm_detector = queue_length_m_detector / detector_length
count_norm_detector = vehicle_count_detector / detector_capacity
```

where:

```text
detector_capacity ≈ detector_length * num_lanes / effective_vehicle_spacing
```

Include an explicit saturation feature:

```text
detector_saturation = 1 if detector is fully or nearly fully queued
```

This lets the model distinguish:

```text
short detector fully saturated
long detector partially saturated
long detector fully saturated
```

only to the extent that the sensor actually observes the difference.

---

## 12.2 Static Scale Features

Even if detector values are normalized by detector capacity, the model should receive static scale context:

```text
lane_group_length
detector_length
num_lanes
speed_limit
freeflow_travel_time
```

This allows it to learn that a saturated 100 m detector on a short link is different from a saturated 100 m detector on a much longer arterial.

---

## 12.3 Movement Demand Normalization

The denominator must match the spatial region over which demand is counted.

If oracle demand is counted only inside the detector region:

```text
oracle_demand_norm = oracle_demand_detector_region / detector_capacity
```

If oracle demand is counted over the full lane group:

```text
oracle_demand_norm = oracle_demand_full_lane_group / lane_group_capacity
```

For v1, prefer detector-region oracle demand so movement demand aligns with the detector-based lane-group state.

---

## 12.4 Normalizer Implementation

Training datasets should store raw numeric features.

During training:

1. fit a normalizer on training data;
2. normalize dynamic numeric features;
3. freeze normalizer statistics;
4. save normalizer with checkpoint;
5. reuse frozen normalizer for evaluation.

This follows the legacy normalizer concept while using the new feature schema.

Recommended first approach:

```text
global normalizer over training networks
frozen for evaluation
static binary/categorical features excluded from normalization
```

Add renormalization tools later only if transfer is unstable.

---

# 13. Dataset Collection

The first dataset should be collected from simulation using the current max-pressure policy as teacher.

At each decision interval, record:

```text
network id / config path
SUMO seed
demand seed
simulation time
traffic light id
TrafficLightProgram metadata or stable movement/phase IDs
current phase state
minimum-green / switch state
LaneGroup feature matrix
Movement feature matrix
LaneGroup ↔ Movement graph edges
phase incidence matrix
teacher movement scores from current max-pressure policy
teacher selected phase
optional queue-policy selected phase for diagnostics
traffic metrics after the step
```

Minimum useful training sample:

```text
graph features → teacher movement scores
```

Minimum useful evaluation sample:

```text
graph features → model movement scores → phase aggregation → selected phase
```

---

## 13.1 Dataset Format

The dataset should support variable graph sizes and variable phase counts.

Recommended stored objects per sample:

```text
x_lane: [num_lane_groups, lane_feature_dim]
x_movement: [num_movements, movement_feature_dim]
edge_index_dict: typed LaneGroup/Movement edge indices
movement_to_phase_incidence: ragged or sparse phase incidence
teacher_movement_scores: [num_movements]
teacher_selected_phase_per_tls
metadata
```

For offline training, the graph can be stored per decision time.

If storage becomes too large, optimize later using:

* static graph stored once per network;
* dynamic features stored per timestep;
* movement/phase metadata referenced by stable IDs.

Do not optimize prematurely.

---

# 14. Imitation Learning Plan

Imitation learning is the first learning milestone.

The IL target is direct movement-score regression.

```text
loss = Huber(model_score[M], teacher_score[M])
```

or:

```text
loss = MSE(model_score[M], teacher_score[M])
```

Huber is preferred initially for robustness.

Do not train phase-level cross-entropy in the first version.

Reason:

* phase choices are derived from movement scores;
* the model’s intended interface is movement scoring;
* preserving movement-score semantics makes debugging easier;
* deterministic aggregation can be inspected directly.

---

## 14.1 Stage 1: Zero-Hop IL

Input context:

```text
movement features
input LaneGroup features
output LaneGroup features
```

No macro-hop propagation.

Goal:

```text
reproduce current max-pressure movement scores
```

Purpose:

* validate feature extraction;
* validate movement indexing;
* validate target generation;
* validate normalization;
* validate model wrapper;
* validate phase aggregation.

Acceptance:

* model overfits a tiny fixed-seed dataset;
* model reproduces teacher movement scores on an overfit batch;
* model reproduces teacher phase choices on the deterministic overfit setup;
* learned policy can run inside the same runtime path as max pressure.

---

## 14.2 Stage 2: One-Hop IL

Enable one macro-hop:

```text
Movement → LaneGroup → Movement
```

Goal:

```text
validate graph construction and message passing
```

Expected result:

```text
similar imitation performance to zero-hop
```

If performance degrades strongly, likely issues include:

* incorrect edge construction;
* incorrect relation typing;
* incorrect batching;
* incorrect graph indexing;
* normalization mismatch;
* accidental leakage/removal of local context.

One-hop IL is primarily a graph sanity check, not an expected performance improvement.

---

# 15. Policy Evaluation Plan

Implement a policy runner that can run:

* max-pressure baseline;
* queue baseline;
* learned IL policy;
* later PPO policy.

Evaluation must compare policies on identical configs and seeds:

```text
for seed in seeds:
    run max_pressure(seed)
    run queue(seed)
    run learned_policy(seed)

compare averaged metrics
```

Metrics:

```text
completed vehicles / throughput
average waiting time
average travel time
max queue length
average queue length
wait density
phase switch frequency
per-junction wait density
spillback indicators
stops before traffic lights
optional green-wave/corridor metrics
```

Legacy references:

```text
src/legacy/training/eval_episode.py
scripts/eval_city.py
```

Evaluation should parse SUMO tripinfo and collect in-simulation queue/wait metrics.

The evaluation harness should produce:

* JSON summary;
* CSV table;
* optional TensorBoard logs;
* per-seed metrics;
* aggregate mean/std across seeds.

Acceptance:

* max pressure and learned policy run on identical demand seeds;
* evaluation reports throughput, waiting time, travel time, queue metrics, and switch frequency;
* generated-grid evaluation is reproducible.

---

# 16. PPO Plan After IL

Only start PPO once IL can reproduce max pressure well enough to be useful.

PPO changes relative to legacy:

* model output is movement scores;
* phase logits are computed by aggregating movement scores;
* action distribution is over local selectable phases;
* action counts are variable per junction;
* critic can be per-junction or graph-level.

The first PPO version should use per-junction categorical distributions over the local phase list after aggregation.

During batching, store:

```text
selected local phase index
phase incidence matrix
phase mask / phase count
log probability
value estimate
reward
done flag
graph sample
```

Legacy references:

```text
src/legacy/training/ppo.py
src/legacy/training/rollout.py
scripts/train_rl.py
```

Reuse:

```text
GAE
clipped PPO objective
entropy bonus
value warmup
rollout buffer concept
checkpointing cadence
evaluation cadence
parallel collection approach later
```

Replace:

```text
fixed NUM_PHASES
fixed phase categorical output
fixed observation vector
```

---

## 16.1 PPO Stage 1: One-Hop RL

Initialize from the one-hop IL checkpoint.

Goal:

```text
learn immediate upstream/downstream coordination
```

Expected possible improvements:

* better avoidance of downstream blocking;
* better response to arriving platoons;
* reduced local spillback;
* improved throughput under uneven demand.

The controller still enforces:

* minimum switch interval;
* valid phases;
* yellow/all-red transitions.

A separate switch penalty is likely unnecessary in v1 because switching is already hard-constrained.

---

## 16.2 PPO Stage 2: Two-Hop RL

Enable two macro-hops.

Goal:

```text
learn corridor coordination and green-wave-like behavior
```

Expected possible improvements:

* fewer stops along arterials;
* better progression;
* smoother discharge across adjacent junctions;
* reduced queue propagation.

Two-hop RL should be compared against:

* max pressure;
* queue baseline;
* zero-hop IL;
* one-hop IL;
* one-hop PPO.

---

# 17. OSM and City Network Plan

After the generated-grid IL/RL path works, port the OSM build path.

Legacy references:

```text
scripts/build_network.py
scripts/tools/diagnose_network.py
scripts/tools/inspect_junctions.py
```

Required changes:

* adapt `scripts/build_network.py` to call movement conflict phase synthesis;
* remove fixed canonical phase generation;
* ensure generated `.tll.xml` programs load cleanly;
* ensure every intended signalized junction has movement programs extracted;
* ensure `src/movement/sumo_adapter.py` can process city networks;
* add a simple run command for city configs with baseline policies;
* add visual verification with SUMO-GUI.

The OSM build should eventually produce:

```text
configs/<city>/<city>.net.xml
configs/<city>/<city>.tll.xml
configs/<city>/<city>.rou.xml
configs/<city>/<city>.add.xml
configs/<city>/<city>.sumocfg
```

Baseline run command should look like:

```powershell
python scripts\run.py --cfg configs\city\city.sumocfg --method max-pressure --gui
```

Learned policy evaluation should use the same config:

```powershell
python scripts\eval_policy.py --cfg configs\city\city.sumocfg --policy checkpoint.pt --baseline max-pressure
```

Acceptance:

* city network builds;
* baseline policy runs visually;
* movement programs are extracted for all intended traffic lights;
* generated phases are valid;
* visual verification confirms plausible signal behavior.

---

# 18. Transfer Learning Plan

Transfer remains a major project objective.

Before OSM work, run generated-grid transfer checks.

Initial transfer check:

1. Train IL on a generated `4x4` grid.
2. Evaluate the checkpoint zero-shot on `5x5` or `6x6`.
3. Compare against max pressure and queue baselines on identical seeds.

Later transfer work:

1. Train on generated `4x4` / `6x6` grids.
2. Evaluate zero-shot on larger generated grids.
3. Train on one OSM city network.
4. Evaluate zero-shot on another OSM network.
5. Train on multiple networks.
6. Evaluate full transfer matrix:

```text
train network set × evaluation network set
```

Normalization matters strongly for transfer.

Possible approaches:

* one global normalizer accumulated over all training networks;
* per-network normalizer for evaluation;
* hybrid global normalization plus explicit static scale features;
* no normalization for static categorical/binary features.

First recommendation:

```text
use one global normalizer for dynamic numeric features during multi-network training
freeze it for evaluation
add renormalization utilities only if zero-shot transfer is unstable
```

Legacy references:

```text
src/legacy/utils/graph_builder.py
scripts/renormalize.py
```

---

# 19. Legacy Code Reference Map

The legacy implementation is in:

```text
src/legacy/
```

It should be treated as a conceptual reference, not as active code.

Some imports inside legacy files may still point to old paths such as:

```text
src.environment
src.model
src.training
```

---

## 19.1 Legacy Environment and Observations

Legacy files:

```text
src/legacy/environment/sumo_env.py
src/legacy/environment/junction_info.py
src/legacy/environment/phase_schema.py
src/legacy/environment/expert.py
```

What they contain:

* `TrafficEnv`;
* SUMO-backed RL/IL environment;
* `reset`, `observe`, `step`;
* reward calculation;
* fixed 45-dimensional observation definition;
* fixed `NUM_PHASES = 8`;
* canonical 3-way/4-way junction geometry parsing;
* `GreedyExpert`.

Reuse conceptually:

* SUMO lifecycle;
* demand randomization;
* observation collection patterns;
* local/global wait-density metrics;
* expert data collection structure;
* min-green and phase-switch accounting;
* tripinfo-based evaluation.

Do not reuse directly:

* fixed `NUM_PHASES = 8`;
* canonical slot phase mapping;
* fixed 45-dimensional observation vector.

---

## 19.2 Legacy Graph Builder and Normalization

Legacy file:

```text
src/legacy/utils/graph_builder.py
```

Relevant classes:

```text
RunningNormalizer
GraphBuilder
```

What it did:

* created one graph node per traffic-light junction;
* built directed edges between neighboring signalized junctions;
* used static edge features `[road_length_m, n_lanes, speed_limit_m_s]`;
* normalized node features online with `RunningNormalizer`;
* froze normalizer statistics for RL/evaluation.

Reuse conceptually:

* normalizer design;
* save/load of normalizer statistics;
* transfer-aware feature scaling;
* graph construction test style.

Replace:

* junction-node graph;
* old node feature schema;
* fixed phase-oriented representation.

The new graph builder should construct `LaneGroup` and `Movement` nodes.

---

## 19.3 Legacy Model

Legacy file:

```text
src/legacy/model/gat_policy.py
```

Relevant class:

```text
GATPolicy
```

What it did:

* encoded per-junction node features;
* ran `GATv2Conv` layers over a junction graph;
* produced per-junction phase logits;
* included a value head for PPO.

Reuse conceptually:

* GAT/GATv2-style message passing ideas;
* residual structure;
* layer norm;
* actor-critic split;
* value head concept.

Replace:

* fixed phase logits;
* junction-only graph;
* fixed action-space classifier.

The new model should output scalar movement scores.

---

## 19.4 Legacy Imitation Learning

Legacy files:

```text
src/legacy/training/imitation.py
scripts/train_il.py
```

What they did:

* drove the environment with a greedy expert;
* built graph observations;
* trained `GATPolicy` with cross-entropy against expert phase labels;
* logged loss, phase match, wait density, switch rate, and evaluation metrics;
* saved model and normalizer checkpoints;
* supported DAgger-style expert/model mixing.

Reuse:

* training loop shape;
* checkpoint layout;
* evaluation cadence;
* TensorBoard logging;
* optional DAgger;
* normalizer save/load;
* overfit-one-batch debugging.

Replace:

* fixed phase labels;
* phase cross-entropy as first IL target;
* fixed observation vector.

New IL target:

```text
movement-score regression against current max-pressure policy
```

---

## 19.5 Legacy PPO

Legacy files:

```text
src/legacy/training/ppo.py
src/legacy/training/rollout.py
scripts/train_rl.py
```

What they did:

* loaded an IL checkpoint;
* collected graph rollouts;
* used GAE;
* used PPO clipped objective;
* trained a value head;
* supported fixed-time burn-in;
* evaluated model vs expert periodically;
* supported parallel SUMO workers.

Reuse:

* rollout buffer structure;
* actor-critic training loop;
* GAE implementation;
* value warmup;
* evaluation cadence;
* checkpointing;
* parallel rollout idea.

Replace:

* fixed categorical over 8 phases;
* fixed phase logits;
* fixed observation vector.

New PPO action distribution:

```text
movement scores → phase logits via aggregation → local categorical over selectable phases
```

---

## 19.6 Legacy Evaluation

Legacy files:

```text
src/legacy/training/eval_episode.py
scripts/eval_city.py
```

What they did:

* ran model and expert on identical demand seeds;
* parsed SUMO tripinfo;
* measured waiting time;
* measured travel time;
* measured throughput;
* collected max queue;
* collected wait density;
* tracked switch frequency;
* tracked progression/green-wave-style metrics such as stops before TLS.

Reuse almost directly.

The policy wrapper changes, but the metric design remains appropriate.

---

## 19.7 Legacy OSM / City Network Build

Legacy files:

```text
scripts/build_network.py
scripts/tools/diagnose_network.py
scripts/tools/inspect_junctions.py
```

What they did:

* downloaded or loaded OSM;
* ran `netconvert`;
* cleaned topology via plain-XML round trips;
* promoted/demoted traffic-light junctions;
* wrote `.tll.xml`, detectors, routes, and `.sumocfg`;
* optionally launched visual verification.

Reuse:

* OSM import and cleanup;
* topology audit tools;
* demand generation;
* config generation;
* visual verification loop.

Replace:

* fixed canonical `.tll.xml` generation;
* old phase assumptions.

New OSM path:

```text
OSM/network build → movement conflict phase synthesis → movement runtime → baseline/learned policy
```

---

## 19.8 Legacy Demand Generation

Legacy file:

```text
src/legacy/utils/demand_generator.py
```

What it did:

* discovered entry/exit edges;
* precomputed valid shortest-path O-D routes;
* generated randomized per-episode route files.

Reuse for:

* IL data diversity;
* PPO training episodes;
* fixed-seed evaluation;
* generated-grid transfer.

---

# 20. Proposed Implementation Order

## Milestone 1: Static LaneGroup/Movement Graph Builder

Deliverables:

```text
src/movement/graph.py
src/movement/graph_schema.py
```

Responsibilities:

* construct directed `LaneGroup` nodes from the SUMO/generator topology;
* construct `Movement` nodes from `TrafficLightProgram.movements`;
* map current `ControlledMovement` objects to graph-level movements;
* build typed `LaneGroup ↔ Movement` edges;
* build movement-to-phase incidence matrices;
* maintain stable IDs for dataset collection and replay.

Acceptance:

* graph can be built for `grid_3x3_dedicated`;
* `L_AB` and `L_BA` are separate where both exist;
* every movement has exactly one input and one output lane group;
* phase incidence aligns with `TrafficLightProgram.selectable_phases`;
* graph construction is deterministic across runs.

---

## Milestone 2: Feature Schema and Extraction

Deliverables:

```text
src/movement/features.py
src/movement/normalization.py
```

Responsibilities:

* define lane-group feature dataclasses;
* define movement feature dataclasses;
* extract dynamic lane-group features from TraCI/runtime;
* extract oracle movement demand;
* extract movement control-state features;
* compute detector-local normalized values;
* store raw features for dataset collection.

Acceptance:

* features can be extracted for generated grids;
* feature rows align with graph node IDs;
* detector-length normalization is tested;
* movement demand rows align with movement IDs;
* fake lane APIs can test feature extraction without SUMO.

---

## Milestone 3: Dataset Collection

Deliverables:

```text
scripts/collect_il_data.py
docs/dataset_schema.md
```

Responsibilities:

* run `MovementControlRuntime`;
* use current max-pressure policy as teacher;
* collect graph features and teacher scores;
* save reproducible samples;
* store metadata for replay/inspection.

Acceptance:

* data can be collected from `3x3` and `6x6` grids;
* samples include lane features, movement features, graph edges, phase incidence, teacher movement scores, selected phases, and metadata;
* replay/inspection script can reconstruct the selected phase from stored teacher scores;
* fixed seeds are reproducible.

---

## Milestone 4: Zero-Hop Offline IL

Deliverables:

```text
src/movement/models/
scripts/train_il.py
```

Responsibilities:

* implement zero-hop movement scorer;
* implement dataset loader;
* implement normalizer fitting;
* implement movement-score regression training;
* save checkpoint and normalizer.

Acceptance:

* model overfits a tiny fixed-seed dataset;
* model reproduces teacher movement scores on the overfit set;
* model reproduces teacher selected phases through deterministic aggregation;
* checkpoint loads in a runner/evaluation script.

---

## Milestone 5: One-Hop GNN IL

Deliverables:

```text
src/movement/models/bipartite_gnn.py
```

Responsibilities:

* implement typed `Movement → LaneGroup → Movement` macro-hop;
* support one macro-hop;
* preserve zero-hop local context;
* train against same max-pressure teacher.

Acceptance:

* one-hop model matches zero-hop imitation performance;
* graph batching works;
* no large degradation from message passing;
* learned policy runs in simulation.

---

## Milestone 6: Policy Evaluation Harness

Deliverables:

```text
scripts/eval_policy.py
src/movement/evaluation/
```

Responsibilities:

* run max-pressure baseline;
* run queue baseline;
* run learned IL policy;
* compare on identical demand seeds;
* parse tripinfo;
* collect simulation metrics;
* write JSON/CSV summaries.

Acceptance:

* evaluation reports throughput, waiting time, travel time, queue metrics, and switch frequency;
* max pressure and learned policy run on identical seeds;
* generated-grid evaluation is reproducible;
* evaluation supports both 0-hop and 1-hop checkpoints.

---

## Milestone 7: Generated-Grid Transfer Check

Deliverables:

```text
scripts/eval_transfer.py
docs/generated_grid_transfer.md
```

Responsibilities:

* train on `4x4`;
* evaluate zero-shot on `5x5` or `6x6`;
* compare against max pressure and queue;
* report degradation or transfer stability.

Acceptance:

* one IL checkpoint trained on one grid size can be evaluated unchanged on a larger grid;
* results clearly show whether transfer preserves max-pressure-like behavior, degrades gracefully, or fails;
* normalizer behavior is documented.

---

## Milestone 8: One-Hop PPO

Deliverables:

```text
src/movement/training/ppo.py
src/movement/training/rollout.py
scripts/train_rl.py
```

Responsibilities:

* load IL checkpoint;
* aggregate movement scores into phase logits;
* sample local phases from variable-size categorical distributions;
* implement PPO rollout buffer;
* implement GAE;
* implement clipped PPO objective;
* train value head.

Acceptance:

* PPO starts from IL checkpoint;
* action distribution handles variable phase counts;
* PPO, IL, max pressure, and queue are evaluated on identical seeds;
* PPO does not break runtime legality constraints.

---

## Milestone 9: Two-Hop PPO

Deliverables:

```text
two-hop model config
two-hop PPO training/eval configs
```

Responsibilities:

* enable two macro-hop propagation;
* train from IL or one-hop PPO checkpoint;
* evaluate corridor/green-wave behavior;
* compare against one-hop PPO.

Acceptance:

* two-hop model runs without batching/indexing issues;
* evaluation includes progression metrics such as stops before TLS;
* results show whether extra context improves coordination.

---

## Milestone 10: OSM Build Integration

Deliverables:

```text
updated scripts/build_network.py
city config generation docs
visual verification command
```

Responsibilities:

* replace fixed canonical traffic-light generation with movement synthesis;
* ensure `.tll.xml` programs load;
* ensure movement programs are extracted for all intended traffic lights;
* generate detectors/routes/configs;
* run baseline policies visually.

Acceptance:

* city network builds;
* max-pressure baseline runs visually;
* generated phases are valid;
* movement programs are extracted cleanly;
* diagnostics identify unsupported/problematic junctions.

---

## Milestone 11: City Evaluation

Deliverables:

```text
city eval configs
city baseline reports
```

Responsibilities:

* run max pressure and queue on city networks;
* run learned generated-grid checkpoints zero-shot if feasible;
* collect tripinfo and queue metrics;
* identify feature/normalization failures.

Acceptance:

* city baseline evaluation is reproducible;
* model policy can be evaluated or failure modes are documented;
* city-specific topology issues are diagnosed.

---

## Milestone 12: Multi-Network Training and Transfer Matrix

Deliverables:

```text
multi-network training configs
transfer matrix report
```

Responsibilities:

* train on multiple generated grids;
* train on generated + city networks later;
* evaluate on held-out networks;
* compare normalizer strategies.

Acceptance:

* results table covers train-network set vs evaluation-network set;
* transfer performance is reported relative to max pressure and queue;
* normalization sensitivity is documented.

---

# 21. Deferred Items

The following are explicitly deferred:

```text
raw lane-level graph nodes
splitting LaneGroups into left/through/right lane-use groups
realistic non-oracle movement demand
conflict edges in the GNN
signal/intersection graph nodes
three-hop message passing
phase-level cross-entropy IL
manual/protected phase generation
full OSM transfer before generated-grid IL works
```

Potential future work:

* turn-lane-based movement demand estimation;
* camera/tracking-style detector assumptions;
* residual max-pressure model;
* explicit conflict/competition edges;
* capacity-capped phase aggregation;
* lane-use group splitting for turn pockets;
* multi-agent credit assignment;
* curriculum from generated grids to OSM cities.

---

# 22. Immediate Next Step

Start with Milestone 1.

The first concrete implementation should produce:

```text
LaneGroup node table
Movement node table
typed LaneGroup ↔ Movement edges
movement feature matrix placeholder
lane-group feature matrix placeholder
phase incidence matrix
stable graph IDs
```

Then Milestone 2 can fill the feature values.

The first end-to-end validation target is:

```text
fixed generated grid
fixed demand seed
current max-pressure teacher
zero-hop model
movement-score overfit
phase-choice reproduction through deterministic aggregation
```

Once that works, the rest of the pipeline can be built incrementally.
