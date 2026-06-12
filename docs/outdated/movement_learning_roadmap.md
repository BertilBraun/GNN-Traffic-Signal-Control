# Movement-Based Learning Roadmap

This document records the current movement-based controller state and the next
development path toward imitation learning, PPO, city networks, and transfer
evaluation. It also points to legacy code that already implemented similar
systems for the old fixed-phase controller.

## Current State

The active implementation is now under `src/movement/`. The legacy fixed
canonical 8-phase stack is kept under `src/legacy/` as reference material only.

### Movement And Phase Representation

The core schema is in `src/movement/schema.py`.

The current control unit is a `TrafficLightProgram`:

- `ControlledMovement`: one SUMO controlled-link movement, identified by a
  movement index, SUMO signal index, incoming lane, and outgoing lane.
- `SelectablePhase`: one selectable green phase, represented by a SUMO signal
  state and the movement indices it enables.
- `TrafficLightProgram`: the movement list and selectable phases for one
  traffic light.

This is the main architectural shift away from the old fixed action-space
design. The controller no longer chooses action `0..7` directly. Instead:

1. A policy scores movements.
2. A phase score is computed by summing the scores of movements enabled by that
   phase.
3. The selected phase is the maximum-scoring phase.

The phase scoring logic is in `src/movement/phase_selection.py`.

This design is the right foundation for irregular city intersections because
the number of phases can vary per junction. A learned model can still share
parameters globally by scoring movements rather than predicting a fixed phase
class.

### Phase Synthesis

Phase synthesis is in `src/movement/phase_synthesis.py`.

The current synthesis algorithm:

- reads SUMO controlled-link indices;
- uses SUMO `areFoes(request_a, request_b)` to prevent conflicting movements;
- adds an extra same-outgoing-edge conflict when movements from different
  incoming lanes enter the same outgoing edge;
- enforces shared incoming-lane completeness, so a phase cannot green only a
  subset of movements from one incoming lane;
- enumerates maximal non-conflicting movement sets;
- exposes the result as `TrafficLightProgram`.

The shared incoming-lane rule is important. If one lane serves both straight
and left, opening only straight is often useless because a left-turning vehicle
at the front blocks the lane. Therefore, all movements from the same incoming
lane must be in sync inside a phase.

The older protected/manual phase-generation mode has been removed. The only
supported path is conflict-driven synthesis.

### Runtime And Signal Transitions

The runtime is in `src/movement/runtime.py`.

`MovementControlRuntime` owns:

- SUMO lifecycle;
- extraction of movement-aware programs from TraCI;
- current signal-state application;
- minimum-green filtering;
- yellow transitions.

The runtime exposes `lane_api` through the `LaneQueueApi` protocol. The current
policy surface only requires:

```python
getLastStepHaltingNumber(lane_id: LaneId) -> int
```

Transition logic is in `src/movement/transition.py`. It preserves greens that
remain green in the next phase; only links that lose green become yellow. This
avoids unnecessary yellow/red cycles for movements that continue to be served.

Minimum green logic is in `src/movement/min_green.py`. The default run path
holds accepted green targets for at least two decision intervals, which means
30 seconds when the decision interval is 15 seconds.

### Baseline Policies

Baseline movement scoring policies are in `src/movement/policies/`.

- `max_pressure.py`: movement score = incoming queue minus outgoing queue.
- `queue.py`: movement score = incoming queue.
- `__init__.py`: policy method enum and dispatch.

The current runner `scripts/run.py` uses one of these movement-scoring policies,
then calls `select_highest_scoring_phase`.

These baselines are not the final learning target, but they are useful for:

- visual debugging;
- collecting imitation targets;
- baseline evaluation;
- sanity checking learned movement scores.

### Generated Grid Networks

The current generated grids are created by `scripts/generate_grid_network.py`.

The generated grid path supports rectangular grids and writes:

- nodes;
- edges;
- lane connections;
- route flows;
- SUMO config;
- synthesized traffic-light programs.

The generated traffic-light programs use the same movement conflict synthesis
that should later be used for OSM/city networks.

Usage examples are in `docs/grid_generation.md`.

### Current Limitations

The current active movement stack does not yet have:

- a learning environment abstraction;
- observation/feature extraction for a neural policy;
- graph construction;
- dataset collection;
- imitation training;
- model evaluation;
- PPO training;
- OSM build integration using the new movement synthesis.

The legacy stack implemented most of these concepts, but against fixed
canonical phases. The next work is to port the useful pieces while replacing
fixed phase labels with movement-score learning.

## Legacy Code Reference Map

The legacy implementation is in `src/legacy/`. It should be treated as a
reference, not as active code. Some imports inside it still point to old
locations such as `src.environment`, `src.model`, and `src.training`.

### Environment And Observations

Legacy files:

- `src/legacy/environment/sumo_env.py`
- `src/legacy/environment/junction_info.py`
- `src/legacy/environment/phase_schema.py`
- `src/legacy/environment/expert.py`

What they contain:

- `TrafficEnv`: SUMO-backed RL/IL environment with `reset`, `observe`, `step`,
  and reward calculation.
- `phase_schema.py`: fixed 45-dimensional observation definition and fixed
  `NUM_PHASES = 8`.
- `junction_info.py`: parsing of 3-way and 4-way junction geometry into
  canonical approach slots and fixed phase strings.
- `GreedyExpert`: expert policy that scored fixed phases from vehicle waiting
  and intended route movements.

What should be reused conceptually:

- SUMO lifecycle and demand randomization patterns;
- observation collection from TraCI;
- local/global wait-density metrics;
- expert data collection structure;
- min-green and phase-switch accounting;
- tripinfo-based evaluation.

What should not be reused directly:

- fixed `NUM_PHASES = 8` as the action space;
- canonical slot phase mapping;
- fixed 45-dimensional phase-class observation as the final model input.

### Graph Builder And Normalization

Legacy file:

- `src/legacy/utils/graph_builder.py`

Relevant classes:

- `RunningNormalizer`
- `GraphBuilder`

What it did:

- created one graph node per traffic-light junction;
- built directed edges between neighboring signalized junctions;
- used static edge features `[road_length_m, n_lanes, speed_limit_m_s]`;
- normalized node features online with `RunningNormalizer`;
- froze normalizer statistics for RL/evaluation.

This is still a good concept. For transfer across networks, feature
normalization is likely necessary because lane lengths, speeds, queue scales,
and junction spacing can differ substantially between generated grids and city
networks.

The current movement-based model will probably still need a normalizer, but
the feature schema will change from fixed phase-class observations to
movement/lane/junction graph features.

### Model

Legacy file:

- `src/legacy/model/gat_policy.py`

Relevant class:

- `GATPolicy`

What it did:

- encoded per-junction node features;
- ran multiple `GATv2Conv` layers over a junction graph;
- produced per-junction phase logits;
- also had a value head for PPO.

What should be reused:

- GATv2 backbone idea;
- edge features;
- residual + layer norm structure;
- actor-critic split for PPO.

What must change:

- output should not be fixed phase logits;
- output should be movement scores, or parameters from which movement scores
  are derived;
- phase selection remains deterministic through the movement-to-phase
  aggregation algorithm.

### Imitation Learning

Legacy files:

- `src/legacy/training/imitation.py`
- `scripts/train_il.py`

What they did:

- drove the environment with a greedy expert;
- built graph observations;
- trained `GATPolicy` with cross-entropy against expert phase labels;
- logged loss, phase match, wait density, switch rate, and evaluation metrics;
- saved model and normalizer checkpoints;
- supported DAgger-style expert/model mixing.

What should be reused:

- training loop shape;
- checkpoint layout;
- evaluation cadence;
- TensorBoard logging ideas;
- optional DAgger;
- normalizer save/load.

What must change:

- labels are no longer fixed phase indices;
- the expert target should be normalized max-pressure movement scores;
- model output must support variable phase sets per junction.

### PPO

Legacy files:

- `src/legacy/training/ppo.py`
- `src/legacy/training/rollout.py`
- `scripts/train_rl.py`

What they did:

- loaded an IL checkpoint;
- collected graph rollouts;
- used GAE and PPO clipped objective;
- trained a value head;
- supported fixed-time burn-in;
- evaluated model vs expert periodically;
- supported parallel SUMO workers.

What should be reused:

- rollout buffer structure;
- actor-critic training loop;
- GAE implementation;
- value warmup idea;
- evaluation and checkpointing cadence;
- parallel collection approach, later.

What must change:

- action distribution cannot be a fixed categorical over 8 phases if the model
  outputs movement scores;
- PPO action distribution should probably be over selectable phases after
  aggregating predicted movement scores into phase logits for each junction;
- variable phase counts require masking or per-junction ragged action
  handling.

### Evaluation

Legacy files:

- `src/legacy/training/eval_episode.py`
- `scripts/eval_city.py`

What they did:

- ran model and expert on identical demand seeds;
- parsed SUMO tripinfo for waiting time, travel time, and throughput;
- collected in-sim max queue, wait density, switch frequency;
- tracked progression/green-wave style metrics such as stops before TLS.

This structure is highly reusable. The policy wrapper changes, but the metric
design is still appropriate.

### OSM / City Network Build

Legacy files:

- `scripts/build_network.py`
- `scripts/tools/diagnose_network.py`
- `scripts/tools/inspect_junctions.py`

What it did:

- downloaded or loaded OSM;
- ran `netconvert`;
- cleaned topology via plain-XML round trips;
- promoted/demoted traffic-light junctions;
- wrote `.tll.xml`, detectors, routes, and `.sumocfg`;
- optionally launched a visual verification run.

What should be reused:

- OSM import and cleanup pipeline;
- topology audit tools;
- demand and config generation ideas;
- verify loop idea.

What must change:

- fixed canonical `.tll.xml` generation must be replaced by movement conflict
  synthesis from `src/movement/phase_synthesis.py`;
- generated city networks should be runnable with `scripts/run.py` and later
  with learned movement policies.

### Demand Generation

Legacy file:

- `src/legacy/utils/demand_generator.py`

What it did:

- discovered entry/exit edges;
- precomputed valid shortest-path O-D routes;
- generated randomized per-episode route files.

This is useful for both IL and PPO because training needs many demand
realizations, and evaluation needs fixed demand seeds for fair comparisons.

## Target Learning Architecture

The intended learning architecture should preserve the current deterministic
phase-selection layer:

```text
network state -> movement feature graph -> model -> movement scores
movement scores + selectable phases -> phase scores -> selected phase
```

This gives us a stable bridge between:

- arbitrary SUMO phase sets;
- generated grids;
- OSM/city networks;
- fixed baselines such as max pressure;
- neural policies.

For imitation learning, phase selection is not a training target. The model
learns only to regress movement scores from the max-pressure teacher. Phase
scores are deterministic sums of movement scores, and the selected phase is the
maximum-scoring selectable phase. PPO may later turn aggregated phase scores
into a probability distribution for exploration, but that is not part of the
initial IL baseline.

## Detection And Model Features

The first next step is defining the feature schema. The active movement system
needs a new feature extractor. It should not inherit the legacy 45-dimensional
fixed-slot vector directly.

### Required Dynamic Features

For each movement, collect at least:

- incoming lane halting queue;
- outgoing lane halting queue;
- pressure = incoming queue - outgoing queue;
- incoming lane vehicle count;
- outgoing lane vehicle count;
- incoming lane waiting time;
- outgoing lane waiting time;
- incoming lane occupancy if available from TraCI;
- outgoing lane occupancy if available from TraCI;
- current signal state for the movement;
- elapsed time since that movement or its current phase became green;
- whether the movement is currently enabled by the active phase.

The feature schema should stay close to realistic traffic-light sensing. In
real deployments, assume detector-area measurements on lanes approaching a
signalized junction, not perfect knowledge of all vehicles on the network and
not route plans for individual cars. The default detector window is the last
200 m before the stop line, or the full lane if the lane is shorter. Generated
grids may read from TraCI lane APIs initially, but the feature meaning should
match this detector-window interpretation where possible.

Movement direction/intention should be inferred only from topology and lane
assignment. For example, a dedicated left-turn lane makes the left-turn
movement observable because cars in that lane can only serve that movement.
The model should not require future vehicle routes beyond what lane placement
implies.

For each lane, collect:

- lane length;
- speed limit;
- lane index/order;
- number of lanes on the edge;
- vehicle count;
- halting vehicle count;
- accumulated waiting time;
- mean speed if available;
- occupancy if available.

For each junction, collect:

- number of movements;
- number of selectable phases;
- current signal state or per-movement enabled mask;
- time since last accepted phase switch;
- minimum-green remaining time;
- phase incidence matrix as metadata for deterministic phase aggregation.

### Static Topology Features

For graph learning and transfer, include static edge/junction metadata:

- edge length;
- number of lanes;
- speed limit;
- distance between signalized intersections;
- shortest-path hop distance between traffic lights;
- whether an edge enters or leaves a signalized junction;
- movement relation from incoming lane to outgoing lane;
- conflict metadata, if useful, such as the number of phases that serve a
  movement or number of movements conflicting with it.

For generated grids, start with a directed graph over signalized junctions.
Each directed graph edge represents traffic flow from one signalized junction
to another along the corresponding road direction. The two physical directions
between a pair of junctions are separate graph edges with separate attributes.
This is important because movement features and downstream pressure are
directional.

For multi-hop context, start with three directed graph hops. Edge attributes
should include physical distance and free-flow travel time, so the model can
learn that a far-away junction reached in three sparse-city hops is not the
same as a nearby junction reached in three dense-grid hops.

For OSM/city networks, graph construction is intentionally deferred. The likely
future direction is still signalized-junction nodes, with graph edges that may
compress paths through non-signalized junctions, but deciding which city
junctions count as main graph neighbors is outside the first IL milestone.

### Candidate Graph Structures

There are three plausible graph granularities.

Recommended first implementation for generated grids: directed
signalized-junction graph with movement heads.

- Node = traffic-light junction.
- Directed edge = road direction from one signalized junction to another.
- Node features = local aggregate traffic-light and detector state.
- Edge features = length, lane count, speed, free-flow travel time, and related
  static road metadata.
- Model output = one score per movement of each junction.
- Advantage = closest to legacy `GraphBuilder`, easiest to get working.
- Tradeoff = movement-level detail must be attached to the junction embedding
  through a movement head rather than propagated as first-class graph nodes.

More expressive later option: movement graph.

- Node = movement.
- Edges connect movements that share lanes, outgoing edges, junctions, or
  upstream/downstream topology.
- Model output = scalar per movement node.
- Advantage = matches the movement-based policy directly.
- Tradeoff = more engineering effort for batching, graph construction, and
  evaluation.

Hybrid option: lane/movement bipartite graph.

- Nodes = lanes and movements.
- Edges encode lane-to-movement and movement-to-outgoing-lane relations.
- Advantage = cleanest representation of shared-lane effects.
- Tradeoff = probably too much complexity before IL works.

For the first IL milestone, use the junction graph with movement-head output.
Keep the schema flexible enough to move to a movement graph later. The exact
neural representation of movement rows, junction embeddings, and directional
incoming/outgoing context still needs one more design pass before model
implementation.

### Current Graph Representation Question

The working conceptual model is:

- graph nodes are signalized junctions;
- graph edges are directed road links between signalized junctions in generated
  grids;
- each junction owns controlled movements;
- each selectable phase owns a set of movement indices;
- phase scores are deterministic sums over movement scores.

The unresolved model question is where information should live during neural
message passing. A simple junction GNN can propagate aggregate upstream and
downstream context between traffic lights, then score each local movement from
the parent junction embedding plus movement-specific lane/direction features.
This is likely the first implementation. A more explicit design could encode
incoming-direction and outgoing-direction context separately before combining
them into a movement score, which may better match max pressure but requires a
more careful graph schema.

One important distinction: physical edge direction and neural message direction
are not necessarily the same thing. A physical directed edge `A -> B` means
traffic flows from junction A toward junction B. For control at A, downstream
state at B can still be relevant to the movement that exits A toward B. The
model design may therefore need bidirectional message passing over directed
road edges, or separate relation types for upstream-to-downstream and
downstream-to-upstream information, while keeping the physical road edge itself
directional.

## Model Definition

The model should be a movement-scoring GNN.

Input:

- graph of signalized junctions;
- directed edge index between signalized junctions;
- node features containing local aggregate dynamic state;
- edge features containing static road topology;
- movement feature rows associated with parent junctions;
- phase incidence data for deterministic phase aggregation outside the model.

Output:

- scalar score per controlled movement.

Phase selection:

```text
phase_score[p] = sum(movement_score[m] for m in phase[p].enabled_movements)
selected_phase = argmax(phase_score)
```

For evaluation, the network policy should plug into the same phase-selection
path as max pressure. That keeps comparison fair and reduces duplicate logic.

Legacy `GATPolicy` can be adapted by:

- keeping the encoder + GATv2 message-passing backbone;
- replacing the fixed classifier head with a movement scoring head;
- optionally adding a value head later for PPO;
- retaining edge features and normalizer support.

Inputs and targets must be normalized. Raw detector/lane features should be
stored in datasets, with a normalizer fitted during training and saved in the
checkpoint. Teacher movement scores should also be normalized for training,
while preserving enough metadata to reconstruct or compare unnormalized
max-pressure behavior during inspection.

## Data Collection

The first dataset should be collected from simulation using the max-pressure
policy as the teacher.

At each decision interval, record:

- network id / config path;
- SUMO seed and demand seed;
- simulation time;
- traffic light id;
- `TrafficLightProgram` metadata or stable IDs for movements and phases;
- current phase state and min-green state;
- movement features;
- selectable phase incidence;
- teacher movement scores from max pressure;
- teacher selected phase;
- optionally queue-policy selected phase for diagnostics;
- traffic metrics after the step.

The minimum useful training sample is:

```text
features -> teacher_movement_scores
```

The minimum useful evaluation sample is:

```text
features -> model_movement_scores -> phase aggregation -> selected phase
```

### Target Choice

The IL target is direct movement-score regression:

- target = max-pressure movement score;
- loss = MSE or Huber over movement scores;
- phase choice emerges from deterministic aggregation;
- simple and aligned with the baseline;
- normalized for stable optimization.

Do not train phase-level ranking, phase classification, or phase cross-entropy
for the initial IL baseline. Those phase decisions are derived deterministically
from movement scores.

## Imitation Learning Plan

### Stage 1: Feature Schema And Collector

Implement:

- movement feature dataclasses;
- graph feature dataclasses;
- collector that runs `MovementControlRuntime`;
- expert policy wrapper for max pressure;
- dataset writer.

Use generated grids first:

- 3x3 for fast debugging;
- 4x4 and 6x6 for training scale;
- fixed seeds for reproducible tests;
- randomized seeds for dataset diversity.

Legacy references:

- `src/legacy/training/imitation.py` for the training loop shape;
- `src/legacy/utils/graph_builder.py` for graph construction and normalizer;
- `src/legacy/utils/demand_generator.py` for randomized demand;
- `src/legacy/training/eval_episode.py` for metrics.

### Stage 2: Movement-Scoring Model

Implement:

- movement-aware graph builder;
- normalizer;
- GNN movement scorer;
- checkpoint format.

Initial model:

- GATv2 backbone over junction graph;
- per-junction movement encoder;
- movement score head;
- optional phase-score computation helper for training/evaluation.

Legacy reference:

- `src/legacy/model/gat_policy.py`

### Stage 3: IL Training

Implement:

- dataset loader;
- train loop for movement-score regression;
- TensorBoard logging;
- model checkpointing;
- overfit test on one small dataset;
- evaluation after each epoch on fixed seeds.

Suggested first losses:

```text
loss = movement_score_huber
```

The first deterministic training check is to overfit one batch collected from a
fixed seed. Evaluation should then run the learned policy and max-pressure
teacher on the same fixed seed. If the overfit model reproduces the teacher's
movement scores and selected phases on that deterministic setup, the collector,
training loop, model wrapper, and evaluation path are all wired correctly.

Legacy reference:

- `src/legacy/training/imitation.py`
- `scripts/train_il.py`

### Stage 4: Policy Evaluation

Implement a policy runner that can run:

- max pressure baseline;
- queue baseline;
- learned network policy.

Evaluation should run policies on identical configs and seeds:

```text
for seed in seeds:
    run max_pressure(seed)
    run network_policy(seed)
compare averaged metrics
```

Metrics:

- completed vehicles / throughput;
- average waiting time;
- average travel time;
- max queue length;
- average queue length;
- wait density;
- phase switch frequency;
- per-junction wait density;
- optional green-wave metrics from legacy evaluation.

Legacy reference:

- `src/legacy/training/eval_episode.py`
- `scripts/eval_city.py`

## PPO Plan After IL

Only start PPO once IL can reproduce max pressure well enough to be useful.

PPO changes relative to legacy:

- model output is movement scores;
- phase logits are computed by aggregating movement scores;
- action distribution is over local selectable phases;
- action counts are variable per junction;
- critic can be per-junction or graph-level.

The first PPO version can use per-junction categorical distributions over the
local phase list after aggregation. During batching, store selected local phase
indices and phase masks/incidence matrices.

Legacy references:

- `src/legacy/training/ppo.py`
- `src/legacy/training/rollout.py`
- `scripts/train_rl.py`

Reuse:

- GAE;
- clipped policy objective;
- entropy bonus;
- value warmup;
- rollout buffer concept;
- checkpointing and evaluation cadence.

Replace:

- fixed `NUM_PHASES`;
- fixed phase categorical output;
- fixed observation vector.

## OSM And City Network Plan

After the generated-grid IL path works, port the OSM build path.

Required changes:

- adapt `scripts/build_network.py` to call movement conflict phase synthesis;
- remove fixed canonical phase generation;
- ensure generated `.tll.xml` programs load cleanly;
- ensure every signalized junction has movement programs extracted by
  `src/movement/sumo_adapter.py`;
- add a simple run command for city configs with baseline policies;
- add visual verification with SUMO-GUI.

Legacy references:

- `scripts/build_network.py`
- `scripts/tools/diagnose_network.py`
- `scripts/tools/inspect_junctions.py`

The OSM build should eventually produce:

```text
configs/<city>/<city>.net.xml
configs/<city>/<city>.tll.xml
configs/<city>/<city>.rou.xml
configs/<city>/<city>.add.xml
configs/<city>/<city>.sumocfg
```

The run command should look similar to:

```powershell
python scripts\run.py --cfg configs\city\city.sumocfg --method max-pressure --gui
```

Later, learned policy evaluation should use the same config:

```powershell
python scripts\eval_policy.py --cfg configs\city\city.sumocfg --policy checkpoint.pt --baseline max-pressure
```

## Transfer Learning Plan

Before starting PPO or OSM/city-network work, run a small generated-grid
transfer check:

1. Train IL on a generated 4x4 grid.
2. Evaluate the same checkpoint zero-shot on a generated 5x5 or 6x6 grid.
3. Compare against max pressure and queue baselines on identical seeds.

Later transfer work:

1. Train on generated 4x4/6x6 grids.
2. Evaluate zero-shot on larger generated grids.
3. Train on one OSM city network.
4. Evaluate zero-shot on another OSM network.
5. Train on multiple networks.
6. Evaluate transfer matrix: train network set vs evaluation network set.

Normalization matters here. The legacy `RunningNormalizer` and
`scripts/renormalize.py` are useful references.

Possible approaches:

- one global normalizer accumulated over all training networks;
- per-network normalizer for evaluation;
- hybrid: global normalization plus explicit static scale features;
- no normalization for static categorical/binary features.

First recommendation:

- use one global normalizer for dynamic numeric features during multi-network
  training;
- freeze it for evaluation;
- add a renormalization utility later only if zero-shot transfer is unstable.

Legacy references:

- `src/legacy/utils/graph_builder.py`
- `scripts/renormalize.py`

## Proposed Implementation Order

### Milestone 1: Movement Feature Schema

Deliverables:

- `src/movement/features.py`
- dataclasses for lane, movement, junction, and graph features;
- feature extraction from TraCI/runtime;
- tests with fake lane APIs and small synthesized programs.

Acceptance:

- features can be extracted for `grid_3x3_dedicated`;
- movement feature rows align exactly with `TrafficLightProgram.movements`;
- phase incidence matrix aligns with `TrafficLightProgram.selectable_phases`.

### Milestone 2: Dataset Collection

Deliverables:

- `scripts/collect_il_data.py`;
- dataset schema documentation;
- max-pressure teacher collection;
- reproducible seed handling;
- small generated sample dataset.

Acceptance:

- can collect from 3x3 and 6x6 grids;
- saved samples include features, teacher movement scores, selected phases,
  and metadata;
- replay/inspection script can print a sample and reconstruct the selected
  phase.

### Milestone 3: Model And Offline IL

Deliverables:

- movement-scoring GNN model;
- graph builder and normalizer;
- offline dataset loader;
- training script;
- checkpoint format.

Acceptance:

- model overfits a tiny dataset;
- model matches max-pressure phase choices on held-out seeds better than a
  random or queue-only baseline;
- model can be loaded by a run/eval script.

### Milestone 4: Evaluation Harness

Deliverables:

- `scripts/eval_policy.py`;
- common metrics dataclass;
- baseline-vs-model same-seed comparison;
- CSV/JSON output.

Acceptance:

- max pressure and learned policy run on identical demand seeds;
- evaluation reports throughput, waiting time, travel time, queue metrics, and
  switch frequency;
- generated-grid evaluation is reproducible.

### Milestone 5: Generated-Grid Transfer Check

Deliverables:

- train-on-4x4/evaluate-on-5x5-or-6x6 command path;
- same-seed baseline comparison against max pressure and queue;
- summarized zero-shot transfer metrics.

Acceptance:

- one IL checkpoint trained on 4x4 can be evaluated unchanged on a larger grid;
- evaluation clearly reports whether transfer preserves max-pressure-like
  behavior, degrades gracefully, or fails.

### Milestone 6: OSM Build Integration

Deliverables:

- updated OSM build script using movement synthesis;
- city run script or documented run command;
- visual verification path.

Acceptance:

- city network builds;
- baseline policy runs visually;
- movement programs are extracted for all intended traffic lights.

### Milestone 7: PPO

Deliverables:

- movement-policy PPO rollout buffer;
- actor-critic model head;
- PPO training loop;
- IL checkpoint warm start.

Acceptance:

- PPO starts from an IL checkpoint;
- action distribution handles variable phase counts;
- evaluation compares PPO, IL, and max pressure on same seeds.

## Open Design Questions

The following should be decided before implementing the model:

1. How exactly should movement rows attach to junction embeddings?

   Current recommendation: use a directed signalized-junction graph for message
   passing, then score flattened movement rows from the parent junction
   embedding plus movement-specific lane/direction features. This avoids a
   fixed global maximum movement count and keeps a path open to a future
   movement graph.

2. Should the model explicitly separate upstream and downstream context?

   Max pressure is naturally directional: incoming pressure and downstream
   outgoing pressure are different quantities. A plain junction GNN may be
   enough for the first baseline, but a more structured model could maintain
   separate embeddings for incoming and outgoing directed edges before
   combining them into movement scores.

3. How should movement-score targets be normalized?

   The target remains pure max pressure, but training should normalize target
   values. The exact target scaling should preserve the ordering needed for
   phase aggregation while avoiding unstable magnitude differences across
   networks.

4. Should normalized features be computed online during collection or offline
   during training?

   Recommendation: collect raw numeric features, compute normalizer during
   training, save normalizer with checkpoint.

5. How much neighboring context is needed after the first baseline?

   Current decision: start with three directed graph hops and include physical
   distance, free-flow travel time, speed, and lane count as edge features.
   Revisit only after generated-grid IL and transfer evaluation produce data.

## Immediate Next Step

Start with Milestone 1: define movement feature extraction.

The first concrete implementation should produce, for each traffic light:

- movement feature matrix `M x F`;
- phase incidence matrix `P x M`;
- static movement metadata;
- current runtime state features;
- teacher movement scores from max pressure.

Once that exists, dataset collection and IL training become straightforward.
