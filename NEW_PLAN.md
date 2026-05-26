# Movement-Aware GNN Reinforcement Learning for Transferable Traffic Signal Control

## 1. Project Goal

The goal is to build a multi-agent reinforcement learning system for traffic signal control that can train across many automatically imported city networks and generalize to unseen city networks without requiring manually defined intersection templates or fixed phase catalogs.

The central idea is:

> Use the legal signal phases already defined by SUMO/OSM-derived traffic-light programs, extract the movements served by each phase, and train a shared neural policy that scores each legal phase based on movement embeddings computed from a directed road-network GNN.

The intended result is not merely a controller that performs well on one synthetic grid, but a representation that can handle:

* irregular intersections,
* T-junctions,
* different numbers of phases,
* different lane counts,
* priority/give-way junctions between traffic lights,
* automatically imported OSM city networks,
* transfer to unseen cities.

## 2. Core Research Hypothesis

A traffic-signal policy that scores simulator-extracted legal phases through movement embeddings derived from a directed, travel-time-aware road-network GNN will generalize better to unseen road networks than policies using fixed intersection-level phase IDs or manually defined intersection templates.

More concretely:

> The model should learn what a phase does, based on the movements it serves and the spatial traffic context around those movements, rather than learning arbitrary phase indices such as `phase_0`, `phase_1`, etc.

## 3. High-Level Architecture

```text
SUMO/OSM road network
    ↓
road-link / lane-group graph
    ↓
directed travel-time-aware GNN
    ↓
contextual link embeddings
    ↓
movement embeddings: incoming link -> outgoing link
    ↓
phase embeddings: aggregate movements enabled by phase
    ↓
shared phase scorer
    ↓
categorical action over legal phases
    ↓
SUMO setPhase() wrapper with min-green/yellow/all-red handling
```

## 4. Scope and Non-Goals

### In Scope

* SUMO-based simulation.
* Automatically imported OSM networks.
* Random and structured synthetic demand generation.
* Multi-agent RL with one agent per controlled traffic light.
* Shared policy across all traffic lights and networks.
* Variable action spaces via per-intersection legal phase scoring.
* Road-topology-aware GNN context.
* Transfer evaluation on held-out city networks.

### Out of Scope Initially

* Real-world deployment.
* Camera/perception integration.
* Exact real-world detector placement.
* Calibrated real-world traffic demand.
* Raw red/yellow/green state-string generation.
* Manually defining a fixed set of intersection types.
* Manually defining phase semantics such as “north-south green.”

## 5. Demand Generation

### 5.1 Initial Random Demand

The first implementation should use randomly generated routes to make the system trainable and debuggable.

Random demand is acceptable initially because the primary research target is representation and transfer across topology, not calibrated real-world traffic modeling.

However, pure random demand is too weak if used alone. It may fail to create realistic directional pressure, bottlenecks, rush-hour patterns, and corridor effects.

### 5.2 Rush-Hour Demand

Rush-hour scenarios should be added early, starting with simple grid-based experiments.

For synthetic grids, rush-hour demand should be deliberately simple and inspectable:

```text
For a 3x3 or 6x6 grid:
  choose one node, corner, edge node, or central node as the demand center
  inbound rush:
    many trips from outer grid nodes -> demand center
  outbound rush:
    many trips from demand center -> outer grid nodes
```

This allows the controller to be tested on directional pressure, corridor formation, queue propagation, and green-wave behavior before moving to real OSM networks.

A later city-network rush-hour model:

```text
For each city network:
  choose one or more activity centers
  morning rush:
    many trips from residential/peripheral areas -> activity center
  evening rush:
    many trips from activity center -> residential/peripheral areas
```

The activity center can be selected by one of several heuristics:

1. **Centrality-based center**

   * choose a high-betweenness or high-closeness region of the road graph.

2. **OSM land-use/POI based**

   * choose areas tagged as commercial, industrial, retail, office, university, etc.

3. **Randomized multiple centers**

   * choose several centers per city to avoid overfitting to single-center flows.

Recommended progression:

```text
1. Grid demand center: manually selected corner/center node
2. Grid demand center: randomized node per episode
3. Synthetic irregular network: graph-centrality center
4. OSM city network: graph-centrality center
5. OSM city network: optional POI/land-use-based centers
```

Use graph-centrality centers first on non-grid networks.
Add OSM POI/land-use semantics later.

### 5.3 Demand Distribution Variations

Training should vary demand across episodes:

* low / medium / high demand,
* uniform random OD,
* center-bound rush hour,
* center-outbound rush hour,
* multi-center demand,
* corridor-heavy demand,
* asymmetric directional demand,
* local short trips vs cross-city trips.

This is important because otherwise the policy may generalize to topology but not to demand distributions.

## 6. Observability and Traffic State

A major design issue is that simulation gives access to far more traffic state than a real-world system would normally have.

Therefore, the project should define observability tiers.

### 6.1 Tier 1: Oracle Simulation State

The controller can observe detailed traffic state on all relevant road links/lanes:

* vehicle count,
* queue length,
* waiting time,
* speed,
* occupancy,
* exact route intentions,
* exact movement demand,
* downstream congestion,
* upstream approaching vehicles.

This is useful for debugging and as an upper-bound experiment.

But it is not realistic.

### 6.2 Tier 2: Detector-Like State

The controller observes only information that could plausibly be available from loop detectors, radar, cameras, or connected infrastructure near controlled junctions:

* incoming lane/link vehicle count,
* queue estimate,
* waiting time estimate,
* speed/occupancy estimate,
* possibly detector readings on selected upstream links,
* no exact route intention.

Movement demand must be estimated:

```text
movement_demand(in -> out)
  = incoming_count(in) × estimated_turn_probability(in -> out)
```

Turning probabilities may come from:

* route-generation priors,
* historical rolling estimates,
* recent observed exits,
* learned latent estimators.

### 6.3 Tier 3: Local-Only State

The controller observes only local intersection approaches:

* incoming approach queue,
* current phase,
* elapsed phase time,
* maybe downstream capacity estimate.

No full-network state is available.

This is the most realistic but may severely limit green-wave anticipation.

### 6.4 Recommended Experimental Strategy

Train and evaluate multiple observability variants:

```text
A. Oracle full-link state
B. Full-link state without exact route intentions
C. Detector-like state near controlled intersections
D. Local-only state
```

This allows a clean answer to:

> How much of the performance depends on unrealistic simulator observability?

## 7. Road Graph Representation

### 7.1 Problem: Raw SUMO Segments Can Be Too Fragmented

OSM/SUMO networks often contain many short road segments caused by geometry rather than actual decision points.

Example:

```text
A road bends slightly:
  segment_1 -> segment_2 -> segment_3 -> segment_4
```

If each segment becomes a separate GNN node, then a 3-hop GNN may only see a few hundred meters of the same road and fail to capture meaningful upstream traffic context.

This would make hop count dependent on arbitrary OSM/SUMO segmentation.

### 7.2 Recommended Solution: Compress Road Chains

Before constructing the GNN graph, merge consecutive road segments that do not introduce a meaningful traffic decision.

Merge segments if:

* the intermediate node has in-degree 1 and out-degree 1,
* there is no traffic light,
* there is no priority/give-way conflict that should be modeled explicitly,
* there is no lane split/merge,
* there is no significant change in lane count,
* there is no route choice introduced,
* the road class is compatible,
* directionality is preserved.

Do not merge across:

* signalized intersections,
* priority/give-way intersections with multiple incoming roads,
* roundabouts,
* merges/splits,
* turns where route choice changes,
* lane count changes,
* road class changes if relevant.

Compressed node:

```text
compressed_link = chain of original SUMO edges/lanes
```

Features should aggregate from the original segments:

* total length,
* free-flow travel time,
* weighted mean speed,
* total vehicle count,
* total queue length,
* total waiting time,
* minimum downstream capacity,
* lane count statistics.

### 7.3 Lane-Level vs Link-Level Nodes

There are two possible granularities.

#### Lane-Level Nodes

Pros:

* most precise,
* captures turn lanes,
* closer to SUMO controlled links,
* better movement demand modeling.

Cons:

* many nodes,
* more fragmented,
* harder batching,
* more unrealistic observability.

#### Link/Lane-Group Nodes

Pros:

* fewer nodes,
* easier GNN inference,
* more robust to noisy segmentation,
* better for transfer.

Cons:

* may lose turn-lane specificity,
* harder to model separate left/through/right queues.

Recommended initial choice:

```text
Use lane groups / compressed directed links as GNN nodes.
Preserve lane-level controlled-link information only where needed for movement extraction.
```

### 7.4 Network Curriculum

The project should not start directly on arbitrary OSM city networks.

Recommended progression:

```text
Stage A: 3x3 synthetic grid
Stage B: 6x6 synthetic grid
Stage C: synthetic irregular network
Stage D: one real OSM/SUMO city patch
Stage E: multiple OSM/SUMO city patches for transfer
```

Reason:

* synthetic grids make extraction, action mapping, reward, and RL debugging tractable;
* 3x3 is small enough to inspect manually;
* 6x6 tests scalability and multi-agent coordination;
* synthetic irregular networks introduce T-junctions, varying lane counts, and nonuniform phases in a controlled setting;
* real city networks should only be used after the phase/movement extraction pipeline is validated;
* multi-city transfer should be the final stage, not the first implementation target.

This curriculum prevents ambiguous OSM import artifacts from being confused with model or RL bugs.

## 8. Handling Uncontrolled Junctions

Uncontrolled junctions include:

* priority junctions,
* give-way junctions,
* stop signs,
* uncontrolled merges,
* minor residential intersections,
* roundabouts without signal control.

They are not RL agents.

They are handled by SUMO’s built-in right-of-way and priority logic during simulation.

In the learned graph representation, they should appear as part of the road topology.

### 8.1 Simple Case: Pass-Through Junction

If the junction does not introduce a meaningful choice or conflict, compress it into a single edge/link chain.

```text
road_segment_before -> uncontrolled junction -> road_segment_after
```

becomes:

```text
compressed_link
```

### 8.2 Choice/Conflict Junction

If the uncontrolled junction introduces possible route choices, merges, priority delays, or conflicts, keep it as graph structure.

Options:

1. Keep as road-link transitions:

```text
incoming_link -> outgoing_link
```

with edge features:

* turn type,
* priority/give-way indicator,
* expected delay,
* conflict count,
* free-flow travel time.

2. Compress between signalized intersections but add corridor-level features:

```text
signal_A -> signal_B edge features:
  distance
  free-flow travel time
  number of uncontrolled junctions
  number of priority conflicts
  number of merges/splits
  estimated travel time
```

Recommended initial approach:

```text
Use compressed road-link graph.
Keep uncontrolled junction effects as edge features when they affect traffic flow or route choice.
Let SUMO itself handle the exact priority dynamics.
```

## 9. Directed GNN Design

### 9.1 Node Definition

Node = compressed directed road link or lane group.

Node features:

```text
static:
  length
  lane count
  speed limit
  capacity estimate
  road type
  distance to next signal
  is incoming to traffic light
  is outgoing from traffic light

dynamic:
  vehicle count
  queue length
  mean waiting time
  mean speed
  occupancy
  estimated inflow
  estimated outflow
```

### 9.2 Edge Definition

Directed edge exists if traffic can flow from road-link node `u` to road-link node `v`.

Edge features:

```text
free-flow travel time
length / connector distance
turn type
priority/give-way flag
merge/split indicator
uncontrolled-junction indicator
intermediate junction count if compressed
capacity ratio
```

### 9.3 Directional Message Passing

Use separate upstream and downstream channels.

```text
Downstream channel:
  messages follow driving direction
  useful for downstream capacity and spillback

Upstream channel:
  messages go against driving direction
  useful for approaching vehicles and expected arrivals
```

For each link:

```text
h_link = concat(h_local, h_upstream, h_downstream)
```

### 9.4 Travel-Time Weighting

Messages should be aware of travel time.

Simple implementation:

```text
message_weight = exp(- travel_time / τ)
```

Learned implementation:

```text
message_weight = attention(h_source, h_target, edge_features)
```

Recommended initial version:

```text
Use edge features including travel time.
Let the message MLP or attention mechanism learn how strongly to use them.
Add explicit travel-time decay later as an ablation.
```

### 9.5 Hop Depth

A GNN with `K` layers gives approximately `K`-hop context.

Initial values:

```text
K = 2, 3, 4
```

Ablation:

```text
No GNN
1-hop
2-hop
3-hop
4-hop
```

Because raw road segmentation can distort hop meaning, this should be evaluated after road-chain compression.

## 10. Movement Extraction

For each traffic light, extract controlled movements from SUMO.

A raw controlled movement is:

```text
raw_movement m = incoming lane/link -> outgoing lane/link
```

Using SUMO traffic-light controlled links, each controlled link maps to:

```text
incoming lane
outgoing lane
via lane, if present
```

After graph compression, map each lane/link to its corresponding compressed road-link or lane-group node.

Then each raw movement becomes:

```text
compressed_in_link/lane_group -> compressed_out_link/lane_group
```

### 10.1 Important Distinction: Road Compression vs Movement Semantics

Road-chain compression must not collapse movements into generic categories such as:

```text
left / straight / right
```

Compression only removes non-decision geometry between relevant junctions.

Movements remain defined by legal SUMO-controlled incoming-to-outgoing connections.

### 10.2 Lane-Blocking-Aware Movement Grouping

A crucial issue is that movement separability depends on lane structure.

If one incoming approach has only a single lane, then left, straight, and right movements are not independently serviceable in practice:

```text
one incoming lane:
  first vehicle wants to turn left
  following vehicles want to go straight/right
  entire lane may be blocked
```

Therefore, for one-lane approaches, it is usually misleading to treat left/straight/right as fully independent demand channels.

Recommended rule:

```text
If multiple outgoing movements originate from the same physical incoming lane
and cannot be served independently because they share the same queue,
represent them as a shared-lane movement group for phase scoring.
```

This does not mean the movements disappear. It means that for scoring and demand estimation, the model should know they are coupled by the same upstream queue.

Example:

```text
raw SUMO movements:
  lane_0 -> west_out   left
  lane_0 -> south_out  straight
  lane_0 -> east_out   right

shared-lane movement group:
  lane_0 -> {west_out, south_out, east_out}
```

If a phase enables all three, the phase should be scored as serving the shared incoming queue, not as independently serving three separate queues.

### 10.3 Multi-Lane / Dedicated-Turn Case

If the incoming approach has separate lane groups, preserve that distinction.

Example:

```text
lane_0: left only
lane_1: straight
lane_2: straight/right
```

Then useful movement groups may be:

```text
left_group:
  lane_0 -> left_out

straight_group:
  lane_1 -> through_out
  lane_2 -> through_out

right_or_shared_group:
  lane_2 -> right_out
```

The grouping should follow actual lane connectivity, not an assumed four-way-intersection template.

### 10.4 Practical Initial Rule

Initial implementation should use the following hierarchy:

```text
SUMO controlled link
    ↓
raw movement: incoming lane -> outgoing lane
    ↓
map lanes to compressed road-link / lane-group nodes
    ↓
group movements only when they share the same physical incoming queue
```

This preserves legal movement semantics while avoiding artificial separation of movements that cannot operate independently.

### 10.5 Movement Features

Movement or movement-group features:

```text
turn type or turn-type distribution
is currently green
queue on incoming lane/group
capacity on outgoing link/group
pressure
estimated movement demand
historical turning probability
shared-lane indicator
number of raw SUMO controlled links in group
```

Pressure:

```text
pressure(m) = normalized_queue(in_link_or_lane_group)
              - normalized_queue(out_link_or_lane_group)
```

If exact turn intention is unavailable, movement demand should be estimated:

```text
movement_demand(in -> out)
  = incoming_queue_or_count(in) × estimated_turn_probability(in -> out)
```

For shared-lane groups, total incoming queue should not be counted multiple times for each possible outgoing direction.

## 11. Phase Extraction

For each traffic light, extract the SUMO traffic-light program.

Each phase has a signal-state string over controlled links.

A selectable green phase is defined as a phase that:

* is not yellow-only,
* is not all-red,
* enables at least one vehicle movement,
* is safe/legal according to SUMO’s traffic-light logic.

For each phase:

```text
M(p) = {movement m | signal_state_for_movement is green}
```

Green states:

```text
G/g -> enabled movement
r/R -> blocked movement
y/Y -> transition, not directly selected as green action
```

The action space of a traffic light is:

```text
A_i = legal selectable green phases at traffic light i
```

This can vary across intersections and cities.

## 12. Movement and Phase Embeddings

### 12.1 Link Embeddings

After GNN message passing:

```text
h_link = contextual embedding of compressed road link
```

### 12.2 Movement Embedding

For each movement:

```text
h_m = MLP_movement([
  h_in_link,
  h_out_link,
  movement_features
])
```

This tells the model:

* what traffic is waiting upstream,
* what capacity exists downstream,
* what kind of turn/movement this is,
* whether the movement has high pressure,
* whether vehicles are expected soon.

### 12.3 Phase Embedding

For each phase:

```text
h_p = sum({h_m | m ∈ M(p)})
```

Initial pooling choice:

```text
sum pooling
```

Reason:

* permutation-invariant,
* simple,
* stable,
* physically interpretable as total served movement benefit.

Later ablations:

* mean pooling,
* max pooling,
* attention pooling,
* pressure-weighted pooling.

### 12.4 Phase Score

For each phase:

```text
logit_p = MLP_phase([
  h_p,
  is_current_phase,
  elapsed_green_time,
  switch_penalty,
  min_green_satisfied
])
```

The same `MLP_phase` is used for all phases, all intersections, and all cities.

## 13. Action Selection and SUMO Control

At each controlled traffic light:

```text
policy outputs logits over legal selectable green phases
```

Training:

```text
sample phase from categorical distribution
```

Evaluation:

```text
choose argmax phase
```

The selected phase is passed to SUMO using the corresponding SUMO phase index.

Control wrapper handles:

* minimum green time,
* yellow transition,
* all-red transition,
* invalid action masking,
* decision interval.

Recommended initial decision interval:

```text
15 seconds
```

Ablate:

```text
10s, 15s, 20s, 30s
```

## 14. Reinforcement Learning Setup

### 14.1 Agents

One agent per controlled traffic light.

All agents share the same actor network.

### 14.2 Policy

The actor is a shared phase scorer.

It supports variable local action spaces because it scores each legal phase independently.

### 14.3 Critic

Initial recommendation:

```text
Use a centralized critic during training.
Use decentralized actors during execution.
```

The critic may receive richer global or regional state than the actor.

This stabilizes multi-agent RL while preserving decentralized control at inference.

### 14.4 Algorithm

Recommended initial algorithm:

```text
PPO / MAPPO-style actor-critic
```

Reason:

* handles discrete actions,
* does not require differentiable environment dynamics,
* stable baseline,
* compatible with shared policy and centralized critic.

## 15. Rewards

Candidate reward components:

```text
local waiting time reduction
local queue length reduction
pressure reduction
throughput increase
spillback penalty
global average waiting time penalty
```

Recommended initial reward:

```text
r_i = - local_average_waiting_time
      - λ_queue * local_queue_length
      - λ_spillback * downstream_spillback
```

Alternative pressure reward:

```text
r_i = - sum absolute movement pressure
```

A global or regional term may be added:

```text
r_i_total = local_reward_i + α * regional_reward_i + β * global_reward
```

Start simple. Reward design can dominate results.

## 16. Baselines

Required baselines:

1. Fixed-time SUMO program.
2. SUMO actuated control, if available.
3. Max-pressure control.

Optionals:

4. PressLight-style pressure RL.
5. Intersection-level GNN/MARL baseline.
6. Proposed model without road-link GNN.
7. Proposed model without movement-based phase scoring.

## 17. Key Ablations

Representation:

```text
fixed phase ID policy
vs movement-based phase scoring
```

GNN context:

```text
no GNN
1-hop
2-hop
3-hop
4-hop
```

Travel-time modeling:

```text
no edge travel time
travel time as edge feature
explicit travel-time decay
```

Observability:

```text
oracle full state
detector-like state
local-only state
```

Turning information:

```text
oracle route intention
estimated turning probabilities
no turn information
```

Demand:

```text
random demand
single-center rush hour
multi-center rush hour
asymmetric corridor demand
```

Transfer:

```text
train single city -> test same city
train single city -> test unseen city
train many cities -> test unseen city
many cities + fine-tune on target city
```

## 18. Implementation Plan

The implementation should progress through increasingly difficult network types. Do not start with arbitrary real city networks.

### Phase 0: Synthetic Network Curriculum Setup

Create or generate a sequence of training/evaluation networks:

```text
A. 3x3 grid
B. 6x6 grid
C. irregular synthetic network
D. one real OSM/SUMO city patch
E. multiple real OSM/SUMO city patches
```

Goal:

```text
Separate model/RL bugs from OSM import and traffic-light-definition problems.
```

### Phase 1: 3x3 Grid Environment Sanity Check

* Build or generate a 3x3 signalized grid.
* Use simple fixed phases.
* Generate simple random demand.
* Extract traffic lights.
* Extract controlled links.
* Extract phases and selectable green phases.
* Build movement sets per phase.
* Implement simple phase-selection wrapper.
* Run fixed-time and max-pressure baselines.
* Manually inspect phase/movement mappings.

Goal:

```text
Environment setup works and max-pressure makes sensible decisions.
```

### Phase 1.5: Basic PPO Phase Selection on 3x3 Grid

* Train basic PPO without GNN.
* Use local traffic features only.
* Use a simple fixed action interval.
* Compare against fixed-time and max-pressure.

Goal:

```text
End-to-end RL loop works before adding movement pooling or GNN context.
```

### Phase 2: Movement-Based Phase Scoring on 3x3 Grid

* Build movement embeddings from local link/lane-group features.
* Implement shared-lane-aware movement grouping.
* Pool movements into phase embeddings.
* Score legal phases with a shared MLP.
* Compare to fixed phase-ID baseline.

Goal:

```text
Variable action-space phase scoring works and does not depend on arbitrary phase IDs.
```

### Phase 3: Scale to 6x6 Grid

* Run the same pipeline on a 6x6 grid.
* Increase number of agents.
* Test multi-agent stability.
* Add simple rush-hour demand.
* Choose one grid node, corner, or central node as a demand center.
* Test inbound rush-hour traffic toward the center.
* Test outbound rush-hour traffic away from the center.
* Randomize the demand center across episodes as a later variant.
* Compare fixed-time, max-pressure, and PPO variants.

Goal:

```text
Verify scaling from toy grid to larger controlled grid.
```

### Phase 4: Road-Link Graph and Compression

* Build directed road-link graph.
* Implement road-chain compression.
* Preserve mapping from SUMO lanes/edges to compressed nodes.
* Add static and dynamic link features.
* Validate compression visually and with printed summaries.

Goal:

```text
Topology representation is robust to SUMO/OSM-style segmentation.
```

### Phase 5: Directed GNN Context

* Add upstream/downstream directed GNN channels.
* Add edge features including travel time.
* Compare no-GNN vs 1/2/3/4-hop GNN.
* Compare edge travel-time feature vs explicit travel-time decay.

Goal:

```text
Spatial anticipation improves control beyond local movement scoring.
```

### Phase 6: Irregular Synthetic Network

* Introduce non-grid geometry.
* Include T-junctions.
* Include different lane counts.
* Include different numbers of legal phases.
* Include shared one-lane approaches.
* Include some uncontrolled priority/give-way junctions.
* Validate movement extraction and shared-lane grouping.

Goal:

```text
Test irregularity without full OSM messiness.
```

### Phase 7: First Real OSM/SUMO City Patch

* Import one small city patch.
* Validate traffic-light programs.
* Validate phase extraction.
* Validate movement extraction.
* Validate road-chain compression.
* Filter malformed, duplicate, pedestrian-only, yellow-only, and all-red phases.
* Train/evaluate with random and rush-hour demand.

Goal:

```text
First real network runs without manual phase templates.
```

### Phase 8: Demand Variation

* Add stronger demand randomization.
* Add center-bound rush hour.
* Add center-outbound rush hour.
* Add multi-center demand.
* Add corridor-heavy demand.
* Add asymmetric directional demand.

Goal:

```text
Policy does not overfit to one synthetic demand pattern.
```

### Phase 9: Multi-City Training

* Automatically import multiple OSM city patches.
* Normalize features across networks.
* Train shared policy across cities.
* Evaluate zero-shot on held-out cities.
* Optionally evaluate few-shot fine-tuning.

Goal:

```text
Measure transfer/generalization.
```

### Phase 10: Observability Ablations

* Oracle state.
* Full-link state without exact route intention.
* Detector-like state.
* Local-only state.
* Route-intention vs estimated-turning vs no-turning variants.

Goal:

```text
Quantify realism/performance trade-off.
```

## 19. Network and Phase Validation

Before training on real OSM/SUMO networks, the import and extraction pipeline must be validated explicitly.

This is a required implementation milestone, not optional cleanup.

### 19.1 Things to Validate Manually on Small Networks

For the 3x3 and 6x6 synthetic networks:

* each traffic light is detected correctly;
* each selectable green phase is extracted correctly;
* yellow/all-red phases are not selectable actions;
* SUMO controlled-link indices map to the expected incoming/outgoing lanes;
* phase-to-movement sets are correct;
* shared-lane movement grouping behaves as intended;
* one-lane approaches do not artificially multiply the same queue across left/straight/right movements;
* the selected SUMO phase produces the expected visual traffic-light behavior;
* max-pressure or heuristic control makes qualitatively sensible decisions.

### 19.2 Things to Validate on First Real OSM Network

For the first OSM/SUMO city patch:

* imported traffic-light programs are sane enough to use;
* weird pedestrian-only or empty phases are filtered;
* duplicate green phases are merged or handled consistently;
* uncontrolled junctions are represented as topology/simulation dynamics, not agents;
* road-chain compression does not merge across actual decision points;
* lane-to-compressed-link mappings remain valid after compression;
* selected phases still map to the correct SUMO phase indices.

### 19.3 Validation Tooling

Build small inspection utilities before large-scale training:

```text
print traffic light summary:
  tls_id
  number of controlled links
  number of raw phases
  number of selectable green phases
  movements per phase

visualize:
  road graph
  compressed graph
  controlled movements
  selected phase movements
  shared-lane groups
```

A minimal visual/debug tool is likely worth more than an early complex RL model.

## 20. Main Risks

### 20.1 Overpowered Simulator Observability

Using full link-level state may produce unrealistic performance.

Mitigation:

```text
Make observability an explicit ablation.
Do not claim real-world deployability from oracle-state results.
```

### 20.2 OSM/SUMO Traffic-Light Programs Are Messy

Imported traffic-light logic may contain odd phase structures.

Mitigation:

```text
Filter yellow/all-red/pedestrian-only phases.
Merge duplicate green phases.
Validate each selectable phase has vehicle movements.
Use SUMO legality as source of truth.
```

### 20.3 Fragmented Road Segments Break Hop Semantics

Raw OSM geometry may create too many small graph nodes.

Mitigation:

```text
Compress non-decision road chains.
Use travel-time edge features.
Compare hop-based vs travel-time-radius contexts.
```

### 20.4 Demand Generation May Dominate Results

Policies may overfit to artificial demand patterns.

Mitigation:

```text
Vary demand aggressively.
Evaluate across multiple demand regimes.
Separate topology transfer from demand transfer.
```

### 20.5 Multi-Agent RL Instability

Training may fail due to nonstationarity.

Mitigation:

```text
Start with small networks.
Use shared actor.
Use centralized critic.
Use strong baselines.
Debug with max-pressure-like rewards.
```

## 21. Current Design Decision Summary

Accepted decisions:

* Use SUMO/OSM imported networks.
* Use one RL agent per controlled traffic light.
* Do not output raw red/yellow/green strings.
* Select among SUMO-defined legal green phases.
* Extract movements from SUMO controlled links.
* Represent phases as sets of movements or shared-lane movement groups.
* Do not manually define a universal left/straight/right phase catalog.
* Score phases using shared neural phase scorer.
* Use directed road-link GNN for spatial traffic context.
* Use road-chain compression to avoid raw OSM segment fragmentation.
* Do not collapse movements during road-chain compression.
* Account for one-lane shared-queue blocking in movement grouping.
* Include uncontrolled junctions as topology/simulation dynamics, not as RL agents.
* Start with 3x3 grid, then 6x6 grid, then irregular synthetic, then real OSM, then multi-city transfer.
* Start with simulation-observable state but explicitly ablate observability.
* Use random demand initially, then add rush-hour and multi-center demand.
* Train across many city networks to test transfer.

Open design questions:

* Lane-level vs lane-group/link-level GNN nodes.
* Exact compression rules for road-chain merging.
* Exact shared-lane movement grouping rules.
* Whether to use explicit travel-time decay or learned edge attention.
* Best reward formulation.
* Whether to use PPO, MAPPO, or another actor-critic variant.
* How realistic detector-like observability should be.
* How to generate robust synthetic demand across many cities.
