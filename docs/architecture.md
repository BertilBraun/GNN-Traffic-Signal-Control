# Architecture and Constraints

## Control objective

The controller maps a SUMO traffic state to one legal target phase per signalized junction. It does not generate raw red/yellow/green strings. Network construction synthesizes compatible phases, and the runtime enforces minimum-green and transition rules around the policy's choices.

```text
traffic state
  -> LaneGroup / Movement graph
  -> shared message-passing GNN
  -> one score per Movement
  -> one summed logit per synthesized phase
  -> legal-action mask
  -> one phase decision per junction
```

## Graph nodes

### LaneGroup

A `LaneGroup` is a directed road corridor between controller-relevant endpoints. Opposite directions are different nodes. Unambiguous corridors through unsignalized junctions may be contracted into one lane group; contraction stops at branches where continuing would introduce a reachability shortcut.

Lane-group features describe the final detector region before the downstream junction and include:

- vehicle, moving-vehicle, and queue counts;
- occupancy, mean speed, detector saturation, and available storage;
- short- and long-window arrival and departure rates;
- moving vehicles and predicted arrivals approaching the queue tail;
- minimum and mean queue-tail ETA;
- corridor length, detector length, lane count, and speed limit.

Dynamic counts are normalized by detector capacity. A detector describes its observed region, not an unobserved full-road queue.

### Movement

A `Movement` is a legal signal-controlled turn from one input lane group to one output lane group through a specific traffic light. Movement nodes contain turn-local information such as demand, turn type, controlled-link count, and whether the movement was green at the previous decision.

Junctions are not GNN nodes. They own movements and selectable phases, but appear in visualizations only as spatial anchors.

## Typed message edges

Every movement has four directed message edges:

```text
input LaneGroup  -> Movement
output LaneGroup -> Movement
Movement         -> input LaneGroup
Movement         -> output LaneGroup
```

Message direction is not traffic direction. The output-to-movement edge carries downstream supply and spillback information back to the turn that would feed it.

At a non-controllable junction, every legal pass-through connection becomes a directed `LaneGroup -> LaneGroup` edge instead of a movement node. Its message is weighted by

```text
exp(-connector_freeflow_time / 30 seconds)
```

This lets information cross unsignalized branches without pretending that the policy selects an action there. Signalized junctions never receive these bypass edges: information must pass through their explicit movement nodes.

One macro-hop updates movements from their input/output lane groups, then updates lane groups from movements and unsignalized lane connectors. It lets a movement observe immediately adjacent traffic context while preserving the distinction between controlled and uncontrolled junctions. The reference checkpoint uses one macro-hop.

## Phase scoring

The GNN outputs one scalar score for every movement in the current city graph. For each junction, a phase-incidence matrix records which local movements receive green in each synthesized phase. These phases are the maximal conflict-free movement sets produced by the [city-building pipeline](city_pipeline.md#signal-and-movement-synthesis), not a fixed universal phase template. Phase logits are sums over enabled movements:

```text
phase_logit[p] = sum(movement_score[m] for m enabled by phase p)
```

This reduction supports any number of movements and phases without changing the learned parameter shapes.

## Legal-action constraints

The selectable phase set already excludes incompatible movement combinations. At runtime, a Boolean action mask further removes phases that cannot be selected at that decision because of control state.

- A junction must satisfy its configured minimum green before switching.
- A switch may insert a deterministic yellow transition before the new green.
- A continued phase does not insert a transition.
- Pass-through or unsupported junctions are not policy decisions.
- Forced one-action decisions train the critic but do not contribute actor or entropy loss.
- Rollout collection and sampled learned-policy evaluation use the same legal-action mask.

The runtime asserts that an action allowed by PPO's mask is accepted by the signal controller. Illegal signal states are therefore excluded by construction rather than learned through penalties.

## Policy inference modes

PPO forms a categorical distribution over the legal phase logits. Training rollouts and the reference evaluation sample from that distribution. The interactive `scripts/run.py` path instead selects the highest-scoring legal phase for a stable visual demonstration. Results must state which inference mode was used.

## Cross-city generalization

The same feature definitions, typed message functions, movement scorer, and phase reduction are reused in every city. No learned city identity, fixed intersection index, fixed graph size, or global fixed phase count is required. A new city supplies a new graph and new per-junction phase-incidence matrices while retaining the learned parameters.

This is the intended generalization mechanism. Its effectiveness must still be evaluated empirically on topology-held-out cities and, for final claims, on fresh seeds and an unseen city that was not used for checkpoint selection.

## Visual examples

![3x3 movement graph](assets/movement-graph-3x3.png)

Green edges are directed lane-group connectors through unsignalized junctions. At signalized junctions, blue input edges and amber output edges route information through explicit movement nodes instead.

[Open the interactive 3×3 graph](assets/movement-graph-3x3.html).

Regenerate the synthetic network and documentation views:

```powershell
uv run python scripts\generate_grid_network.py --rows 3 --cols 3 --out configs\grid_3x3_dedicated
uv run python scripts\visualize_movement_graph.py `
  --cfg configs\grid_3x3_dedicated\grid.sumocfg `
  --out reports\movement_graph_3x3.html
Copy-Item reports\movement_graph_3x3.html docs\assets\movement-graph-3x3.html
uv run python scripts\plot_movement_graph_examples.py
```
