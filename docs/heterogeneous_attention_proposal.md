# Heterogeneous Attention GNN Proposal

## Status

Deferred experiment. The current mean-aggregation movement GNN remains the
baseline until PPO demonstrates a meaningful learning signal with the revised
interval-integrated delay reward.

This proposal changes only the neural message-passing architecture. It does not
change graph construction, traffic features, valid phase synthesis, signal
transitions, or PPO.

## Objective

Allow the policy to assign state-dependent importance to neighboring lane
groups and movements.

The current model transforms messages separately by relation type but averages
all incoming messages with equal weight. Attention would allow the model to
learn that, for example:

- a nearly full outgoing lane matters more than an empty outgoing lane;
- a large approaching platoon matters more than distant low-volume traffic;
- different neighboring movements matter under different congestion states;
- road length, free-flow time, and capacity affect how useful remote context is.

## Graph Schema

Preserve the existing heterogeneous bipartite graph:

```text
LaneGroup <-> Movement <-> LaneGroup
```

Preserve the four directed relation types:

```text
input_lane_to_movement
output_lane_to_movement
movement_to_input_lane
movement_to_output_lane
```

No junction nodes are added. Unsignalized corridor contraction and all static
and dynamic features remain unchanged.

## Proposed Layer

Replace each `MovementLaneHop` mean-aggregation block with one heterogeneous
multi-head attention macro-hop:

```text
Movement -> LaneGroup
LaneGroup -> Movement
```

Each relation has independent query, key, value, and attention parameters. A
message from source node `j` to target node `i` for relation `r` is:

```text
q_i^r = W_q^r h_i
k_j^r = W_k^r h_j
v_j^r = W_v^r h_j

e_ij^r = attention(q_i^r, k_j^r, edge_ij)
alpha_ij^r = softmax_j(e_ij^r)
message_i^r = sum_j(alpha_ij^r * v_j^r)
```

The initial implementation does not need separate learned edge embeddings.
Road geometry and traffic state already reside on the `LaneGroup` nodes.
Explicit edge features should be added only if an ablation shows that node
features are insufficient.

## Multi-Head Attention

Use four heads initially. Each head computes independent attention weights and
value projections:

```text
head_i = relation_attention_i(target, neighbors)
combined = concatenate(head_1, head_2, head_3, head_4)
```

The concatenated output is projected back to the configured hidden dimension.
For a hidden dimension of 64, four 16-dimensional heads preserve the existing
embedding width.

Heads are not assigned fixed meanings. They may learn different traffic
relationships such as queue pressure, downstream supply, approaching flow, or
corridor progression.

## Node Updates

Update lane groups using separate messages from movements entering and leaving
the lane group:

```text
lane_message =
    concat(
        attention(movement_to_input_lane),
        attention(movement_to_output_lane)
    )

lane_next =
    LayerNorm(
        lane_current + lane_update_mlp(lane_current, lane_message)
    )
```

Update movements using separate messages from their input and output lane
groups:

```text
movement_message =
    concat(
        attention(input_lane_to_movement),
        attention(output_lane_to_movement)
    )

movement_next =
    LayerNorm(
        movement_current
        + movement_update_mlp(movement_current, movement_message)
    )
```

Use residual connections and layer normalization for both node types. Apply
dropout only if training or evaluation demonstrates overfitting.

## Actor And Critic

Keep the current actor interface:

```text
movement embedding -> movement score
phase score = sum(enabled movement scores)
```

Keep the current critic initially:

```text
junction embedding = mean(movement embeddings owned by junction)
junction embedding -> value head -> V(s)
```

Attention pooling for the critic is a separate experiment. It should not be
introduced in the first architecture comparison.

## Initial Configuration

```text
hidden dimension:       64
attention heads:         4
head dimension:         16
macro-hops:              1
relation parameters:     independent
residual connections:    yes
layer normalization:     yes
attention dropout:       0.0
feature dropout:         0.0
```

One macro-hop remains the first target because it matches the current model's
receptive field. Additional hops must be tested separately.

## Implementation Strategy

1. Introduce a message-passing interface shared by mean and attention models.
2. Preserve the existing mean aggregator as the baseline implementation.
3. Add a relation-aware attention macro-hop.
4. Add model type and attention-head metadata to checkpoints.
5. Train new IL checkpoints for each architecture.
6. Compare PPO from equally trained IL initializations.

Old checkpoints should not be silently loaded into a different architecture.

## Required Ablations

Compare one change at a time:

| Experiment | Aggregation | Heads | Hops | Residual/Norm |
|---|---|---:|---:|---|
| Baseline | Mean | 0 | 1 | No |
| Stabilized mean | Mean | 0 | 1 | Yes |
| Single-head attention | Attention | 1 | 1 | Yes |
| Multi-head attention | Attention | 4 | 1 | Yes |
| Wider context | Attention | 4 | 2 | Yes |

The stabilized mean model is important. If residual connections and
normalization provide the improvement, attention should not receive the credit.

## Evaluation Criteria

An attention model is worth retaining only if it demonstrates:

- equal or better IL phase match on held-out seeds;
- lower held-out waiting time or time loss than the mean baseline;
- consistent PPO improvement across multiple seeds;
- no substantial increase in teleporting or unfinished vehicles;
- useful attention variation rather than nearly uniform weights;
- acceptable simulation and training runtime.

Attention weights should be logged by relation and head. Useful diagnostics
include attention entropy, maximum neighbor weight, and attention grouped by
lane occupancy or downstream available storage.

## Decision Gate

Do not implement this proposal solely because the current PPO run fails.

Proceed only after:

1. the revised reward has been tested on a fixed-seed overfit run;
2. PPO reward and evaluation metrics have been checked for alignment;
3. the current mean model has either learned weakly or a specific
   representation limitation has been demonstrated.

If PPO still cannot improve a deterministic fixed scenario, investigate the
action parameterization and training pipeline before increasing model
complexity.
