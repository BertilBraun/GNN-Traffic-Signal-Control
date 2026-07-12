# Movement Imitation Dataset Schema

Each collected sample represents one controller decision time.

The current storage format is JSONL: one serialized ^MovementDatasetSample^ per
line. Graphs and phase counts may vary between samples, so arrays are stored as
ragged lists.

LaneGroups may contain an ordered corridor of multiple SUMO edges when
unsignalized junctions can be contracted without introducing a topology
shortcut. Their static length and free-flow time cover the complete corridor.

Lane-group dynamic values are detector-local. The detector covers the final
^min(200 m, lane-group corridor length)^ before the downstream junction and includes
total vehicle count, moving vehicle count, queue extent, occupancy, mean speed,
detector arrival/departure rates over 15 and 60 seconds, and ETA-to-queue-tail
features for moving vehicles that are approaching the back of the current queue.
The queue tail is estimated as one effective vehicle spacing behind the most
upstream stopped detector vehicle; if there is no stopped vehicle, it is the
junction stop line. The detector can span multiple underlying edges. Raw halting counts
are intentionally excluded from neural inputs because they would make the
max-pressure imitation target an exact subtraction shortcut. For an output
LaneGroup, this region is near the next junction rather than immediately after
the current junction.

Movement feature rows contain no graph, lane, edge, or traffic-light IDs. Typed
graph edges provide endpoint indices to the model without exposing identifiers
as numeric features. The dynamic movement state includes whether the movement
was green at the previous controller decision.

## Fields

^^^text
x_lane
    [num_lane_groups, lane_feature_dim]

x_movement
    [num_movements, movement_feature_dim]

edge_index_dict
    typed LaneGroup/Movement edge lists:
    input_lane_to_movement
    output_lane_to_movement
    movement_to_input_lane
    movement_to_output_lane

phase_incidences
    per-traffic-light selectable phase incidence rows

teacher_movement_scores
    graph-level max-pressure scores, aligned with movement IDs

teacher_selected_phase_by_tls
    local selectable phase index selected by graph-level teacher scores

metadata
    config path, network path, seed, simulation time, and teacher name
^^^

## Teacher Target

The first imitation target is graph-level max pressure:

^^^text
score(M) = halting_count(input LaneGroup) - halting_count(output LaneGroup)
^^^

The target generator may use detector halting counts, but those exact operands
are not stored in ^x_lane^. IL therefore has to infer useful pressure-like
behavior from queue extent, moving traffic, occupancy, storage, flow history,
movement demand, and graph context. Training combines movement-score regression
with phase-ranking cross-entropy.

## Replay Check

^src.movement.dataset.replay_teacher_selected_phases^ recomputes selected local
phase indices from stored graph-level scores and stored phase incidence. This is
the minimum inspection check for collected samples.

## Collection Command

^^^powershell
python scripts\collect_il_data.py --cfg configs\grid_3x3_dedicated\grid.sumocfg --out data\il\grid_3x3_seed42.jsonl --steps 3600 --decision-interval 15 --seed 42 --initial-occupancy 0.06
^^^

Add ^--time-to-teleport -1^ when collection should disable SUMO gridlock
teleporting completely.

This schema is incompatible with movement datasets and checkpoints created
before the ETA-to-queue-tail feature update. Regenerate both. The current lane
feature vector has 29 values.

The standard multi-seed 3x3 regeneration and training run is:

^^^powershell
python scripts\train_il.py ^
  --cfg configs\grid_3x3_dedicated\grid.sumocfg ^
  --samples 4800 ^
  --samples-per-simulation 240 ^
  --epochs 100 ^
  --eval-cfg configs\grid_3x3_dedicated\grid.sumocfg ^
  --eval-every-epochs 10 ^
  --eval-seeds 100 101 ^
  --time-to-teleport -1
^^^

The combined JSONL dataset is retained as ^training_samples.jsonl^ in the
checkpoint directory. ^movement_policy_eval_best.pt^ tracks the periodic
evaluation checkpoint with the lowest learned wait density.

Collection automatically uses consecutive seeds beginning at 42. Each
simulation starts with randomized traffic occupancy, and collection prints the
minimum, mean, and maximum active vehicle count. Before collection, two
same-seed max-pressure simulations are compared exactly for deterministic
vehicle trajectories and actions. SUMO advances one insertion step before the
first policy decision so the randomized initial population is present in the
first stored sample; this is not a stabilization or burn-in period.
