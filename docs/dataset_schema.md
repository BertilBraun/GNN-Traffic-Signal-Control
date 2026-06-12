# Movement Imitation Dataset Schema

Each collected sample represents one controller decision time.

The current storage format is JSONL: one serialized `MovementDatasetSample` per
line. Graphs and phase counts may vary between samples, so arrays are stored as
ragged lists.

## Fields

```text
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
```

## Teacher Target

The first imitation target is graph-level max pressure over the LaneGroup
features visible to the zero-hop model:

```text
score(M) = halting_count(input LaneGroup) - halting_count(output LaneGroup)
```

This differs intentionally from the lower-level SUMO controlled-link teacher.
The model sees collapsed LaneGroup features, not per-controlled-link lane
queues, so the stored regression target must be defined at the same graph level
or the exact teacher is partially unobservable.

## Replay Check

`src.movement.dataset.replay_teacher_selected_phases` recomputes selected local
phase indices from stored graph-level scores and stored phase incidence. This is
the minimum inspection check for collected samples.

## Collection Command

```powershell
python scripts\collect_il_data.py --cfg configs\grid_3x3_dedicated\grid.sumocfg --out data\il\grid_3x3_seed42.jsonl --steps 1800 --decision-interval 15 --seed 42
```
