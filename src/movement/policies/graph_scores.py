"""Graph-level baseline scoring policies."""

from __future__ import annotations

from src.movement.features import MovementFeatureFrame
from src.movement.graph_schema import MovementGraph
from src.movement.policies import MovementScoringMethod


def compute_graph_movement_scores(
    graph: MovementGraph,
    feature_frame: MovementFeatureFrame,
    method: MovementScoringMethod,
) -> tuple[float, ...]:
    """Compute movement scores from detector-aware LaneGroup features."""
    halting_by_lane_group = {
        row.lane_group_id: row.dynamic.halting_count_detector for row in feature_frame.lane_group_rows
    }
    match method:
        case MovementScoringMethod.MAX_PRESSURE:
            return tuple(
                float(
                    halting_by_lane_group[movement.input_lane_group_id]
                    - halting_by_lane_group[movement.output_lane_group_id]
                )
                for movement in graph.movements
            )
        case MovementScoringMethod.QUEUE:
            return tuple(
                float(halting_by_lane_group[movement.input_lane_group_id])
                for movement in graph.movements
            )
        case _:
            raise ValueError(f'Unsupported control method: {method}')
