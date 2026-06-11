"""Max-pressure movement scoring policy."""
from __future__ import annotations

from src.movement.runtime import LaneQueueApi
from src.movement.schema import MovementIndex, TrafficLightProgram


def compute_max_pressure_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
) -> dict[MovementIndex, float]:
    scores: dict[MovementIndex, float] = {}
    for movement in program.movements:
        incoming_queue = lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        outgoing_queue = lane_api.getLastStepHaltingNumber(movement.outgoing_lane_id)
        scores[movement.movement_index] = float(incoming_queue - outgoing_queue)
    return scores
