"""Dispatch movement scoring to the selected baseline policy."""
from __future__ import annotations

from src.movement.policies.max_pressure import compute_max_pressure_scores
from src.movement.policies.queue import compute_queue_scores
from src.movement.policies.types import LaneQueueApi, MovementScoringMethod
from src.movement.schema import MovementIndex, TrafficLightProgram


def compute_movement_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
    method: MovementScoringMethod,
) -> dict[MovementIndex, float]:
    match method:
        case MovementScoringMethod.MAX_PRESSURE:
            return compute_max_pressure_scores(program, lane_api)
        case MovementScoringMethod.QUEUE:
            return compute_queue_scores(program, lane_api)
        case _:
            raise ValueError(f"Unsupported control method: {method}")
