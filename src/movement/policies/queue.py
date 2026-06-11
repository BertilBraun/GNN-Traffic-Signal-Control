"""Incoming queue length movement scoring policy."""
from __future__ import annotations

from src.movement.policies.types import LaneQueueApi
from src.movement.schema import MovementIndex, TrafficLightProgram


def compute_queue_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
) -> dict[MovementIndex, float]:
    scores: dict[MovementIndex, float] = {}
    for movement in program.movements:
        scores[movement.movement_index] = float(
            lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        )
    return scores
