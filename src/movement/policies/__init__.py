"""Baseline movement scoring policies."""
from __future__ import annotations

from enum import Enum
from typing import Protocol

from src.movement.schema import LaneId, MovementIndex, TrafficLightProgram


class MovementScoringMethod(str, Enum):
    MAX_PRESSURE = "max-pressure"
    QUEUE = "queue"


class LaneQueueApi(Protocol):
    def getLastStepHaltingNumber(self, lane_id: LaneId) -> int:
        ...


def compute_movement_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
    method: MovementScoringMethod,
) -> dict[MovementIndex, float]:
    match method:
        case MovementScoringMethod.MAX_PRESSURE:
            return _compute_max_pressure_scores(program, lane_api)
        case MovementScoringMethod.QUEUE:
            return _compute_queue_scores(program, lane_api)
        case _:
            raise ValueError(f"Unsupported control method: {method}")


def _compute_max_pressure_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
) -> dict[MovementIndex, float]:
    scores: dict[MovementIndex, float] = {}
    for movement in program.movements:
        incoming_queue = lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        outgoing_queue = lane_api.getLastStepHaltingNumber(movement.outgoing_lane_id)
        scores[movement.movement_index] = float(incoming_queue - outgoing_queue)
    return scores


def _compute_queue_scores(
    program: TrafficLightProgram,
    lane_api: LaneQueueApi,
) -> dict[MovementIndex, float]:
    scores: dict[MovementIndex, float] = {}
    for movement in program.movements:
        scores[movement.movement_index] = float(
            lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        )
    return scores
