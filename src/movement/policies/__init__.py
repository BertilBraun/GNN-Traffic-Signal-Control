"""Baseline movement scoring policies."""

from __future__ import annotations

from enum import Enum

from src.movement.features import LaneFeatureApi
from src.movement.policies.max_pressure import compute_max_pressure_scores
from src.movement.policies.queue import compute_queue_scores
from src.movement.schema import MovementIndex, TrafficLightProgram


class MovementScoringMethod(str, Enum):
    MAX_PRESSURE = 'max-pressure'
    QUEUE = 'queue'


def compute_movement_scores(
    program: TrafficLightProgram,
    lane_api: LaneFeatureApi,
    method: MovementScoringMethod,
) -> dict[MovementIndex, float]:
    match method:
        case MovementScoringMethod.MAX_PRESSURE:
            return compute_max_pressure_scores(program, lane_api)
        case MovementScoringMethod.QUEUE:
            return compute_queue_scores(program, lane_api)
        case _:
            raise ValueError(f'Unsupported control method: {method}')


__all__ = [
    'LaneFeatureApi',
    'MovementScoringMethod',
    'compute_max_pressure_scores',
    'compute_movement_scores',
    'compute_queue_scores',
]
