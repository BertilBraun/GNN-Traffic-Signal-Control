"""Shared types for movement scoring policies."""
from __future__ import annotations

from enum import Enum
from typing import Protocol

from src.movement.schema import LaneId


class MovementScoringMethod(str, Enum):
    MAX_PRESSURE = "max-pressure"
    QUEUE = "queue"


class LaneQueueApi(Protocol):
    def getLastStepHaltingNumber(self, lane_id: LaneId) -> int:
        ...
