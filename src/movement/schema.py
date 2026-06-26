"""Data structures for movement-based traffic signal control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

TrafficLightId = NewType('TrafficLightId', str)
LaneId = NewType('LaneId', str)
EdgeId = NewType('EdgeId', str)
MovementIndex = NewType('MovementIndex', int)
SignalIndex = NewType('SignalIndex', int)
SumoPhaseIndex = NewType('SumoPhaseIndex', int)
LocalPhaseIndex = NewType('LocalPhaseIndex', int)
PhaseState = NewType('PhaseState', str)


@dataclass(frozen=True)
class ControlledMovement:
    """One SUMO controlled-link movement."""

    movement_index: MovementIndex
    signal_index: SignalIndex
    incoming_lane_id: LaneId
    outgoing_lane_id: LaneId
    via_lane_id: LaneId | None = None


@dataclass(frozen=True)
class SelectablePhase:
    """A SUMO green phase that can be selected by a controller."""

    sumo_phase_index: SumoPhaseIndex
    state: PhaseState
    enabled_movement_indices: tuple[MovementIndex, ...]


@dataclass(frozen=True)
class TrafficLightProgram:
    """Movement-aware view of one traffic-light program."""

    traffic_light_id: TrafficLightId
    movements: tuple[ControlledMovement, ...]
    selectable_phases: tuple[SelectablePhase, ...]
