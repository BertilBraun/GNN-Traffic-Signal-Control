"""Data structures for movement-based traffic signal control."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledMovement:
    """One SUMO controlled-link movement."""

    movement_index: int
    signal_index: int
    incoming_lane_id: str
    outgoing_lane_id: str
    via_lane_id: str | None = None


@dataclass(frozen=True)
class SelectablePhase:
    """A SUMO green phase that can be selected by a controller."""

    sumo_phase_index: int
    state: str
    enabled_movement_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrafficLightProgram:
    """Movement-aware view of one traffic-light program."""

    tls_id: str
    movements: tuple[ControlledMovement, ...]
    selectable_phases: tuple[SelectablePhase, ...]
