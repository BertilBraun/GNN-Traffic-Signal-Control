"""Max-pressure scoring over extracted movement-aware phases."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .schema import SelectablePhase, TrafficLightProgram


@dataclass(frozen=True)
class PhaseSelection:
    """Selected phase and its aggregate score."""

    sumo_phase_index: int
    local_phase_index: int
    score: float


def score_phase_pressure(
    phase: SelectablePhase,
    movement_pressure: Mapping[int, float],
) -> float:
    """Sum pressures for all movements enabled by `phase`."""
    return sum(
        float(movement_pressure.get(movement_idx, 0.0))
        for movement_idx in phase.enabled_movement_indices
    )


def score_phase_pressure_by_incoming_lane(
    phase: SelectablePhase,
    program: TrafficLightProgram,
    movement_pressure: Mapping[int, float],
) -> float:
    """Aggregate pressure once per incoming lane served by `phase`.

    If several active movements share an incoming lane, use the largest
    movement pressure for that lane. This avoids counting the same queue once
    per possible turn when route intentions are unknown.
    """
    best_by_lane: dict[str, float] = {}
    movements_by_index = {
        movement.movement_index: movement for movement in program.movements
    }
    for movement_idx in phase.enabled_movement_indices:
        movement = movements_by_index[movement_idx]
        pressure = float(movement_pressure.get(movement_idx, 0.0))
        lane_id = movement.incoming_lane_id
        if lane_id not in best_by_lane or pressure > best_by_lane[lane_id]:
            best_by_lane[lane_id] = pressure
    return sum(best_by_lane.values())


def score_program_phases(
    program: TrafficLightProgram,
    movement_pressure: Mapping[int, float],
    aggregate_by: str = "movement",
) -> dict[int, float]:
    """Return `{sumo_phase_index: aggregate_pressure}` for selectable phases."""
    if aggregate_by not in {"movement", "incoming_lane"}:
        raise ValueError(f"Unsupported phase pressure aggregation: {aggregate_by}")
    return {
        phase.sumo_phase_index: (
            score_phase_pressure(phase, movement_pressure)
            if aggregate_by == "movement"
            else score_phase_pressure_by_incoming_lane(phase, program, movement_pressure)
        )
        for phase in program.selectable_phases
    }


def select_max_pressure_phase(
    program: TrafficLightProgram,
    movement_pressure: Mapping[int, float],
    aggregate_by: str = "movement",
) -> PhaseSelection:
    """Select the highest-pressure phase with stable program-order tie break."""
    if not program.selectable_phases:
        raise ValueError(f"Traffic light {program.tls_id} has no selectable phases.")
    if aggregate_by not in {"movement", "incoming_lane"}:
        raise ValueError(f"Unsupported phase pressure aggregation: {aggregate_by}")

    def score(phase: SelectablePhase) -> float:
        if aggregate_by == "movement":
            return score_phase_pressure(phase, movement_pressure)
        return score_phase_pressure_by_incoming_lane(phase, program, movement_pressure)

    best_local_idx = 0
    best_phase = program.selectable_phases[0]
    best_score = score(best_phase)

    for local_idx, phase in enumerate(program.selectable_phases[1:], start=1):
        phase_score = score(phase)
        if phase_score > best_score:
            best_local_idx = local_idx
            best_phase = phase
            best_score = phase_score

    return PhaseSelection(
        sumo_phase_index=best_phase.sumo_phase_index,
        local_phase_index=best_local_idx,
        score=best_score,
    )
