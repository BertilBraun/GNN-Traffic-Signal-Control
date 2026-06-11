"""Generic phase selection over extracted movement-aware phases."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .schema import (
    LocalPhaseIndex,
    MovementIndex,
    SelectablePhase,
    SumoPhaseIndex,
    TrafficLightProgram,
)


@dataclass(frozen=True)
class PhaseSelection:
    """Selected phase and its aggregate score."""

    sumo_phase_index: SumoPhaseIndex
    local_phase_index: LocalPhaseIndex
    score: float


def score_phase(
    phase: SelectablePhase,
    movement_scores: Mapping[MovementIndex, float],
) -> float:
    """Sum scores for all movements enabled by `phase`."""
    return sum(
        float(movement_scores.get(movement_idx, 0.0))
        for movement_idx in phase.enabled_movement_indices
    )


def score_program_phases(
    program: TrafficLightProgram,
    movement_scores: Mapping[MovementIndex, float],
) -> dict[SumoPhaseIndex, float]:
    """Return `{sumo_phase_index: aggregate_score}` for selectable phases."""
    return {
        phase.sumo_phase_index: score_phase(phase, movement_scores)
        for phase in program.selectable_phases
    }


def select_highest_scoring_phase(
    program: TrafficLightProgram,
    movement_scores: Mapping[MovementIndex, float],
) -> PhaseSelection:
    """Select the highest-scoring phase with stable program-order tie break."""
    if not program.selectable_phases:
        raise ValueError(
            f"Traffic light {program.traffic_light_id} has no selectable phases."
        )

    best_local_idx = LocalPhaseIndex(0)
    best_phase = program.selectable_phases[0]
    best_score = score_phase(best_phase, movement_scores)

    for local_idx, phase in enumerate(program.selectable_phases[1:], start=1):
        phase_score = score_phase(phase, movement_scores)
        if phase_score > best_score:
            best_local_idx = LocalPhaseIndex(local_idx)
            best_phase = phase
            best_score = phase_score

    return PhaseSelection(
        sumo_phase_index=best_phase.sumo_phase_index,
        local_phase_index=best_local_idx,
        score=best_score,
    )
