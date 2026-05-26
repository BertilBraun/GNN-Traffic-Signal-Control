"""Adapters between TraCI-style APIs and movement-aware control primitives."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .extraction import extract_traffic_light_program
from .max_pressure import select_max_pressure_phase
from .schema import TrafficLightProgram


def extract_programs_from_trafficlight_api(trafficlight_api: Any) -> dict[str, TrafficLightProgram]:
    """Extract movement-aware programs for every TraCI traffic light."""
    programs: dict[str, TrafficLightProgram] = {}
    for tls_id in trafficlight_api.getIDList():
        logics = trafficlight_api.getAllProgramLogics(tls_id)
        if not logics:
            continue
        active_program_id = trafficlight_api.getProgram(tls_id)
        logic = next(
            (
                candidate
                for candidate in logics
                if getattr(candidate, "programID", None) == active_program_id
            ),
            logics[0],
        )
        phase_states = [phase.state for phase in logic.phases]
        program = extract_traffic_light_program(
            tls_id=tls_id,
            phase_states=phase_states,
            controlled_links=trafficlight_api.getControlledLinks(tls_id),
        )
        if program.selectable_phases:
            programs[tls_id] = program
    return programs


def compute_movement_pressures(
    program: TrafficLightProgram,
    lane_api: Any,
) -> dict[int, float]:
    """Compute movement pressure as incoming halting queue minus outgoing queue."""
    pressures: dict[int, float] = {}
    for movement in program.movements:
        incoming_queue = lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        outgoing_queue = lane_api.getLastStepHaltingNumber(movement.outgoing_lane_id)
        pressures[movement.movement_index] = float(incoming_queue - outgoing_queue)
    return pressures


def select_max_pressure_actions(
    programs: Mapping[str, TrafficLightProgram],
    lane_api: Any,
    aggregate_by: str = "incoming_lane",
) -> dict[str, int]:
    """Return `{tls_id: sumo_phase_index}` selected by max-pressure scoring."""
    actions: dict[str, int] = {}
    for tls_id, program in programs.items():
        movement_pressures = compute_movement_pressures(program, lane_api)
        actions[tls_id] = select_max_pressure_phase(
            program,
            movement_pressures,
            aggregate_by=aggregate_by,
        ).sumo_phase_index
    return actions


def select_max_pressure_states(
    programs: Mapping[str, TrafficLightProgram],
    lane_api: Any,
    aggregate_by: str = "incoming_lane",
) -> dict[str, str]:
    """Return `{tls_id: phase_state}` for directly holding selected greens."""
    states: dict[str, str] = {}
    for tls_id, program in programs.items():
        movement_pressures = compute_movement_pressures(program, lane_api)
        selection = select_max_pressure_phase(
            program,
            movement_pressures,
            aggregate_by=aggregate_by,
        )
        states[tls_id] = program.selectable_phases[selection.local_phase_index].state
    return states
