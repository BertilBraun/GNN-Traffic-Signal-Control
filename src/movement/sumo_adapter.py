"""Adapters between TraCI-style APIs and movement-aware control primitives."""
from __future__ import annotations

from typing import Any

from .extraction import extract_traffic_light_program
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
