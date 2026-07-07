"""Adapters between TraCI-style APIs and movement-aware control primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .extraction import extract_traffic_light_program
from .schema import TrafficLightProgram


class PhaseLike(Protocol):
    state: str


class ProgramLogicLike(Protocol):
    programID: str
    phases: Sequence[PhaseLike]


class TrafficLightApi(Protocol):
    def getIDList(self) -> Sequence[str]: ...

    def getAllProgramLogics(self, traffic_light_id: str) -> Sequence[ProgramLogicLike]: ...

    def getProgram(self, traffic_light_id: str) -> str: ...

    def getControlledLinks(self, traffic_light_id: str) -> Sequence[Sequence[Sequence[str]]]: ...


def extract_programs_from_trafficlight_api(
    trafficlight_api: TrafficLightApi,
) -> dict[str, TrafficLightProgram]:
    """Extract movement-aware programs for every TraCI traffic light."""
    programs: dict[str, TrafficLightProgram] = {}
    for tls_id in trafficlight_api.getIDList():
        logics = trafficlight_api.getAllProgramLogics(tls_id)
        if not logics:
            continue
        active_program_id = trafficlight_api.getProgram(tls_id)
        logic = next(
            (candidate for candidate in logics if candidate.programID == active_program_id),
            logics[0],
        )
        phase_states = [phase.state for phase in logic.phases]
        program = extract_traffic_light_program(
            tls_id=tls_id,
            phase_states=phase_states,
            controlled_links=trafficlight_api.getControlledLinks(tls_id),
        )
        if len(program.selectable_phases) > 1:
            programs[tls_id] = program
    return programs
