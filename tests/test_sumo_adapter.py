from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.sumo_adapter import (
    extract_programs_from_trafficlight_api,
)


class FakePhase:
    def __init__(self, state: str) -> None:
        self.state = state


class FakeLogic:
    def __init__(self, states: list[str]) -> None:
        self.phases = [FakePhase(state) for state in states]


class FakeTrafficLightApi:
    def getIDList(self) -> list[str]:
        return ["J0"]

    def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str | None]]]:
        assert tls_id == "J0"
        return [
            [("north_0", "south_0", None), ("north_0", "east_0", ":via_0")],
            [("east_0", "west_0", None)],
        ]

    def getAllProgramLogics(self, tls_id: str) -> list[FakeLogic]:
        assert tls_id == "J0"
        return [FakeLogic(["Gr", "rG", "yy"])]

    def getProgram(self, tls_id: str) -> str:
        assert tls_id == "J0"
        return "0"


class NamedFakeLogic(FakeLogic):
    def __init__(self, program_id: str, states: list[str]) -> None:
        super().__init__(states)
        self.programID = program_id


class FakeTrafficLightApiWithActiveProgram(FakeTrafficLightApi):
    def getAllProgramLogics(self, tls_id: str) -> list[NamedFakeLogic]:
        assert tls_id == "J0"
        return [
            NamedFakeLogic("0", ["Gr", "yy"]),
            NamedFakeLogic("canonical", ["rG", "Gr"]),
        ]

    def getProgram(self, tls_id: str) -> str:
        assert tls_id == "J0"
        return "canonical"


def test_extract_programs_from_trafficlight_api_uses_current_logic_phases() -> None:
    programs = extract_programs_from_trafficlight_api(FakeTrafficLightApi())

    program = programs["J0"]
    assert [phase.sumo_phase_index for phase in program.selectable_phases] == [0, 1]
    assert program.selectable_phases[0].enabled_movement_indices == (0, 1)
    assert program.selectable_phases[1].enabled_movement_indices == (2,)


def test_extract_programs_from_trafficlight_api_uses_active_program_id() -> None:
    programs = extract_programs_from_trafficlight_api(FakeTrafficLightApiWithActiveProgram())

    program = programs["J0"]
    assert [phase.state for phase in program.selectable_phases] == ["rG", "Gr"]

