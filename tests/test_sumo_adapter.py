from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.sumo_adapter import (
    compute_movement_pressures,
    compute_movement_queues,
    extract_programs_from_trafficlight_api,
    select_control_states,
    select_max_pressure_actions,
    select_max_pressure_states,
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


class FakeLaneApi:
    queues = {
        "north_0": 7,
        "south_0": 2,
        "east_0": 3,
        "west_0": 8,
    }

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return self.queues[lane_id]


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


def test_compute_movement_pressures_uses_incoming_minus_outgoing_queue() -> None:
    program = extract_programs_from_trafficlight_api(FakeTrafficLightApi())["J0"]

    pressures = compute_movement_pressures(program, FakeLaneApi())

    assert pressures == {
        0: 5.0,
        1: 4.0,
        2: -5.0,
    }


def test_compute_movement_queues_uses_incoming_queue_only() -> None:
    program = extract_programs_from_trafficlight_api(FakeTrafficLightApi())["J0"]

    queues = compute_movement_queues(program, FakeLaneApi())

    assert queues == {
        0: 7.0,
        1: 7.0,
        2: 3.0,
    }


def test_select_max_pressure_actions_returns_sumo_phase_indices() -> None:
    programs = extract_programs_from_trafficlight_api(FakeTrafficLightApi())

    actions = select_max_pressure_actions(programs, FakeLaneApi())

    assert actions == {"J0": 0}


def test_select_max_pressure_states_returns_green_state_to_hold() -> None:
    programs = extract_programs_from_trafficlight_api(FakeTrafficLightApi())

    states = select_max_pressure_states(programs, FakeLaneApi())

    assert states == {"J0": "Gr"}


def test_select_control_states_can_use_longest_queue_method() -> None:
    class QueueTrafficLightApi(FakeTrafficLightApi):
        def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str | None]]]:
            assert tls_id == "J0"
            return [
                [("north_0", "west_0", None)],
                [("east_0", "west_0", None)],
            ]

        def getAllProgramLogics(self, tls_id: str) -> list[FakeLogic]:
            assert tls_id == "J0"
            return [FakeLogic(["Gr", "rG"])]

    class QueueLaneApi:
        queues = {
            "north_0": 3,
            "east_0": 6,
            "west_0": 10,
        }

        def getLastStepHaltingNumber(self, lane_id: str) -> int:
            return self.queues[lane_id]

    programs = extract_programs_from_trafficlight_api(QueueTrafficLightApi())

    states = select_control_states(programs, QueueLaneApi(), method="queue")

    assert states == {"J0": "rG"}


def test_select_control_states_rejects_unknown_method() -> None:
    programs = extract_programs_from_trafficlight_api(FakeTrafficLightApi())

    try:
        select_control_states(programs, FakeLaneApi(), method="unknown")
    except ValueError as exc:
        assert "Unsupported control method" in str(exc)
    else:
        raise AssertionError("expected unknown method to raise ValueError")


def test_select_max_pressure_actions_aggregates_shared_incoming_lanes_once() -> None:
    class SharedLaneTrafficLightApi(FakeTrafficLightApi):
        def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str | None]]]:
            assert tls_id == "J0"
            return [
                [("north_0", "west_0", None)],
                [("north_0", "south_0", None)],
                [("east_0", "west_0", None)],
            ]

        def getAllProgramLogics(self, tls_id: str) -> list[FakeLogic]:
            assert tls_id == "J0"
            return [FakeLogic(["GGr", "rrG"])]

    class SharedLaneApi:
        queues = {
            "north_0": 5,
            "east_0": 6,
            "west_0": 0,
            "south_0": 0,
        }

        def getLastStepHaltingNumber(self, lane_id: str) -> int:
            return self.queues[lane_id]

    programs = extract_programs_from_trafficlight_api(SharedLaneTrafficLightApi())

    actions = select_max_pressure_actions(programs, SharedLaneApi())

    assert actions == {"J0": 1}
