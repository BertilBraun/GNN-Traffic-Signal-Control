from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.runtime import MovementControlRuntime


class FakePhase:
    def __init__(self, state: str) -> None:
        self.state = state


class FakeLogic:
    programID = "0"

    def __init__(self, states: list[str]) -> None:
        self.phases = [FakePhase(state) for state in states]


class FakeTrafficLightApi:
    def __init__(self) -> None:
        self.applied_states: list[tuple[str, str]] = []

    def getIDList(self) -> list[str]:
        return ["J0"]

    def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str | None]]]:
        assert tls_id == "J0"
        return [
            [("north_0", "south_0", None)],
            [("east_0", "west_0", None)],
        ]

    def getAllProgramLogics(self, tls_id: str) -> list[FakeLogic]:
        assert tls_id == "J0"
        return [FakeLogic(["Gr", "rG"])]

    def getProgram(self, tls_id: str) -> str:
        assert tls_id == "J0"
        return "0"

    def setRedYellowGreenState(self, tls_id: str, state: str) -> None:
        self.applied_states.append((tls_id, state))


class FakeLaneApi:
    pass


class FakeSimulationApi:
    def __init__(self) -> None:
        self.steps = 0

    def simulationStep(self) -> None:
        self.steps += 1

    def getMinExpectedNumber(self) -> int:
        return 1


class FakeTraciModule:
    def __init__(self) -> None:
        self.trafficlight = FakeTrafficLightApi()
        self.lane = FakeLaneApi()
        self.simulation = FakeSimulationApi()
        self.started_command: list[str] | None = None
        self.closed = False

    def start(self, command: list[str]) -> None:
        self.started_command = command

    def simulationStep(self) -> None:
        self.simulation.simulationStep()

    def close(self) -> None:
        self.closed = True


class FakeSumolibModule:
    def checkBinary(self, name: str) -> str:
        return f"{name}-bin"


def test_runtime_starts_sumo_and_extracts_movement_programs() -> None:
    fake_traci = FakeTraciModule()

    runtime = MovementControlRuntime(
        cfg_path="network.sumocfg",
        seed=7,
        traci_module=fake_traci,
        sumolib_module=FakeSumolibModule(),
    )
    runtime.start()

    assert fake_traci.started_command == [
        "sumo-bin",
        "-c",
        "network.sumocfg",
        "--seed",
        "7",
        "--no-step-log",
        "true",
    ]
    assert sorted(runtime.programs) == ["J0"]
    assert runtime.lane_api is fake_traci.lane


def test_runtime_filters_targets_applies_states_and_steps_simulation() -> None:
    fake_traci = FakeTraciModule()
    runtime = MovementControlRuntime(
        cfg_path="network.sumocfg",
        yellow_duration=0,
        min_green_steps=2,
        traci_module=fake_traci,
        sumolib_module=FakeSumolibModule(),
    )
    runtime.start()

    assert runtime.request_targets({"J0": "Gr"}) == {"J0": "Gr"}
    runtime.step()
    assert runtime.request_targets({"J0": "rG"}) == {"J0": "Gr"}
    runtime.step()
    assert runtime.request_targets({"J0": "rG"}) == {"J0": "rG"}
    runtime.step()

    assert fake_traci.trafficlight.applied_states == [
        ("J0", "Gr"),
        ("J0", "Gr"),
        ("J0", "rG"),
    ]
    assert fake_traci.simulation.steps == 3


def test_runtime_closes_traci() -> None:
    fake_traci = FakeTraciModule()
    runtime = MovementControlRuntime(
        cfg_path="network.sumocfg",
        traci_module=fake_traci,
        sumolib_module=FakeSumolibModule(),
    )

    runtime.close()

    assert fake_traci.closed
