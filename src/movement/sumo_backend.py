"""Centralized SUMO backend selection for TraCI-compatible APIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .schema import LaneId


class SumoBackendKind(str, Enum):
    TRACI = 'traci'
    LIBSUMO = 'libsumo'


class LaneApi(Protocol):
    def getLastStepVehicleNumber(self, lane_id: LaneId | str) -> int: ...

    def getLastStepMeanSpeed(self, lane_id: LaneId | str) -> float: ...

    def getLastStepVehicleIDs(self, lane_id: LaneId | str) -> tuple[str, ...]: ...

    def getLastStepHaltingNumber(self, lane_id: LaneId | str) -> int: ...

    def getWaitingTime(self, lane_id: LaneId | str) -> float: ...

    def getLength(self, lane_id: LaneId | str) -> float: ...

    def getShape(self, lane_id: LaneId | str) -> tuple[tuple[float, float], ...]: ...


class SimulationApi(Protocol):
    def getMinExpectedNumber(self) -> int: ...

    def getStartingTeleportNumber(self) -> int: ...

    def getDepartedNumber(self) -> int: ...

    def getArrivedIDList(self) -> tuple[str, ...]: ...


class PhaseLike(Protocol):
    state: str


class ProgramLogicLike(Protocol):
    programID: str
    phases: Sequence[PhaseLike]


class TrafficLightApi(Protocol):
    def getIDList(self) -> tuple[str, ...]: ...

    def getAllProgramLogics(self, traffic_light_id: str) -> Sequence[ProgramLogicLike]: ...

    def getProgram(self, traffic_light_id: str) -> str: ...

    def getControlledLinks(self, traffic_light_id: str) -> Sequence[Sequence[Sequence[str]]]: ...

    def setRedYellowGreenState(self, traffic_light_id: str, state: str) -> None: ...


class VehicleApi(Protocol):
    def getIDList(self) -> tuple[str, ...]: ...

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]: ...

    def subscribe(self, vehicle_id: str, variables: Sequence[int]) -> None: ...

    def getAllSubscriptionResults(self) -> dict[str, dict[int, str | int | float]]: ...

    def getSpeed(self, vehicle_id: str) -> float: ...

    def getLaneID(self, vehicle_id: str) -> str: ...

    def getLanePosition(self, vehicle_id: str) -> float: ...

    def getNextTLS(self, vehicle_id: str) -> tuple[object, ...]: ...


class SumoModule(Protocol):
    lane: LaneApi
    simulation: SimulationApi
    trafficlight: TrafficLightApi
    vehicle: VehicleApi

    def start(self, command: Sequence[str]) -> None: ...

    def close(self) -> None: ...

    def simulationStep(self) -> None: ...


@dataclass(frozen=True)
class SumoBackend:
    kind: SumoBackendKind
    module: SumoModule

    @property
    def lane(self) -> LaneApi:
        return self.module.lane

    @property
    def simulation(self) -> SimulationApi:
        return self.module.simulation

    @property
    def trafficlight(self) -> TrafficLightApi:
        return self.module.trafficlight

    @property
    def vehicle(self) -> VehicleApi:
        return self.module.vehicle

    def start(self, command: Sequence[str]) -> None:
        self.module.start(command)

    def close(self) -> None:
        self.module.close()

    def simulation_step(self) -> None:
        self.module.simulationStep()


def create_sumo_backend(kind: SumoBackendKind) -> SumoBackend:
    match kind:
        case SumoBackendKind.TRACI:
            import traci

            return SumoBackend(kind=kind, module=traci)
        case SumoBackendKind.LIBSUMO:
            import libsumo

            return SumoBackend(kind=kind, module=libsumo)


def check_sumo_binary(gui: bool) -> str:
    import sumolib

    return str(sumolib.checkBinary('sumo-gui' if gui else 'sumo'))


def vehicle_subscription_variables() -> tuple[int, int, int, int, int]:
    from traci import constants as traci_constants

    return (
        traci_constants.VAR_LANE_ID,
        traci_constants.VAR_LANEPOSITION,
        traci_constants.VAR_SPEED,
        traci_constants.VAR_LENGTH,
        traci_constants.VAR_ROUTE_INDEX,
    )


def subscription_lane_id_key() -> int:
    from traci import constants as traci_constants

    return int(traci_constants.VAR_LANE_ID)


def subscription_lane_position_key() -> int:
    from traci import constants as traci_constants

    return int(traci_constants.VAR_LANEPOSITION)


def subscription_speed_key() -> int:
    from traci import constants as traci_constants

    return int(traci_constants.VAR_SPEED)


def subscription_length_key() -> int:
    from traci import constants as traci_constants

    return int(traci_constants.VAR_LENGTH)


def subscription_route_index_key() -> int:
    from traci import constants as traci_constants

    return int(traci_constants.VAR_ROUTE_INDEX)


def resolve_sumo_config_path(path: str | Path) -> Path:
    return Path(path)
