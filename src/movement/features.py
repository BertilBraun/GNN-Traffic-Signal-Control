"""Feature extraction for movement-score learning samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from traci._vehicle import VehicleDomain

from .graph_schema import GraphMovementId, LaneGroupId, MovementGraph
from .schema import EdgeId, LaneId

DEFAULT_DETECTOR_LENGTH_M = 200.0
EFFECTIVE_VEHICLE_SPACING_M = 8.0
DEFAULT_SPEED_LIMIT_MPS = 13.89
HALTING_SPEED_THRESHOLD_MPS = 0.1


class LaneFeatureApi(Protocol):
    def getLastStepHaltingNumber(self, lane_id: LaneId | str) -> int: ...


@dataclass(frozen=True)
class LaneGroupGeometry:
    """Static geometry for one directed lane group."""

    length_m: float
    num_lanes: int
    speed_limit_mps: float = DEFAULT_SPEED_LIMIT_MPS


@dataclass(frozen=True)
class StaticLaneGroupFeatures:
    length_m: float
    detector_length_m: float
    num_lanes: float
    speed_limit_mps: float
    freeflow_travel_time_s: float
    estimated_storage_capacity: float
    is_short_link: float


@dataclass(frozen=True)
class DynamicLaneGroupFeatures:
    vehicle_count_detector: float
    moving_count_detector: float
    halting_count_detector: float
    queue_length_m_detector: float
    queue_length_vehicles_detector: float
    occupancy_detector: float
    mean_speed_detector: float
    density_detector: float
    available_storage_detector_ratio: float
    arrival_rate_15s: float
    departure_rate_15s: float
    arrival_rate_60s: float
    departure_rate_60s: float
    detector_saturation: float
    vehicle_count_norm_detector: float
    moving_count_norm_detector: float
    queue_length_norm_detector: float


@dataclass(frozen=True)
class StaticMovementFeatures:
    turn_type: str
    num_underlying_controlled_links: float
    saturation_flow_estimate: float
    input_lane_group_id: LaneGroupId
    output_lane_group_id: LaneGroupId


@dataclass(frozen=True)
class DynamicMovementFeatures:
    oracle_movement_demand: float
    oracle_movement_demand_norm: float
    is_currently_enabled: float
    was_enabled_last_decision: float
    time_since_enabled_s: float


@dataclass(frozen=True)
class LaneGroupFeatureRow:
    lane_group_id: LaneGroupId
    static: StaticLaneGroupFeatures
    dynamic: DynamicLaneGroupFeatures


@dataclass(frozen=True)
class MovementFeatureRow:
    movement_id: GraphMovementId
    static: StaticMovementFeatures
    dynamic: DynamicMovementFeatures


@dataclass(frozen=True)
class MovementFeatureFrame:
    lane_group_rows: tuple[LaneGroupFeatureRow, ...]
    movement_rows: tuple[MovementFeatureRow, ...]


@dataclass(frozen=True)
class MovementControlState:
    current_enabled_movement_ids: tuple[GraphMovementId, ...] = ()
    previous_enabled_movement_ids: tuple[GraphMovementId, ...] = ()
    time_since_enabled_s: Mapping[GraphMovementId, float] | None = None


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: str
    lane_id: LaneId | str
    next_edge_id: EdgeId | str | None
    lane_position_m: float
    speed_mps: float
    length_m: float


def vehicle_snapshots_from_api(vehicle_api: VehicleDomain) -> tuple[VehicleSnapshot, ...]:
    """Capture vehicle state used by detector-local and movement features."""
    snapshots: list[VehicleSnapshot] = []
    for vehicle_id in vehicle_api.getIDList():
        route = tuple(str(edge_id) for edge_id in vehicle_api.getRoute(vehicle_id))
        route_index = int(vehicle_api.getRouteIndex(vehicle_id))
        next_edge = route[route_index + 1] if route_index + 1 < len(route) else None
        snapshots.append(
            VehicleSnapshot(
                vehicle_id=str(vehicle_id),
                lane_id=LaneId(str(vehicle_api.getLaneID(vehicle_id))),
                next_edge_id=EdgeId(next_edge) if next_edge is not None else None,
                lane_position_m=float(vehicle_api.getLanePosition(vehicle_id)),
                speed_mps=float(vehicle_api.getSpeed(vehicle_id)),
                length_m=float(vehicle_api.getLength(vehicle_id)),
            )
        )
    return tuple(snapshots)


def detector_length(lane_group_length: float) -> float:
    """Return the downstream detector length for a lane group."""
    return min(DEFAULT_DETECTOR_LENGTH_M, max(0.0, float(lane_group_length)))


def detector_capacity(
    detector_length_m: float,
    num_lanes: int,
    effective_vehicle_spacing_m: float = EFFECTIVE_VEHICLE_SPACING_M,
) -> float:
    """Approximate detector vehicle storage capacity."""
    if effective_vehicle_spacing_m <= 0.0:
        raise ValueError('effective_vehicle_spacing_m must be positive.')
    return max(0.0, float(detector_length_m) * float(num_lanes) / effective_vehicle_spacing_m)


def build_feature_frame(
    graph: MovementGraph,
    lane_ids_by_edge: Mapping[EdgeId | str, Sequence[LaneId | str]],
    lane_geometries: Mapping[EdgeId | str, LaneGroupGeometry],
    control_state: MovementControlState,
    vehicles: Sequence[VehicleSnapshot],
) -> MovementFeatureFrame:
    """Extract lane-group and movement feature rows aligned with graph IDs."""
    geometries = {EdgeId(str(edge_id)): geometry for edge_id, geometry in lane_geometries.items()}
    lanes_by_edge = {
        EdgeId(str(edge_id)): tuple(LaneId(str(lane_id)) for lane_id in lane_ids)
        for edge_id, lane_ids in lane_ids_by_edge.items()
    }

    lane_rows = tuple(
        _lane_group_row(
            lane_group_id=lane_group.lane_group_id,
            edge_id=lane_group.edge_id,
            lane_ids=lanes_by_edge.get(lane_group.edge_id, ()),
            geometry=geometries.get(lane_group.edge_id),
            vehicles=vehicles,
        )
        for lane_group in graph.lane_groups
    )
    lane_capacity_by_group = {row.lane_group_id: row.static.estimated_storage_capacity for row in lane_rows}
    movement_rows = tuple(
        _movement_row(
            graph=graph,
            movement_id=movement.movement_id,
            input_lane_group_id=movement.input_lane_group_id,
            output_lane_group_id=movement.output_lane_group_id,
            num_controlled_links=len(movement.controlled_movement_indices),
            control_state=control_state,
            vehicles=vehicles,
            lane_capacity_by_group=lane_capacity_by_group,
        )
        for movement in graph.movements
    )
    return MovementFeatureFrame(
        lane_group_rows=lane_rows,
        movement_rows=movement_rows,
    )


def _lane_group_row(
    lane_group_id: LaneGroupId,
    edge_id: EdgeId,
    lane_ids: tuple[LaneId, ...],
    geometry: LaneGroupGeometry | None,
    vehicles: Sequence[VehicleSnapshot],
) -> LaneGroupFeatureRow:
    num_lanes = geometry.num_lanes if geometry is not None else max(1, len(lane_ids))
    length_m = geometry.length_m if geometry is not None else 0.0
    speed_limit_mps = geometry.speed_limit_mps if geometry is not None else DEFAULT_SPEED_LIMIT_MPS
    detector_length_m = detector_length(length_m)
    capacity = detector_capacity(detector_length_m, num_lanes)
    detector_vehicles = _detector_vehicles(
        edge_id=edge_id,
        lane_ids=lane_ids,
        lane_group_length_m=length_m,
        detector_length_m=detector_length_m,
        vehicles=vehicles,
    )
    moving_vehicles = tuple(vehicle for vehicle in detector_vehicles if vehicle.speed_mps > HALTING_SPEED_THRESHOLD_MPS)
    halting_vehicles = tuple(
        vehicle for vehicle in detector_vehicles if vehicle.speed_mps <= HALTING_SPEED_THRESHOLD_MPS
    )
    vehicle_count = float(len(detector_vehicles))
    moving_count = float(len(moving_vehicles))
    halting_count = float(len(halting_vehicles))
    queue_length_m = _queue_length_m(
        halting_vehicles=halting_vehicles,
        lane_group_length_m=length_m,
    )
    occupancy = _detector_occupancy_percent(
        vehicles=detector_vehicles,
        detector_length_m=detector_length_m,
        num_lanes=num_lanes,
    )
    mean_speed = _mean_vehicle_speed(detector_vehicles)
    vehicle_count_norm = _safe_div(vehicle_count, capacity)
    moving_count_norm = _safe_div(moving_count, capacity)
    queue_length_norm = _safe_div(queue_length_m, detector_length_m)
    saturation = 1.0 if vehicle_count_norm >= 0.95 or queue_length_norm >= 0.95 else 0.0
    static = StaticLaneGroupFeatures(
        length_m=float(length_m),
        detector_length_m=detector_length_m,
        num_lanes=float(num_lanes),
        speed_limit_mps=float(speed_limit_mps),
        freeflow_travel_time_s=_safe_div(length_m, speed_limit_mps),
        estimated_storage_capacity=capacity,
        is_short_link=1.0 if length_m <= DEFAULT_DETECTOR_LENGTH_M else 0.0,
    )
    dynamic = DynamicLaneGroupFeatures(
        vehicle_count_detector=vehicle_count,
        moving_count_detector=moving_count,
        halting_count_detector=halting_count,
        queue_length_m_detector=queue_length_m,
        queue_length_vehicles_detector=halting_count,
        occupancy_detector=occupancy,
        mean_speed_detector=mean_speed,
        density_detector=vehicle_count_norm,
        available_storage_detector_ratio=max(0.0, 1.0 - vehicle_count_norm),
        arrival_rate_15s=0.0,
        departure_rate_15s=0.0,
        arrival_rate_60s=0.0,
        departure_rate_60s=0.0,
        detector_saturation=saturation,
        vehicle_count_norm_detector=vehicle_count_norm,
        moving_count_norm_detector=moving_count_norm,
        queue_length_norm_detector=queue_length_norm,
    )
    return LaneGroupFeatureRow(
        lane_group_id=lane_group_id,
        static=static,
        dynamic=dynamic,
    )


def _movement_row(
    graph: MovementGraph,
    movement_id: GraphMovementId,
    input_lane_group_id: LaneGroupId,
    output_lane_group_id: LaneGroupId,
    num_controlled_links: int,
    control_state: MovementControlState,
    vehicles: Sequence[VehicleSnapshot],
    lane_capacity_by_group: Mapping[LaneGroupId, float],
) -> MovementFeatureRow:
    demand = _oracle_movement_demand(
        graph=graph,
        input_lane_group_id=input_lane_group_id,
        output_lane_group_id=output_lane_group_id,
        vehicles=vehicles,
    )
    capacity = lane_capacity_by_group.get(input_lane_group_id, 0.0)
    current_enabled = set(control_state.current_enabled_movement_ids)
    previous_enabled = set(control_state.previous_enabled_movement_ids)
    time_since_enabled = control_state.time_since_enabled_s or {}
    static = StaticMovementFeatures(
        turn_type='unknown',
        num_underlying_controlled_links=float(num_controlled_links),
        saturation_flow_estimate=float(num_controlled_links),
        input_lane_group_id=input_lane_group_id,
        output_lane_group_id=output_lane_group_id,
    )
    dynamic = DynamicMovementFeatures(
        oracle_movement_demand=demand,
        oracle_movement_demand_norm=_safe_div(demand, capacity),
        is_currently_enabled=1.0 if movement_id in current_enabled else 0.0,
        was_enabled_last_decision=1.0 if movement_id in previous_enabled else 0.0,
        time_since_enabled_s=float(time_since_enabled.get(movement_id, 0.0)),
    )
    return MovementFeatureRow(
        movement_id=movement_id,
        static=static,
        dynamic=dynamic,
    )


def _oracle_movement_demand(
    graph: MovementGraph,
    input_lane_group_id: LaneGroupId,
    output_lane_group_id: LaneGroupId,
    vehicles: Sequence[VehicleSnapshot],
) -> float:
    input_edge = graph.lane_groups[input_lane_group_id].edge_id
    output_edge = graph.lane_groups[output_lane_group_id].edge_id
    return float(
        sum(
            1
            for vehicle in vehicles
            if _lane_edge(vehicle.lane_id) == input_edge
            and vehicle.next_edge_id is not None
            and EdgeId(str(vehicle.next_edge_id)) == output_edge
        )
    )


def _lane_edge(lane_id: LaneId | str) -> EdgeId:
    text = str(lane_id)
    edge_text, separator, lane_index = text.rpartition('_')
    if separator and lane_index.isdigit() and edge_text:
        return EdgeId(edge_text)
    return EdgeId(text)


def _detector_vehicles(
    edge_id: EdgeId,
    lane_ids: tuple[LaneId, ...],
    lane_group_length_m: float,
    detector_length_m: float,
    vehicles: Sequence[VehicleSnapshot],
) -> tuple[VehicleSnapshot, ...]:
    detector_start_m = max(0.0, lane_group_length_m - detector_length_m)
    lane_id_set = set(lane_ids)
    return tuple(
        vehicle
        for vehicle in vehicles
        if (vehicle.lane_id in lane_id_set or (not lane_id_set and _lane_edge(vehicle.lane_id) == edge_id))
        and detector_start_m <= vehicle.lane_position_m <= lane_group_length_m
    )


def _queue_length_m(
    halting_vehicles: Sequence[VehicleSnapshot],
    lane_group_length_m: float,
) -> float:
    if not halting_vehicles:
        return 0.0
    queue_start_m = min(vehicle.lane_position_m for vehicle in halting_vehicles)
    return max(0.0, lane_group_length_m - queue_start_m)


def _detector_occupancy_percent(
    vehicles: Sequence[VehicleSnapshot],
    detector_length_m: float,
    num_lanes: int,
) -> float:
    observed_lane_length_m = detector_length_m * num_lanes
    occupied_length_m = sum(vehicle.length_m for vehicle in vehicles)
    return min(100.0, 100.0 * _safe_div(occupied_length_m, observed_lane_length_m))


def _mean_vehicle_speed(vehicles: Sequence[VehicleSnapshot]) -> float:
    if not vehicles:
        return 0.0
    return sum(vehicle.speed_mps for vehicle in vehicles) / len(vehicles)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)
