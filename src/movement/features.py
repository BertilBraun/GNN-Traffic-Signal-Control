"""Feature extraction for movement-score learning samples."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Protocol

from .graph_schema import GraphMovementId, LaneGroupId, MovementGraph
from .schema import EdgeId, LaneId

DEFAULT_DETECTOR_LENGTH_M = 200.0
EFFECTIVE_VEHICLE_SPACING_M = 8.0
DEFAULT_SPEED_LIMIT_MPS = 13.89


class LaneFeatureApi(Protocol):
    def getLastStepHaltingNumber(self, lane_id: LaneId | str) -> int:
        ...


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
    next_lane_id: LaneId | str | None


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
        raise ValueError("effective_vehicle_spacing_m must be positive.")
    return max(0.0, float(detector_length_m) * float(num_lanes) / effective_vehicle_spacing_m)


def build_feature_frame(
    graph: MovementGraph,
    lane_ids_by_edge: Mapping[EdgeId | str, Sequence[LaneId | str]],
    lane_geometries: Mapping[EdgeId | str, LaneGroupGeometry],
    lane_api: LaneFeatureApi,
    control_state: MovementControlState | None = None,
    vehicles: Sequence[VehicleSnapshot] = (),
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
            lane_api=lane_api,
        )
        for lane_group in graph.lane_groups
    )
    lane_capacity_by_group = {
        row.lane_group_id: row.static.estimated_storage_capacity
        for row in lane_rows
    }
    movement_rows = tuple(
        _movement_row(
            graph=graph,
            movement_id=movement.movement_id,
            input_lane_group_id=movement.input_lane_group_id,
            output_lane_group_id=movement.output_lane_group_id,
            num_controlled_links=len(movement.controlled_movement_indices),
            control_state=control_state or MovementControlState(),
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
    lane_api: LaneFeatureApi,
) -> LaneGroupFeatureRow:
    num_lanes = geometry.num_lanes if geometry is not None else max(1, len(lane_ids))
    length_m = geometry.length_m if geometry is not None else 0.0
    speed_limit_mps = (
        geometry.speed_limit_mps
        if geometry is not None
        else DEFAULT_SPEED_LIMIT_MPS
    )
    detector_length_m = detector_length(length_m)
    capacity = detector_capacity(detector_length_m, num_lanes)
    vehicle_count = _sum_lane_metric(lane_api, "getLastStepVehicleNumber", lane_ids)
    halting_count = _sum_lane_metric(lane_api, "getLastStepHaltingNumber", lane_ids)
    queue_length_m = _sum_lane_metric(lane_api, "getLastStepLength", lane_ids)
    occupancy = _mean_lane_metric(lane_api, "getLastStepOccupancy", lane_ids)
    mean_speed = _mean_lane_metric(lane_api, "getLastStepMeanSpeed", lane_ids)
    vehicle_count_norm = _safe_div(vehicle_count, capacity)
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
        turn_type="unknown",
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
            and vehicle.next_lane_id is not None
            and _lane_edge(vehicle.next_lane_id) == output_edge
        )
    )


def _lane_edge(lane_id: LaneId | str) -> EdgeId:
    text = str(lane_id)
    edge_text, separator, lane_index = text.rpartition("_")
    if separator and lane_index.isdigit() and edge_text:
        return EdgeId(edge_text)
    return EdgeId(text)


def _sum_lane_metric(
    lane_api: LaneFeatureApi,
    method_name: str,
    lane_ids: tuple[LaneId, ...],
) -> float:
    method = getattr(lane_api, method_name, None)
    if method is None:
        return 0.0
    return float(sum(float(method(lane_id)) for lane_id in lane_ids))


def _mean_lane_metric(
    lane_api: LaneFeatureApi,
    method_name: str,
    lane_ids: tuple[LaneId, ...],
) -> float:
    if not lane_ids:
        return 0.0
    method = getattr(lane_api, method_name, None)
    if method is None:
        return 0.0
    values = [float(method(lane_id)) for lane_id in lane_ids]
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)
