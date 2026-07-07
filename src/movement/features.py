"""Feature extraction for movement-score learning samples."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .graph_schema import GraphMovementId, LaneGroupId, MovementGraph
from .schema import EdgeId, LaneId, TrafficLightProgram
from .sumo_backend import (
    VehicleApi,
    subscription_lane_id_key,
    subscription_lane_position_key,
    subscription_length_key,
    subscription_route_index_key,
    subscription_speed_key,
    vehicle_subscription_variables,
)

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
    approaching_queue_tail_count: float
    fast_approaching_queue_tail_count: float
    min_eta_to_queue_tail_s: float
    mean_eta_to_queue_tail_s: float
    predicted_arrivals_to_queue_tail_5s: float
    predicted_arrivals_to_queue_tail_10s: float
    predicted_arrivals_to_queue_tail_15s: float


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
    was_green_last_decision: float


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
    green_movement_ids_last_decision: tuple[GraphMovementId, ...] = ()


@dataclass(frozen=True)
class LaneGroupFlowRates:
    arrival_rate_15s: float
    departure_rate_15s: float
    arrival_rate_60s: float
    departure_rate_60s: float


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: str
    lane_id: LaneId | str
    next_edge_id: EdgeId | str | None
    lane_position_m: float
    speed_mps: float
    length_m: float
    route_edge_ids_ahead: tuple[EdgeId, ...] = ()


@dataclass(frozen=True)
class PositionedVehicleSnapshot:
    vehicle: VehicleSnapshot
    position_m: float


@dataclass(frozen=True)
class LaneGroupDetectorContext:
    edge_offsets_m: dict[EdgeId, float]
    detector_start_m: float
    total_length_m: float


@dataclass(frozen=True)
class VehicleFeatureIndex:
    detector_vehicles_by_lane_group: dict[LaneGroupId, tuple[PositionedVehicleSnapshot, ...]]
    movement_demand_by_lane_group_pair: dict[tuple[LaneGroupId, LaneGroupId], float]


class VehicleSnapshotCollector:
    """Collect vehicle state through persistent TraCI subscriptions."""

    def __init__(self, vehicle_api: VehicleApi) -> None:
        self._vehicle_api = vehicle_api
        self._subscribed_vehicle_ids: set[str] = set()
        self._route_by_vehicle_id: dict[str, tuple[str, ...]] = {}

    def capture(self) -> tuple[VehicleSnapshot, ...]:
        vehicle_ids = tuple(str(vehicle_id) for vehicle_id in self._vehicle_api.getIDList())
        active_vehicle_ids = set(vehicle_ids)
        self._subscribed_vehicle_ids.intersection_update(active_vehicle_ids)
        self._route_by_vehicle_id = {
            vehicle_id: route
            for vehicle_id, route in self._route_by_vehicle_id.items()
            if vehicle_id in active_vehicle_ids
        }
        for vehicle_id in active_vehicle_ids - self._subscribed_vehicle_ids:
            self._route_by_vehicle_id[vehicle_id] = tuple(
                str(edge_id) for edge_id in self._vehicle_api.getRoute(vehicle_id)
            )
            self._vehicle_api.subscribe(
                vehicle_id,
                vehicle_subscription_variables(),
            )
            self._subscribed_vehicle_ids.add(vehicle_id)

        subscription_results = self._vehicle_api.getAllSubscriptionResults()
        lane_id_key = subscription_lane_id_key()
        lane_position_key = subscription_lane_position_key()
        speed_key = subscription_speed_key()
        length_key = subscription_length_key()
        route_index_key = subscription_route_index_key()
        snapshots: list[VehicleSnapshot] = []
        for vehicle_id in vehicle_ids:
            result = subscription_results[vehicle_id]
            route = self._route_by_vehicle_id[vehicle_id]
            route_index = int(result[route_index_key])
            next_edge = route[route_index + 1] if route_index + 1 < len(route) else None
            snapshots.append(
                VehicleSnapshot(
                    vehicle_id=vehicle_id,
                    lane_id=LaneId(str(result[lane_id_key])),
                    next_edge_id=EdgeId(next_edge) if next_edge is not None else None,
                    lane_position_m=float(result[lane_position_key]),
                    speed_mps=float(result[speed_key]),
                    length_m=float(result[length_key]),
                    route_edge_ids_ahead=tuple(EdgeId(edge_id) for edge_id in route[route_index + 1 :]),
                )
            )
        return tuple(snapshots)


class LaneGroupFlowTracker:
    """Estimate detector arrivals and departures between decision snapshots."""

    def __init__(
        self,
        graph: MovementGraph,
        lane_ids_by_edge: Mapping[EdgeId | str, Sequence[LaneId | str]],
        lane_geometries: Mapping[EdgeId | str, LaneGroupGeometry],
        decision_interval_s: float,
    ) -> None:
        if decision_interval_s <= 0.0:
            raise ValueError('decision_interval_s must be positive.')
        self._graph = graph
        self._lane_ids_by_edge = {
            EdgeId(str(edge_id)): tuple(LaneId(str(lane_id)) for lane_id in lane_ids)
            for edge_id, lane_ids in lane_ids_by_edge.items()
        }
        self._lane_geometries = {EdgeId(str(edge_id)): geometry for edge_id, geometry in lane_geometries.items()}
        self._decision_interval_s = float(decision_interval_s)
        self._previous_vehicle_ids: dict[LaneGroupId, set[str]] | None = None
        self._arrival_history: dict[LaneGroupId, deque[int]] = {
            lane_group.lane_group_id: deque(maxlen=4) for lane_group in graph.lane_groups
        }
        self._departure_history: dict[LaneGroupId, deque[int]] = {
            lane_group.lane_group_id: deque(maxlen=4) for lane_group in graph.lane_groups
        }

    def observe(self, vehicle_index: VehicleFeatureIndex) -> dict[LaneGroupId, LaneGroupFlowRates]:
        """Update detector membership and return 15/60-second flow rates."""
        current_vehicle_ids = {
            lane_group.lane_group_id: {
                positioned.vehicle.vehicle_id
                for positioned in vehicle_index.detector_vehicles_by_lane_group[lane_group.lane_group_id]
            }
            for lane_group in self._graph.lane_groups
        }
        if self._previous_vehicle_ids is None:
            arrivals = {lane_group_id: 0 for lane_group_id in current_vehicle_ids}
            departures = {lane_group_id: 0 for lane_group_id in current_vehicle_ids}
        else:
            arrivals = {
                lane_group_id: len(vehicle_ids - self._previous_vehicle_ids[lane_group_id])
                for lane_group_id, vehicle_ids in current_vehicle_ids.items()
            }
            departures = {
                lane_group_id: len(self._previous_vehicle_ids[lane_group_id] - vehicle_ids)
                for lane_group_id, vehicle_ids in current_vehicle_ids.items()
            }
        self._previous_vehicle_ids = current_vehicle_ids
        rates: dict[LaneGroupId, LaneGroupFlowRates] = {}
        for lane_group_id in current_vehicle_ids:
            self._arrival_history[lane_group_id].append(arrivals[lane_group_id])
            self._departure_history[lane_group_id].append(departures[lane_group_id])
            history_duration_s = self._decision_interval_s * len(self._arrival_history[lane_group_id])
            rates[lane_group_id] = LaneGroupFlowRates(
                arrival_rate_15s=arrivals[lane_group_id] / self._decision_interval_s,
                departure_rate_15s=departures[lane_group_id] / self._decision_interval_s,
                arrival_rate_60s=sum(self._arrival_history[lane_group_id]) / history_duration_s,
                departure_rate_60s=sum(self._departure_history[lane_group_id]) / history_duration_s,
            )
        return rates


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


def movement_control_state_from_targets(
    graph: MovementGraph,
    programs: Mapping[str, TrafficLightProgram],
    target_states: Mapping[str, str],
) -> MovementControlState:
    """Encode movements served by the green targets used at the last decision."""
    enabled_movement_ids: list[GraphMovementId] = []
    for traffic_light_id, target_state in target_states.items():
        program = programs[traffic_light_id]
        local_phase_index = next(
            index for index, phase in enumerate(program.selectable_phases) if str(phase.state) == target_state
        )
        incidence = graph.phase_incidences[program.traffic_light_id]
        enabled_movement_ids.extend(
            movement_id
            for enabled, movement_id in zip(
                incidence.rows[local_phase_index],
                incidence.movement_ids,
            )
            if enabled
        )
    return MovementControlState(
        green_movement_ids_last_decision=tuple(enabled_movement_ids),
    )


def build_feature_frame(
    graph: MovementGraph,
    lane_ids_by_edge: Mapping[EdgeId | str, Sequence[LaneId | str]],
    lane_geometries: Mapping[EdgeId | str, LaneGroupGeometry],
    control_state: MovementControlState,
    vehicles: Sequence[VehicleSnapshot],
    lane_flow_rates: Mapping[LaneGroupId, LaneGroupFlowRates] | None = None,
    vehicle_index: VehicleFeatureIndex | None = None,
) -> MovementFeatureFrame:
    """Extract lane-group and movement feature rows aligned with graph IDs."""
    geometries = {EdgeId(str(edge_id)): geometry for edge_id, geometry in lane_geometries.items()}
    lanes_by_edge = {
        EdgeId(str(edge_id)): tuple(LaneId(str(lane_id)) for lane_id in lane_ids)
        for edge_id, lane_ids in lane_ids_by_edge.items()
    }
    feature_index = (
        vehicle_index
        if vehicle_index is not None
        else build_vehicle_feature_index(
            graph=graph,
            lane_ids_by_edge=lanes_by_edge,
            lane_geometries=geometries,
            vehicles=vehicles,
        )
    )

    lane_rows = tuple(
        _lane_group_row(
            lane_group_id=lane_group.lane_group_id,
            edge_ids=lane_group.edge_ids,
            lane_ids_by_edge=lanes_by_edge,
            lane_geometries=geometries,
            positioned_detector_vehicles=feature_index.detector_vehicles_by_lane_group[lane_group.lane_group_id],
            flow_rates=(lane_flow_rates.get(lane_group.lane_group_id) if lane_flow_rates is not None else None),
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
            movement_demand_by_lane_group_pair=feature_index.movement_demand_by_lane_group_pair,
            lane_capacity_by_group=lane_capacity_by_group,
        )
        for movement in graph.movements
    )
    return MovementFeatureFrame(
        lane_group_rows=lane_rows,
        movement_rows=movement_rows,
    )


def build_vehicle_feature_index(
    graph: MovementGraph,
    lane_ids_by_edge: Mapping[EdgeId | str, Sequence[LaneId | str]],
    lane_geometries: Mapping[EdgeId | str, LaneGroupGeometry],
    vehicles: Sequence[VehicleSnapshot],
) -> VehicleFeatureIndex:
    lanes_by_edge = {
        EdgeId(str(edge_id)): tuple(LaneId(str(lane_id)) for lane_id in lane_ids)
        for edge_id, lane_ids in lane_ids_by_edge.items()
    }
    geometries = {EdgeId(str(edge_id)): geometry for edge_id, geometry in lane_geometries.items()}
    lane_to_edge = {lane_id: edge_id for edge_id, lane_ids in lanes_by_edge.items() for lane_id in lane_ids}
    detector_context_by_lane_group = _detector_context_by_lane_group(graph=graph, lane_geometries=geometries)
    detector_vehicles: dict[LaneGroupId, list[PositionedVehicleSnapshot]] = {
        lane_group.lane_group_id: [] for lane_group in graph.lane_groups
    }
    movement_demand_by_lane_group_pair: dict[tuple[LaneGroupId, LaneGroupId], float] = {}
    for vehicle in vehicles:
        edge_id = lane_to_edge.get(LaneId(str(vehicle.lane_id)), _lane_edge(vehicle.lane_id))
        lane_group_id = graph.lane_group_id_by_edge.get(edge_id)
        if lane_group_id is None:
            continue
        context = detector_context_by_lane_group[lane_group_id]
        edge_offset_m = context.edge_offsets_m.get(edge_id)
        if edge_offset_m is not None:
            position_m = edge_offset_m + vehicle.lane_position_m
            if context.detector_start_m <= position_m <= context.total_length_m:
                detector_vehicles[lane_group_id].append(
                    PositionedVehicleSnapshot(vehicle=vehicle, position_m=position_m)
                )
        next_lane_group_id = _next_lane_group_id(graph=graph, vehicle=vehicle, current_lane_group_id=lane_group_id)
        if next_lane_group_id is not None:
            key = (lane_group_id, next_lane_group_id)
            movement_demand_by_lane_group_pair[key] = movement_demand_by_lane_group_pair.get(key, 0.0) + 1.0
    return VehicleFeatureIndex(
        detector_vehicles_by_lane_group={
            lane_group_id: tuple(positioned_vehicles)
            for lane_group_id, positioned_vehicles in detector_vehicles.items()
        },
        movement_demand_by_lane_group_pair=movement_demand_by_lane_group_pair,
    )


def _detector_context_by_lane_group(
    graph: MovementGraph,
    lane_geometries: Mapping[EdgeId, LaneGroupGeometry],
) -> dict[LaneGroupId, LaneGroupDetectorContext]:
    contexts: dict[LaneGroupId, LaneGroupDetectorContext] = {}
    for lane_group in graph.lane_groups:
        edge_offsets: dict[EdgeId, float] = {}
        total_length_m = 0.0
        for edge_id in lane_group.edge_ids:
            edge_offsets[edge_id] = total_length_m
            total_length_m += lane_geometries.get(
                edge_id,
                LaneGroupGeometry(length_m=0.0, num_lanes=1),
            ).length_m
        contexts[lane_group.lane_group_id] = LaneGroupDetectorContext(
            edge_offsets_m=edge_offsets,
            detector_start_m=max(0.0, total_length_m - detector_length(total_length_m)),
            total_length_m=total_length_m,
        )
    return contexts


def _lane_group_row(
    lane_group_id: LaneGroupId,
    edge_ids: tuple[EdgeId, ...],
    lane_ids_by_edge: Mapping[EdgeId, tuple[LaneId, ...]],
    lane_geometries: Mapping[EdgeId, LaneGroupGeometry],
    positioned_detector_vehicles: Sequence[PositionedVehicleSnapshot],
    flow_rates: LaneGroupFlowRates | None,
) -> LaneGroupFeatureRow:
    segment_geometries = tuple(
        lane_geometries.get(
            edge_id,
            LaneGroupGeometry(
                length_m=0.0,
                num_lanes=max(1, len(lane_ids_by_edge.get(edge_id, ()))),
            ),
        )
        for edge_id in edge_ids
    )
    length_m = sum(geometry.length_m for geometry in segment_geometries)
    freeflow_travel_time_s = sum(
        _safe_div(geometry.length_m, geometry.speed_limit_mps) for geometry in segment_geometries
    )
    speed_limit_mps = _safe_div(length_m, freeflow_travel_time_s)
    lane_length_m = sum(geometry.length_m * geometry.num_lanes for geometry in segment_geometries)
    num_lanes = _safe_div(lane_length_m, length_m)
    detector_length_m = detector_length(length_m)
    detector_lane_length_m = _detector_lane_length_m(segment_geometries, detector_length_m)
    capacity = detector_lane_length_m / EFFECTIVE_VEHICLE_SPACING_M
    detector_vehicles = tuple(positioned.vehicle for positioned in positioned_detector_vehicles)
    moving_vehicles = tuple(vehicle for vehicle in detector_vehicles if vehicle.speed_mps > HALTING_SPEED_THRESHOLD_MPS)
    halting_positions = tuple(
        positioned.position_m
        for positioned in positioned_detector_vehicles
        if positioned.vehicle.speed_mps <= HALTING_SPEED_THRESHOLD_MPS
    )
    vehicle_count = float(len(detector_vehicles))
    moving_count = float(len(moving_vehicles))
    halting_count = float(len(halting_positions))
    queue_length_m = max(0.0, length_m - min(halting_positions)) if halting_positions else 0.0
    occupancy = _detector_occupancy_percent(
        vehicles=detector_vehicles,
        observed_lane_length_m=detector_lane_length_m,
    )
    mean_speed = _mean_vehicle_speed(detector_vehicles)
    vehicle_count_norm = _safe_div(vehicle_count, capacity)
    moving_count_norm = _safe_div(moving_count, capacity)
    queue_length_norm = _safe_div(queue_length_m, detector_length_m)
    queue_tail_position_m = _queue_tail_position_m(
        total_length_m=length_m,
        detector_start_m=max(0.0, length_m - detector_length_m),
        halting_positions=halting_positions,
    )
    queue_tail_etas = _queue_tail_etas(
        positioned_detector_vehicles=positioned_detector_vehicles,
        queue_tail_position_m=queue_tail_position_m,
    )
    fast_queue_tail_etas = tuple(eta for vehicle, eta in queue_tail_etas if vehicle.speed_mps >= 0.5 * speed_limit_mps)
    eta_values = tuple(eta for _vehicle, eta in queue_tail_etas)
    saturation = 1.0 if vehicle_count_norm >= 0.95 or queue_length_norm >= 0.95 else 0.0
    static = StaticLaneGroupFeatures(
        length_m=float(length_m),
        detector_length_m=detector_length_m,
        num_lanes=float(num_lanes),
        speed_limit_mps=float(speed_limit_mps),
        freeflow_travel_time_s=freeflow_travel_time_s,
        estimated_storage_capacity=capacity,
        is_short_link=1.0 if length_m <= DEFAULT_DETECTOR_LENGTH_M else 0.0,
    )
    dynamic = DynamicLaneGroupFeatures(
        vehicle_count_detector=vehicle_count,
        moving_count_detector=moving_count,
        halting_count_detector=halting_count,
        queue_length_m_detector=queue_length_m,
        occupancy_detector=occupancy,
        mean_speed_detector=mean_speed,
        density_detector=vehicle_count_norm,
        available_storage_detector_ratio=max(0.0, 1.0 - vehicle_count_norm),
        arrival_rate_15s=flow_rates.arrival_rate_15s if flow_rates is not None else 0.0,
        departure_rate_15s=flow_rates.departure_rate_15s if flow_rates is not None else 0.0,
        arrival_rate_60s=flow_rates.arrival_rate_60s if flow_rates is not None else 0.0,
        departure_rate_60s=flow_rates.departure_rate_60s if flow_rates is not None else 0.0,
        detector_saturation=saturation,
        vehicle_count_norm_detector=vehicle_count_norm,
        moving_count_norm_detector=moving_count_norm,
        queue_length_norm_detector=queue_length_norm,
        approaching_queue_tail_count=float(len(queue_tail_etas)),
        fast_approaching_queue_tail_count=float(len(fast_queue_tail_etas)),
        min_eta_to_queue_tail_s=min(eta_values, default=0.0),
        mean_eta_to_queue_tail_s=_mean_float(eta_values),
        predicted_arrivals_to_queue_tail_5s=float(sum(1 for eta in eta_values if eta <= 5.0)),
        predicted_arrivals_to_queue_tail_10s=float(sum(1 for eta in eta_values if eta <= 10.0)),
        predicted_arrivals_to_queue_tail_15s=float(sum(1 for eta in eta_values if eta <= 15.0)),
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
    movement_demand_by_lane_group_pair: Mapping[tuple[LaneGroupId, LaneGroupId], float],
    lane_capacity_by_group: Mapping[LaneGroupId, float],
) -> MovementFeatureRow:
    del graph
    demand = movement_demand_by_lane_group_pair.get((input_lane_group_id, output_lane_group_id), 0.0)
    capacity = lane_capacity_by_group.get(input_lane_group_id, 0.0)
    green_last_decision = set(control_state.green_movement_ids_last_decision)
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
        was_green_last_decision=1.0 if movement_id in green_last_decision else 0.0,
    )
    return MovementFeatureRow(
        movement_id=movement_id,
        static=static,
        dynamic=dynamic,
    )


def _next_lane_group_id(
    graph: MovementGraph,
    vehicle: VehicleSnapshot,
    current_lane_group_id: LaneGroupId,
) -> LaneGroupId | None:
    route_edges = vehicle.route_edge_ids_ahead
    if not route_edges and vehicle.next_edge_id is not None:
        route_edges = (EdgeId(str(vehicle.next_edge_id)),)
    for edge_id in route_edges:
        lane_group_id = graph.lane_group_id_by_edge.get(EdgeId(str(edge_id)))
        if lane_group_id is not None and lane_group_id != current_lane_group_id:
            return lane_group_id
    return None


def _lane_edge(lane_id: LaneId | str) -> EdgeId:
    text = str(lane_id)
    edge_text, separator, lane_index = text.rpartition('_')
    if separator and lane_index.isdigit() and edge_text:
        return EdgeId(edge_text)
    return EdgeId(text)


def _detector_lane_length_m(
    segment_geometries: Sequence[LaneGroupGeometry],
    detector_length_m: float,
) -> float:
    remaining_m = detector_length_m
    lane_length_m = 0.0
    for geometry in reversed(segment_geometries):
        observed_length_m = min(remaining_m, geometry.length_m)
        lane_length_m += observed_length_m * geometry.num_lanes
        remaining_m -= observed_length_m
        if remaining_m <= 0.0:
            break
    return lane_length_m


def _queue_tail_position_m(
    total_length_m: float,
    detector_start_m: float,
    halting_positions: Sequence[float],
) -> float:
    if not halting_positions:
        return total_length_m
    return max(detector_start_m, min(halting_positions) - EFFECTIVE_VEHICLE_SPACING_M)


def _queue_tail_etas(
    positioned_detector_vehicles: Sequence[PositionedVehicleSnapshot],
    queue_tail_position_m: float,
) -> tuple[tuple[VehicleSnapshot, float], ...]:
    etas: list[tuple[VehicleSnapshot, float]] = []
    for positioned in positioned_detector_vehicles:
        vehicle = positioned.vehicle
        if vehicle.speed_mps <= HALTING_SPEED_THRESHOLD_MPS:
            continue
        distance_to_queue_tail_m = queue_tail_position_m - positioned.position_m
        if distance_to_queue_tail_m <= 0.0:
            continue
        etas.append((vehicle, distance_to_queue_tail_m / vehicle.speed_mps))
    return tuple(etas)


def _detector_occupancy_percent(
    vehicles: Sequence[VehicleSnapshot],
    observed_lane_length_m: float,
) -> float:
    occupied_length_m = sum(vehicle.length_m for vehicle in vehicles)
    return min(100.0, 100.0 * _safe_div(occupied_length_m, observed_lane_length_m))


def _mean_vehicle_speed(vehicles: Sequence[VehicleSnapshot]) -> float:
    if not vehicles:
        return 0.0
    return sum(vehicle.speed_mps for vehicle in vehicles) / len(vehicles)


def _mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)
