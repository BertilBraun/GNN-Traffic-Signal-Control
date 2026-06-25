"""Episode runner for movement-policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import sumolib
import torch
import traci

from src.movement.dataset import build_dataset_sample
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.evaluation.metrics import EvaluationMetrics, parse_tripinfo_metrics
from src.movement.evaluation.progression import GreenWaveTracker
from src.movement.features import (
    LaneGroupFlowTracker,
    LaneGroupGeometry,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
    movement_control_state_from_targets,
)
from src.movement.graph import build_movement_graph
from src.movement.graph_schema import MovementGraph
from src.movement.initial_traffic import generate_initial_traffic_population, sample_target_occupancy
from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.policies import MovementScoringMethod
from src.movement.policies.graph_scores import compute_graph_movement_scores
from src.movement.runtime import MovementControlRuntime
from src.movement.schema import TrafficLightProgram
from src.movement.training.il.checkpoint import (
    load_movement_checkpoint,
    normalizer_from_state,
)
from src.movement.training.il.tensors import (
    edge_tensors_from_sample,
    tensors_from_sample,
)


class EvaluationPolicy(str, Enum):
    MAX_PRESSURE = 'max-pressure'
    QUEUE = 'queue'
    LEARNED = 'learned'


@dataclass(frozen=True)
class LearnedPolicyConfig:
    checkpoint_path: Path
    device: str


@dataclass(frozen=True)
class LearnedPolicyContext:
    model: MovementScorer
    graph: MovementGraph
    lane_ids_by_edge: dict[str, tuple[str, ...]]
    lane_geometries: dict[str, LaneGroupGeometry]
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer
    vehicle_snapshot_collector: VehicleSnapshotCollector
    lane_flow_tracker: LaneGroupFlowTracker
    device: str


@dataclass(frozen=True)
class BaselinePolicyContext:
    graph: MovementGraph
    lane_ids_by_edge: dict[str, tuple[str, ...]]
    lane_geometries: dict[str, LaneGroupGeometry]
    vehicle_snapshot_collector: VehicleSnapshotCollector


def run_evaluation_episode(
    cfg_path: str | Path,
    policy: EvaluationPolicy,
    seed: int,
    steps: int,
    decision_interval: int,
    yellow_duration: int,
    min_green_steps: int,
    learned_policy_config: LearnedPolicyConfig | None,
    demand_scale: float,
    initial_occupancy_min: float,
    initial_occupancy_max: float,
    time_to_teleport: int | None = None,
) -> EvaluationMetrics:
    """Run one SUMO episode under one movement policy."""
    net_path = resolve_sumocfg_net_path(cfg_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    lane_ids = tuple(lane_id for lanes in lane_ids_by_edge.values() for lane_id in lanes)
    total_lane_length_m = _total_lane_length(lane_geometries)

    with tempfile.NamedTemporaryFile(suffix='.xml', prefix='movement_eval_tripinfo_', delete=False) as handle:
        tripinfo_path = Path(handle.name)

    demand_route_files = route_files_for_demand_scale(
        cfg_path=cfg_path,
        demand_scale=demand_scale,
    )
    initial_population = generate_initial_traffic_population(
        cfg_path=cfg_path,
        net_path=net_path,
        target_occupancy=sample_target_occupancy(
            minimum_occupancy=initial_occupancy_min,
            maximum_occupancy=initial_occupancy_max,
            seed=seed,
        ),
        seed=seed,
    )
    additional_sumo_args = (
        '--tripinfo-output',
        str(tripinfo_path),
        *route_file_sumo_args(
            (
                *demand_route_files.route_files,
                initial_population.route_file,
            )
        ),
    )
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=False,
        seed=seed,
        yellow_duration=yellow_duration,
        min_green_steps=min_green_steps,
        time_to_teleport=time_to_teleport,
        additional_sumo_args=additional_sumo_args,
    )
    queue_sum = 0.0
    max_queue = 0.0
    wait_density_sum = 0.0
    per_junction_wait_density_sum: dict[str, float] = {}
    per_junction_max_queue: dict[str, float] = {}
    per_junction_phase_counts: dict[str, list[int]] = {}
    switch_count = 0
    departed_vehicle_count = 0
    teleport_count = 0
    vehicles_remaining = 0
    simulated_steps = 0
    accepted_targets: dict[str, str] = {}
    progression_tracker = GreenWaveTracker(approach_distance_m=150.0, stop_speed_mps=0.1)

    try:
        runtime.start()
        per_junction_wait_density_sum = {traffic_light_id: 0.0 for traffic_light_id in runtime.programs}
        per_junction_max_queue = {traffic_light_id: 0.0 for traffic_light_id in runtime.programs}
        per_junction_phase_counts = {
            traffic_light_id: [0 for _phase in program.selectable_phases]
            for traffic_light_id, program in runtime.programs.items()
        }
        incoming_lanes_by_junction = _incoming_lanes_by_junction(runtime.programs)
        runtime.step()
        simulated_steps = 1
        departed_vehicle_count += int(traci.simulation.getDepartedNumber())
        teleport_count += int(traci.simulation.getStartingTeleportNumber())
        queue_values = tuple(float(traci.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lane_ids)
        if queue_values:
            queue_sum += sum(queue_values) / len(queue_values)
            max_queue = max(max_queue, max(queue_values))
        wait_density_sum += _wait_density(lane_ids, total_lane_length_m)
        _update_per_junction_metrics(
            incoming_lanes_by_junction=incoming_lanes_by_junction,
            per_junction_wait_density_sum=per_junction_wait_density_sum,
            per_junction_max_queue=per_junction_max_queue,
        )
        _update_progression_tracker(progression_tracker)
        learned_context = _learned_context(
            policy=policy,
            learned_policy_config=learned_policy_config,
            programs=runtime.programs,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            decision_interval=decision_interval,
            net_path=net_path,
        )
        baseline_context = _baseline_context(
            policy=policy,
            programs=runtime.programs,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            net_path=net_path,
        )
        for step in range(1, steps):
            if (step - 1) % decision_interval == 0:
                desired_states = _desired_states(
                    policy=policy,
                    programs=runtime.programs,
                    baseline_context=baseline_context,
                    learned_context=learned_context,
                    accepted_targets=accepted_targets,
                )
                next_accepted_targets = runtime.request_targets(desired_states)
                _record_phase_counts(
                    per_junction_phase_counts=per_junction_phase_counts,
                    programs=runtime.programs,
                    desired_states=desired_states,
                )
                switch_count += _count_switches(accepted_targets, next_accepted_targets)
                accepted_targets = dict(next_accepted_targets)

            runtime.step()
            simulated_steps += 1
            departed_vehicle_count += int(traci.simulation.getDepartedNumber())
            teleport_count += int(traci.simulation.getStartingTeleportNumber())
            queue_values = tuple(float(traci.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lane_ids)
            if queue_values:
                queue_sum += sum(queue_values) / len(queue_values)
                max_queue = max(max_queue, max(queue_values))
            wait_density_sum += _wait_density(lane_ids, total_lane_length_m)
            _update_per_junction_metrics(
                incoming_lanes_by_junction=incoming_lanes_by_junction,
                per_junction_wait_density_sum=per_junction_wait_density_sum,
                per_junction_max_queue=per_junction_max_queue,
            )
            _update_progression_tracker(progression_tracker)
            if not runtime.is_running():
                break
        vehicles_remaining = len(traci.vehicle.getIDList())
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()

    completed, throughput, average_wait, average_travel, average_time_loss = parse_tripinfo_metrics(
        tripinfo_path=tripinfo_path,
        episode_length_s=max(simulated_steps, 1),
    )
    tripinfo_path.unlink(missing_ok=True)
    progression = progression_tracker.metric_values()
    return EvaluationMetrics(
        departed_vehicles=departed_vehicle_count,
        completed_vehicles=completed,
        vehicles_remaining=vehicles_remaining,
        completion_rate=completed / departed_vehicle_count if departed_vehicle_count > 0 else 0.0,
        teleport_count=teleport_count,
        throughput_per_hour=throughput,
        average_waiting_time_s=average_wait,
        average_travel_time_s=average_travel,
        average_time_loss_s=average_time_loss,
        average_queue_length_vehicles=queue_sum / max(simulated_steps, 1),
        max_queue_length_vehicles=max_queue,
        average_wait_density_s_per_m=wait_density_sum / max(simulated_steps, 1),
        phase_switch_frequency_per_junction_per_minute=_switch_frequency(
            switch_count=switch_count,
            traffic_light_count=len(runtime.programs),
            simulated_steps=simulated_steps,
        ),
        average_tls_passes_per_vehicle=progression[0],
        average_stops_before_tls_per_vehicle=progression[1],
        nonstop_tls_pass_rate=progression[2],
        average_best_nonstop_tls_streak=progression[3],
        per_junction_wait_density_s_per_m={
            traffic_light_id: value / max(simulated_steps, 1)
            for traffic_light_id, value in per_junction_wait_density_sum.items()
        },
        per_junction_max_queue_length_vehicles=per_junction_max_queue,
        per_junction_phase_counts={
            traffic_light_id: tuple(counts) for traffic_light_id, counts in per_junction_phase_counts.items()
        },
    )


def resolve_sumocfg_net_path(cfg_path: str | Path) -> Path:
    """Resolve the net-file referenced by a SUMO config."""
    config_path = Path(cfg_path)
    root = ET.parse(config_path).getroot()
    net_file = root.find('./input/net-file')
    if net_file is None or 'value' not in net_file.attrib:
        raise ValueError(f'{config_path} does not define input/net-file.')
    net_path = Path(net_file.attrib['value'])
    if not net_path.is_absolute():
        net_path = config_path.parent / net_path
    return net_path


def lane_inputs_from_net(
    net_path: str | Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, LaneGroupGeometry]]:
    """Extract lane IDs and static geometry keyed by SUMO edge id."""
    net = sumolib.net.readNet(str(net_path), withConnections=True)
    lane_ids_by_edge: dict[str, tuple[str, ...]] = {}
    lane_geometries: dict[str, LaneGroupGeometry] = {}
    for edge in net.getEdges():
        edge_id = str(edge.getID())
        if edge_id.startswith(':'):
            continue
        lanes = tuple(str(lane.getID()) for lane in edge.getLanes())
        lane_ids_by_edge[edge_id] = lanes
        lane_geometries[edge_id] = LaneGroupGeometry(
            length_m=float(edge.getLength()),
            num_lanes=len(lanes),
            speed_limit_mps=float(edge.getSpeed()),
        )
    return lane_ids_by_edge, lane_geometries


def _desired_states(
    policy: EvaluationPolicy,
    programs: Mapping[str, TrafficLightProgram],
    baseline_context: BaselinePolicyContext | None,
    learned_context: LearnedPolicyContext | None,
    accepted_targets: Mapping[str, str],
) -> dict[str, str]:
    match policy:
        case EvaluationPolicy.MAX_PRESSURE:
            return _baseline_states(programs, baseline_context, accepted_targets, MovementScoringMethod.MAX_PRESSURE)
        case EvaluationPolicy.QUEUE:
            return _baseline_states(programs, baseline_context, accepted_targets, MovementScoringMethod.QUEUE)
        case EvaluationPolicy.LEARNED:
            if learned_context is None:
                raise ValueError('learned_policy_config is required for learned evaluation.')
            control_state = movement_control_state_from_targets(
                graph=learned_context.graph,
                programs=programs,
                target_states=accepted_targets,
            )
            return _learned_states(
                programs=programs,
                learned_context=learned_context,
                control_state=control_state,
            )


def _baseline_states(
    programs: Mapping[str, TrafficLightProgram],
    baseline_context: BaselinePolicyContext | None,
    accepted_targets: Mapping[str, str],
    method: MovementScoringMethod,
) -> dict[str, str]:
    if baseline_context is None:
        raise ValueError('baseline_context is required for baseline evaluation.')
    control_state = movement_control_state_from_targets(
        graph=baseline_context.graph,
        programs=programs,
        target_states=accepted_targets,
    )
    vehicles = baseline_context.vehicle_snapshot_collector.capture()
    feature_frame = build_feature_frame(
        graph=baseline_context.graph,
        lane_ids_by_edge=baseline_context.lane_ids_by_edge,
        lane_geometries=baseline_context.lane_geometries,
        control_state=control_state,
        vehicles=vehicles,
    )
    graph_movement_scores = compute_graph_movement_scores(
        graph=baseline_context.graph,
        feature_frame=feature_frame,
        method=method,
    )
    states: dict[str, str] = {}
    for traffic_light_id, program in programs.items():
        incidence = baseline_context.graph.phase_incidences[program.traffic_light_id]
        movement_ids = tuple(int(value) for value in incidence.movement_ids)
        best_local_idx = 0
        best_score = _phase_score(incidence.rows[0], movement_ids, graph_movement_scores)
        for local_idx, row in enumerate(incidence.rows[1:], start=1):
            score = _phase_score(row, movement_ids, graph_movement_scores)
            if score > best_score:
                best_local_idx = local_idx
                best_score = score
        states[traffic_light_id] = program.selectable_phases[best_local_idx].state
    return states


def _phase_score(
    row: tuple[int, ...],
    movement_ids: tuple[int, ...],
    graph_movement_scores: tuple[float, ...],
) -> float:
    return sum(graph_movement_scores[movement_id] for enabled, movement_id in zip(row, movement_ids) if enabled)


def _baseline_context(
    policy: EvaluationPolicy,
    programs: Mapping[str, TrafficLightProgram],
    lane_ids_by_edge: dict[str, tuple[str, ...]],
    lane_geometries: dict[str, LaneGroupGeometry],
    net_path: Path,
) -> BaselinePolicyContext | None:
    if policy == EvaluationPolicy.LEARNED:
        return None
    return BaselinePolicyContext(
        graph=build_movement_graph(programs, net_path=net_path),
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        vehicle_snapshot_collector=VehicleSnapshotCollector(traci.vehicle),
    )


def _learned_context(
    policy: EvaluationPolicy,
    learned_policy_config: LearnedPolicyConfig | None,
    programs: Mapping[str, TrafficLightProgram],
    lane_ids_by_edge: dict[str, tuple[str, ...]],
    lane_geometries: dict[str, LaneGroupGeometry],
    decision_interval: int,
    net_path: Path,
) -> LearnedPolicyContext | None:
    if policy != EvaluationPolicy.LEARNED:
        return None
    if learned_policy_config is None:
        raise ValueError('learned_policy_config is required for learned evaluation.')
    model, metadata = load_movement_checkpoint(
        learned_policy_config.checkpoint_path,
        device=learned_policy_config.device,
    )
    graph = build_movement_graph(programs, net_path=net_path)
    return LearnedPolicyContext(
        model=model,
        graph=graph,
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        lane_normalizer=normalizer_from_state(metadata.lane_normalizer),
        movement_normalizer=normalizer_from_state(metadata.movement_normalizer),
        vehicle_snapshot_collector=VehicleSnapshotCollector(traci.vehicle),
        lane_flow_tracker=LaneGroupFlowTracker(
            graph=graph,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            decision_interval_s=decision_interval,
        ),
        device=learned_policy_config.device,
    )


def _learned_states(
    programs: Mapping[str, TrafficLightProgram],
    learned_context: LearnedPolicyContext,
    control_state: MovementControlState,
) -> dict[str, str]:
    vehicles = learned_context.vehicle_snapshot_collector.capture()
    sample = build_dataset_sample(
        graph=learned_context.graph,
        feature_frame=build_feature_frame(
            graph=learned_context.graph,
            lane_ids_by_edge=learned_context.lane_ids_by_edge,
            lane_geometries=learned_context.lane_geometries,
            control_state=control_state,
            vehicles=vehicles,
            lane_flow_rates=learned_context.lane_flow_tracker.observe(vehicles),
        ),
        programs=programs,
        teacher_controlled_scores={traffic_light_id: {} for traffic_light_id in programs},
        metadata={},
    )
    x_lane, x_movement, _target = tensors_from_sample(
        sample=sample,
        lane_normalizer=learned_context.lane_normalizer,
        movement_normalizer=learned_context.movement_normalizer,
        device=learned_context.device,
    )
    learned_context.model.eval()
    with torch.no_grad():
        movement_scores = tuple(
            float(value)
            for value in learned_context.model(
                x_lane=x_lane,
                x_movement=x_movement,
                edge_index_dict=edge_tensors_from_sample(sample, device=learned_context.device),
            ).cpu()
        )
    return _graph_score_states(
        programs=programs,
        graph=learned_context.graph,
        movement_scores=movement_scores,
    )


def _graph_score_states(
    programs: Mapping[str, TrafficLightProgram],
    graph: MovementGraph,
    movement_scores: Sequence[float],
) -> dict[str, str]:
    states: dict[str, str] = {}
    for traffic_light_id, program in programs.items():
        incidence = graph.phase_incidences[program.traffic_light_id]
        movement_ids = tuple(int(value) for value in incidence.movement_ids)
        best_local_index = 0
        best_score = _phase_score(incidence.rows[0], movement_ids, movement_scores)
        for local_index, row in enumerate(incidence.rows[1:], start=1):
            score = _phase_score(row, movement_ids, movement_scores)
            if score > best_score:
                best_local_index = local_index
                best_score = score
        states[traffic_light_id] = program.selectable_phases[best_local_index].state
    return states


def _phase_score(
    row: Sequence[int],
    movement_ids: Sequence[int],
    movement_scores: Sequence[float],
) -> float:
    return sum(float(movement_scores[movement_id]) for enabled, movement_id in zip(row, movement_ids) if enabled)


def _record_phase_counts(
    per_junction_phase_counts: dict[str, list[int]],
    programs: Mapping[str, TrafficLightProgram],
    desired_states: Mapping[str, str],
) -> None:
    for traffic_light_id, desired_state in desired_states.items():
        phases = programs[traffic_light_id].selectable_phases
        for local_phase_index, phase in enumerate(phases):
            if phase.state == desired_state:
                per_junction_phase_counts[traffic_light_id][local_phase_index] += 1
                break


def _incoming_lanes_by_junction(
    programs: Mapping[str, TrafficLightProgram],
) -> dict[str, tuple[str, ...]]:
    return {
        traffic_light_id: tuple(dict.fromkeys(str(movement.incoming_lane_id) for movement in program.movements))
        for traffic_light_id, program in programs.items()
    }


def _update_per_junction_metrics(
    incoming_lanes_by_junction: Mapping[str, tuple[str, ...]],
    per_junction_wait_density_sum: dict[str, float],
    per_junction_max_queue: dict[str, float],
) -> None:
    for traffic_light_id, lane_ids in incoming_lanes_by_junction.items():
        total_lane_length = sum(float(traci.lane.getLength(lane_id)) for lane_id in lane_ids)
        per_junction_wait_density_sum[traffic_light_id] += _wait_density(lane_ids, total_lane_length)
        queue_values = tuple(float(traci.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lane_ids)
        if queue_values:
            per_junction_max_queue[traffic_light_id] = max(
                per_junction_max_queue[traffic_light_id],
                max(queue_values),
            )


def _total_lane_length(lane_geometries: Mapping[str, LaneGroupGeometry]) -> float:
    return sum(geometry.length_m * geometry.num_lanes for geometry in lane_geometries.values())


def _wait_density(lane_ids: Sequence[str], total_lane_length_m: float) -> float:
    if total_lane_length_m <= 0.0:
        return 0.0
    total_waiting_time_s = sum(float(traci.lane.getWaitingTime(lane_id)) for lane_id in lane_ids)
    return total_waiting_time_s / total_lane_length_m


def _count_switches(
    previous_targets: Mapping[str, str],
    next_targets: Mapping[str, str],
) -> int:
    return sum(
        1
        for traffic_light_id, state in next_targets.items()
        if traffic_light_id in previous_targets and previous_targets[traffic_light_id] != state
    )


def _switch_frequency(
    switch_count: int,
    traffic_light_count: int,
    simulated_steps: int,
) -> float:
    if traffic_light_count <= 0 or simulated_steps <= 0:
        return 0.0
    return float(switch_count) / float(traffic_light_count) / (float(simulated_steps) / 60.0)


def _update_progression_tracker(tracker: GreenWaveTracker) -> None:
    vehicle_ids = tuple(str(vehicle_id) for vehicle_id in traci.vehicle.getIDList())
    next_tls_by_vehicle: dict[str, tuple[object, ...]] = {}
    speed_by_vehicle: dict[str, float] = {}
    for vehicle_id in vehicle_ids:
        try:
            next_tls_by_vehicle[vehicle_id] = tuple(traci.vehicle.getNextTLS(vehicle_id))
            speed_by_vehicle[vehicle_id] = float(traci.vehicle.getSpeed(vehicle_id))
        except traci.exceptions.TraCIException:
            continue
    tracker.update(
        vehicle_ids=vehicle_ids,
        next_tls_by_vehicle=next_tls_by_vehicle,
        speed_by_vehicle=speed_by_vehicle,
        arrived_vehicle_ids=tuple(str(vehicle_id) for vehicle_id in traci.simulation.getArrivedIDList()),
    )
