"""Collect movement-score imitation samples from SUMO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sumolib  # noqa: E402
import traci  # noqa: E402

from src.movement.dataset import build_dataset_sample, save_jsonl_samples  # noqa: E402
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale  # noqa: E402
from src.movement.features import (  # noqa: E402
    LaneGroupGeometry,
    LaneGroupFlowTracker,
    MovementFeatureFrame,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
    movement_control_state_from_targets,
)
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_schema import MovementGraph  # noqa: E402
from src.movement.initial_traffic import generate_initial_traffic_population  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402


@dataclass(frozen=True)
class VehicleTrajectoryState:
    vehicle_id: str
    lane_id: str
    lane_position_m: float
    speed_mps: float
    next_edge_id: str | None


@dataclass(frozen=True)
class DecisionTrajectoryState:
    vehicles: tuple[VehicleTrajectoryState, ...]
    target_states: tuple[tuple[str, str], ...]


def resolve_sumocfg_net_path(cfg_path: str | Path) -> Path:
    """Resolve the net-file referenced by a SUMO config."""
    cfg = Path(cfg_path)
    root = ET.parse(cfg).getroot()
    net_file = root.find('./input/net-file')
    if net_file is None or 'value' not in net_file.attrib:
        raise ValueError(f'{cfg} does not define input/net-file.')
    net_path = Path(net_file.attrib['value'])
    if not net_path.is_absolute():
        net_path = cfg.parent / net_path
    return net_path


def lane_inputs_from_net(
    net_path: str | Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, LaneGroupGeometry]]:
    """Extract lane IDs and static geometry keyed by SUMO edge id."""
    net = sumolib.net.readNet(str(net_path), withConnections=True)
    lane_ids_by_edge: dict[str, tuple[str, ...]] = {}
    lane_geometries: dict[str, LaneGroupGeometry] = {}
    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(':'):
            continue
        lanes = tuple(lane.getID() for lane in edge.getLanes())
        lane_ids_by_edge[edge_id] = lanes
        lane_geometries[edge_id] = LaneGroupGeometry(
            length_m=float(edge.getLength()),
            num_lanes=len(lanes),
            speed_limit_mps=float(edge.getSpeed()),
        )
    return lane_ids_by_edge, lane_geometries


def graph_max_pressure_scores_from_features(
    graph: MovementGraph,
    feature_frame: MovementFeatureFrame,
) -> tuple[float, ...]:
    """Compute graph-level max-pressure scores from visible LaneGroup features."""
    halting_by_lane_group = {
        row.lane_group_id: row.dynamic.halting_count_detector for row in feature_frame.lane_group_rows
    }
    return tuple(
        float(
            halting_by_lane_group[movement.input_lane_group_id] - halting_by_lane_group[movement.output_lane_group_id]
        )
        for movement in graph.movements
    )


def collect_samples(
    cfg_path: str | Path,
    output_path: str | Path,
    steps: int,
    decision_interval: int,
    seed: int,
    gui: bool = False,
    demand_scale: float = 1.0,
    initial_occupancy: float = 0.06,
) -> int:
    """Run max-pressure control and write one sample per decision time."""
    net_path = resolve_sumocfg_net_path(cfg_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    demand_route_files = route_files_for_demand_scale(
        cfg_path=cfg_path,
        demand_scale=demand_scale,
    )
    initial_population = generate_initial_traffic_population(
        cfg_path=cfg_path,
        net_path=net_path,
        target_occupancy=initial_occupancy,
        seed=seed,
    )
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=gui,
        seed=seed,
        additional_sumo_args=route_file_sumo_args(
            (
                *demand_route_files.route_files,
                initial_population.route_file,
            )
        ),
    )
    samples = []
    vehicle_counts: list[int] = []
    try:
        runtime.start()
        runtime.step()
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
        flow_tracker = LaneGroupFlowTracker(
            graph=graph,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            decision_interval_s=decision_interval,
        )
        control_state = MovementControlState()
        for step in range(steps):
            if step % decision_interval == 0:
                vehicles = vehicle_snapshot_collector.capture()
                vehicle_counts.append(len(vehicles))
                feature_frame = build_feature_frame(
                    graph=graph,
                    lane_ids_by_edge=lane_ids_by_edge,
                    lane_geometries=lane_geometries,
                    control_state=control_state,
                    vehicles=vehicles,
                    lane_flow_rates=flow_tracker.observe(vehicles),
                )
                sample = build_dataset_sample(
                    graph=graph,
                    feature_frame=feature_frame,
                    programs=runtime.programs,
                    teacher_controlled_scores={tls_id: {} for tls_id in runtime.programs},
                    teacher_graph_scores=graph_max_pressure_scores_from_features(
                        graph,
                        feature_frame,
                    ),
                    metadata={
                        'cfg_path': str(cfg_path),
                        'network_path': str(net_path),
                        'seed': seed,
                        'simulation_time_s': step,
                        'vehicle_count': len(vehicles),
                        'initial_occupancy': initial_occupancy,
                        'teacher': 'max-pressure',
                    },
                )
                samples.append(sample)
                desired_targets = {
                    traffic_light_id: str(runtime.programs[traffic_light_id].selectable_phases[local_phase_index].state)
                    for traffic_light_id, local_phase_index in sample.teacher_selected_phase_by_tls.items()
                }
                accepted_targets = runtime.request_targets(desired_targets)
                control_state = movement_control_state_from_targets(
                    graph=graph,
                    programs=runtime.programs,
                    target_states=accepted_targets,
                )
            runtime.step()
            if not runtime.is_running():
                break
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()

    save_jsonl_samples(output_path, samples)
    if vehicle_counts:
        print(
            f'  collection seed={seed} vehicles '
            f'min={min(vehicle_counts)} mean={sum(vehicle_counts) / len(vehicle_counts):.1f} '
            f'max={max(vehicle_counts)}'
        )
    return len(samples)


def verify_max_pressure_determinism(
    cfg_path: str | Path,
    decision_samples: int,
    decision_interval: int,
    seed: int,
    demand_scale: float,
    initial_occupancy: float,
) -> None:
    """Verify exact same-seed max-pressure vehicle trajectories."""
    first = _max_pressure_trajectory(
        cfg_path=cfg_path,
        decision_samples=decision_samples,
        decision_interval=decision_interval,
        seed=seed,
        demand_scale=demand_scale,
        initial_occupancy=initial_occupancy,
    )
    second = _max_pressure_trajectory(
        cfg_path=cfg_path,
        decision_samples=decision_samples,
        decision_interval=decision_interval,
        seed=seed,
        demand_scale=demand_scale,
        initial_occupancy=initial_occupancy,
    )
    if first != second:
        mismatch_index = min(len(first), len(second))
        for index, (first_state, second_state) in enumerate(zip(first, second)):
            if first_state != second_state:
                mismatch_index = index
                break
        raise RuntimeError(f'Max-pressure determinism check failed at decision {mismatch_index}.')
    print(
        f'Determinism check passed: seed={seed} decisions={len(first)} '
        f'final_vehicles={len(first[-1].vehicles) if first else 0}'
    )


def _max_pressure_trajectory(
    cfg_path: str | Path,
    decision_samples: int,
    decision_interval: int,
    seed: int,
    demand_scale: float,
    initial_occupancy: float,
) -> tuple[DecisionTrajectoryState, ...]:
    net_path = resolve_sumocfg_net_path(cfg_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    demand_route_files = route_files_for_demand_scale(cfg_path=cfg_path, demand_scale=demand_scale)
    initial_population = generate_initial_traffic_population(
        cfg_path=cfg_path,
        net_path=net_path,
        target_occupancy=initial_occupancy,
        seed=seed,
    )
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=False,
        seed=seed,
        additional_sumo_args=route_file_sumo_args(
            (
                *demand_route_files.route_files,
                initial_population.route_file,
            )
        ),
    )
    trajectory: list[DecisionTrajectoryState] = []
    try:
        runtime.start()
        runtime.step()
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
        flow_tracker = LaneGroupFlowTracker(
            graph=graph,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            decision_interval_s=decision_interval,
        )
        control_state = MovementControlState()
        for _decision in range(decision_samples):
            if not runtime.is_running():
                break
            vehicles = vehicle_snapshot_collector.capture()
            feature_frame = build_feature_frame(
                graph=graph,
                lane_ids_by_edge=lane_ids_by_edge,
                lane_geometries=lane_geometries,
                control_state=control_state,
                vehicles=vehicles,
                lane_flow_rates=flow_tracker.observe(vehicles),
            )
            sample = build_dataset_sample(
                graph=graph,
                feature_frame=feature_frame,
                programs=runtime.programs,
                teacher_controlled_scores={traffic_light_id: {} for traffic_light_id in runtime.programs},
                teacher_graph_scores=graph_max_pressure_scores_from_features(graph, feature_frame),
                metadata={},
            )
            desired_targets = {
                traffic_light_id: str(runtime.programs[traffic_light_id].selectable_phases[local_phase_index].state)
                for traffic_light_id, local_phase_index in sample.teacher_selected_phase_by_tls.items()
            }
            accepted_targets = runtime.request_targets(desired_targets)
            trajectory.append(
                DecisionTrajectoryState(
                    vehicles=tuple(
                        sorted(
                            (
                                VehicleTrajectoryState(
                                    vehicle_id=vehicle.vehicle_id,
                                    lane_id=str(vehicle.lane_id),
                                    lane_position_m=vehicle.lane_position_m,
                                    speed_mps=vehicle.speed_mps,
                                    next_edge_id=(
                                        str(vehicle.next_edge_id) if vehicle.next_edge_id is not None else None
                                    ),
                                )
                                for vehicle in vehicles
                            ),
                            key=lambda vehicle: vehicle.vehicle_id,
                        )
                    ),
                    target_states=tuple(sorted((str(key), str(value)) for key, value in accepted_targets.items())),
                )
            )
            control_state = movement_control_state_from_targets(
                graph=graph,
                programs=runtime.programs,
                target_states=accepted_targets,
            )
            for _step in range(decision_interval):
                runtime.step()
                if not runtime.is_running():
                    break
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()
    return tuple(trajectory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect movement-score imitation samples.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', required=True, type=Path, help='SUMO .sumocfg path')
    parser.add_argument('--out', required=True, type=Path, help='Output JSONL path')
    parser.add_argument('--steps', type=int, default=1800, help='Maximum simulation seconds')
    parser.add_argument('--decision-interval', type=int, default=15, help='Sample interval')
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=1.0,
        help='Multiplier applied to route-file flow demand at runtime',
    )
    parser.add_argument('--gui', action='store_true', help='Run sumo-gui')
    parser.add_argument(
        '--initial-occupancy',
        type=float,
        default=0.06,
        help='Initial randomized network occupancy',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = collect_samples(
        cfg_path=args.cfg,
        output_path=args.out,
        steps=args.steps,
        decision_interval=args.decision_interval,
        seed=args.seed,
        gui=args.gui,
        demand_scale=args.demand_scale,
        initial_occupancy=args.initial_occupancy,
    )
    print(f'Wrote {count} samples to {args.out}')


if __name__ == '__main__':
    main()
