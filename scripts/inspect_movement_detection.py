"""Compare movement LaneGroup detector features against raw TraCI lane data."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import traci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import lane_inputs_from_net, resolve_sumocfg_net_path  # noqa: E402
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale  # noqa: E402
from src.movement.features import (  # noqa: E402
    HALTING_SPEED_THRESHOLD_MPS,
    LaneGroupFlowTracker,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
    build_vehicle_feature_index,
    detector_length,
)
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_schema import LaneGroupId, LaneGroupNode, MovementGraph  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402


@dataclass(frozen=True)
class RawLaneGroupDetector:
    vehicle_ids: tuple[str, ...]
    halting_vehicle_ids: tuple[str, ...]


@dataclass(frozen=True)
class LaneGroupComparison:
    lane_group_id: LaneGroupId
    edge_ids: tuple[str, ...]
    feature_vehicle_count: int
    raw_vehicle_count: int
    feature_halting_count: int
    raw_halting_count: int
    raw_vehicle_ids: tuple[str, ...]

    @property
    def has_mismatch(self) -> bool:
        return (
            self.feature_vehicle_count != self.raw_vehicle_count or self.feature_halting_count != self.raw_halting_count
        )


def inspect_detection(
    cfg_path: Path,
    steps: int,
    sample_every: int,
    seed: int,
    demand_scale: float,
    time_to_teleport: int,
    top: int,
) -> None:
    """Run a SUMO config and compare movement features with raw lane state."""
    net_path = resolve_sumocfg_net_path(cfg_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    demand_route_files = route_files_for_demand_scale(
        cfg_path=cfg_path,
        demand_scale=demand_scale,
    )
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=False,
        seed=seed,
        time_to_teleport=time_to_teleport,
        additional_sumo_args=route_file_sumo_args(demand_route_files.route_files),
    )
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
        flow_tracker = LaneGroupFlowTracker(
            graph=graph,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
            decision_interval_s=max(1, sample_every),
        )
        samples: list[tuple[int, tuple[LaneGroupComparison, ...], tuple[str, ...]]] = []
        for step in range(steps + 1):
            if step % sample_every == 0:
                vehicles = vehicle_snapshot_collector.capture()
                vehicle_index = build_vehicle_feature_index(
                    graph=graph,
                    lane_ids_by_edge=lane_ids_by_edge,
                    lane_geometries=lane_geometries,
                    vehicles=vehicles,
                )
                feature_frame = build_feature_frame(
                    graph=graph,
                    lane_ids_by_edge=lane_ids_by_edge,
                    lane_geometries=lane_geometries,
                    control_state=MovementControlState(),
                    vehicles=vehicles,
                    lane_flow_rates=flow_tracker.observe(vehicle_index),
                    vehicle_index=vehicle_index,
                )
                comparisons = _compare_lane_groups(
                    graph=graph,
                    lane_ids_by_edge=lane_ids_by_edge,
                    lane_geometries=lane_geometries,
                    feature_frame=feature_frame,
                )
                samples.append(
                    (
                        step,
                        comparisons,
                        _unmapped_active_vehicles(graph=graph),
                    )
                )
            if step < steps:
                runtime.step()
            if not runtime.is_running():
                break
        _print_report(samples=samples, top=top)
    finally:
        runtime.close()
        demand_route_files.cleanup()


def _compare_lane_groups(
    graph: MovementGraph,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
    lane_geometries: dict[str, object],
    feature_frame,
) -> tuple[LaneGroupComparison, ...]:
    raw_by_lane_group = {
        lane_group.lane_group_id: _raw_lane_group_detector(
            lane_group=lane_group,
            lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries,
        )
        for lane_group in graph.lane_groups
    }
    edge_ids_by_lane_group = {
        lane_group.lane_group_id: tuple(str(edge_id) for edge_id in lane_group.edge_ids)
        for lane_group in graph.lane_groups
    }
    return tuple(
        LaneGroupComparison(
            lane_group_id=row.lane_group_id,
            edge_ids=edge_ids_by_lane_group[row.lane_group_id],
            feature_vehicle_count=round(row.dynamic.vehicle_count_detector),
            raw_vehicle_count=len(raw_by_lane_group[row.lane_group_id].vehicle_ids),
            feature_halting_count=round(row.dynamic.halting_count_detector),
            raw_halting_count=len(raw_by_lane_group[row.lane_group_id].halting_vehicle_ids),
            raw_vehicle_ids=raw_by_lane_group[row.lane_group_id].vehicle_ids,
        )
        for row in feature_frame.lane_group_rows
    )


def _raw_lane_group_detector(
    lane_group: LaneGroupNode,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
    lane_geometries: dict[str, object],
) -> RawLaneGroupDetector:
    edge_offsets: dict[str, float] = {}
    total_length_m = 0.0
    for edge_id in lane_group.edge_ids:
        edge_text = str(edge_id)
        edge_offsets[edge_text] = total_length_m
        total_length_m += float(lane_geometries[edge_text].length_m)
    detector_start_m = max(0.0, total_length_m - detector_length(total_length_m))

    vehicle_ids: list[str] = []
    halting_vehicle_ids: list[str] = []
    for edge_id in lane_group.edge_ids:
        edge_text = str(edge_id)
        edge_offset = edge_offsets[edge_text]
        for lane_id in lane_ids_by_edge.get(edge_text, ()):
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                vehicle_text = str(vehicle_id)
                position_m = edge_offset + float(traci.vehicle.getLanePosition(vehicle_text))
                if detector_start_m <= position_m <= total_length_m:
                    vehicle_ids.append(vehicle_text)
                    if float(traci.vehicle.getSpeed(vehicle_text)) <= HALTING_SPEED_THRESHOLD_MPS:
                        halting_vehicle_ids.append(vehicle_text)
    return RawLaneGroupDetector(
        vehicle_ids=tuple(sorted(vehicle_ids)),
        halting_vehicle_ids=tuple(sorted(halting_vehicle_ids)),
    )


def _unmapped_active_vehicles(graph: MovementGraph) -> tuple[str, ...]:
    unmapped: list[str] = []
    lane_group_edge_ids = {str(edge_id) for edge_id in graph.lane_group_id_by_edge}
    for vehicle_id in traci.vehicle.getIDList():
        lane_id = str(traci.vehicle.getLaneID(vehicle_id))
        edge_id = lane_id.rsplit('_', 1)[0] if lane_id.rsplit('_', 1)[-1].isdigit() else lane_id
        if edge_id.startswith(':'):
            continue
        if edge_id not in lane_group_edge_ids:
            unmapped.append(f'{vehicle_id}@{edge_id}')
    return tuple(sorted(unmapped))


def _print_report(
    samples: list[tuple[int, tuple[LaneGroupComparison, ...], tuple[str, ...]]],
    top: int,
) -> None:
    print('Movement detector inspection')
    print(f'  sampled steps: {len(samples)}')
    mismatch_counts = Counter()
    unmapped_counts = Counter()
    occupied_rows: list[tuple[int, LaneGroupComparison]] = []
    for step, comparisons, unmapped_vehicle_ids in samples:
        mismatches = tuple(comparison for comparison in comparisons if comparison.has_mismatch)
        mismatch_counts[step] = len(mismatches)
        unmapped_counts[step] = len(unmapped_vehicle_ids)
        occupied_rows.extend(
            (step, comparison)
            for comparison in comparisons
            if comparison.feature_vehicle_count > 0 or comparison.raw_vehicle_count > 0
        )

    max_mismatches = max(mismatch_counts.values(), default=0)
    max_unmapped = max(unmapped_counts.values(), default=0)
    print(f'  max lane-group count mismatches in a sample: {max_mismatches}')
    print(f'  max active vehicles on non-GNN edges: {max_unmapped}')
    print('  mismatches by sample:')
    for step, comparisons, _unmapped_vehicle_ids in samples:
        mismatches = tuple(comparison for comparison in comparisons if comparison.has_mismatch)
        if not mismatches:
            continue
        print(f'    t={step}s: {len(mismatches)} mismatches')
        for comparison in mismatches[:top]:
            _print_comparison('      ', comparison)
    if max_mismatches == 0:
        print('    none')

    print('  top occupied lane groups:')
    for step, comparison in sorted(
        occupied_rows,
        key=lambda item: (item[1].raw_vehicle_count, item[1].feature_vehicle_count),
        reverse=True,
    )[:top]:
        print(f'    t={step}s')
        _print_comparison('      ', comparison)
    if not occupied_rows:
        print('    none')

    print('  unmapped active vehicles:')
    for step, _comparisons, unmapped_vehicle_ids in samples:
        if unmapped_vehicle_ids:
            shown = ', '.join(unmapped_vehicle_ids[:top])
            suffix = ' ...' if len(unmapped_vehicle_ids) > top else ''
            print(f'    t={step}s: {len(unmapped_vehicle_ids)} vehicles ({shown}{suffix})')
    if max_unmapped == 0:
        print('    none')


def _print_comparison(prefix: str, comparison: LaneGroupComparison) -> None:
    vehicle_ids = ', '.join(comparison.raw_vehicle_ids[:8])
    suffix = ' ...' if len(comparison.raw_vehicle_ids) > 8 else ''
    print(
        f'{prefix}L{int(comparison.lane_group_id)} '
        f'feature={comparison.feature_vehicle_count}/{comparison.feature_halting_count} '
        f'raw={comparison.raw_vehicle_count}/{comparison.raw_halting_count} '
        f'edges={" -> ".join(comparison.edge_ids)} '
        f'vehicles=[{vehicle_ids}{suffix}]'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify movement LaneGroup detector features against raw TraCI lane state.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', type=Path, required=True, help='SUMO .sumocfg path')
    parser.add_argument('--steps', type=int, default=120, help='Simulation seconds to inspect')
    parser.add_argument('--sample-every', type=int, default=10, help='Seconds between detector checks')
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument('--demand-scale', type=float, default=1.0, help='Runtime demand scale')
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=-1,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    parser.add_argument('--top', type=int, default=10, help='Maximum rows to print in each section')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_detection(
        cfg_path=args.cfg,
        steps=args.steps,
        sample_every=args.sample_every,
        seed=args.seed,
        demand_scale=args.demand_scale,
        time_to_teleport=args.time_to_teleport,
        top=args.top,
    )


if __name__ == '__main__':
    main()
