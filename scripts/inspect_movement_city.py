"""Inspect movement graph extraction for a SUMO city configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import sumolib
import traci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path  # noqa: E402
from src.movement.extraction import extract_traffic_light_program  # noqa: E402
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_schema import MovementGraph  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402


@dataclass(frozen=True)
class SkippedTrafficLight:
    traffic_light_id: str
    reason: str


@dataclass(frozen=True)
class InspectionReport:
    traffic_light_count: int
    selectable_traffic_light_count: int
    lane_group_count: int
    movement_count: int
    phase_counts_by_traffic_light: tuple[tuple[str, int], ...]
    skipped_traffic_lights: tuple[SkippedTrafficLight, ...]
    suspicious_lane_groups: tuple[str, ...]
    suspicious_movements: tuple[str, ...]


def inspect_city_config(
    cfg_path: Path,
    seed: int,
    time_to_teleport: int,
) -> InspectionReport:
    """Load a SUMO config and inspect movement runtime graph extraction."""
    net_path = resolve_sumocfg_net_path(cfg_path)
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=False,
        seed=seed,
        time_to_teleport=time_to_teleport,
    )
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        traffic_light_ids = tuple(str(traffic_light_id) for traffic_light_id in traci.trafficlight.getIDList())
        skipped = tuple(
            SkippedTrafficLight(
                traffic_light_id=traffic_light_id,
                reason=_skip_reason(traffic_light_id),
            )
            for traffic_light_id in traffic_light_ids
            if traffic_light_id not in runtime.programs
        )
        network = sumolib.net.readNet(str(net_path), withConnections=True)
        return InspectionReport(
            traffic_light_count=len(traffic_light_ids),
            selectable_traffic_light_count=len(runtime.programs),
            lane_group_count=len(graph.lane_groups),
            movement_count=len(graph.movements),
            phase_counts_by_traffic_light=tuple(
                (traffic_light_id, len(program.selectable_phases))
                for traffic_light_id, program in sorted(runtime.programs.items())
            ),
            skipped_traffic_lights=skipped,
            suspicious_lane_groups=_suspicious_lane_groups(graph=graph, network=network),
            suspicious_movements=_suspicious_movements(graph),
        )
    finally:
        runtime.close()


def _skip_reason(traffic_light_id: str) -> str:
    logics = traci.trafficlight.getAllProgramLogics(traffic_light_id)
    if not logics:
        return 'no traffic-light program logics'
    active_program_id = traci.trafficlight.getProgram(traffic_light_id)
    logic = next(
        (candidate for candidate in logics if candidate.programID == active_program_id),
        logics[0],
    )
    controlled_links = traci.trafficlight.getControlledLinks(traffic_light_id)
    if not controlled_links:
        return 'no controlled links'
    phase_states = [phase.state for phase in logic.phases]
    if not phase_states:
        return 'active traffic-light program has no phases'
    try:
        program = extract_traffic_light_program(
            tls_id=traffic_light_id,
            phase_states=phase_states,
            controlled_links=controlled_links,
        )
    except ValueError as exc:
        return str(exc)
    if not program.movements:
        return 'no controlled movements after extraction'
    if not program.selectable_phases:
        return 'no selectable green phases in active program'
    return 'unknown extraction skip'


def _suspicious_lane_groups(graph: MovementGraph, network) -> tuple[str, ...]:
    network_edges = {str(edge.getID()): edge for edge in network.getEdges() if not str(edge.getID()).startswith(':')}
    warnings: list[str] = []
    for lane_group in graph.lane_groups:
        missing = tuple(edge_id for edge_id in lane_group.edge_ids if str(edge_id) not in network_edges)
        if missing:
            warnings.append(f'L{int(lane_group.lane_group_id)} missing edges: {missing}')
            continue
        edges = tuple(network_edges[str(edge_id)] for edge_id in lane_group.edge_ids)
        total_length = sum(float(edge.getLength()) for edge in edges)
        if total_length <= 0.0:
            warnings.append(f'L{int(lane_group.lane_group_id)} has non-positive length')
        if any(int(edge.getLaneNumber()) <= 0 for edge in edges):
            warnings.append(f'L{int(lane_group.lane_group_id)} has an edge with no lanes')
        for first, second in zip(edges, edges[1:]):
            if str(first.getToNode().getID()) != str(second.getFromNode().getID()):
                warnings.append(f'L{int(lane_group.lane_group_id)} discontinuity: {first.getID()} -> {second.getID()}')
    return tuple(warnings)


def _suspicious_movements(graph: MovementGraph) -> tuple[str, ...]:
    served_movement_ids = {
        movement_id
        for incidence in graph.phase_incidences.values()
        for row in incidence.rows
        for enabled, movement_id in zip(row, incidence.movement_ids)
        if enabled
    }
    warnings: list[str] = []
    for movement in graph.movements:
        movement_id = int(movement.movement_id)
        if movement.input_lane_group_id == movement.output_lane_group_id:
            warnings.append(f'M{movement_id} enters and exits the same LaneGroup')
        if not movement.controlled_movement_indices:
            warnings.append(f'M{movement_id} has no controlled links')
        if movement.movement_id not in served_movement_ids:
            warnings.append(f'M{movement_id} is not enabled by any selectable phase')
    return tuple(warnings)


def print_report(report: InspectionReport) -> None:
    """Print a compact human-readable movement graph inspection report."""
    print('Movement city inspection')
    print(f'  traffic lights in SUMO: {report.traffic_light_count}')
    print(f'  traffic lights with selectable phases: {report.selectable_traffic_light_count}')
    print(f'  lane groups: {report.lane_group_count}')
    print(f'  movements: {report.movement_count}')
    print('  selectable phases per traffic light:')
    for traffic_light_id, phase_count in report.phase_counts_by_traffic_light:
        print(f'    {traffic_light_id}: {phase_count}')
    print('  unsupported/skipped traffic lights:')
    if report.skipped_traffic_lights:
        for skipped in report.skipped_traffic_lights:
            print(f'    {skipped.traffic_light_id}: {skipped.reason}')
    else:
        print('    none')
    print('  suspicious lane groups:')
    _print_warning_lines(report.suspicious_lane_groups)
    print('  suspicious movements:')
    _print_warning_lines(report.suspicious_movements)


def _print_warning_lines(warnings: tuple[str, ...]) -> None:
    if warnings:
        for warning in warnings:
            print(f'    {warning}')
    else:
        print('    none')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate movement graph extraction for a SUMO city config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', type=Path, required=True, help='SUMO .sumocfg path')
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=-1,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_report(
        inspect_city_config(
            cfg_path=args.cfg,
            seed=args.seed,
            time_to_teleport=args.time_to_teleport,
        )
    )


if __name__ == '__main__':
    main()
