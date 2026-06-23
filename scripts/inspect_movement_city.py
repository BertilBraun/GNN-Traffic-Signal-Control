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
    lane_lane_connector_count: int
    pass_through_traffic_light_count: int
    phase_counts_by_traffic_light: tuple[tuple[str, int], ...]
    skipped_traffic_lights: tuple[SkippedTrafficLight, ...]
    connectivity: 'ConnectivityReport'
    suspicious_lane_groups: tuple[str, ...]
    suspicious_movements: tuple[str, ...]
    signalized_connector_errors: tuple[str, ...]


@dataclass(frozen=True)
class GraphComponent:
    lane_group_ids: tuple[int, ...]
    movement_ids: tuple[int, ...]
    traffic_light_ids: tuple[str, ...]
    lane_group_edges: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ConnectivityReport:
    component_count: int
    largest_component_lane_groups: int
    largest_component_movements: int
    components: tuple[GraphComponent, ...]
    input_only_lane_groups: tuple[int, ...]
    output_only_lane_groups: tuple[int, ...]
    unused_lane_groups: tuple[int, ...]


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
            lane_lane_connector_count=len(graph.lane_lane_connectors),
            pass_through_traffic_light_count=len(graph.pass_through_traffic_light_ids),
            phase_counts_by_traffic_light=tuple(
                (traffic_light_id, len(program.selectable_phases))
                for traffic_light_id, program in sorted(runtime.programs.items())
            ),
            skipped_traffic_lights=skipped,
            connectivity=_connectivity_report(graph),
            suspicious_lane_groups=_suspicious_lane_groups(graph=graph, network=network),
            suspicious_movements=_suspicious_movements(graph),
            signalized_connector_errors=_signalized_connector_errors(graph),
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


def _connectivity_report(graph: MovementGraph) -> ConnectivityReport:
    lane_group_ids = {int(lane_group.lane_group_id) for lane_group in graph.lane_groups}
    movement_ids = {int(movement.movement_id) for movement in graph.movements}
    neighbors: dict[str, set[str]] = {
        **{f'L{lane_group_id}': set() for lane_group_id in lane_group_ids},
        **{f'M{movement_id}': set() for movement_id in movement_ids},
    }
    input_lane_group_ids = {int(movement.input_lane_group_id) for movement in graph.movements}
    output_lane_group_ids = {int(movement.output_lane_group_id) for movement in graph.movements}
    connector_source_lane_group_ids = {int(connector.source_lane_group_id) for connector in graph.lane_lane_connectors}
    connector_target_lane_group_ids = {int(connector.target_lane_group_id) for connector in graph.lane_lane_connectors}
    movement_by_id = {int(movement.movement_id): movement for movement in graph.movements}
    lane_group_by_id = {int(lane_group.lane_group_id): lane_group for lane_group in graph.lane_groups}
    for movement in graph.movements:
        movement_node = f'M{int(movement.movement_id)}'
        for lane_group_id in (int(movement.input_lane_group_id), int(movement.output_lane_group_id)):
            lane_node = f'L{lane_group_id}'
            neighbors[movement_node].add(lane_node)
            neighbors[lane_node].add(movement_node)
    for connector in graph.lane_lane_connectors:
        source_node = f'L{int(connector.source_lane_group_id)}'
        target_node = f'L{int(connector.target_lane_group_id)}'
        neighbors[source_node].add(target_node)
        neighbors[target_node].add(source_node)

    components: list[GraphComponent] = []
    seen: set[str] = set()
    for node in sorted(neighbors, key=_component_node_sort_key):
        if node in seen:
            continue
        pending = [node]
        component_nodes: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            component_nodes.add(current)
            pending.extend(sorted(neighbors[current] - seen, key=_component_node_sort_key))
        component_lane_group_ids = tuple(
            sorted(int(component_node[1:]) for component_node in component_nodes if component_node.startswith('L'))
        )
        component_movement_ids = tuple(
            sorted(int(component_node[1:]) for component_node in component_nodes if component_node.startswith('M'))
        )
        traffic_light_ids = tuple(
            sorted(
                {str(movement_by_id[movement_id].traffic_light_id) for movement_id in component_movement_ids},
                key=str,
            )
        )
        lane_group_edges = tuple(
            tuple(str(edge_id) for edge_id in lane_group_by_id[lane_group_id].edge_ids)
            for lane_group_id in component_lane_group_ids
        )
        components.append(
            GraphComponent(
                lane_group_ids=component_lane_group_ids,
                movement_ids=component_movement_ids,
                traffic_light_ids=traffic_light_ids,
                lane_group_edges=lane_group_edges,
            )
        )
    components = sorted(
        components,
        key=lambda component: (-len(component.lane_group_ids) - len(component.movement_ids), component.lane_group_ids),
    )
    largest = components[0] if components else GraphComponent((), (), (), ())
    return ConnectivityReport(
        component_count=len(components),
        largest_component_lane_groups=len(largest.lane_group_ids),
        largest_component_movements=len(largest.movement_ids),
        components=tuple(components),
        input_only_lane_groups=tuple(sorted(input_lane_group_ids - output_lane_group_ids)),
        output_only_lane_groups=tuple(sorted(output_lane_group_ids - input_lane_group_ids)),
        unused_lane_groups=tuple(
            sorted(
                lane_group_ids
                - input_lane_group_ids
                - output_lane_group_ids
                - connector_source_lane_group_ids
                - connector_target_lane_group_ids
            )
        ),
    )


def _signalized_connector_errors(graph: MovementGraph) -> tuple[str, ...]:
    controllable_ids = {str(movement.traffic_light_id) for movement in graph.movements}
    return tuple(
        f'L{int(connector.source_lane_group_id)} -> L{int(connector.target_lane_group_id)} via {connector.via_junction_id}'
        for connector in graph.lane_lane_connectors
        if connector.via_junction_id in controllable_ids
    )


def _component_node_sort_key(node: str) -> tuple[str, int]:
    return (node[0], int(node[1:]))


def print_report(report: InspectionReport) -> None:
    """Print a compact human-readable movement graph inspection report."""
    print('Movement city inspection')
    print(f'  traffic lights in SUMO: {report.traffic_light_count}')
    print(f'  traffic lights with selectable phases: {report.selectable_traffic_light_count}')
    print(f'  pass-through/single-phase traffic lights: {report.pass_through_traffic_light_count}')
    print(f'  lane groups: {report.lane_group_count}')
    print(f'  movements: {report.movement_count}')
    print(f'  lane-lane connector edges: {report.lane_lane_connector_count}')
    print('  selectable phases per traffic light:')
    for traffic_light_id, phase_count in report.phase_counts_by_traffic_light:
        print(f'    {traffic_light_id}: {phase_count}')
    print('  unsupported/skipped traffic lights:')
    if report.skipped_traffic_lights:
        for skipped in report.skipped_traffic_lights:
            print(f'    {skipped.traffic_light_id}: {skipped.reason}')
    else:
        print('    none')
    print('  graph connectivity:')
    print(f'    message graph components: {report.connectivity.component_count}')
    print(
        '    largest component: '
        f'{report.connectivity.largest_component_lane_groups} lane groups, '
        f'{report.connectivity.largest_component_movements} movements'
    )
    print(f'    input-only lane groups: {len(report.connectivity.input_only_lane_groups)}')
    print(f'    output-only lane groups: {len(report.connectivity.output_only_lane_groups)}')
    print(f'    unused lane groups: {len(report.connectivity.unused_lane_groups)}')
    print('    smallest components:')
    for component in sorted(
        report.connectivity.components,
        key=lambda item: (len(item.lane_group_ids) + len(item.movement_ids), item.lane_group_ids),
    )[:10]:
        edge_preview = ', '.join(' -> '.join(edge_ids) for edge_ids in component.lane_group_edges[:3])
        suffix = ' ...' if len(component.lane_group_edges) > 3 else ''
        print(
            f'      L={len(component.lane_group_ids)} M={len(component.movement_ids)} '
            f'TLS={len(component.traffic_light_ids)} edges=[{edge_preview}{suffix}]'
        )
    print('  suspicious lane groups:')
    _print_warning_lines(report.suspicious_lane_groups)
    print('  suspicious movements:')
    _print_warning_lines(report.suspicious_movements)
    print('  signalized connector errors:')
    _print_warning_lines(report.signalized_connector_errors)


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
