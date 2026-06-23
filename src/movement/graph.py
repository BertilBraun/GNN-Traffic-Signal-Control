"""Build static LaneGroup/Movement graphs from movement-aware programs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import sumolib

from .graph_schema import (
    GraphMovementId,
    LaneLaneConnectorEdge,
    LaneGroupId,
    LaneGroupNode,
    LaneMovementEdgeMetadata,
    MovementGraph,
    MovementNode,
    PhaseIncidence,
    TypedMovementEdges,
)
from .schema import EdgeId, LaneId, MovementIndex, TrafficLightId, TrafficLightProgram


def build_movement_graph(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    net_path: str | Path | None = None,
) -> MovementGraph:
    """Build a deterministic static graph from extracted traffic-light programs."""
    controllable_programs = _controllable_programs(programs)
    pass_through_traffic_light_ids = tuple(
        sorted(
            TrafficLightId(str(program.traffic_light_id))
            for program in programs.values()
            if len(program.selectable_phases) <= 1
        )
    )
    controlled_edges = _collect_lane_group_edges(controllable_programs)
    network = sumolib.net.readNet(str(net_path), withConnections=True) if net_path is not None else None
    lane_group_edge_ids = (
        _collect_network_lane_group_edges(network=network, controlled_edges=controlled_edges)
        if network is not None
        else controlled_edges
    )
    lane_groups = tuple(
        LaneGroupNode(lane_group_id=LaneGroupId(index), edge_ids=(edge_id,))
        for index, edge_id in enumerate(lane_group_edge_ids)
    )
    lane_group_id_by_edge = {
        edge_id: lane_group.lane_group_id for lane_group in lane_groups for edge_id in lane_group.edge_ids
    }

    movement_keys = _collect_movement_keys(controllable_programs)
    movement_id_by_key = {key: GraphMovementId(index) for index, key in enumerate(movement_keys)}
    controlled_indices_by_key = _collect_controlled_indices_by_key(controllable_programs)
    movements = tuple(
        MovementNode(
            movement_id=movement_id_by_key[key],
            traffic_light_id=key[0],
            input_lane_group_id=lane_group_id_by_edge[key[1]],
            output_lane_group_id=lane_group_id_by_edge[key[2]],
            controlled_movement_indices=controlled_indices_by_key[key],
        )
        for key in movement_keys
    )

    return MovementGraph(
        lane_groups=lane_groups,
        movements=movements,
        edges=_build_edges(movements),
        lane_lane_connectors=(
            _build_lane_lane_connectors(
                network=network,
                lane_group_id_by_edge=lane_group_id_by_edge,
                controllable_traffic_light_ids={
                    str(program.traffic_light_id) for program in controllable_programs.values()
                },
            )
            if network is not None
            else ()
        ),
        lane_movement_metadata=_build_lane_movement_metadata(
            movements=movements,
            lane_groups=lane_groups,
            network=network,
        ),
        phase_incidences=_build_phase_incidences(
            programs=controllable_programs,
            movement_id_by_key=movement_id_by_key,
        ),
        lane_group_id_by_edge=lane_group_id_by_edge,
        movement_id_by_key=movement_id_by_key,
        pass_through_traffic_light_ids=pass_through_traffic_light_ids,
    )


def _controllable_programs(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> dict[str | TrafficLightId, TrafficLightProgram]:
    return {
        traffic_light_id: program
        for traffic_light_id, program in programs.items()
        if len(program.selectable_phases) > 1
    }


def _collect_network_lane_group_edges(
    network: object,
    controlled_edges: tuple[EdgeId, ...],
) -> tuple[EdgeId, ...]:
    edge_ids = {EdgeId(str(edge.getID())) for edge in network.getEdges() if not str(edge.getID()).startswith(':')}
    edge_ids.update(controlled_edges)
    return tuple(sorted(edge_ids, key=str))


def _build_lane_lane_connectors(
    network: object,
    lane_group_id_by_edge: Mapping[EdgeId, LaneGroupId],
    controllable_traffic_light_ids: set[str],
) -> tuple[LaneLaneConnectorEdge, ...]:
    network_edges = {
        EdgeId(str(edge.getID())): edge for edge in network.getEdges() if not str(edge.getID()).startswith(':')
    }
    connectors: dict[tuple[LaneGroupId, LaneGroupId, str], LaneLaneConnectorEdge] = {}
    for source_edge_id, source_edge in sorted(network_edges.items(), key=lambda item: str(item[0])):
        for connection in _outgoing_connections(source_edge):
            target_edge_id = EdgeId(str(connection.getTo().getID()))
            if str(target_edge_id).startswith(':') or target_edge_id not in lane_group_id_by_edge:
                continue
            via_junction_id = str(source_edge.getToNode().getID())
            if via_junction_id in controllable_traffic_light_ids:
                continue
            source_lane_group_id = lane_group_id_by_edge[source_edge_id]
            target_lane_group_id = lane_group_id_by_edge[target_edge_id]
            target_edge = network_edges[target_edge_id]
            key = (source_lane_group_id, target_lane_group_id, via_junction_id)
            connectors[key] = _connector_edge(
                source_lane_group_id=source_lane_group_id,
                target_lane_group_id=target_lane_group_id,
                source_edge_id=source_edge_id,
                target_edge_id=target_edge_id,
                via_junction_id=via_junction_id,
                source_edge=source_edge,
                target_edge=target_edge,
            )
    return tuple(connectors[key] for key in sorted(connectors, key=lambda item: (int(item[0]), int(item[1]), item[2])))


def _outgoing_connections(edge: object) -> tuple[object, ...]:
    return tuple(connection for values in edge.getOutgoing().values() for connection in values)


def _connector_edge(
    source_lane_group_id: LaneGroupId,
    target_lane_group_id: LaneGroupId,
    source_edge_id: EdgeId,
    target_edge_id: EdgeId,
    via_junction_id: str,
    source_edge: object,
    target_edge: object,
) -> LaneLaneConnectorEdge:
    distance_m = 0.5 * float(source_edge.getLength()) + 0.5 * float(target_edge.getLength())
    freeflow_time_s = 0.5 * _freeflow_time_s(source_edge) + 0.5 * _freeflow_time_s(target_edge)
    lane_count = float(min(int(source_edge.getLaneNumber()), int(target_edge.getLaneNumber())))
    return LaneLaneConnectorEdge(
        source_lane_group_id=source_lane_group_id,
        target_lane_group_id=target_lane_group_id,
        source_edge_id=source_edge_id,
        target_edge_id=target_edge_id,
        via_junction_id=via_junction_id,
        distance_m=distance_m,
        freeflow_time_s=freeflow_time_s,
        lane_count=lane_count,
        connector_type='unsignalized',
    )


def _build_lane_movement_metadata(
    movements: tuple[MovementNode, ...],
    lane_groups: tuple[LaneGroupNode, ...],
    network: object | None,
) -> tuple[LaneMovementEdgeMetadata, ...]:
    network_edges = (
        {EdgeId(str(edge.getID())): edge for edge in network.getEdges() if not str(edge.getID()).startswith(':')}
        if network is not None
        else {}
    )
    lane_group_by_id = {lane_group.lane_group_id: lane_group for lane_group in lane_groups}
    metadata: list[LaneMovementEdgeMetadata] = []
    for movement in movements:
        for lane_group_id, connector_type in (
            (movement.input_lane_group_id, 'signalized_input'),
            (movement.output_lane_group_id, 'signalized_output'),
        ):
            edge_id = lane_group_by_id[lane_group_id].edge_id
            edge = network_edges.get(edge_id)
            metadata.append(
                LaneMovementEdgeMetadata(
                    lane_group_id=lane_group_id,
                    movement_id=movement.movement_id,
                    distance_m=0.0,
                    freeflow_time_s=0.0,
                    lane_count=0.0 if edge is None else float(edge.getLaneNumber()),
                    connector_type=connector_type,
                )
            )
    return tuple(metadata)


def _freeflow_time_s(edge: object) -> float:
    speed = float(edge.getSpeed())
    if speed <= 0.0:
        return 0.0
    return float(edge.getLength()) / speed


def _collect_corridors(
    controlled_edges: tuple[EdgeId, ...],
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    net_path: str | Path,
) -> tuple[tuple[EdgeId, ...], ...]:
    network = sumolib.net.readNet(str(net_path), withConnections=True)
    edges_by_id = {
        EdgeId(str(edge.getID())): edge for edge in network.getEdges() if not str(edge.getID()).startswith(':')
    }
    signalized_ids = {str(traffic_light_id) for traffic_light_id in programs}
    corridor_by_controlled_edge: dict[EdgeId, tuple[EdgeId, ...]] = {}
    for edge_id in controlled_edges:
        if edge_id not in edges_by_id:
            corridor_by_controlled_edge[edge_id] = (edge_id,)
            continue
        corridor = _corridor_for_edge(
            edge=edges_by_id[edge_id],
            signalized_ids=signalized_ids,
        )
        corridor_by_controlled_edge[edge_id] = tuple(EdgeId(str(edge.getID())) for edge in corridor)
    resolved_corridors = _resolve_overlapping_corridors(corridor_by_controlled_edge)
    return tuple(
        sorted(
            set(resolved_corridors.values()),
            key=lambda edge_ids: tuple(str(edge_id) for edge_id in edge_ids),
        )
    )


def _resolve_overlapping_corridors(
    corridor_by_controlled_edge: Mapping[EdgeId, tuple[EdgeId, ...]],
) -> dict[EdgeId, tuple[EdgeId, ...]]:
    resolved = dict(corridor_by_controlled_edge)
    while True:
        corridors_by_edge: dict[EdgeId, set[tuple[EdgeId, ...]]] = {}
        for _controlled_edge_id, corridor in resolved.items():
            for edge_id in corridor:
                corridors_by_edge.setdefault(edge_id, set()).add(corridor)
        overlapping_controlled_edges = {
            controlled_edge_id
            for corridors in corridors_by_edge.values()
            if len(corridors) > 1
            for corridor in corridors
            for controlled_edge_id, controlled_corridor in resolved.items()
            if controlled_corridor == corridor
        }
        if not overlapping_controlled_edges:
            return resolved
        next_resolved = {
            controlled_edge_id: (
                (controlled_edge_id,) if controlled_edge_id in overlapping_controlled_edges else corridor
            )
            for controlled_edge_id, corridor in resolved.items()
        }
        if next_resolved == resolved:
            return resolved
        resolved = next_resolved


def _corridor_for_edge(edge: object, signalized_ids: set[str]) -> tuple[object, ...]:
    upstream = _trace_corridor(edge, signalized_ids, downstream=False)
    downstream = _trace_corridor(edge, signalized_ids, downstream=True)
    return tuple((*reversed(upstream), edge, *downstream))


def _trace_corridor(
    start_edge: object,
    signalized_ids: set[str],
    downstream: bool,
) -> tuple[object, ...]:
    path: list[object] = []
    visited = {str(start_edge.getID())}
    current = start_edge
    while True:
        junction = current.getToNode() if downstream else current.getFromNode()
        if str(junction.getID()) in signalized_ids:
            break
        candidates = _viable_continuations(
            current=current,
            signalized_ids=signalized_ids,
            downstream=downstream,
            visited=visited,
        )
        continuation = _select_continuation(candidates)
        if continuation is None:
            break
        next_edge = continuation[0]
        next_edge_id = str(next_edge.getID())
        if next_edge_id in visited:
            break
        path.append(next_edge)
        visited.add(next_edge_id)
        current = next_edge
    return tuple(path)


def _viable_continuations(
    current: object,
    signalized_ids: set[str],
    downstream: bool,
    visited: set[str],
) -> tuple[tuple[object, tuple[str, ...]], ...]:
    connections = (
        tuple(connection for values in current.getOutgoing().values() for connection in values)
        if downstream
        else tuple(connection for values in current.getIncoming().values() for connection in values)
    )
    candidates: dict[str, tuple[object, set[str]]] = {}
    for connection in connections:
        candidate = connection.getTo() if downstream else connection.getFrom()
        candidate_id = str(candidate.getID())
        if candidate_id.startswith(':') or candidate_id in visited:
            continue
        if not _reaches_signal(
            edge=candidate,
            signalized_ids=signalized_ids,
            downstream=downstream,
            visited=frozenset((*visited, candidate_id)),
        ):
            continue
        directions = candidates.setdefault(candidate_id, (candidate, set()))[1]
        directions.add(str(connection.getDirection()))
    return tuple(
        (candidate, tuple(sorted(directions)))
        for candidate, directions in sorted(candidates.values(), key=lambda item: str(item[0].getID()))
    )


def _select_continuation(
    candidates: tuple[tuple[object, tuple[str, ...]], ...],
) -> tuple[object, tuple[str, ...]] | None:
    if len(candidates) == 1:
        return candidates[0]
    straight = tuple(candidate for candidate in candidates if 's' in candidate[1])
    return straight[0] if len(straight) == 1 else None


def _reaches_signal(
    edge: object,
    signalized_ids: set[str],
    downstream: bool,
    visited: frozenset[str],
) -> bool:
    pending = [edge]
    seen = set(visited)
    while pending:
        current = pending.pop()
        junction = current.getToNode() if downstream else current.getFromNode()
        if str(junction.getID()) in signalized_ids:
            return True
        connections = (
            tuple(connection for values in current.getOutgoing().values() for connection in values)
            if downstream
            else tuple(connection for values in current.getIncoming().values() for connection in values)
        )
        for connection in connections:
            candidate = connection.getTo() if downstream else connection.getFrom()
            candidate_id = str(candidate.getID())
            if candidate_id.startswith(':') or candidate_id in seen:
                continue
            seen.add(candidate_id)
            pending.append(candidate)
    return False


def edge_id_from_lane_id(lane_id: LaneId | str) -> EdgeId:
    """Return the SUMO edge id for a lane id such as `edge_with_underscores_0`."""
    text = str(lane_id)
    edge_text, separator, lane_index = text.rpartition('_')
    if separator and lane_index.isdigit() and edge_text:
        return EdgeId(edge_text)
    return EdgeId(text)


def _collect_lane_group_edges(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> tuple[EdgeId, ...]:
    edge_ids: set[EdgeId] = set()
    for program in programs.values():
        for movement in program.movements:
            edge_ids.add(edge_id_from_lane_id(movement.incoming_lane_id))
            edge_ids.add(edge_id_from_lane_id(movement.outgoing_lane_id))
    return tuple(sorted(edge_ids, key=str))


def _collect_movement_keys(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> tuple[tuple[TrafficLightId, EdgeId, EdgeId], ...]:
    keys: list[tuple[TrafficLightId, EdgeId, EdgeId]] = []
    seen: set[tuple[TrafficLightId, EdgeId, EdgeId]] = set()
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
            key = _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def _collect_controlled_indices_by_key(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> dict[tuple[TrafficLightId, EdgeId, EdgeId], tuple[MovementIndex, ...]]:
    indices_by_key: dict[tuple[TrafficLightId, EdgeId, EdgeId], list[MovementIndex]] = {}
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
            key = _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
            indices_by_key.setdefault(key, []).append(movement.movement_index)
    return {key: tuple(indices) for key, indices in indices_by_key.items()}


def _movement_key(
    traffic_light_id: TrafficLightId,
    incoming_lane_id: LaneId,
    outgoing_lane_id: LaneId,
) -> tuple[TrafficLightId, EdgeId, EdgeId]:
    return (
        traffic_light_id,
        edge_id_from_lane_id(incoming_lane_id),
        edge_id_from_lane_id(outgoing_lane_id),
    )


def _build_edges(movements: tuple[MovementNode, ...]) -> TypedMovementEdges:
    return TypedMovementEdges(
        input_lane_to_movement=tuple((movement.input_lane_group_id, movement.movement_id) for movement in movements),
        output_lane_to_movement=tuple((movement.output_lane_group_id, movement.movement_id) for movement in movements),
        movement_to_input_lane=tuple((movement.movement_id, movement.input_lane_group_id) for movement in movements),
        movement_to_output_lane=tuple((movement.movement_id, movement.output_lane_group_id) for movement in movements),
    )


def _build_phase_incidences(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> dict[TrafficLightId, PhaseIncidence]:
    incidences: dict[TrafficLightId, PhaseIncidence] = {}
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        local_movement_ids = _local_movement_ids(program, movement_id_by_key)
        rows = []
        controlled_to_graph_id = _controlled_to_graph_id(program, movement_id_by_key)
        for phase in program.selectable_phases:
            enabled = {controlled_to_graph_id[movement_index] for movement_index in phase.enabled_movement_indices}
            rows.append(tuple(1 if movement_id in enabled else 0 for movement_id in local_movement_ids))
        incidences[program.traffic_light_id] = PhaseIncidence(
            traffic_light_id=program.traffic_light_id,
            sumo_phase_indices=tuple(int(phase.sumo_phase_index) for phase in program.selectable_phases),
            movement_ids=local_movement_ids,
            rows=tuple(rows),
        )
    return incidences


def _local_movement_ids(
    program: TrafficLightProgram,
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> tuple[GraphMovementId, ...]:
    movement_ids: list[GraphMovementId] = []
    seen: set[GraphMovementId] = set()
    for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
        key = _movement_key(
            program.traffic_light_id,
            movement.incoming_lane_id,
            movement.outgoing_lane_id,
        )
        movement_id = movement_id_by_key[key]
        if movement_id in seen:
            continue
        seen.add(movement_id)
        movement_ids.append(movement_id)
    return tuple(movement_ids)


def _controlled_to_graph_id(
    program: TrafficLightProgram,
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> dict[MovementIndex, GraphMovementId]:
    return {
        movement.movement_index: movement_id_by_key[
            _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
        ]
        for movement in program.movements
    }
