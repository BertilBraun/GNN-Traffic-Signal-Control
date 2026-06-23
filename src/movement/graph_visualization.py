"""Export the movement GNN and supporting SUMO topology for visualization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict
import sumolib

from src.movement.graph_schema import MovementGraph
from src.movement.schema import TrafficLightProgram


class JunctionVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    junction_id: str
    x: float
    y: float
    junction_type: str
    is_signalized: bool
    selectable_phase_count: int


class RoadVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_junction_id: str
    to_junction_id: str
    lane_count: int
    is_lane_group: bool


class LaneGroupVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    lane_group_id: int
    component_id: int
    edge_ids: tuple[str, ...]
    junction_ids: tuple[str, ...]
    from_junction_id: str
    to_junction_id: str
    length_m: float
    effective_lane_count: float
    effective_speed_limit_mps: float


class MovementVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    movement_id: int
    component_id: int
    traffic_light_id: str
    input_lane_group_id: int
    output_lane_group_id: int
    controlled_link_count: int


class LaneConnectorVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_lane_group_id: int
    target_lane_group_id: int
    via_junction_id: str
    distance_m: float
    freeflow_time_s: float
    lane_count: float
    connector_type: str


class MovementGraphVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_name: str
    junctions: tuple[JunctionVisualization, ...]
    roads: tuple[RoadVisualization, ...]
    lane_groups: tuple[LaneGroupVisualization, ...]
    movements: tuple[MovementVisualization, ...]
    lane_connectors: tuple[LaneConnectorVisualization, ...]


def build_graph_visualization(
    net_path: Path,
    graph: MovementGraph,
    programs: Mapping[str, TrafficLightProgram],
) -> MovementGraphVisualization:
    """Combine the exact learning graph with its surrounding SUMO topology."""
    network = sumolib.net.readNet(str(net_path), withConnections=True)
    signalized_ids = {str(traffic_light_id) for traffic_light_id in programs}
    junctions = tuple(
        JunctionVisualization(
            junction_id=str(node.getID()),
            x=float(node.getCoord()[0]),
            y=float(node.getCoord()[1]),
            junction_type=str(node.getType()),
            is_signalized=str(node.getID()) in signalized_ids,
            selectable_phase_count=(
                len(programs[str(node.getID())].selectable_phases) if str(node.getID()) in programs else 0
            ),
        )
        for node in sorted(network.getNodes(), key=lambda item: str(item.getID()))
        if not str(node.getID()).startswith(':')
    )
    lane_group_edge_ids = {str(edge_id) for lane_group in graph.lane_groups for edge_id in lane_group.edge_ids}
    network_edges = {str(edge.getID()): edge for edge in network.getEdges() if not str(edge.getID()).startswith(':')}
    component_by_lane_group, component_by_movement = _component_ids(graph)
    roads = tuple(
        RoadVisualization(
            edge_id=edge_id,
            from_junction_id=str(edge.getFromNode().getID()),
            to_junction_id=str(edge.getToNode().getID()),
            lane_count=int(edge.getLaneNumber()),
            is_lane_group=edge_id in lane_group_edge_ids,
        )
        for edge_id, edge in sorted(network_edges.items())
    )
    lane_groups = tuple(
        LaneGroupVisualization(
            lane_group_id=int(lane_group.lane_group_id),
            component_id=component_by_lane_group[int(lane_group.lane_group_id)],
            edge_ids=tuple(str(edge_id) for edge_id in lane_group.edge_ids),
            junction_ids=(
                str(network_edges[str(lane_group.edge_ids[0])].getFromNode().getID()),
                *(str(network_edges[str(edge_id)].getToNode().getID()) for edge_id in lane_group.edge_ids),
            ),
            from_junction_id=str(network_edges[str(lane_group.edge_ids[0])].getFromNode().getID()),
            to_junction_id=str(network_edges[str(lane_group.edge_ids[-1])].getToNode().getID()),
            length_m=sum(float(network_edges[str(edge_id)].getLength()) for edge_id in lane_group.edge_ids),
            effective_lane_count=_effective_lane_count(lane_group.edge_ids, network_edges),
            effective_speed_limit_mps=_effective_speed_limit(lane_group.edge_ids, network_edges),
        )
        for lane_group in graph.lane_groups
    )
    movements = tuple(
        MovementVisualization(
            movement_id=int(movement.movement_id),
            component_id=component_by_movement[int(movement.movement_id)],
            traffic_light_id=str(movement.traffic_light_id),
            input_lane_group_id=int(movement.input_lane_group_id),
            output_lane_group_id=int(movement.output_lane_group_id),
            controlled_link_count=len(movement.controlled_movement_indices),
        )
        for movement in graph.movements
    )
    lane_connectors = tuple(
        LaneConnectorVisualization(
            source_lane_group_id=int(connector.source_lane_group_id),
            target_lane_group_id=int(connector.target_lane_group_id),
            via_junction_id=connector.via_junction_id,
            distance_m=connector.distance_m,
            freeflow_time_s=connector.freeflow_time_s,
            lane_count=connector.lane_count,
            connector_type=connector.connector_type,
        )
        for connector in graph.lane_lane_connectors
    )
    return MovementGraphVisualization(
        network_name=net_path.parent.name,
        junctions=junctions,
        roads=roads,
        lane_groups=lane_groups,
        movements=movements,
        lane_connectors=lane_connectors,
    )


def _component_ids(graph: MovementGraph) -> tuple[dict[int, int], dict[int, int]]:
    neighbors: dict[str, set[str]] = {
        **{f'L{int(lane_group.lane_group_id)}': set() for lane_group in graph.lane_groups},
        **{f'M{int(movement.movement_id)}': set() for movement in graph.movements},
    }
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

    components: list[tuple[set[int], set[int]]] = []
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
        components.append(
            (
                {int(component_node[1:]) for component_node in component_nodes if component_node.startswith('L')},
                {int(component_node[1:]) for component_node in component_nodes if component_node.startswith('M')},
            )
        )
    components = sorted(components, key=lambda item: (-(len(item[0]) + len(item[1])), min(item[0], default=-1)))
    component_by_lane_group: dict[int, int] = {}
    component_by_movement: dict[int, int] = {}
    for component_id, (lane_group_ids, movement_ids) in enumerate(components):
        component_by_lane_group.update({lane_group_id: component_id for lane_group_id in lane_group_ids})
        component_by_movement.update({movement_id: component_id for movement_id in movement_ids})
    return component_by_lane_group, component_by_movement


def _component_node_sort_key(node: str) -> tuple[str, int]:
    return (node[0], int(node[1:]))


def _effective_lane_count(edge_ids: tuple[object, ...], network_edges: Mapping[str, object]) -> float:
    total_length_m = sum(float(network_edges[str(edge_id)].getLength()) for edge_id in edge_ids)
    lane_length_m = sum(
        float(network_edges[str(edge_id)].getLength()) * int(network_edges[str(edge_id)].getLaneNumber())
        for edge_id in edge_ids
    )
    return lane_length_m / total_length_m if total_length_m > 0.0 else 0.0


def _effective_speed_limit(edge_ids: tuple[object, ...], network_edges: Mapping[str, object]) -> float:
    total_length_m = sum(float(network_edges[str(edge_id)].getLength()) for edge_id in edge_ids)
    freeflow_time_s = sum(
        float(network_edges[str(edge_id)].getLength()) / float(network_edges[str(edge_id)].getSpeed())
        for edge_id in edge_ids
        if float(network_edges[str(edge_id)].getSpeed()) > 0.0
    )
    return total_length_m / freeflow_time_s if freeflow_time_s > 0.0 else 0.0
