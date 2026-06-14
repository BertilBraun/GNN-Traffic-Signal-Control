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
    traffic_light_id: str
    input_lane_group_id: int
    output_lane_group_id: int
    controlled_link_count: int


class MovementGraphVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_name: str
    junctions: tuple[JunctionVisualization, ...]
    roads: tuple[RoadVisualization, ...]
    lane_groups: tuple[LaneGroupVisualization, ...]
    movements: tuple[MovementVisualization, ...]


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
            traffic_light_id=str(movement.traffic_light_id),
            input_lane_group_id=int(movement.input_lane_group_id),
            output_lane_group_id=int(movement.output_lane_group_id),
            controlled_link_count=len(movement.controlled_movement_indices),
        )
        for movement in graph.movements
    )
    return MovementGraphVisualization(
        network_name=net_path.parent.name,
        junctions=junctions,
        roads=roads,
        lane_groups=lane_groups,
        movements=movements,
    )


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
