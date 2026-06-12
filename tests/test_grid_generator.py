from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_grid_network import (
    build_boundary_stub_specs,
    build_connection_specs,
    build_edge_specs,
    build_node_specs,
    build_route_flows,
)
from src.movement.phase_synthesis import (
    TrafficLightLinkSpec,
    build_conflict_phase_states,
)


def test_3x3_grid_marks_only_actual_intersections_as_traffic_lights() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)

    traffic_lights = {node.node_id for node in nodes if node.node_type == "traffic_light"}

    assert traffic_lights == {"N0_1", "N1_0", "N1_1", "N1_2", "N2_1"}
    assert {node.node_id for node in nodes if node.node_type is None} == {
        "N0_0",
        "N0_2",
        "N2_0",
        "N2_2",
    }


def test_edge_lanes_match_target_junction_complexity() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)
    lanes_by_edge = {edge.edge_id: edge.lanes for edge in edges}

    assert lanes_by_edge["N0_1_to_N1_1"] == 3
    assert lanes_by_edge["N1_1_to_N0_1"] == 2
    assert lanes_by_edge["N1_1_to_N2_1"] == 2
    assert lanes_by_edge["N0_0_to_N0_1"] == 2
    assert lanes_by_edge["N0_1_to_N0_0"] == 2


def test_edges_exclude_outer_perimeter_roads_when_stubs_are_present() -> None:
    nodes = build_node_specs(rows=4, cols=4, spacing=200.0)
    stubs = build_boundary_stub_specs(nodes, rows=4, cols=4, spacing=200.0)
    edges = build_edge_specs(nodes + stubs)
    edge_ids = {edge.edge_id for edge in edges}

    assert "N0_0_to_N0_1" not in edge_ids
    assert "N0_1_to_N0_0" not in edge_ids
    assert "N3_2_to_N3_3" not in edge_ids
    assert "N3_3_to_N3_2" not in edge_ids
    assert "N1_0_to_N2_0" not in edge_ids
    assert "N2_0_to_N1_0" not in edge_ids
    assert "N1_3_to_N2_3" not in edge_ids
    assert "N2_3_to_N1_3" not in edge_ids
    assert "S_top_1_to_N0_1" in edge_ids
    assert "N0_1_to_N1_1" in edge_ids


def test_center_four_way_has_dedicated_left_straight_right_connections() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    connections = [
        connection
        for connection in build_connection_specs(nodes, edges)
        if connection.node_id == "N1_1" and connection.from_edge == "N0_1_to_N1_1"
    ]

    assert [(c.from_lane, c.to_edge) for c in connections] == [
        (0, "N1_1_to_N1_0"),
        (1, "N1_1_to_N2_1"),
        (2, "N1_1_to_N1_2"),
    ]


def test_corner_bend_connects_both_lanes_to_single_outgoing_road() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    connections = [
        connection
        for connection in build_connection_specs(nodes, edges)
        if connection.node_id == "N0_0" and connection.from_edge == "N0_1_to_N0_0"
    ]

    assert [(c.from_lane, c.to_lane, c.to_edge) for c in connections] == [
        (0, 0, "N0_0_to_N1_0"),
        (1, 1, "N0_0_to_N1_0"),
    ]


def test_t_junction_connects_extra_center_bound_lane() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    to_center = [
        connection
        for connection in build_connection_specs(nodes, edges)
        if connection.node_id == "N0_1"
        and connection.from_edge == "N0_0_to_N0_1"
        and connection.to_edge == "N0_1_to_N1_1"
    ]

    assert {connection.to_lane for connection in to_center} <= {0, 1, 2}
    assert 2 in {connection.to_lane for connection in to_center}


def test_connections_only_reference_existing_source_lanes_with_stubs() -> None:
    nodes = build_node_specs(rows=4, cols=4, spacing=200.0)
    stubs = build_boundary_stub_specs(nodes, rows=4, cols=4, spacing=200.0)
    edges = build_edge_specs(nodes + stubs)
    lanes_by_edge = {edge.edge_id: edge.lanes for edge in edges}

    connections = build_connection_specs(nodes + stubs, edges)

    assert all(
        connection.from_lane < lanes_by_edge[connection.from_edge]
        for connection in connections
    )


def test_default_route_flows_are_nonempty() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    flows = build_route_flows(edges)

    assert flows
    assert all(flow.from_edge != flow.to_edge for flow in flows)


def test_boundary_stubs_add_external_spawn_nodes_on_every_boundary_approach() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    stubs = build_boundary_stub_specs(nodes, rows=3, cols=3, spacing=200.0)

    assert len(stubs) == 12
    assert {stub.node_type for stub in stubs} == {None}
    assert {stub.node_id for stub in stubs} >= {
        "S_top_0",
        "S_bottom_2",
        "S_left_1",
        "S_right_1",
    }


def test_default_route_flows_use_external_stubs_on_all_boundary_sides() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    stubs = build_boundary_stub_specs(nodes, rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes + stubs)

    flows = build_route_flows(edges)
    source_nodes = {_edge_nodes(flow.from_edge)[0] for flow in flows}
    destination_nodes = {_edge_nodes(flow.to_edge)[1] for flow in flows}

    assert len(flows) == 12
    assert source_nodes == {stub.node_id for stub in stubs}
    assert destination_nodes == {stub.node_id for stub in stubs}
    assert {node.split("_", 2)[1] for node in source_nodes} == {
        "top",
        "bottom",
        "left",
        "right",
    }


def test_conflict_phase_states_use_sumo_foes_and_same_outgoing_edge() -> None:
    links = [
        TrafficLightLinkSpec(0, outgoing_edge_id="west_out", request_index=0),
        TrafficLightLinkSpec(1, outgoing_edge_id="south_out", request_index=1),
        TrafficLightLinkSpec(2, outgoing_edge_id="east_out", request_index=2),
        TrafficLightLinkSpec(3, outgoing_edge_id="north_out", request_index=3),
        TrafficLightLinkSpec(4, outgoing_edge_id="west_out", request_index=4),
        TrafficLightLinkSpec(5, outgoing_edge_id="south_out", request_index=5),
    ]
    sumo_foes = {
        frozenset({1, 5}),
        frozenset({2, 4}),
    }

    def are_foes(first: int, second: int) -> bool:
        return frozenset({first, second}) in sumo_foes

    states = [
        str(state)
        for state in build_conflict_phase_states(
            links,
            number_of_links=6,
            are_foes=are_foes,
        )
    ]

    assert states == [
        "GGGGrr",
        "GrGGrG",
        "rGrGGr",
        "rrrGGG",
    ]


def _edge_nodes(edge_id: str) -> tuple[str, str]:
    from_node, to_node = edge_id.split("_to_", 1)
    return from_node, to_node
