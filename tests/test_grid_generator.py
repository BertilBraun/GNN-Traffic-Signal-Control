from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_grid_network import (
    build_connection_specs,
    build_edge_specs,
    build_node_specs,
    build_route_flows,
    build_safe_phase_states,
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


def test_default_route_flows_are_nonempty() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    flows = build_route_flows(edges)

    assert flows
    assert all(flow.from_edge != flow.to_edge for flow in flows)


def test_default_route_flows_include_bottom_and_right_boundary_sources() -> None:
    nodes = build_node_specs(rows=3, cols=3, spacing=200.0)
    edges = build_edge_specs(nodes)

    flows = build_route_flows(edges)
    source_edges = {flow.from_edge for flow in flows}
    destination_edges = {flow.to_edge for flow in flows}

    assert any(edge.startswith("N2_2_to_") for edge in source_edges)
    assert any(edge.endswith("_to_N2_2") for edge in destination_edges)


def test_safe_phase_states_separate_left_turns_from_through_movements() -> None:
    links = [
        (0, "vertical", "r"),
        (1, "vertical", "s"),
        (2, "vertical", "l"),
        (3, "horizontal", "r"),
        (4, "horizontal", "s"),
        (5, "horizontal", "l"),
    ]

    states = build_safe_phase_states(links, n_links=6)

    assert states == [
        "GGrrrr",
        "rrGrrr",
        "rrrGGr",
        "rrrrrG",
    ]
