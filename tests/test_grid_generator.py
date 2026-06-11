from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_grid_network import (
    TLLinkSpec,
    build_connection_specs,
    build_conflict_phase_states,
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
        TLLinkSpec(0, "north", "r"),
        TLLinkSpec(1, "north", "s"),
        TLLinkSpec(2, "north", "l"),
        TLLinkSpec(3, "east", "r"),
        TLLinkSpec(4, "east", "s"),
        TLLinkSpec(5, "east", "l"),
    ]

    states = build_safe_phase_states(links, n_links=6)

    assert states == [
        "GGrrrr",
        "rrGGrr",
        "rrrGGr",
        "GrrrrG",
        "GGGrrr",
        "rrrGGG",
    ]


def test_safe_phase_states_include_richer_conflict_valid_four_way_candidates() -> None:
    links = [
        TLLinkSpec(0, "north", "r"),
        TLLinkSpec(1, "north", "s"),
        TLLinkSpec(2, "north", "l"),
        TLLinkSpec(3, "south", "r"),
        TLLinkSpec(4, "south", "s"),
        TLLinkSpec(5, "south", "l"),
        TLLinkSpec(6, "east", "r"),
        TLLinkSpec(7, "east", "s"),
        TLLinkSpec(8, "east", "l"),
        TLLinkSpec(9, "west", "r"),
        TLLinkSpec(10, "west", "s"),
        TLLinkSpec(11, "west", "l"),
    ]

    states = build_safe_phase_states(links, n_links=12)

    assert states == [
        "GGrGGrrrrrrr",
        "rrGrrGGrrGrr",
        "rrrrrrGGrGGr",
        "GrrGrrrrGrrG",
        "GGGrrrrrrrrr",
        "rrrGGGrrrrrr",
        "rrrrrrGGGrrr",
        "rrrrrrrrrGGG",
    ]


def test_conflict_phase_states_use_sumo_foes_and_same_outgoing_edge() -> None:
    links = [
        TLLinkSpec(0, "north", "r", "west_out", 0),
        TLLinkSpec(1, "north", "s", "south_out", 1),
        TLLinkSpec(2, "north", "l", "east_out", 2),
        TLLinkSpec(3, "east", "r", "north_out", 3),
        TLLinkSpec(4, "east", "s", "west_out", 4),
        TLLinkSpec(5, "east", "l", "south_out", 5),
    ]
    sumo_foes = {
        frozenset({1, 5}),
        frozenset({2, 4}),
    }

    def are_foes(first: int, second: int) -> bool:
        return frozenset({first, second}) in sumo_foes

    states = build_conflict_phase_states(links, n_links=6, are_foes=are_foes)

    assert states == [
        "GGGGrr",
        "GrGGrG",
        "rGrGGr",
        "rrrGGG",
    ]
