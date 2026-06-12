from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.extraction import extract_traffic_light_program
from src.movement.graph import build_movement_graph


def test_build_graph_keeps_opposite_lane_groups_separate() -> None:
    programs = {
        "B": extract_traffic_light_program(
            tls_id="B",
            phase_states=["G"],
            controlled_links=[("A_to_B_0", "B_to_C_0", None)],
        ),
        "A": extract_traffic_light_program(
            tls_id="A",
            phase_states=["G"],
            controlled_links=[("B_to_A_0", "A_to_D_0", None)],
        ),
    }

    graph = build_movement_graph(programs)

    lane_group_edges = {lane_group.edge_id for lane_group in graph.lane_groups}
    assert {"A_to_B", "B_to_A"} <= lane_group_edges
    assert graph.lane_group_id_by_edge["A_to_B"] != graph.lane_group_id_by_edge["B_to_A"]


def test_build_graph_groups_controlled_links_into_graph_movements() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["Gr", "rG"],
        controlled_links=[
            [("north_in_0", "south_out_0", None), ("north_in_1", "south_out_1", None)],
            [("east_in_0", "west_out_0", None)],
        ],
    )

    graph = build_movement_graph({"J0": program})

    assert len(graph.movements) == 2
    first = graph.movements[0]
    assert first.traffic_light_id == "J0"
    assert first.input_lane_group_id == graph.lane_group_id_by_edge["north_in"]
    assert first.output_lane_group_id == graph.lane_group_id_by_edge["south_out"]
    assert first.controlled_movement_indices == (0, 1)


def test_build_graph_edges_and_phase_incidence_align_with_program_order() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["Gr", "rG"],
        controlled_links=[
            [("north_in_0", "south_out_0", None), ("north_in_1", "south_out_1", None)],
            [("east_in_0", "west_out_0", None)],
        ],
    )

    graph = build_movement_graph({"J0": program})

    assert graph.edges.input_lane_to_movement == (
        (graph.lane_group_id_by_edge["north_in"], 0),
        (graph.lane_group_id_by_edge["east_in"], 1),
    )
    assert graph.edges.output_lane_to_movement == (
        (graph.lane_group_id_by_edge["south_out"], 0),
        (graph.lane_group_id_by_edge["west_out"], 1),
    )
    assert graph.edges.movement_to_input_lane == (
        (0, graph.lane_group_id_by_edge["north_in"]),
        (1, graph.lane_group_id_by_edge["east_in"]),
    )
    assert graph.edges.movement_to_output_lane == (
        (0, graph.lane_group_id_by_edge["south_out"]),
        (1, graph.lane_group_id_by_edge["west_out"]),
    )
    assert graph.phase_incidences["J0"].rows == ((1, 0), (0, 1))
    assert graph.phase_incidences["J0"].sumo_phase_indices == (0, 1)


def test_build_graph_is_deterministic_regardless_of_program_mapping_order() -> None:
    program_a = extract_traffic_light_program(
        tls_id="A",
        phase_states=["G"],
        controlled_links=[("B_to_A_0", "A_to_C_0", None)],
    )
    program_b = extract_traffic_light_program(
        tls_id="B",
        phase_states=["G"],
        controlled_links=[("A_to_B_0", "B_to_D_0", None)],
    )

    first = build_movement_graph({"B": program_b, "A": program_a})
    second = build_movement_graph({"A": program_a, "B": program_b})

    assert first == second
