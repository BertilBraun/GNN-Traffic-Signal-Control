from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.extraction import extract_traffic_light_program
from src.movement.graph import (
    _merge_tiny_controlled_edge_corridors,
    _resolve_overlapping_corridors,
    _select_continuation,
    build_movement_graph,
)
from src.movement.runtime import MovementControlRuntime


GRID_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
GRID_NET = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.net.xml'


def test_build_graph_keeps_opposite_lane_groups_separate() -> None:
    programs = {
        'B': extract_traffic_light_program(
            tls_id='B',
            phase_states=['G', 'g'],
            controlled_links=[('A_to_B_0', 'B_to_C_0', None)],
        ),
        'A': extract_traffic_light_program(
            tls_id='A',
            phase_states=['G', 'g'],
            controlled_links=[('B_to_A_0', 'A_to_D_0', None)],
        ),
    }

    graph = build_movement_graph(programs)

    lane_group_edges = {lane_group.edge_id for lane_group in graph.lane_groups}
    assert {'A_to_B', 'B_to_A'} <= lane_group_edges
    assert graph.lane_group_id_by_edge['A_to_B'] != graph.lane_group_id_by_edge['B_to_A']


def test_build_graph_groups_controlled_links_into_graph_movements() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['Gr', 'rG'],
        controlled_links=[
            [('north_in_0', 'south_out_0', None), ('north_in_1', 'south_out_1', None)],
            [('east_in_0', 'west_out_0', None)],
        ],
    )

    graph = build_movement_graph({'J0': program})

    assert len(graph.movements) == 2
    first = graph.movements[0]
    assert first.traffic_light_id == 'J0'
    assert first.input_lane_group_id == graph.lane_group_id_by_edge['north_in']
    assert first.output_lane_group_id == graph.lane_group_id_by_edge['south_out']
    assert first.controlled_movement_indices == (0, 1)


def test_build_graph_edges_and_phase_incidence_align_with_program_order() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['Gr', 'rG'],
        controlled_links=[
            [('north_in_0', 'south_out_0', None), ('north_in_1', 'south_out_1', None)],
            [('east_in_0', 'west_out_0', None)],
        ],
    )

    graph = build_movement_graph({'J0': program})

    assert graph.edges.input_lane_to_movement == (
        (graph.lane_group_id_by_edge['north_in'], 0),
        (graph.lane_group_id_by_edge['east_in'], 1),
    )
    assert graph.edges.output_lane_to_movement == (
        (graph.lane_group_id_by_edge['south_out'], 0),
        (graph.lane_group_id_by_edge['west_out'], 1),
    )
    assert graph.edges.movement_to_input_lane == (
        (0, graph.lane_group_id_by_edge['north_in']),
        (1, graph.lane_group_id_by_edge['east_in']),
    )
    assert graph.edges.movement_to_output_lane == (
        (0, graph.lane_group_id_by_edge['south_out']),
        (1, graph.lane_group_id_by_edge['west_out']),
    )
    assert graph.phase_incidences['J0'].rows == ((1, 0), (0, 1))
    assert graph.phase_incidences['J0'].sumo_phase_indices == (0, 1)


def test_build_graph_is_deterministic_regardless_of_program_mapping_order() -> None:
    program_a = extract_traffic_light_program(
        tls_id='A',
        phase_states=['G', 'g'],
        controlled_links=[('B_to_A_0', 'A_to_C_0', None)],
    )
    program_b = extract_traffic_light_program(
        tls_id='B',
        phase_states=['G', 'g'],
        controlled_links=[('A_to_B_0', 'B_to_D_0', None)],
    )

    first = build_movement_graph({'B': program_b, 'A': program_a})
    second = build_movement_graph({'A': program_a, 'B': program_b})

    assert first == second


def test_grid_graph_uses_unsignalized_lane_connectors_without_corridor_contraction() -> None:
    runtime = MovementControlRuntime(cfg_path=GRID_CFG, gui=False, seed=42)
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=GRID_NET)
    finally:
        runtime.close()

    source_lane_group_id = graph.lane_group_id_by_edge['N0_1_to_N0_0']
    through_lane_group_id = graph.lane_group_id_by_edge['N0_0_to_N1_0']
    assert through_lane_group_id != source_lane_group_id
    assert graph.lane_groups[source_lane_group_id].edge_ids == ('N0_1_to_N0_0',)
    assert any(
        connector.source_lane_group_id == source_lane_group_id
        and connector.target_lane_group_id == through_lane_group_id
        and connector.via_junction_id == 'N0_0'
        and connector.connector_type == 'unsignalized'
        and connector.freeflow_time_s > 0.0
        for connector in graph.lane_lane_connectors
    )
    assert sum(1 for connector in graph.lane_lane_connectors if connector.via_junction_id == 'N0_0') > 1
    assert any(
        movement.traffic_light_id == 'N0_1' and movement.output_lane_group_id == source_lane_group_id
        for movement in graph.movements
    )
    assert any(
        movement.traffic_light_id == 'N1_0' and movement.input_lane_group_id == through_lane_group_id
        for movement in graph.movements
    )
    assert all(
        connector.via_junction_id not in {str(movement.traffic_light_id) for movement in graph.movements}
        for connector in graph.lane_lane_connectors
    )
    assert len(graph.lane_groups) == 48
    assert len(graph.movements) == 60


def test_single_phase_traffic_light_is_pass_through_not_policy_control() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['G'],
        controlled_links=[('north_in_0', 'south_out_0', None)],
    )

    graph = build_movement_graph({'J0': program})

    assert graph.movements == ()
    assert graph.phase_incidences == {}
    assert graph.pass_through_traffic_light_ids == ('J0',)


def test_corridor_branch_requires_one_unique_straight_continuation() -> None:
    turning_edge = object()
    straight_edge = object()

    assert _select_continuation(
        (
            (turning_edge, ('l',)),
            (straight_edge, ('s',)),
        )
    ) == (straight_edge, ('s',))
    assert (
        _select_continuation(
            (
                (turning_edge, ('l',)),
                (object(), ('r',)),
            )
        )
        is None
    )


class EdgeStub:
    def __init__(self, length: float) -> None:
        self.length = length

    def getLength(self) -> float:
        return self.length


def test_overlapping_corridors_fall_back_to_controlled_edges() -> None:
    resolved = _resolve_overlapping_corridors(
        {
            'a_in': ('a_in', 'shared', 'signal_out'),
            'b_in': ('b_in', 'shared', 'signal_out'),
            'clear_in': ('clear_in', 'clear_out'),
        }
    )

    assert resolved == {
        'a_in': ('a_in',),
        'b_in': ('b_in',),
        'clear_in': ('clear_in', 'clear_out'),
    }


def test_tiny_controlled_edges_keep_shared_upstream_corridor() -> None:
    raw = {
        'turn_left_stub': ('shared_upstream', 'turn_left_stub'),
        'turn_right_stub': ('shared_upstream', 'turn_right_stub'),
        'normal_in': ('normal_in', 'signal_out'),
    }
    resolved = {
        'turn_left_stub': ('turn_left_stub',),
        'turn_right_stub': ('turn_right_stub',),
        'normal_in': ('normal_in', 'signal_out'),
    }

    merged = _merge_tiny_controlled_edge_corridors(
        corridor_by_controlled_edge=raw,
        resolved_corridors=resolved,
        edges_by_id={
            'turn_left_stub': EdgeStub(0.2),
            'turn_right_stub': EdgeStub(0.2),
            'normal_in': EdgeStub(80.0),
        },
    )

    assert merged == {
        'turn_left_stub': ('shared_upstream', 'turn_left_stub', 'turn_right_stub'),
        'turn_right_stub': ('shared_upstream', 'turn_left_stub', 'turn_right_stub'),
        'normal_in': ('normal_in', 'signal_out'),
    }
