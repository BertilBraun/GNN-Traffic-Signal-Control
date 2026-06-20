from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.phase_synthesis import (
    TrafficLightLinkSpec,
    build_conflict_phase_states,
)
from src.movement.schema import LaneId


def test_conflict_phases_do_not_partially_open_shared_incoming_lane() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            incoming_lane_id=LaneId('lane_a'),
            outgoing_edge_id='edge_x',
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            incoming_lane_id=LaneId('lane_a'),
            outgoing_edge_id='edge_y',
            request_index=1,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=2,
            incoming_lane_id=LaneId('lane_b'),
            outgoing_edge_id='edge_z',
            request_index=2,
        ),
    ]

    def are_foes(first: int, second: int) -> bool:
        return {first, second} == {0, 2}

    states = build_conflict_phase_states(
        links=links,
        number_of_links=3,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['GGr', 'rrG']


def test_conflict_phases_allow_same_incoming_lane_to_same_outgoing_edge() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            incoming_lane_id=LaneId('lane_a'),
            outgoing_edge_id='edge_x',
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            incoming_lane_id=LaneId('lane_a'),
            outgoing_edge_id='edge_x',
            request_index=1,
        ),
    ]

    def are_foes(first: int, second: int) -> bool:
        return False

    states = build_conflict_phase_states(
        links=links,
        number_of_links=2,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['GG']


def test_conflict_phases_allow_parallel_lanes_from_same_approach_to_same_outgoing_edge() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            incoming_lane_id=LaneId('north_in_0'),
            outgoing_edge_id='south_out',
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            incoming_lane_id=LaneId('north_in_1'),
            outgoing_edge_id='south_out',
            request_index=1,
        ),
    ]

    def are_foes(first: int, second: int) -> bool:
        return False

    states = build_conflict_phase_states(
        links=links,
        number_of_links=2,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['GG']


def test_conflict_phases_reject_same_outgoing_edge_from_different_approaches() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            incoming_lane_id=LaneId('north_in_0'),
            outgoing_edge_id='south_out',
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            incoming_lane_id=LaneId('east_in_0'),
            outgoing_edge_id='south_out',
            request_index=1,
        ),
    ]

    def are_foes(first: int, second: int) -> bool:
        return False

    states = build_conflict_phase_states(
        links=links,
        number_of_links=2,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['Gr', 'rG']


def test_conflict_phases_do_not_generate_left_plus_right_without_parallel_straight() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            incoming_lane_id=LaneId('north_in_0'),
            outgoing_edge_id='east_out',
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            incoming_lane_id=LaneId('north_in_1'),
            outgoing_edge_id='south_out',
            request_index=1,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=2,
            incoming_lane_id=LaneId('north_in_2'),
            outgoing_edge_id='south_out',
            request_index=2,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=3,
            incoming_lane_id=LaneId('north_in_2'),
            outgoing_edge_id='west_out',
            request_index=3,
        ),
    ]

    def are_foes(first: int, second: int) -> bool:
        return (first == 0 and second in {1, 2}) or (second == 0 and first in {1, 2})

    states = build_conflict_phase_states(
        links=links,
        number_of_links=4,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['rGGG', 'Grrr']
    assert 'GrrG' not in {str(state) for state in states}


def test_conflict_phase_synthesis_groups_shared_lanes_before_search() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=link_index,
            incoming_lane_id=LaneId(f'lane_{link_index // 3}'),
            outgoing_edge_id=f'edge_{link_index}',
            request_index=link_index,
        )
        for link_index in range(24)
    ]

    def are_foes(first: int, second: int) -> bool:
        return False

    states = build_conflict_phase_states(
        links=links,
        number_of_links=24,
        are_foes=are_foes,
    )

    assert [str(state) for state in states] == ['G' * 24]
