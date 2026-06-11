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
            approach="unknown",
            direction="s",
            incoming_lane_id=LaneId("lane_a"),
            outgoing_edge_id="edge_x",
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            approach="unknown",
            direction="l",
            incoming_lane_id=LaneId("lane_a"),
            outgoing_edge_id="edge_y",
            request_index=1,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=2,
            approach="unknown",
            direction="r",
            incoming_lane_id=LaneId("lane_b"),
            outgoing_edge_id="edge_z",
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

    assert [str(state) for state in states] == ["GGr", "rrG"]


def test_conflict_phases_allow_same_incoming_lane_to_same_outgoing_edge() -> None:
    links = [
        TrafficLightLinkSpec(
            traffic_light_link_index=0,
            approach="unknown",
            direction="r",
            incoming_lane_id=LaneId("lane_a"),
            outgoing_edge_id="edge_x",
            request_index=0,
        ),
        TrafficLightLinkSpec(
            traffic_light_link_index=1,
            approach="unknown",
            direction="r",
            incoming_lane_id=LaneId("lane_a"),
            outgoing_edge_id="edge_x",
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

    assert [str(state) for state in states] == ["GG"]
