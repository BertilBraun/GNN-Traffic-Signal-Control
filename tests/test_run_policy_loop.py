from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run import compute_movement_scores, select_control_states
from src.movement.extraction import extract_traffic_light_program


class FakeLaneApi:
    queues = {
        "north_0": 7,
        "south_0": 2,
        "east_0": 3,
        "west_0": 8,
    }

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return self.queues[lane_id]


def test_compute_movement_scores_uses_incoming_minus_outgoing_for_max_pressure() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["GGr", "rrG"],
        controlled_links=[
            ("north_0", "south_0", None),
            ("north_0", "east_0", None),
            ("east_0", "west_0", None),
        ],
    )

    scores = compute_movement_scores(program, FakeLaneApi(), method="max-pressure")

    assert scores == {
        0: 5.0,
        1: 4.0,
        2: -5.0,
    }


def test_compute_movement_scores_uses_incoming_queue_for_queue_method() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["GGr", "rrG"],
        controlled_links=[
            ("north_0", "south_0", None),
            ("north_0", "east_0", None),
            ("east_0", "west_0", None),
        ],
    )

    scores = compute_movement_scores(program, FakeLaneApi(), method="queue")

    assert scores == {
        0: 7.0,
        1: 7.0,
        2: 3.0,
    }


def test_select_control_states_picks_highest_scoring_phase_state() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["Gr", "rG"],
        controlled_links=[
            ("north_0", "west_0", None),
            ("east_0", "west_0", None),
        ],
    )

    states = select_control_states({"J0": program}, FakeLaneApi(), method="queue")

    assert states == {"J0": "Gr"}


def test_select_control_states_rejects_unknown_method() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["G"],
        controlled_links=[("north_0", "south_0", None)],
    )

    try:
        select_control_states({"J0": program}, FakeLaneApi(), method="unknown")
    except ValueError as exc:
        assert "Unsupported control method" in str(exc)
    else:
        raise AssertionError("expected unknown method to raise ValueError")
