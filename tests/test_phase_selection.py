from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.extraction import extract_traffic_light_program
from src.movement.phase_selection import (
    score_phase,
    score_program_phases,
    select_highest_scoring_phase,
)


def test_score_phase_sums_active_movement_scores() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["GGr", "rrG"],
        controlled_links=[
            ("north_0", "south_0", None),
            ("north_1", "east_0", None),
            ("east_0", "west_0", None),
        ],
    )
    phase = program.selectable_phases[0]
    movement_scores = {0: 4.5, 1: 2.0, 2: 99.0}

    assert score_phase(phase, movement_scores) == 6.5


def test_score_program_phases_uses_zero_for_missing_movement_score() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["GGr", "rrG"],
        controlled_links=[
            ("north_0", "south_0", None),
            ("north_1", "east_0", None),
            ("east_0", "west_0", None),
        ],
    )

    scores = score_program_phases(program, {0: 4.0})

    assert scores == {0: 4.0, 1: 0.0}


def test_select_highest_scoring_phase_returns_sumo_phase_index_with_stable_tie_break() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["Grr", "rGr", "rrG"],
        controlled_links=[
            ("north_0", "south_0", None),
            ("east_0", "west_0", None),
            ("south_0", "north_0", None),
        ],
    )

    selected = select_highest_scoring_phase(program, {0: 3.0, 1: 3.0, 2: 1.0})

    assert selected.sumo_phase_index == 0
    assert selected.score == 3.0


def test_select_highest_scoring_phase_can_score_unique_incoming_lanes_once() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["GGr", "rrG"],
        controlled_links=[
            ("north_0", "west_0", None),
            ("north_0", "south_0", None),
            ("east_0", "west_0", None),
        ],
    )

    selected = select_highest_scoring_phase(
        program,
        {0: 5.0, 1: 5.0, 2: 6.0},
        phase_score_aggregation="incoming_lane",
    )

    assert selected.sumo_phase_index == 1
    assert selected.score == 6.0
