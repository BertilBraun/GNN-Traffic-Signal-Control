from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.grid_study import (
    GRID_COVERAGE_SCENARIOS,
    MATCHED_GRID_SCENARIOS,
    GridStudyRole,
    balanced_rollout_allocation,
    controllable_node_ids,
)


@pytest.mark.parametrize(
    ('rows', 'cols', 'controller_count'),
    [
        (2, 3, 2),
        (3, 2, 2),
        (2, 5, 6),
        (3, 3, 5),
        (4, 4, 12),
        (5, 3, 11),
        (3, 6, 14),
        (5, 5, 21),
        (6, 6, 32),
        (7, 7, 45),
    ],
)
def test_controllable_node_count_matches_grid_geometry(
    rows: int,
    cols: int,
    controller_count: int,
) -> None:
    assert len(controllable_node_ids(rows=rows, cols=cols)) == controller_count


def test_training_rollouts_are_balanced_by_action_samples() -> None:
    allocations = balanced_rollout_allocation(
        scenarios=MATCHED_GRID_SCENARIOS,
        target_action_samples_per_scenario=21_000,
        decisions_per_rollout=200,
    )

    assert {allocation.scenario_name: allocation.rollout_jobs for allocation in allocations} == {
        'matched_grid_5x2_tall': 18,
        'matched_grid_3x3_square': 21,
        'matched_grid_4x4_square': 9,
        'matched_grid_5x3_tall': 10,
        'matched_grid_5x5_square': 5,
    }
    assert max(allocation.action_sample_count for allocation in allocations) <= 22_000
    assert min(allocation.action_sample_count for allocation in allocations) >= 21_000


def test_transposed_rectangles_are_evaluation_only() -> None:
    roles = {scenario.name: scenario.role for scenario in MATCHED_GRID_SCENARIOS}

    assert roles['matched_grid_5x2_tall'] is GridStudyRole.TRAIN
    assert roles['matched_grid_2x5_wide'] is GridStudyRole.EVALUATION_ONLY
    assert roles['matched_grid_5x3_tall'] is GridStudyRole.TRAIN
    assert roles['matched_grid_3x5_wide'] is GridStudyRole.EVALUATION_ONLY


def test_coverage_scenarios_are_nested_and_preserve_geometry() -> None:
    signal_sets = tuple(scenario.signalized_node_ids for scenario in GRID_COVERAGE_SCENARIOS)

    assert tuple(len(signal_set) for signal_set in signal_sets) == (32, 24, 16, 8)
    assert signal_sets[3] < signal_sets[2] < signal_sets[1] < signal_sets[0]
    assert all(scenario.rows == 6 and scenario.cols == 6 for scenario in GRID_COVERAGE_SCENARIOS)
