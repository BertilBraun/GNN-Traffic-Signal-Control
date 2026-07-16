from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_grid_generalization_suite import ValidationSuite, validation_scenarios
from src.movement.grid_study import GRID_COVERAGE_SCENARIOS


def test_validation_suite_includes_all_matched_and_coverage_scenarios() -> None:
    matched = validation_scenarios(ValidationSuite.MATCHED)
    coverage = validation_scenarios(ValidationSuite.COVERAGE)
    all_scenarios = validation_scenarios(ValidationSuite.ALL)

    assert len(matched) == 10
    assert len(coverage) == 4
    assert all_scenarios == matched + coverage


def test_coverage_scenarios_use_controller_eligible_denominator() -> None:
    assert tuple(scenario.name for scenario in GRID_COVERAGE_SCENARIOS) == (
        'coverage_grid_6x6_eligible_signals_32_of_32',
        'coverage_grid_6x6_eligible_signals_24_of_32',
        'coverage_grid_6x6_eligible_signals_16_of_32',
        'coverage_grid_6x6_eligible_signals_08_of_32',
    )
