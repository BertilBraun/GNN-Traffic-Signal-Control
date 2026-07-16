from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_grid_generalization_suite import (
    ValidationScenario,
    ValidationSuite,
    _topology_title,
    run_validation_suite,
    validation_scenarios,
)
from src.movement.grid_study import GRID_COVERAGE_SCENARIOS


def test_validation_suite_includes_all_matched_and_coverage_scenarios() -> None:
    matched = validation_scenarios(ValidationSuite.MATCHED)
    coverage = validation_scenarios(ValidationSuite.COVERAGE)
    all_scenarios = validation_scenarios(ValidationSuite.ALL)

    assert len(matched) == 10
    assert len(coverage) == 4
    assert all_scenarios == matched + coverage


def test_validation_suite_includes_coverage_generalization_variants() -> None:
    scenarios = validation_scenarios(ValidationSuite.COVERAGE_GENERALIZATION_4X4)

    assert len(scenarios) == 22
    assert all(scenario.rows == 4 and scenario.cols == 4 for scenario in scenarios)


def test_coverage_scenarios_use_controller_eligible_denominator() -> None:
    assert tuple(scenario.name for scenario in GRID_COVERAGE_SCENARIOS) == (
        'coverage_grid_6x6_eligible_signals_32_of_32',
        'coverage_grid_6x6_eligible_signals_24_of_32',
        'coverage_grid_6x6_eligible_signals_16_of_32',
        'coverage_grid_6x6_eligible_signals_08_of_32',
    )


def test_topology_title_uses_controller_eligible_denominator() -> None:
    scenario = ValidationScenario(
        name='coverage_grid_6x6_eligible_signals_08_of_32',
        rows=6,
        cols=6,
    )

    assert _topology_title(scenario=scenario, signalized_count=8).endswith('8/32 eligible junctions signalized')


def test_validation_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='workers must be positive'):
        run_validation_suite(
            scenarios=(),
            configuration_root=tmp_path,
            output_directory=tmp_path,
            demand_scales=(0.7,),
            simulation_seed=9101,
            simulation_steps=1,
            skip_simulation=True,
            workers=0,
        )
