from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_grid_training_matrix_configs import (
    MATRIX_TRAINING_DESIGNS,
    MatrixTrainingDesign,
    build_matrix_configuration,
)
from src.movement.experiment_config import load_experiment_configuration


@pytest.mark.parametrize('design', MATRIX_TRAINING_DESIGNS)
def test_generated_matrix_configuration_loads(design: MatrixTrainingDesign) -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / design.output_filename,
        project_root=ROOT,
    )

    assert configuration.name == design.experiment_name
    assert configuration.proximal_policy_optimization.rollouts_per_update == sum(
        allocation.rollout_jobs_per_iteration for allocation in design.allocations
    )


def test_matrix_designs_match_total_action_sample_budget() -> None:
    base_configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_shape_generalization_mixed_2hop_gate_30.yaml',
        project_root=ROOT,
    )
    action_sample_totals = []
    for design in MATRIX_TRAINING_DESIGNS:
        configuration = build_matrix_configuration(
            base_configuration=base_configuration,
            design=design,
        )
        action_sample_totals.append(
            sum(
                city.rollout_jobs_per_iteration
                * _controller_count(city.name)
                * configuration.proximal_policy_optimization.steps_per_rollout
                for city in configuration.train_cities
            )
        )

    assert min(action_sample_totals) >= 108_000
    assert max(action_sample_totals) <= 110_000


def test_matrix_design_keeps_six_by_six_for_selection() -> None:
    base_configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_shape_generalization_mixed_2hop_gate_30.yaml',
        project_root=ROOT,
    )

    configuration = build_matrix_configuration(
        base_configuration=base_configuration,
        design=MATRIX_TRAINING_DESIGNS[0],
    )

    assert configuration.held_out_city.name == 'matched_grid_6x6_square_validation'
    assert tuple(city.name for city in configuration.train_cities) == ('matched_grid_3x3_square',)


def _controller_count(city_name: str) -> int:
    match city_name:
        case 'matched_grid_5x2_tall':
            return 6
        case 'matched_grid_3x3_square':
            return 5
        case 'matched_grid_4x4_square':
            return 12
        case 'matched_grid_5x3_tall':
            return 11
        case 'matched_grid_5x5_square':
            return 21
        case _:
            raise ValueError(f'Unknown training city: {city_name}')
