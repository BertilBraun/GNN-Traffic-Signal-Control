"""Generate action-sample-matched training designs for the grid study matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.experiment_config import (
    CitySplit,
    ExperimentCityConfiguration,
    ExperimentConfiguration,
    load_experiment_configuration,
)


@dataclass(frozen=True)
class TrainingScenarioAllocation:
    scenario_name: str
    rollout_jobs_per_iteration: int


@dataclass(frozen=True)
class MatrixTrainingDesign:
    experiment_name: str
    output_filename: str
    allocations: tuple[TrainingScenarioAllocation, ...]


MATRIX_TRAINING_DESIGNS: tuple[MatrixTrainingDesign, ...] = (
    MatrixTrainingDesign(
        experiment_name='grid_shape_generalization_train_3x3_2hop_30',
        output_filename='grid_shape_generalization_train_3x3_2hop_30.yaml',
        allocations=(TrainingScenarioAllocation('matched_grid_3x3_square', 108),),
    ),
    MatrixTrainingDesign(
        experiment_name='grid_shape_generalization_train_4x4_2hop_30',
        output_filename='grid_shape_generalization_train_4x4_2hop_30.yaml',
        allocations=(TrainingScenarioAllocation('matched_grid_4x4_square', 45),),
    ),
    MatrixTrainingDesign(
        experiment_name='grid_shape_generalization_train_5x5_2hop_30',
        output_filename='grid_shape_generalization_train_5x5_2hop_30.yaml',
        allocations=(TrainingScenarioAllocation('matched_grid_5x5_square', 26),),
    ),
    MatrixTrainingDesign(
        experiment_name='grid_shape_generalization_train_rectangles_2hop_30',
        output_filename='grid_shape_generalization_train_rectangles_2hop_30.yaml',
        allocations=(
            TrainingScenarioAllocation('matched_grid_5x2_tall', 45),
            TrainingScenarioAllocation('matched_grid_5x3_tall', 25),
        ),
    ),
)


def build_matrix_configuration(
    base_configuration: ExperimentConfiguration,
    design: MatrixTrainingDesign,
) -> ExperimentConfiguration:
    known_scenarios = frozenset(city.name for city in base_configuration.cities)
    unknown_scenarios = frozenset(allocation.scenario_name for allocation in design.allocations) - known_scenarios
    if unknown_scenarios:
        raise ValueError(f'Unknown matrix training scenarios: {", ".join(sorted(unknown_scenarios))}')
    cities = tuple(
        _matrix_city(
            city=city,
            allocations=design.allocations,
        )
        for city in base_configuration.cities
    )
    ppo_configuration = base_configuration.proximal_policy_optimization.model_copy(
        update={'rollouts_per_update': sum(allocation.rollout_jobs_per_iteration for allocation in design.allocations)}
    )
    return base_configuration.model_copy(
        update={
            'name': design.experiment_name,
            'cities': cities,
            'proximal_policy_optimization': ppo_configuration,
        }
    )


def _matrix_city(
    city: ExperimentCityConfiguration,
    allocations: tuple[TrainingScenarioAllocation, ...],
) -> ExperimentCityConfiguration:
    rollout_jobs = _rollout_jobs_for_city(city_name=city.name, allocations=allocations)
    if rollout_jobs is not None:
        return city.model_copy(
            update={
                'split': CitySplit.TRAIN,
                'rollout_jobs_per_iteration': rollout_jobs,
            }
        )
    if city.split is CitySplit.HELD_OUT:
        return city
    return city.model_copy(
        update={
            'split': CitySplit.EVALUATION_ONLY,
            'rollout_jobs_per_iteration': 0,
        }
    )


def _rollout_jobs_for_city(
    city_name: str,
    allocations: tuple[TrainingScenarioAllocation, ...],
) -> int | None:
    return next(
        (allocation.rollout_jobs_per_iteration for allocation in allocations if allocation.scenario_name == city_name),
        None,
    )


def write_matrix_configuration(
    configuration: ExperimentConfiguration,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = configuration.model_dump(mode='json', by_alias=True, exclude_none=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding='utf-8',
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base-configuration',
        type=Path,
        default=ROOT / 'configs' / 'training' / 'grid_shape_generalization_mixed_2hop_gate_30.yaml',
    )
    parser.add_argument(
        '--output-directory',
        type=Path,
        default=ROOT / 'configs' / 'training',
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    base_configuration = load_experiment_configuration(
        configuration_path=arguments.base_configuration,
        project_root=ROOT,
    )
    for design in MATRIX_TRAINING_DESIGNS:
        configuration = build_matrix_configuration(
            base_configuration=base_configuration,
            design=design,
        )
        output_path = arguments.output_directory / design.output_filename
        write_matrix_configuration(configuration=configuration, output_path=output_path)
        print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
