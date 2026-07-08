from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.experiment_config import (  # noqa: E402
    CitySplit,
    ExperimentEvaluationPolicy,
    load_experiment_configuration,
)


def test_valid_city_first_pass_config_loads() -> None:
    configuration_path = ROOT / 'configs' / 'training' / 'city_first_pass.yaml'

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert configuration.name == 'city_first_pass'
    assert tuple(city.name for city in configuration.train_cities) == (
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
    )
    assert configuration.held_out_city.name == 'freiburg_altstadt'
    assert configuration.held_out_city.split == CitySplit.HELD_OUT
    assert configuration.held_out_city.rollout_workers == 0
    assert configuration.held_out_city.rollout_jobs_per_iteration == 0
    assert configuration.evaluation.policies == (
        ExperimentEvaluationPolicy.LEARNED,
        ExperimentEvaluationPolicy.MAX_PRESSURE,
        ExperimentEvaluationPolicy.QUEUE,
    )


def test_duplicate_city_names_fail(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_workers: 1
  - name: alpha
    split: held_out
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_workers: 0
""",
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='duplicate city names'):
        load_experiment_configuration(
            configuration_path=configuration_path,
            project_root=tmp_path,
        )


def test_held_out_city_with_rollout_workers_fails(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_workers: 1
  - name: beta
    split: held_out
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_workers: 1
""",
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='held-out city beta must define rollout_jobs_per_iteration: 0'):
        load_experiment_configuration(
            configuration_path=configuration_path,
            project_root=tmp_path,
        )


def test_city_rollout_jobs_per_iteration_field_loads(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_jobs_per_iteration: 3
    rollout_priority: 7
  - name: beta
    split: held_out
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_jobs_per_iteration: 0
""",
        ),
        encoding='utf-8',
    )

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=tmp_path,
    )

    assert configuration.train_cities[0].rollout_jobs_per_iteration == 3
    assert configuration.train_cities[0].rollout_workers == 3
    assert configuration.train_cities[0].rollout_priority == 7


@pytest.mark.parametrize(
    ('city_entries', 'expected_message'),
    (
        (
            """
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_workers: 1
  - name: beta
    split: train
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_workers: 1
""",
            'exactly one held-out city is required, found 0',
        ),
        (
            """
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_workers: 1
  - name: beta
    split: held_out
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_workers: 0
  - name: gamma
    split: held_out
    sumo_config: gamma.sumocfg
    build_config: gamma.build.yaml
    rollout_workers: 0
""",
            'exactly one held-out city is required, found 2',
        ),
    ),
)
def test_held_out_city_count_must_be_exactly_one(
    tmp_path: Path,
    city_entries: str,
    expected_message: str,
) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(city_entries=city_entries),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=expected_message):
        load_experiment_configuration(
            configuration_path=configuration_path,
            project_root=tmp_path,
        )


def test_missing_referenced_files_fail_clearly(tmp_path: Path) -> None:
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_workers: 1
  - name: beta
    split: held_out
    sumo_config: beta.sumocfg
    build_config: beta.build.yaml
    rollout_workers: 0
""",
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='experiment configuration references missing files'):
        load_experiment_configuration(
            configuration_path=configuration_path,
            project_root=tmp_path,
        )


def _write_referenced_files(project_root: Path) -> None:
    for city_name in ('alpha', 'beta', 'gamma'):
        (project_root / f'{city_name}.sumocfg').write_text('<configuration />', encoding='utf-8')
        (project_root / f'{city_name}.build.yaml').write_text(f'name: {city_name}\n', encoding='utf-8')


def _experiment_yaml(city_entries: str) -> str:
    return f"""name: test_experiment

cities:
{city_entries}
simulation:
  decision_interval: 10
  time_to_teleport: -1
  yellow_duration: 3
  min_green_steps: 2
  initial_occupancy_min: 0.05
  initial_occupancy_max: 0.08

demand:
  train_scale_min: 0.8
  train_scale_max: 1.2
  eval_scales: [0.8, 1.0, 1.2]

imitation_learning:
  samples_per_city: 4800
  samples_per_simulation: 240
  collection_workers: 8
  epochs: 400
  samples_per_batch: 32
  phase_loss_coefficient: 1.0

ppo:
  iterations: 1000
  steps_per_rollout: 1800
  rollouts_per_update: 8
  rollout_workers: 8
  eval_every_iterations: 25
  save_every_iterations: 25

evaluation:
  policies: [learned, max-pressure, queue]
  seeds: [100, 101, 102]
  steps: 1800
"""
