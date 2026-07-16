from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.experiment_config import (  # noqa: E402
    CitySplit,
    ExperimentEvaluationPolicy,
    ExperimentLearnedEvaluationActionMode,
    ExperimentPpoRewardObjective,
    ExperimentSpeedChangeMode,
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
    assert (
        configuration.proximal_policy_optimization.evaluation_learned_action_mode
        == ExperimentLearnedEvaluationActionMode.DETERMINISTIC
    )
    assert configuration.proximal_policy_optimization.evaluation_learned_temperature == 1.0


def test_throughput_scratch_config_uses_sampled_learned_eval() -> None:
    configuration_path = ROOT / 'configs' / 'training' / 'city_first_pass_throughput_scratch_32_worker.yaml'

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert (
        configuration.proximal_policy_optimization.evaluation_learned_action_mode
        == ExperimentLearnedEvaluationActionMode.SAMPLE
    )
    assert configuration.proximal_policy_optimization.evaluation_learned_temperature == 1.0
    assert configuration.proximal_policy_optimization.steps_per_rollout == 350
    assert configuration.proximal_policy_optimization.progress_reward_weight == 0.25
    assert configuration.proximal_policy_optimization.gridlock_penalty_weight == 0.08


def test_five_second_control_config_uses_late_yellow_and_expanded_baselines() -> None:
    configuration_path = ROOT / 'configs' / 'training' / 'city_first_pass_throughput_scratch_32_worker_5s_control.yaml'

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert configuration.simulation.decision_interval == 5
    assert configuration.simulation.yellow_duration == 3
    assert configuration.simulation.yellow_start_delay == 2
    assert configuration.proximal_policy_optimization.steps_per_rollout == 500
    assert configuration.evaluation.fixed_time_phase_duration == 10
    assert configuration.evaluation.queue_pressure_phase_duration == 10
    assert configuration.evaluation.policies == (
        ExperimentEvaluationPolicy.LEARNED,
        ExperimentEvaluationPolicy.MAX_PRESSURE,
        ExperimentEvaluationPolicy.QUEUE,
        ExperimentEvaluationPolicy.UNIFORM_RANDOM,
        ExperimentEvaluationPolicy.FIXED_TIME,
    )


def test_balanced_five_second_config_uses_immediate_yellow_and_dual_learned_eval() -> None:
    configuration_path = (
        ROOT / 'configs' / 'training' / 'city_first_pass_throughput_scratch_32_worker_5s_balanced_300.yaml'
    )

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert configuration.simulation.decision_interval == 5
    assert configuration.simulation.yellow_duration == 3
    assert configuration.simulation.yellow_start_delay == 0
    assert configuration.proximal_policy_optimization.iterations == 300
    assert configuration.proximal_policy_optimization.update_epochs == 4
    assert configuration.proximal_policy_optimization.throughput_reward_weight == 0.1
    assert configuration.proximal_policy_optimization.progress_reward_weight == 27.0
    assert configuration.proximal_policy_optimization.speed_change_weight == 15.0
    assert configuration.proximal_policy_optimization.switch_penalty_weight == 0.003
    assert configuration.proximal_policy_optimization.entropy_coefficient == 0.001
    assert configuration.evaluation.policies[:2] == (
        ExperimentEvaluationPolicy.LEARNED,
        ExperimentEvaluationPolicy.LEARNED_GREEDY,
    )


def test_negated_reward_sanity_config_minimizes_environment_reward() -> None:
    configuration_path = (
        ROOT / 'configs' / 'training' / 'city_first_pass_throughput_scratch_32_worker_5s_negated_reward_sanity_20.yaml'
    )

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert configuration.proximal_policy_optimization.iterations == 20
    assert configuration.proximal_policy_optimization.reward_objective == ExperimentPpoRewardObjective.MINIMIZE


def test_grid_three_local_reward_validation_config_loads() -> None:
    configuration_path = ROOT / 'configs' / 'training' / 'grid_3x3_local_reward_2hop_30.yaml'

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert tuple(city.name for city in configuration.train_cities) == ('grid_3x3',)
    assert configuration.train_cities[0].build_config is None
    assert configuration.simulation.warmup_steps == 15
    assert configuration.demand.minimum_train_scale == 0.6
    assert configuration.demand.maximum_train_scale == 0.8
    assert configuration.proximal_policy_optimization.iterations == 30
    assert configuration.proximal_policy_optimization.steps_per_rollout == 200
    assert configuration.proximal_policy_optimization.rollouts_per_update == 100
    assert configuration.proximal_policy_optimization.progress_reward_weight == 1.0
    assert configuration.proximal_policy_optimization.discharge_reward_weight == 10.0
    assert configuration.proximal_policy_optimization.speed_change_weight == 10.0
    assert configuration.proximal_policy_optimization.speed_change_mode == ExperimentSpeedChangeMode.BRAKING


def test_grid_shape_generalization_config_balances_action_samples_and_separates_splits() -> None:
    configuration_path = ROOT / 'configs' / 'training' / 'grid_shape_generalization_mixed_2hop_gate_30.yaml'

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=ROOT,
    )

    assert tuple(city.rollout_jobs_per_iteration for city in configuration.train_cities) == (18, 21, 9, 8, 5)
    assert configuration.held_out_city.name == 'matched_grid_6x6_square_validation'
    assert tuple(city.name for city in configuration.cities if city.split is CitySplit.EVALUATION_ONLY) == (
        'matched_grid_2x3_wide',
        'matched_grid_3x2_tall',
        'matched_grid_5x2_tall',
        'matched_grid_6x3_tall',
        'matched_grid_7x7_square_zero_shot',
    )
    assert configuration.proximal_policy_optimization.action_samples_per_batch == 16384
    assert configuration.demand.evaluation_scales == (0.7,)


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

    with pytest.raises(ValueError, match='non-training city beta must define rollout_jobs_per_iteration: 0'):
        load_experiment_configuration(
            configuration_path=configuration_path,
            project_root=tmp_path,
        )


def test_evaluation_only_city_is_evaluated_but_never_used_for_rollouts(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    rollout_workers: 1
  - name: beta
    split: held_out
    sumo_config: beta.sumocfg
    rollout_workers: 0
  - name: gamma
    split: evaluation_only
    sumo_config: gamma.sumocfg
    rollout_workers: 0
""",
        ),
        encoding='utf-8',
    )

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=tmp_path,
    )

    assert tuple(city.name for city in configuration.train_cities) == ('alpha',)
    assert configuration.held_out_city.name == 'beta'
    assert configuration.cities[2].split == CitySplit.EVALUATION_ONLY


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


def test_train_city_can_be_excluded_from_ppo_rollouts(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    build_config: alpha.build.yaml
    rollout_jobs_per_iteration: 0
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

    assert configuration.train_cities[0].rollout_jobs_per_iteration == 0


def test_experiment_can_omit_held_out_city(tmp_path: Path) -> None:
    _write_referenced_files(project_root=tmp_path)
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_path.write_text(
        _experiment_yaml(
            city_entries="""
  - name: alpha
    split: train
    sumo_config: alpha.sumocfg
    rollout_workers: 1
  - name: beta
    split: train
    sumo_config: beta.sumocfg
    rollout_workers: 1
""",
        ),
        encoding='utf-8',
    )

    configuration = load_experiment_configuration(
        configuration_path=configuration_path,
        project_root=tmp_path,
    )

    assert tuple(city.name for city in configuration.train_cities) == ('alpha', 'beta')
    assert all(city.build_config is None for city in configuration.train_cities)


def test_experiment_rejects_multiple_held_out_cities(tmp_path: Path) -> None:
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
    rollout_workers: 0
  - name: gamma
    split: held_out
    sumo_config: gamma.sumocfg
    build_config: gamma.build.yaml
    rollout_workers: 0
""",
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='at most one held-out city is allowed, found 2'):
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
