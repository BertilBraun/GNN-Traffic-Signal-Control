from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.experiment_config import load_experiment_configuration
from src.movement.grid_study import (
    COVERAGE_GENERALIZATION_4X4_EVALUATION_SCENARIOS,
    COVERAGE_GENERALIZATION_4X4_TRAINING_SCENARIOS,
    COVERAGE_GENERALIZATION_4X4_VALIDATION_SCENARIO,
)
from src.movement.training.ppo.rollout import rollout_seed


def test_coverage_generalization_training_config_balances_action_samples() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_coverage_generalization_4x4_train50_2hop_30.yaml',
        project_root=ROOT,
    )

    assert tuple(city.name for city in configuration.train_cities) == tuple(
        scenario.name for scenario in COVERAGE_GENERALIZATION_4X4_TRAINING_SCENARIOS
    )
    assert configuration.held_out_city.name == COVERAGE_GENERALIZATION_4X4_VALIDATION_SCENARIO.name
    assert configuration.proximal_policy_optimization.rollouts_per_update == 90
    assert sum(city.rollout_jobs_per_iteration * 6 * 200 for city in configuration.train_cities) == 108_000


def test_coverage_generalization_training_config_keeps_local_reward() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_coverage_generalization_4x4_train50_2hop_30.yaml',
        project_root=ROOT,
    )
    proximal_policy_optimization = configuration.proximal_policy_optimization

    assert proximal_policy_optimization.iterations == 30
    assert proximal_policy_optimization.global_reward_weight == 0.0
    assert proximal_policy_optimization.flow_reward_weight == 0.0
    assert proximal_policy_optimization.throughput_reward_weight == 0.0
    assert proximal_policy_optimization.progress_reward_weight == 1.0
    assert proximal_policy_optimization.discharge_reward_weight == 10.0
    assert proximal_policy_optimization.gridlock_penalty_weight == 0.02
    assert proximal_policy_optimization.speed_change_weight == 10.0
    assert proximal_policy_optimization.switch_penalty_weight == 0.0


def test_three_hop_coverage_training_config_preserves_samples_and_reward() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_coverage_generalization_4x4_train50_3hop_40.yaml',
        project_root=ROOT,
    )
    proximal_policy_optimization = configuration.proximal_policy_optimization

    assert proximal_policy_optimization.iterations == 40
    assert proximal_policy_optimization.rollouts_per_update == 90
    assert sum(city.rollout_jobs_per_iteration * 6 * 200 for city in configuration.train_cities) == 108_000
    assert proximal_policy_optimization.progress_reward_weight == 1.0
    assert proximal_policy_optimization.discharge_reward_weight == 10.0
    assert proximal_policy_optimization.gridlock_penalty_weight == 0.02
    assert proximal_policy_optimization.speed_change_weight == 10.0


def test_three_hop_coverage_run_uses_unique_rollout_seeds() -> None:
    seeds = tuple(
        rollout_seed(
            training_seed=6101,
            iteration=iteration,
            rollout_index=rollout_index,
            rollouts_per_update=90,
            fixed_rollout_seed=None,
        )
        for iteration in range(1, 41)
        for rollout_index in range(90)
    )

    assert len(seeds) == 3_600
    assert len(set(seeds)) == len(seeds)
    assert seeds[0] == 6191
    assert seeds[-1] == 9790


def test_coverage_generalization_evaluation_config_uses_fresh_masks() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'grid_coverage_generalization_4x4_evaluation.yaml',
        project_root=ROOT,
    )
    configured_city_names = {city.name for city in configuration.cities}

    assert configured_city_names == {scenario.name for scenario in COVERAGE_GENERALIZATION_4X4_EVALUATION_SCENARIOS}
    assert configuration.evaluation.seeds == (8401, 8402, 8403)
    assert configuration.proximal_policy_optimization.evaluation_learned_action_mode.value == 'sample'
