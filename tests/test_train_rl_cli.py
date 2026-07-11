from pathlib import Path
from hashlib import sha256
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_rl
from src.movement.evaluation import EvaluationPolicy, LearnedEvaluationActionMode
from src.movement.experiment_config import CitySplit


def test_train_rl_cli_accepts_movement_ppo_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--iterations',
            '2',
            '--steps-per-rollout',
            '3',
            '--rollouts-per-update',
            '4',
            '--num-workers',
            '2',
            '--value-warmup-iterations',
            '1',
            '--demand-scale',
            '0.5',
            '--demand-scale-min',
            '0.4',
            '--demand-scale-max',
            '0.85',
            '--eval-demand-scale',
            '0.75',
            '--eval-workers',
            '3',
            '--eval-learned-device',
            'cpu',
            '--eval-learned-action-mode',
            'sample',
            '--eval-learned-temperature',
            '0.7',
            '--gui',
            '--initial-occupancy-min',
            '0.15',
            '--initial-occupancy-max',
            '0.25',
            '--fixed-rollout-seed',
            '123',
            '--time-to-teleport',
            '-1',
            '--speed-change-weight',
            '0.03',
            '--flow-reward-weight',
            '0.12',
            '--reward-sample-interval',
            '5',
            '--sumo-backend',
            'libsumo',
        ],
    )

    args = train_rl.parse_args()

    assert args.il_checkpoint == Path('checkpoints/il/unit/movement_policy_best.pt')
    assert args.iterations == 2
    assert args.steps_per_rollout == 3
    assert args.rollouts_per_update == 4
    assert args.num_workers == 2
    assert args.value_warmup_iterations == 1
    assert args.demand_scale == 0.5
    assert args.demand_scale_min == 0.4
    assert args.demand_scale_max == 0.85
    assert args.eval_demand_scale == 0.75
    assert args.eval_workers == 3
    assert args.eval_learned_device == 'cpu'
    assert args.eval_learned_action_mode == 'sample'
    assert args.eval_learned_temperature == 0.7
    assert args.gui is True
    assert args.initial_occupancy_min == 0.15
    assert args.initial_occupancy_max == 0.25
    assert args.fixed_rollout_seed == 123
    assert args.resume_checkpoint is None
    assert args.time_to_teleport == -1
    assert args.flow_reward_weight == 0.12
    assert args.speed_change_weight == 0.03
    assert args.reward_sample_interval == 5
    assert args.sumo_backend == 'libsumo'


def test_train_rl_cli_accepts_resume_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--resume-checkpoint',
            'checkpoints/rl/unit/movement_ppo_latest.pt',
            '--iterations',
            '300',
        ],
    )

    args = train_rl.parse_args()

    assert args.il_checkpoint is None
    assert args.resume_checkpoint == Path('checkpoints/rl/unit/movement_ppo_latest.pt')
    assert args.iterations == 300
    assert args.allow_resume_config_mismatch is False


def test_train_rl_cli_accepts_random_scratch_initialization(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--scratch-random',
            '--scratch-lane-feature-dim',
            '29',
            '--scratch-movement-feature-dim',
            '4',
            '--scratch-hidden-dim',
            '64',
            '--scratch-num-hops',
            '1',
        ],
    )

    args = train_rl.parse_args()
    scratch_config = train_rl.scratch_model_config(args)

    assert args.il_checkpoint is None
    assert args.scratch_checkpoint is None
    assert args.scratch_random is True
    assert train_rl.initialization_checkpoint_path(args) is None
    assert scratch_config is not None
    assert scratch_config.lane_feature_dim == 29
    assert scratch_config.movement_feature_dim == 4
    assert scratch_config.hidden_dim == 64
    assert scratch_config.num_hops == 1


def test_train_rl_cli_accepts_resume_config_mismatch_override(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--resume-checkpoint',
            'checkpoints/rl/unit/movement_ppo_latest.pt',
            '--allow-resume-config-mismatch',
        ],
    )

    args = train_rl.parse_args()

    assert args.allow_resume_config_mismatch is True


def test_train_rl_cli_uses_stable_ppo_defaults(monkeypatch) -> None:
    monkeypatch.setattr(train_rl.os, 'cpu_count', lambda: 12)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
        ],
    )

    args = train_rl.parse_args()

    assert args.iterations == 300
    assert args.steps_per_rollout == 360
    assert args.decision_interval == 10
    assert args.lr == 2e-4
    assert args.clip == 0.1
    assert args.entropy_coeff == 0.01
    assert args.value_warmup_iterations == 20
    assert args.transitions_per_batch == 32
    assert args.cfg.name == 'grid.sumocfg'
    assert args.cfg.parent.name == 'grid_3x3_dedicated'
    assert args.demand_scale == 1.0
    assert args.demand_scale_min is None
    assert args.demand_scale_max is None
    assert args.eval_demand_scale is None
    assert train_rl.evaluation_demand_scale(args) == 1.0
    assert args.eval_steps is None
    assert args.eval_seeds is None
    assert args.eval_policies is None
    assert args.eval_every is None
    assert args.eval_workers == 1
    assert args.eval_learned_device == 'cpu'
    assert args.eval_learned_action_mode == 'deterministic'
    assert args.eval_learned_temperature == 1.0
    assert args.scratch_lane_feature_dim == 29
    assert args.scratch_movement_feature_dim == 4
    assert args.scratch_hidden_dim == 64
    assert args.scratch_num_hops == 1
    assert args.initial_occupancy_min == 0.05
    assert args.initial_occupancy_max == 0.08
    assert args.global_reward_weight == 0.1
    assert args.flow_reward_weight == 0.1
    assert args.speed_change_weight == 0.02
    assert args.reward_sample_interval == 5
    assert args.reward_clip == 1.0
    assert args.teleport_penalty == 0.0
    assert args.max_teleports_per_rollout == 999
    assert args.target_kl == 0.03
    assert args.rollouts_per_update == 3
    assert args.rollouts_per_update_explicit is False
    assert args.num_workers == -1
    assert train_rl.num_workers(args, None) == 12
    assert args.fixed_rollout_seed is None
    assert args.time_to_teleport == -1
    assert args.sumo_backend == 'libsumo'


def test_train_rl_cli_falls_back_to_one_worker_when_cpu_count_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(train_rl.os, 'cpu_count', lambda: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--num-workers',
            '-1',
        ],
    )

    assert train_rl.num_workers(train_rl.parse_args(), None) == 1


def test_train_rl_cli_maps_fixed_demand_to_min_max(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--demand-scale',
            '0.65',
        ],
    )

    assert train_rl.demand_scale_bounds(train_rl.parse_args(), None) == (0.65, 0.65)


def test_train_rl_cli_allows_demand_scale_range(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--demand-scale',
            '0.65',
            '--demand-scale-min',
            '0.4',
            '--demand-scale-max',
            '0.85',
        ],
    )

    assert train_rl.demand_scale_bounds(train_rl.parse_args(), None) == (0.4, 0.85)


def test_train_rl_cli_uses_experiment_rollout_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--experiment-config',
            'configs/training/city_first_pass.yaml',
        ],
    )

    args = train_rl.parse_args()
    experiment_configuration = train_rl.experiment_config(args.experiment_config)
    assert experiment_configuration is not None

    rollout_cities = train_rl.experiment_rollout_cities(experiment_configuration)

    assert train_rl.rollouts_per_update(args, experiment_configuration) == 8
    assert train_rl.num_workers(args, experiment_configuration) == 8
    assert train_rl.reward_sample_interval(args, experiment_configuration) == 5
    assert train_rl.demand_scale_bounds(args, experiment_configuration) == (0.8, 1.2)
    assert train_rl.evaluation_demand_scales(args, experiment_configuration) == (0.8, 1.0, 1.2)
    assert train_rl.evaluation_steps(args, experiment_configuration) == 1800
    assert train_rl.evaluation_seeds(args, experiment_configuration) == (100, 101, 102)
    assert train_rl.evaluation_policies(args, experiment_configuration) == (
        EvaluationPolicy.LEARNED,
        EvaluationPolicy.MAX_PRESSURE,
        EvaluationPolicy.QUEUE,
    )
    assert tuple(city.city_name for city in rollout_cities) == (
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
    )
    assert all(city.city_split == CitySplit.TRAIN for city in rollout_cities)
    assert tuple(city.rollout_workers for city in rollout_cities) == (2, 2, 2, 2)
    assert tuple(city.rollout_jobs_per_iteration for city in rollout_cities) == (2, 2, 2, 2)


def test_train_rl_cli_uses_experiment_ppo_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--experiment-config',
            'configs/training/city_first_pass_4_worker.yaml',
        ],
    )

    args = train_rl.parse_args()
    experiment_configuration = train_rl.experiment_config(args.experiment_config)
    assert experiment_configuration is not None
    settings = train_rl.ppo_command_settings(args=args, experiment_configuration=experiment_configuration)

    assert settings.iterations == 1000
    assert settings.steps_per_rollout == 500
    assert settings.update_epochs == 2
    assert settings.value_warmup_iterations == 2
    assert settings.warmup_epochs == 2
    assert settings.transitions_per_batch == 256
    assert settings.eval_every == 0
    assert settings.eval_workers == 1
    assert settings.eval_learned_device == 'cpu'
    assert settings.eval_learned_action_mode == LearnedEvaluationActionMode.DETERMINISTIC
    assert settings.eval_learned_temperature == 1.0
    assert settings.save_every == 1
    assert train_rl.rollouts_per_update(args, experiment_configuration) == 30
    assert train_rl.num_workers(args, experiment_configuration) == 20
    assert tuple(city.rollout_workers for city in train_rl.experiment_rollout_cities(experiment_configuration)) == (
        10,
        0,
        10,
        10,
    )
    assert tuple(city.rollout_priority for city in train_rl.experiment_rollout_cities(experiment_configuration)) == (
        2,
        3,
        5,
        4,
    )


def test_train_rl_cli_allows_explicit_rollout_total_override(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--experiment-config',
            'configs/training/city_first_pass_4_worker.yaml',
            '--rollouts-per-update',
            '5',
        ],
    )

    args = train_rl.parse_args()
    experiment_configuration = train_rl.experiment_config(args.experiment_config)
    assert experiment_configuration is not None

    assert args.rollouts_per_update_explicit is True
    assert train_rl.rollouts_per_update(args, experiment_configuration) == 5


def test_train_rl_cli_overrides_experiment_ppo_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-checkpoint',
            'checkpoints/il/unit/movement_policy_best.pt',
            '--experiment-config',
            'configs/training/city_first_pass_4_worker.yaml',
            '--steps-per-rollout',
            '120',
            '--value-warmup-iterations',
            '1',
            '--epochs',
            '1',
            '--eval-every',
            '10',
            '--eval-steps',
            '300',
            '--eval-seeds',
            '7',
            '8',
            '--eval-policies',
            'max-pressure',
            'queue',
            '--eval-demand-scale',
            '1.0',
            '--eval-workers',
            '4',
            '--eval-learned-device',
            'cuda',
            '--eval-learned-action-mode',
            'sample',
            '--eval-learned-temperature',
            '0.8',
        ],
    )

    args = train_rl.parse_args()
    experiment_configuration = train_rl.experiment_config(args.experiment_config)
    assert experiment_configuration is not None
    settings = train_rl.ppo_command_settings(args=args, experiment_configuration=experiment_configuration)

    assert settings.steps_per_rollout == 120
    assert settings.value_warmup_iterations == 1
    assert settings.update_epochs == 1
    assert settings.eval_every == 10
    assert settings.eval_workers == 4
    assert settings.eval_learned_device == 'cuda'
    assert settings.eval_learned_action_mode == LearnedEvaluationActionMode.SAMPLE
    assert settings.eval_learned_temperature == 0.8
    assert train_rl.evaluation_steps(args, experiment_configuration) == 300
    assert train_rl.evaluation_seeds(args, experiment_configuration) == (7, 8)
    assert train_rl.evaluation_policies(args, experiment_configuration) == (
        EvaluationPolicy.MAX_PRESSURE,
        EvaluationPolicy.QUEUE,
    )
    assert train_rl.evaluation_demand_scales(args, experiment_configuration) == (1.0,)


def test_train_rl_experiment_hash_uses_configuration_text(tmp_path: Path) -> None:
    configuration_path = tmp_path / 'experiment.yaml'
    configuration_text = 'name: unit\n'
    configuration_path.write_text(configuration_text, encoding='utf-8')

    assert train_rl.experiment_configuration_text(configuration_path) == configuration_text
    assert (
        train_rl.experiment_configuration_sha256(configuration_path)
        == sha256(configuration_text.encode('utf-8')).hexdigest()
    )
