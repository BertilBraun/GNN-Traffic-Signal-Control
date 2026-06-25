from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_rl
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
    assert args.gui is True
    assert args.initial_occupancy_min == 0.15
    assert args.initial_occupancy_max == 0.25
    assert args.fixed_rollout_seed == 123
    assert args.resume_checkpoint is None
    assert args.time_to_teleport == -1
    assert args.speed_change_weight == 0.03


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


def test_train_rl_cli_uses_stable_ppo_defaults(monkeypatch) -> None:
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
    assert args.eval_demand_scale == 1.0
    assert args.initial_occupancy_min == 0.05
    assert args.initial_occupancy_max == 0.08
    assert args.global_reward_weight == 0.1
    assert args.speed_change_weight == 0.02
    assert args.reward_clip == 1.0
    assert args.teleport_penalty == 0.0
    assert args.max_teleports_per_rollout == 999
    assert args.target_kl == 0.03
    assert args.rollouts_per_update == 3
    assert args.num_workers == 3
    assert args.fixed_rollout_seed is None
    assert args.time_to_teleport == -1


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
    assert train_rl.demand_scale_bounds(args, experiment_configuration) == (0.8, 1.2)
    assert train_rl.evaluation_demand_scales(args, experiment_configuration) == (0.8, 1.0, 1.2)
    assert tuple(city.city_name for city in rollout_cities) == (
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
    )
    assert all(city.city_split == CitySplit.TRAIN for city in rollout_cities)
    assert tuple(city.rollout_workers for city in rollout_cities) == (2, 2, 2, 2)
