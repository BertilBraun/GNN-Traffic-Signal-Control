from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_rl


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
            '--value-warmup-iterations',
            '1',
            '--demand-scale',
            '0.5',
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
        ],
    )

    args = train_rl.parse_args()

    assert args.il_checkpoint == Path('checkpoints/il/unit/movement_policy_best.pt')
    assert args.iterations == 2
    assert args.steps_per_rollout == 3
    assert args.value_warmup_iterations == 1
    assert args.demand_scale == 0.5
    assert args.eval_demand_scale == 0.75
    assert args.gui is True
    assert args.initial_occupancy_min == 0.15
    assert args.initial_occupancy_max == 0.25
    assert args.fixed_rollout_seed == 123
    assert args.resume_checkpoint is None
    assert args.time_to_teleport == -1


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

    assert args.lr == 1e-4
    assert args.clip == 0.1
    assert args.entropy_coeff == 0.01
    assert args.cfg.name == 'grid.sumocfg'
    assert args.cfg.parent.name == 'grid_3x3_dedicated'
    assert args.demand_scale == 1.0
    assert args.eval_demand_scale == 1.0
    assert args.initial_occupancy_min == 0.05
    assert args.initial_occupancy_max == 0.08
    assert args.global_reward_weight == 0.1
    assert args.reward_clip == 1.0
    assert args.teleport_penalty == 0.0
    assert args.max_teleports_per_rollout == 10
    assert args.target_kl == 0.02
    assert args.fixed_rollout_seed is None
    assert args.time_to_teleport is None
