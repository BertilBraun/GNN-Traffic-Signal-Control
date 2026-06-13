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
