from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_il, train_rl


def test_train_il_accepts_city_eval_and_demand_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_il.py',
            '--periodic-eval-seeds',
            '5',
            '--demand-min-rate',
            '1.5',
            '--resume-from',
            'checkpoints/il/example',
            '--resume-tag',
            'final',
        ],
    )

    args = train_il.parse_args()

    assert args.periodic_eval_seeds == 5
    assert args.demand_min_rate == 1.5
    assert args.resume_from == 'checkpoints/il/example'
    assert args.resume_tag == 'final'


def test_train_rl_accepts_demand_min_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_rl.py',
            '--il-ckpt',
            'checkpoints/il/example',
            '--demand-min-rate',
            '1.5',
        ],
    )

    args = train_rl.parse_args()

    assert args.demand_min_rate == 1.5
