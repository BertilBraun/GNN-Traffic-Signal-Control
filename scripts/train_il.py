"""Train a movement-score imitation model from JSONL samples."""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.training.il import (  # noqa: E402
    MovementILLoss,
    MovementILTrainingConfig,
    train_movement_il_from_jsonl,
)
from scripts.collect_il_data import collect_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train movement-score imitation from collected JSONL data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', type=Path, default=None, help='Input JSONL dataset')
    parser.add_argument(
        '--cfg',
        dest='sumo_config_path',
        type=Path,
        default=None,
        help='SUMO config to collect before training',
    )
    parser.add_argument('--samples', type=int, default=120, help='Number of decision samples to collect with --cfg')
    parser.add_argument('--decision-interval', type=int, default=15, help='Seconds between collected samples')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='MLP hidden dimension')
    parser.add_argument('--num-hops', type=int, default=1, help='LaneGroup/Movement macro-hops')
    parser.add_argument(
        '--loss',
        choices=('huber', 'mse'),
        default='huber',
        help='Movement-score regression loss',
    )
    parser.add_argument('--seed', type=int, default=42, help='Torch random seed')
    parser.add_argument('--device', default='cpu', help='Torch device')
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10,
        help='Print loss every N epochs (0 disables progress output)',
    )
    parser.add_argument(
        '--ckpt-dir',
        type=Path,
        default=None,
        help='Checkpoint directory (default: checkpoints/il/<timestamp>)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = args.ckpt_dir or ROOT / 'checkpoints' / 'il' / stamp
    if args.data is None and args.sumo_config_path is None:
        raise SystemExit('Either --data or --cfg is required.')
    config = MovementILTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        loss=MovementILLoss(args.loss),
        device=args.device,
        progress_every=args.progress_every,
        num_hops=args.num_hops,
    )
    if args.data is not None:
        result = train_movement_il_from_jsonl(dataset_path=args.data, config=config)
    else:
        with tempfile.TemporaryDirectory(prefix='movement_il_') as temporary_directory:
            dataset_path = Path(temporary_directory) / 'samples.jsonl'
            collect_samples(
                cfg_path=args.sumo_config_path,
                output_path=dataset_path,
                steps=args.samples * args.decision_interval,
                decision_interval=args.decision_interval,
                seed=args.seed,
            )
            result = train_movement_il_from_jsonl(dataset_path=dataset_path, config=config)
    print(
        f'Training complete: epochs={result.epochs} '
        f'final_loss={result.final_loss:.6f} checkpoint={result.checkpoint_path}'
    )


if __name__ == '__main__':
    main()
