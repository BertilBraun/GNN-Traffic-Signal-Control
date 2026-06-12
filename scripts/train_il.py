"""Train a zero-hop movement-score imitation model from JSONL samples."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.training.il import (  # noqa: E402
    ZeroHopTrainingConfig,
    train_zero_hop_il_from_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train zero-hop movement-score imitation from collected JSONL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, type=Path, help="Input JSONL dataset")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension")
    parser.add_argument(
        "--loss",
        choices=("huber", "mse"),
        default="huber",
        help="Movement-score regression loss",
    )
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print loss every N epochs (0 disables progress output)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: checkpoints/il/<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = args.ckpt_dir or ROOT / "checkpoints" / "il" / stamp
    result = train_zero_hop_il_from_jsonl(
        dataset_path=args.data,
        config=ZeroHopTrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            checkpoint_dir=checkpoint_dir,
            seed=args.seed,
            loss=args.loss,
            device=args.device,
            progress_every=args.progress_every,
        ),
    )
    print(
        f"Training complete: epochs={result.epochs} "
        f"final_loss={result.final_loss:.6f} checkpoint={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
