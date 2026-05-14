"""Entry point: Imitation Learning on the irregular 4×4 grid (PLAN Stage 2).

Trains a GATv2 policy to mimic the GreedyExpert controller using
cross-entropy loss on (graph, expert_phase) pairs collected from live
SUMO episodes.

Usage
-----
    # Default: 50 episodes × 1200 s, grid_4x4
    python scripts/train_il.py

    # Custom run
    python scripts/train_il.py \\
        --cfg      configs/grid_4x4/grid.sumocfg \\
        --episodes 100 \\
        --ep-len   3600 \\
        --lr       3e-4 \\
        --log-dir  runs/il_grid \\
        --ckpt-dir checkpoints/il_grid

    # Watch training in TensorBoard (separate terminal)
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.imitation import train_il

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CFG = str(ROOT / 'configs' / 'grid_4x4' / 'grid.sumocfg')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Imitation Learning — GATv2 policy on the irregular 4×4 grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--cfg',
        default=DEFAULT_CFG,
        help='Path to .sumocfg file',
    )
    p.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of training episodes',
    )
    p.add_argument(
        '--ep-len',
        type=int,
        default=1200,
        help='Simulation seconds per episode (Stage 2 default: 1200 s)',
    )
    p.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Adam learning rate',
    )
    p.add_argument(
        '--log-dir',
        default=None,
        help='TensorBoard log directory (default: runs/il/<timestamp>)',
    )
    p.add_argument(
        '--ckpt-dir',
        default=None,
        help='Checkpoint directory (default: checkpoints/il/<timestamp>)',
    )
    p.add_argument(
        '--device',
        default=None,
        help="PyTorch device (e.g. 'cpu', 'cuda').  Auto-detected if omitted.",
    )
    p.add_argument(
        '--eval-every',
        type=int,
        default=5,
        help='Run model-vs-expert eval every N episodes (0 = only at end)',
    )
    p.add_argument(
        '--print-every',
        type=int,
        default=1,
        help='Print episode summary every N episodes',
    )
    p.add_argument(
        '--n-eval-seeds',
        type=int,
        default=5,
        help='Number of demand seeds averaged in the final evaluation',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Generate a single timestamp so log_dir and ckpt_dir are always paired,
    # but only when the user hasn't provided explicit paths.
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = args.log_dir or str(ROOT / 'runs' / 'il' / stamp)
    ckpt_dir = args.ckpt_dir or str(ROOT / 'checkpoints' / 'il' / stamp)

    train_il(
        cfg_path=args.cfg,
        n_episodes=args.episodes,
        episode_length=args.ep_len,
        lr=args.lr,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=args.device,
        eval_every=args.eval_every,
        print_every=args.print_every,
        n_eval_seeds=args.n_eval_seeds,
    )


if __name__ == '__main__':
    main()
