"""Entry point: PPO reinforcement learning on the irregular 4×4 grid (PLAN Stage 4).

Fine-tunes a GATv2 policy that was pre-trained via imitation learning (Stage 2)
using Proximal Policy Optimisation with a per-junction actor-critic architecture.

Usage
-----
    # Quickstart: point at your IL checkpoint
    python scripts/train_rl.py --il-ckpt checkpoints/il/<timestamp>

    # Full custom run
    python scripts/train_rl.py \\
        --il-ckpt      checkpoints/il/<timestamp> \\
        --cfg          configs/grid_4x4/grid.sumocfg \\
        --iterations   500 \\
        --ep-len       3600 \\
        --burn-in      20 \\
        --episodes     1 \\
        --lr           3e-4 \\
        --epochs       10 \\
        --clip         0.2 \\
        --eval-every   50 \\
        --save-every   50 \\
        --log-dir      runs/rl/my_run \\
        --ckpt-dir     checkpoints/rl/my_run \\
        --device       cuda

    # Resume from latest RL checkpoint (re-use same log/ckpt dirs)
    python scripts/train_rl.py \\
        --il-ckpt  checkpoints/rl/<timestamp> \\
        ...

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

from src.training.ppo import train_rl

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CFG = str(ROOT / 'configs' / 'grid_4x4' / 'grid.sumocfg')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='PPO RL fine-tuning — GATv2 policy on the irregular 4×4 grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        '--il-ckpt',
        required=True,
        metavar='DIR',
        help='Directory containing il_policy.pt and normalizer.npz (IL checkpoint)',
    )

    # Environment
    p.add_argument('--cfg', default=DEFAULT_CFG, help='Path to .sumocfg file')
    p.add_argument('--ep-len', type=int, default=3600, help='Simulation seconds per episode')
    p.add_argument(
        '--flow-range',
        nargs=2,
        type=int,
        default=[300, 1000],
        metavar=('MIN', 'MAX'),
        help='Total vehicles/hour range for demand randomisation (default: 300 1000)',
    )
    p.add_argument('--burn-in', type=int, default=20, help='Fixed-time warm-up steps before GNN takes over (× 15 s)')

    # PPO knobs
    p.add_argument('--iterations', type=int, default=500, help='Total collect→update cycles')
    p.add_argument('--episodes', type=int, default=1, help='Episodes collected per iteration')
    p.add_argument('--workers', type=int, default=1, help='Parallel SUMO processes for collection (1 = sequential)')
    p.add_argument(
        '--min-green-steps',
        type=int,
        default=2,
        help='Minimum decision steps before a phase switch is allowed (2 × 15 s = 30 s)',
    )
    p.add_argument('--lr', type=float, default=3e-4, help='Adam learning rate')
    p.add_argument('--gamma', type=float, default=0.99, help='Discount factor γ')
    p.add_argument('--lam', type=float, default=0.95, help='GAE smoothing λ')
    p.add_argument('--clip', type=float, default=0.2, help='PPO clip ε')
    p.add_argument('--epochs', type=int, default=4, help='PPO update epochs per iteration')
    p.add_argument(
        '--warmup', type=int, default=20, help='Iterations to train value head only (backbone frozen) before PPO starts'
    )
    p.add_argument(
        '--warmup-epochs',
        type=int,
        default=10,
        help='PPO epochs per iteration during value warmup (more than n_epochs to converge faster)',
    )
    p.add_argument('--value-coeff', type=float, default=0.5, help='Value loss weight')
    p.add_argument('--entropy-coeff', type=float, default=0.05, help='Entropy bonus weight')
    p.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clip norm')
    p.add_argument(
        '--minibatch', type=int, default=256, help='Target junction-transitions per minibatch (rounded to whole graphs)'
    )

    # Logging / checkpointing
    p.add_argument('--eval-every', type=int, default=5, help='Run model-vs-expert eval every N iterations (0 = never)')
    p.add_argument(
        '--n-eval-seeds', type=int, default=5, help='Number of demand seeds averaged in the final evaluation'
    )
    p.add_argument('--save-every', type=int, default=20, help='Save numbered checkpoint every N iterations')
    p.add_argument('--print-every', type=int, default=1, help='Print console summary every N iterations')
    p.add_argument('--log-dir', default=None, help='TensorBoard log directory (default: runs/rl/<timestamp>)')
    p.add_argument('--ckpt-dir', default=None, help='Checkpoint directory (default: checkpoints/rl/<timestamp>)')

    # Hardware
    p.add_argument('--device', default=None, help="PyTorch device ('cpu', 'cuda').  Auto-detected if omitted.")

    # Reproducibility / debugging
    p.add_argument(
        '--demand-seed',
        type=int,
        default=None,
        help='Fix demand RNG to this seed every episode (deterministic traffic). Omit for randomised demand.',
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = args.log_dir or str(ROOT / 'runs' / 'rl' / stamp)
    ckpt_dir = args.ckpt_dir or str(ROOT / 'checkpoints' / 'rl' / stamp)

    train_rl(
        cfg_path=args.cfg,
        il_checkpoint_dir=args.il_ckpt,
        n_iterations=args.iterations,
        episodes_per_update=args.episodes,
        n_workers=args.workers,
        episode_length=args.ep_len,
        burn_in_steps=args.burn_in,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip,
        n_epochs=args.epochs,
        value_warmup_iters=args.warmup,
        warmup_epochs=args.warmup_epochs,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.grad_clip,
        min_green_steps=args.min_green_steps,
        minibatch_junctions=args.minibatch,
        eval_every=args.eval_every,
        n_eval_seeds=args.n_eval_seeds,
        save_every=args.save_every,
        print_every=args.print_every,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=args.device,
        demand_seed=args.demand_seed,
        flow_range=tuple(args.flow_range),
    )


if __name__ == '__main__':
    main()
