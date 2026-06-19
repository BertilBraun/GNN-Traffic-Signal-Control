"""Train a movement-score policy with PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.evaluation import EvaluationPolicy
from src.movement.training.ppo import MovementPpoConfig, train_movement_ppo

DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PPO fine-tuning for movement-score traffic signal policies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--il-checkpoint', type=Path, help='IL movement checkpoint for a new PPO run')
    initialization.add_argument(
        '--resume-checkpoint',
        type=Path,
        help='PPO checkpoint whose model, optimizer, RNG, and iteration state will be resumed',
    )
    parser.add_argument('--cfg', type=Path, default=DEFAULT_CFG, help='SUMO .sumocfg path')
    parser.add_argument('--iterations', type=int, default=300, help='Final target PPO iteration')
    parser.add_argument('--steps-per-rollout', type=int, default=360, help='Decision steps collected per iteration')
    parser.add_argument('--rollouts-per-update', type=int, default=3, help='Independent rollouts collected per PPO update')
    parser.add_argument('--num-workers', type=int, default=3, help='Parallel SUMO rollout worker processes')
    parser.add_argument('--decision-interval', type=int, default=10, help='Simulation seconds per decision')
    parser.add_argument('--lr', type=float, default=2e-4, help='Adam learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip', type=float, default=0.1, help='PPO clipping epsilon')
    parser.add_argument('--epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--value-warmup-iterations', type=int, default=20, help='Value-only warmup iterations')
    parser.add_argument('--warmup-epochs', type=int, default=8, help='Update epochs during value warmup')
    parser.add_argument('--value-coeff', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Entropy bonus coefficient')
    parser.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clipping norm')
    parser.add_argument('--transitions-per-batch', type=int, default=32, help='Decision transitions per minibatch')
    parser.add_argument('--yellow-duration', type=int, default=3, help='Yellow transition duration')
    parser.add_argument('--min-green-steps', type=int, default=2, help='Minimum accepted green decision intervals')
    parser.add_argument('--demand-scale', type=float, default=1.0, help='Rollout demand multiplier')
    parser.add_argument('--global-reward-weight', type=float, default=0.1, help='Global delay-density reward weight')
    parser.add_argument('--reward-clip', type=float, default=1.0, help='Absolute per-decision reward limit')
    parser.add_argument('--teleport-penalty', type=float, default=0.0, help='Global reward penalty per teleport')
    parser.add_argument(
        '--max-teleports-per-rollout',
        type=int,
        default=999,
        help='Skip an optimizer update when its rollout exceeds this teleport count',
    )
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=-1,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    parser.add_argument('--target-kl', type=float, default=0.03, help='Stop PPO epochs above this approximate KL')
    parser.add_argument('--gui', action='store_true', help='Run PPO rollout collection in SUMO-GUI')
    parser.add_argument(
        '--initial-occupancy-min',
        type=float,
        default=0.05,
        help='Minimum sampled initial network occupancy',
    )
    parser.add_argument(
        '--initial-occupancy-max',
        type=float,
        default=0.08,
        help='Maximum sampled initial network occupancy',
    )
    parser.add_argument('--eval-every', type=int, default=10, help='Evaluate every N iterations')
    parser.add_argument('--eval-steps', type=int, default=600, help='Evaluation simulation seconds')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[42], help='Evaluation SUMO seeds')
    parser.add_argument(
        '--eval-policies',
        nargs='+',
        choices=tuple(policy.value for policy in EvaluationPolicy),
        default=[
            EvaluationPolicy.MAX_PRESSURE.value,
            EvaluationPolicy.QUEUE.value,
            EvaluationPolicy.LEARNED.value,
        ],
        help='Policies included in periodic evaluation',
    )
    parser.add_argument('--eval-demand-scale', type=float, default=1.0, help='Evaluation demand multiplier')
    parser.add_argument('--save-every', type=int, default=10, help='Save numbered checkpoint every N iterations')
    parser.add_argument('--print-every', type=int, default=1, help='Print every N iterations')
    parser.add_argument('--ckpt-dir', type=Path, default=None, help='Checkpoint directory')
    parser.add_argument('--log-dir', type=Path, default=None, help='TensorBoard log directory')
    parser.add_argument('--device', default='cpu', help='Torch device')
    parser.add_argument('--seed', type=int, default=42, help='Torch and SUMO seed')
    parser.add_argument(
        '--fixed-rollout-seed',
        type=int,
        default=None,
        help='Use the same SUMO demand and initial population for every rollout',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = args.resume_checkpoint.parent.name if args.resume_checkpoint is not None else stamp
    checkpoint_dir = args.ckpt_dir or ROOT / 'checkpoints' / 'rl' / run_name
    log_dir = args.log_dir or ROOT / 'runs' / 'rl' / run_name
    result = train_movement_ppo(
        MovementPpoConfig(
            cfg_path=args.cfg,
            il_checkpoint_path=args.il_checkpoint,
            iterations=args.iterations,
            steps_per_rollout=args.steps_per_rollout,
            rollouts_per_update=args.rollouts_per_update,
            num_workers=args.num_workers,
            decision_interval=args.decision_interval,
            learning_rate=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            clip_epsilon=args.clip,
            update_epochs=args.epochs,
            value_warmup_iterations=args.value_warmup_iterations,
            warmup_epochs=args.warmup_epochs,
            value_coefficient=args.value_coeff,
            entropy_coefficient=args.entropy_coeff,
            max_grad_norm=args.grad_clip,
            transitions_per_batch=args.transitions_per_batch,
            yellow_duration=args.yellow_duration,
            min_green_steps=args.min_green_steps,
            demand_scale=args.demand_scale,
            global_reward_weight=args.global_reward_weight,
            reward_clip=args.reward_clip,
            teleport_penalty=args.teleport_penalty,
            max_teleports_per_rollout=args.max_teleports_per_rollout,
            time_to_teleport=args.time_to_teleport,
            target_kl=args.target_kl,
            gui=args.gui,
            initial_occupancy_min=args.initial_occupancy_min,
            initial_occupancy_max=args.initial_occupancy_max,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            eval_seeds=tuple(args.eval_seeds),
            eval_policies=tuple(EvaluationPolicy(policy) for policy in args.eval_policies),
            eval_demand_scale=args.eval_demand_scale,
            save_every=args.save_every,
            print_every=args.print_every,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=args.device,
            seed=args.seed,
            fixed_rollout_seed=args.fixed_rollout_seed,
            resume_checkpoint_path=args.resume_checkpoint,
        )
    )
    print(f'PPO training complete: iterations={result.iterations} checkpoint={result.checkpoint_path}')


if __name__ == '__main__':
    main()
