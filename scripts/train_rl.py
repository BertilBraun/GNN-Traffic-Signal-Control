"""Train a movement-score policy with PPO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.evaluation import EvaluationPolicy
from src.movement.experiment_config import (
    ExperimentConfiguration,
    load_experiment_configuration,
    resolve_experiment_path,
)
from src.movement.sumo_backend import SumoBackendKind
from src.movement.training.ppo import MovementPpoConfig, train_movement_ppo
from src.movement.training.ppo.types import RolloutCity

DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
DEFAULT_ITERATIONS = 300
DEFAULT_STEPS_PER_ROLLOUT = 360
DEFAULT_ROLLOUTS_PER_UPDATE = 3
DEFAULT_NUM_WORKERS = -1
DEFAULT_UPDATE_EPOCHS = 4
DEFAULT_VALUE_WARMUP_ITERATIONS = 20
DEFAULT_WARMUP_EPOCHS = 8
DEFAULT_TRANSITIONS_PER_BATCH = 32
DEFAULT_UPDATE_BATCH_WORKERS = 0
DEFAULT_REWARD_SAMPLE_INTERVAL = 5
DEFAULT_SUMO_BACKEND = SumoBackendKind.LIBSUMO
DEFAULT_EVAL_EVERY = 10
DEFAULT_SAVE_EVERY = 10


@dataclass(frozen=True)
class PpoCommandSettings:
    iterations: int
    steps_per_rollout: int
    update_epochs: int
    value_warmup_iterations: int
    warmup_epochs: int
    transitions_per_batch: int
    update_batch_workers: int
    eval_every: int
    save_every: int


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
    parser.add_argument('--experiment-config', type=Path, default=None, help='Multi-city experiment YAML path')
    parser.add_argument('--cfg', type=Path, default=DEFAULT_CFG, help='SUMO .sumocfg path')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help='Final target PPO iteration')
    parser.add_argument(
        '--steps-per-rollout',
        type=int,
        default=DEFAULT_STEPS_PER_ROLLOUT,
        help='Decision steps collected per iteration',
    )
    parser.add_argument(
        '--rollouts-per-update',
        type=int,
        default=DEFAULT_ROLLOUTS_PER_UPDATE,
        help='Independent rollouts collected per PPO update',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help='Parallel SUMO rollout worker processes; -1 uses all CPUs',
    )
    parser.add_argument('--decision-interval', type=int, default=10, help='Simulation seconds per decision')
    parser.add_argument('--lr', type=float, default=2e-4, help='Adam learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip', type=float, default=0.1, help='PPO clipping epsilon')
    parser.add_argument('--epochs', type=int, default=DEFAULT_UPDATE_EPOCHS, help='PPO update epochs')
    parser.add_argument(
        '--value-warmup-iterations',
        type=int,
        default=DEFAULT_VALUE_WARMUP_ITERATIONS,
        help='Value-only warmup iterations',
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help='Update epochs during value warmup',
    )
    parser.add_argument('--value-coeff', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Entropy bonus coefficient')
    parser.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clipping norm')
    parser.add_argument(
        '--transitions-per-batch',
        type=int,
        default=DEFAULT_TRANSITIONS_PER_BATCH,
        help='Decision transitions per minibatch',
    )
    parser.add_argument(
        '--update-batch-workers',
        type=int,
        default=DEFAULT_UPDATE_BATCH_WORKERS,
        help='PyTorch DataLoader workers for packing PPO update minibatches',
    )
    parser.add_argument('--yellow-duration', type=int, default=3, help='Yellow transition duration')
    parser.add_argument('--min-green-steps', type=int, default=2, help='Minimum accepted green decision intervals')
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=1.0,
        help='Fixed rollout demand multiplier used when min/max are not set',
    )
    parser.add_argument(
        '--demand-scale-min',
        type=float,
        default=None,
        help='Minimum rollout demand multiplier sampled per rollout',
    )
    parser.add_argument(
        '--demand-scale-max',
        type=float,
        default=None,
        help='Maximum rollout demand multiplier sampled per rollout',
    )
    parser.add_argument('--global-reward-weight', type=float, default=0.1, help='Global delay-density reward weight')
    parser.add_argument(
        '--speed-change-weight',
        type=float,
        default=0.02,
        help='Auxiliary penalty weight for lane mean-speed changes on incoming lanes',
    )
    parser.add_argument(
        '--reward-sample-interval',
        type=int,
        default=DEFAULT_REWARD_SAMPLE_INTERVAL,
        help='SUMO simulation steps between reward TraCI lane/vehicle queries',
    )
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
        '--sumo-backend',
        choices=tuple(backend.value for backend in SumoBackendKind),
        default=DEFAULT_SUMO_BACKEND.value,
        help='TraCI-compatible backend used for PPO rollout collection',
    )
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
    parser.add_argument('--eval-every', type=int, default=DEFAULT_EVAL_EVERY, help='Evaluate every N iterations')
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
    parser.add_argument(
        '--save-every',
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help='Save numbered checkpoint every N iterations',
    )
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
    experiment_configuration = experiment_config(args.experiment_config)
    demand_scale_min, demand_scale_max = demand_scale_bounds(args, experiment_configuration)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = args.resume_checkpoint.parent.name if args.resume_checkpoint is not None else stamp
    checkpoint_dir = args.ckpt_dir or ROOT / 'checkpoints' / 'rl' / run_name
    log_dir = args.log_dir or ROOT / 'runs' / 'rl' / run_name
    rollout_cities = experiment_rollout_cities(experiment_configuration)
    ppo_settings = ppo_command_settings(args=args, experiment_configuration=experiment_configuration)
    rollout_count = rollouts_per_update(args, experiment_configuration)
    worker_count = num_workers(args, experiment_configuration)
    result = train_movement_ppo(
        MovementPpoConfig(
            cfg_path=cfg_path(args, experiment_configuration),
            il_checkpoint_path=args.il_checkpoint,
            iterations=ppo_settings.iterations,
            steps_per_rollout=ppo_settings.steps_per_rollout,
            rollouts_per_update=rollout_count,
            num_workers=worker_count,
            decision_interval=args.decision_interval,
            learning_rate=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            clip_epsilon=args.clip,
            update_epochs=ppo_settings.update_epochs,
            value_warmup_iterations=ppo_settings.value_warmup_iterations,
            warmup_epochs=ppo_settings.warmup_epochs,
            value_coefficient=args.value_coeff,
            entropy_coefficient=args.entropy_coeff,
            max_grad_norm=args.grad_clip,
            transitions_per_batch=ppo_settings.transitions_per_batch,
            update_batch_workers=ppo_settings.update_batch_workers,
            yellow_duration=args.yellow_duration,
            min_green_steps=args.min_green_steps,
            demand_scale_min=demand_scale_min,
            demand_scale_max=demand_scale_max,
            global_reward_weight=args.global_reward_weight,
            speed_change_weight=args.speed_change_weight,
            reward_sample_interval=reward_sample_interval(args, experiment_configuration),
            reward_clip=args.reward_clip,
            teleport_penalty=args.teleport_penalty,
            max_teleports_per_rollout=args.max_teleports_per_rollout,
            time_to_teleport=args.time_to_teleport,
            target_kl=args.target_kl,
            gui=args.gui,
            sumo_backend=SumoBackendKind(args.sumo_backend),
            initial_occupancy_min=args.initial_occupancy_min,
            initial_occupancy_max=args.initial_occupancy_max,
            eval_every=ppo_settings.eval_every,
            eval_steps=args.eval_steps,
            eval_seeds=tuple(args.eval_seeds),
            eval_policies=tuple(EvaluationPolicy(policy) for policy in args.eval_policies),
            eval_demand_scale=args.eval_demand_scale,
            eval_demand_scales=evaluation_demand_scales(args, experiment_configuration),
            save_every=ppo_settings.save_every,
            print_every=args.print_every,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=args.device,
            seed=args.seed,
            fixed_rollout_seed=args.fixed_rollout_seed,
            resume_checkpoint_path=args.resume_checkpoint,
            rollout_cities=rollout_cities,
            experiment_configuration=experiment_configuration,
            experiment_configuration_path=args.experiment_config,
            experiment_configuration_text=experiment_configuration_text(args.experiment_config),
            experiment_configuration_sha256=experiment_configuration_sha256(args.experiment_config),
            project_root=ROOT,
        )
    )
    print(f'PPO training complete: iterations={result.iterations} checkpoint={result.checkpoint_path}')


def experiment_config(experiment_configuration_path: Path | None) -> ExperimentConfiguration | None:
    if experiment_configuration_path is None:
        return None
    return load_experiment_configuration(
        configuration_path=experiment_configuration_path,
        project_root=ROOT,
    )


def experiment_configuration_text(experiment_configuration_path: Path | None) -> str | None:
    if experiment_configuration_path is None:
        return None
    return experiment_configuration_path.read_text(encoding='utf-8-sig')


def experiment_configuration_sha256(experiment_configuration_path: Path | None) -> str | None:
    configuration_text = experiment_configuration_text(experiment_configuration_path)
    if configuration_text is None:
        return None
    return sha256(configuration_text.encode('utf-8')).hexdigest()


def demand_scale_bounds(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> tuple[float, float]:
    if (
        experiment_configuration is not None
        and args.demand_scale == 1.0
        and args.demand_scale_min is None
        and args.demand_scale_max is None
    ):
        return (
            experiment_configuration.demand.minimum_train_scale,
            experiment_configuration.demand.maximum_train_scale,
        )
    return (
        args.demand_scale_min if args.demand_scale_min is not None else args.demand_scale,
        args.demand_scale_max if args.demand_scale_max is not None else args.demand_scale,
    )


def cfg_path(args: argparse.Namespace, experiment_configuration: ExperimentConfiguration | None) -> Path:
    if experiment_configuration is None:
        return args.cfg
    return resolve_experiment_path(
        path=experiment_configuration.train_cities[0].sumo_config,
        project_root=ROOT,
    )


def ppo_command_settings(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> PpoCommandSettings:
    if experiment_configuration is None:
        return PpoCommandSettings(
            iterations=args.iterations,
            steps_per_rollout=args.steps_per_rollout,
            update_epochs=args.epochs,
            value_warmup_iterations=args.value_warmup_iterations,
            warmup_epochs=args.warmup_epochs,
            transitions_per_batch=args.transitions_per_batch,
            update_batch_workers=args.update_batch_workers,
            eval_every=args.eval_every,
            save_every=args.save_every,
        )
    ppo = experiment_configuration.proximal_policy_optimization
    return PpoCommandSettings(
        iterations=ppo.iterations if args.iterations == DEFAULT_ITERATIONS else args.iterations,
        steps_per_rollout=(
            ppo.steps_per_rollout if args.steps_per_rollout == DEFAULT_STEPS_PER_ROLLOUT else args.steps_per_rollout
        ),
        update_epochs=ppo.update_epochs if args.epochs == DEFAULT_UPDATE_EPOCHS else args.epochs,
        value_warmup_iterations=(
            ppo.value_warmup_iterations
            if args.value_warmup_iterations == DEFAULT_VALUE_WARMUP_ITERATIONS
            else args.value_warmup_iterations
        ),
        warmup_epochs=ppo.warmup_epochs if args.warmup_epochs == DEFAULT_WARMUP_EPOCHS else args.warmup_epochs,
        transitions_per_batch=(
            ppo.transitions_per_batch
            if args.transitions_per_batch == DEFAULT_TRANSITIONS_PER_BATCH
            else args.transitions_per_batch
        ),
        update_batch_workers=(
            ppo.update_batch_workers
            if args.update_batch_workers == DEFAULT_UPDATE_BATCH_WORKERS
            else args.update_batch_workers
        ),
        eval_every=ppo.evaluate_every_iterations if args.eval_every == DEFAULT_EVAL_EVERY else args.eval_every,
        save_every=ppo.save_every_iterations if args.save_every == DEFAULT_SAVE_EVERY else args.save_every,
    )


def experiment_rollout_cities(
    experiment_configuration: ExperimentConfiguration | None,
) -> tuple[RolloutCity, ...]:
    if experiment_configuration is None:
        return ()
    return tuple(
        RolloutCity(
            city_name=city.name,
            city_split=city.split,
            sumo_config_path=resolve_experiment_path(path=city.sumo_config, project_root=ROOT),
            rollout_workers=city.rollout_workers,
        )
        for city in experiment_configuration.train_cities
    )


def rollouts_per_update(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if experiment_configuration is not None and args.rollouts_per_update == DEFAULT_ROLLOUTS_PER_UPDATE:
        return experiment_configuration.proximal_policy_optimization.rollouts_per_update
    return args.rollouts_per_update


def reward_sample_interval(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if experiment_configuration is not None and args.reward_sample_interval == DEFAULT_REWARD_SAMPLE_INTERVAL:
        return experiment_configuration.proximal_policy_optimization.reward_sample_interval
    return args.reward_sample_interval


def num_workers(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if experiment_configuration is not None and args.num_workers == DEFAULT_NUM_WORKERS:
        return experiment_configuration.proximal_policy_optimization.rollout_workers
    if args.num_workers == -1:
        cpu_count = os.cpu_count()
        return cpu_count if cpu_count is not None else 1
    return args.num_workers


def evaluation_demand_scales(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> tuple[float, ...]:
    if experiment_configuration is not None and args.eval_demand_scale == 1.0:
        return experiment_configuration.demand.evaluation_scales
    return (args.eval_demand_scale,)


if __name__ == '__main__':
    main()
