"""Train a movement-score policy with PPO."""

from __future__ import annotations

import argparse
import builtins
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import sys
from typing import Sequence, TextIO

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.evaluation import EvaluationPolicy, LearnedEvaluationActionMode
from src.movement.experiment_config import (
    ExperimentConfiguration,
    load_experiment_configuration,
    resolve_experiment_path,
)
from src.movement.sumo_backend import SumoBackendKind
from src.movement.training.ppo import MovementPpoConfig, train_movement_ppo
from src.movement.training.ppo.types import (
    PpoRewardMode,
    PpoRewardObjective,
    PpoSpeedChangeMode,
    RolloutCity,
    ScratchModelConfig,
)

DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
DEFAULT_ITERATIONS = 300
DEFAULT_STEPS_PER_ROLLOUT = 360
DEFAULT_ROLLOUTS_PER_UPDATE = 3
DEFAULT_NUM_WORKERS = -1
DEFAULT_DECISION_INTERVAL = 10
DEFAULT_UPDATE_EPOCHS = 4
DEFAULT_VALUE_WARMUP_ITERATIONS = 20
DEFAULT_WARMUP_EPOCHS = 8
DEFAULT_TRANSITIONS_PER_BATCH = 32
DEFAULT_UPDATE_BATCH_WORKERS = 0
DEFAULT_WARMUP_STEPS = 0
DEFAULT_YELLOW_DURATION = 3
DEFAULT_MIN_GREEN_STEPS = 2
DEFAULT_TIME_TO_TELEPORT = -1
DEFAULT_INITIAL_OCCUPANCY_MIN = 0.05
DEFAULT_INITIAL_OCCUPANCY_MAX = 0.08
DEFAULT_REWARD_SAMPLE_INTERVAL = 5
DEFAULT_REWARD_MODE = PpoRewardMode.DELAY_DENSITY
DEFAULT_GLOBAL_REWARD_WEIGHT = 0.1
DEFAULT_FLOW_REWARD_WEIGHT = 0.1
DEFAULT_THROUGHPUT_REWARD_WEIGHT = 1.0
DEFAULT_PROGRESS_REWARD_WEIGHT = 0.03
DEFAULT_DISCHARGE_REWARD_WEIGHT = 0.0
DEFAULT_GRIDLOCK_PENALTY_WEIGHT = 0.02
DEFAULT_SPEED_CHANGE_WEIGHT = 0.02
DEFAULT_SPEED_CHANGE_MODE = PpoSpeedChangeMode.ABSOLUTE
DEFAULT_SWITCH_PENALTY_WEIGHT = 0.0
DEFAULT_ENTROPY_COEFFICIENT = 0.01
DEFAULT_SUMO_BACKEND = SumoBackendKind.LIBSUMO
DEFAULT_EVAL_EVERY = 10
DEFAULT_EVAL_WORKERS = 1
DEFAULT_EVAL_LEARNED_DEVICE = 'cpu'
DEFAULT_EVAL_LEARNED_ACTION_MODE = LearnedEvaluationActionMode.DETERMINISTIC
DEFAULT_EVAL_LEARNED_TEMPERATURE = 1.0
DEFAULT_SAVE_EVERY = 10
DEFAULT_SCRATCH_LANE_FEATURE_DIM = 29
DEFAULT_SCRATCH_MOVEMENT_FEATURE_DIM = 4
DEFAULT_SCRATCH_HIDDEN_DIM = 64
DEFAULT_SCRATCH_NUM_HOPS = 1
ORIGINAL_PRINT = builtins.print


@dataclass(frozen=True)
class PpoCommandSettings:
    iterations: int
    steps_per_rollout: int
    update_epochs: int
    value_warmup_iterations: int
    warmup_epochs: int
    transitions_per_batch: int
    action_samples_per_batch: int | None
    update_batch_workers: int
    warmup_steps: int
    decision_interval: int
    yellow_duration: int
    min_green_steps: int
    time_to_teleport: int
    initial_occupancy_min: float
    initial_occupancy_max: float
    eval_every: int
    eval_workers: int
    eval_learned_device: str
    eval_learned_action_mode: LearnedEvaluationActionMode
    eval_learned_temperature: float
    save_every: int


def install_timestamped_prints() -> None:
    builtins.print = timestamped_print


def timestamped_print(
    *values: object,
    sep: str | None = ' ',
    end: str | None = '\n',
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    ORIGINAL_PRINT(f'[{timestamp}]', *values, sep=sep, end=end, file=file, flush=flush)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PPO fine-tuning for movement-score traffic signal policies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--il-checkpoint', type=Path, help='IL movement checkpoint for a new PPO run')
    initialization.add_argument(
        '--scratch-checkpoint',
        type=Path,
        help='Movement checkpoint used only for architecture and normalizers; PPO weights start random',
    )
    initialization.add_argument(
        '--scratch-random',
        action='store_true',
        help='Start PPO from random actor-critic weights without loading any checkpoint',
    )
    initialization.add_argument(
        '--resume-checkpoint',
        type=Path,
        help='PPO checkpoint whose model, optimizer, RNG, and iteration state will be resumed',
    )
    parser.add_argument(
        '--allow-resume-config-mismatch',
        action='store_true',
        help='Resume a PPO checkpoint even when the current experiment YAML hash differs from the checkpoint',
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
    parser.add_argument(
        '--decision-interval',
        type=int,
        default=DEFAULT_DECISION_INTERVAL,
        help='Simulation seconds per decision',
    )
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
    parser.add_argument(
        '--entropy-coeff',
        type=float,
        default=DEFAULT_ENTROPY_COEFFICIENT,
        help='Entropy bonus coefficient',
    )
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
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help='Native SUMO simulation steps before policy control in training and evaluation',
    )
    parser.add_argument(
        '--yellow-duration',
        type=int,
        default=DEFAULT_YELLOW_DURATION,
        help='Yellow transition duration',
    )
    parser.add_argument(
        '--yellow-start-delay',
        type=int,
        default=None,
        help='Simulation seconds to retain the current green before starting yellow',
    )
    parser.add_argument(
        '--min-green-steps',
        type=int,
        default=DEFAULT_MIN_GREEN_STEPS,
        help='Minimum accepted green decision intervals',
    )
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
    parser.add_argument(
        '--global-reward-weight',
        type=float,
        default=DEFAULT_GLOBAL_REWARD_WEIGHT,
        help='Global delay-density reward weight',
    )
    parser.add_argument(
        '--reward-mode',
        choices=tuple(mode.value for mode in PpoRewardMode),
        default=DEFAULT_REWARD_MODE.value,
        help='PPO rollout reward formula',
    )
    parser.add_argument(
        '--flow-reward-weight',
        type=float,
        default=DEFAULT_FLOW_REWARD_WEIGHT,
        help='Global arrived-vehicle flow reward weight normalized per signal and simulated second',
    )
    parser.add_argument(
        '--throughput-reward-weight',
        type=float,
        default=DEFAULT_THROUGHPUT_REWARD_WEIGHT,
        help='Throughput-mode reward weight for vehicles arriving during the decision interval',
    )
    parser.add_argument(
        '--progress-reward-weight',
        type=float,
        default=DEFAULT_PROGRESS_REWARD_WEIGHT,
        help='Throughput-mode dense progress weight from local lane speed fractions',
    )
    parser.add_argument(
        '--discharge-reward-weight',
        type=float,
        default=DEFAULT_DISCHARGE_REWARD_WEIGHT,
        help='Throughput-mode local stop-line discharge-density reward weight',
    )
    parser.add_argument(
        '--gridlock-penalty-weight',
        type=float,
        default=DEFAULT_GRIDLOCK_PENALTY_WEIGHT,
        help='Throughput-mode local/global delay-density penalty weight',
    )
    parser.add_argument(
        '--speed-change-weight',
        type=float,
        default=DEFAULT_SPEED_CHANGE_WEIGHT,
        help='Auxiliary penalty weight for lane mean-speed changes on incoming lanes',
    )
    parser.add_argument(
        '--speed-change-mode',
        choices=tuple(mode.value for mode in PpoSpeedChangeMode),
        default=DEFAULT_SPEED_CHANGE_MODE.value,
        help='Whether speed-change cost penalizes absolute changes or braking only',
    )
    parser.add_argument(
        '--switch-penalty-weight',
        type=float,
        default=DEFAULT_SWITCH_PENALTY_WEIGHT,
        help='Penalty applied when a traffic light changes its accepted phase target',
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
        default=DEFAULT_TIME_TO_TELEPORT,
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
        default=DEFAULT_INITIAL_OCCUPANCY_MIN,
        help='Minimum sampled initial network occupancy',
    )
    parser.add_argument(
        '--initial-occupancy-max',
        type=float,
        default=DEFAULT_INITIAL_OCCUPANCY_MAX,
        help='Maximum sampled initial network occupancy',
    )
    parser.add_argument('--eval-every', type=int, default=None, help='Evaluate every N iterations')
    parser.add_argument(
        '--eval-workers',
        type=int,
        default=DEFAULT_EVAL_WORKERS,
        help='Parallel worker processes for periodic evaluation',
    )
    parser.add_argument(
        '--eval-learned-device',
        default=DEFAULT_EVAL_LEARNED_DEVICE,
        help='Torch device used by learned policy evaluation workers',
    )
    parser.add_argument(
        '--eval-learned-action-mode',
        choices=tuple(mode.value for mode in LearnedEvaluationActionMode),
        default=DEFAULT_EVAL_LEARNED_ACTION_MODE.value,
        help='Action selection used by learned policy evaluation',
    )
    parser.add_argument(
        '--eval-learned-temperature',
        type=float,
        default=DEFAULT_EVAL_LEARNED_TEMPERATURE,
        help='Softmax temperature for sampled learned policy evaluation',
    )
    parser.add_argument('--eval-steps', type=int, default=None, help='Evaluation simulation seconds')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=None, help='Evaluation SUMO seeds')
    parser.add_argument(
        '--eval-fixed-time-phase-duration',
        type=int,
        default=None,
        help='Seconds each phase remains selected by the fixed-time evaluation baseline',
    )
    parser.add_argument(
        '--eval-queue-pressure-phase-duration',
        type=int,
        default=None,
        help='Seconds between queue and max-pressure evaluation decisions',
    )
    parser.add_argument(
        '--eval-policies',
        nargs='+',
        choices=tuple(policy.value for policy in EvaluationPolicy),
        default=None,
        help='Policies included in periodic evaluation',
    )
    parser.add_argument('--eval-demand-scale', type=float, default=None, help='Evaluation demand multiplier')
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
        '--scratch-lane-feature-dim',
        type=int,
        default=DEFAULT_SCRATCH_LANE_FEATURE_DIM,
        help='Lane feature dimension for --scratch-random',
    )
    parser.add_argument(
        '--scratch-movement-feature-dim',
        type=int,
        default=DEFAULT_SCRATCH_MOVEMENT_FEATURE_DIM,
        help='Movement feature dimension for --scratch-random',
    )
    parser.add_argument(
        '--scratch-hidden-dim',
        type=int,
        default=DEFAULT_SCRATCH_HIDDEN_DIM,
        help='GNN hidden dimension for --scratch-random',
    )
    parser.add_argument(
        '--scratch-num-hops',
        type=int,
        default=DEFAULT_SCRATCH_NUM_HOPS,
        help='GNN message-passing hops for --scratch-random',
    )
    parser.add_argument(
        '--fixed-rollout-seed',
        type=int,
        default=None,
        help='Use the same SUMO demand and initial population for every rollout',
    )
    args = parser.parse_args()
    args.rollouts_per_update_explicit = cli_option_was_passed(
        option_name='--rollouts-per-update',
        arguments=sys.argv[1:],
    )
    return args


def cli_option_was_passed(option_name: str, arguments: Sequence[str]) -> bool:
    option_prefix = f'{option_name}='
    return any(argument == option_name or argument.startswith(option_prefix) for argument in arguments)


def initialization_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.scratch_checkpoint is not None:
        return args.scratch_checkpoint
    return args.il_checkpoint


def scratch_model_config(args: argparse.Namespace) -> ScratchModelConfig | None:
    if not args.scratch_random:
        return None
    return ScratchModelConfig(
        lane_feature_dim=args.scratch_lane_feature_dim,
        movement_feature_dim=args.scratch_movement_feature_dim,
        hidden_dim=args.scratch_hidden_dim,
        num_hops=args.scratch_num_hops,
    )


def main() -> None:
    install_timestamped_prints()
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
            il_checkpoint_path=initialization_checkpoint_path(args),
            scratch_model_config=scratch_model_config(args),
            scratch_initialization=args.scratch_checkpoint is not None,
            iterations=ppo_settings.iterations,
            steps_per_rollout=ppo_settings.steps_per_rollout,
            rollouts_per_update=rollout_count,
            num_workers=worker_count,
            decision_interval=ppo_settings.decision_interval,
            learning_rate=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            clip_epsilon=args.clip,
            update_epochs=ppo_settings.update_epochs,
            value_warmup_iterations=ppo_settings.value_warmup_iterations,
            warmup_epochs=ppo_settings.warmup_epochs,
            value_coefficient=args.value_coeff,
            entropy_coefficient=entropy_coefficient(args, experiment_configuration),
            max_grad_norm=args.grad_clip,
            transitions_per_batch=ppo_settings.transitions_per_batch,
            update_batch_workers=ppo_settings.update_batch_workers,
            warmup_steps=ppo_settings.warmup_steps,
            yellow_duration=ppo_settings.yellow_duration,
            yellow_start_delay=yellow_start_delay(args, experiment_configuration),
            min_green_steps=ppo_settings.min_green_steps,
            demand_scale_min=demand_scale_min,
            demand_scale_max=demand_scale_max,
            global_reward_weight=global_reward_weight(args, experiment_configuration),
            flow_reward_weight=flow_reward_weight(args, experiment_configuration),
            reward_mode=reward_mode(args, experiment_configuration),
            reward_objective=reward_objective(experiment_configuration),
            throughput_reward_weight=throughput_reward_weight(args, experiment_configuration),
            progress_reward_weight=progress_reward_weight(args, experiment_configuration),
            discharge_reward_weight=discharge_reward_weight(args, experiment_configuration),
            gridlock_penalty_weight=gridlock_penalty_weight(args, experiment_configuration),
            speed_change_weight=speed_change_weight(args, experiment_configuration),
            speed_change_mode=speed_change_mode(args, experiment_configuration),
            switch_penalty_weight=switch_penalty_weight(args, experiment_configuration),
            reward_sample_interval=reward_sample_interval(args, experiment_configuration),
            reward_clip=args.reward_clip,
            teleport_penalty=args.teleport_penalty,
            max_teleports_per_rollout=args.max_teleports_per_rollout,
            time_to_teleport=ppo_settings.time_to_teleport,
            target_kl=args.target_kl,
            gui=args.gui,
            sumo_backend=SumoBackendKind(args.sumo_backend),
            initial_occupancy_min=ppo_settings.initial_occupancy_min,
            initial_occupancy_max=ppo_settings.initial_occupancy_max,
            eval_every=ppo_settings.eval_every,
            eval_steps=evaluation_steps(args, experiment_configuration),
            eval_seeds=evaluation_seeds(args, experiment_configuration),
            eval_policies=evaluation_policies(args, experiment_configuration),
            eval_fixed_time_phase_duration=evaluation_fixed_time_phase_duration(args, experiment_configuration),
            eval_queue_pressure_phase_duration=evaluation_queue_pressure_phase_duration(args, experiment_configuration),
            eval_worker_count=ppo_settings.eval_workers,
            eval_learned_device=ppo_settings.eval_learned_device,
            eval_learned_action_mode=ppo_settings.eval_learned_action_mode,
            eval_learned_temperature=ppo_settings.eval_learned_temperature,
            eval_demand_scale=evaluation_demand_scale(args),
            eval_demand_scales=evaluation_demand_scales(args, experiment_configuration),
            save_every=ppo_settings.save_every,
            print_every=args.print_every,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=args.device,
            seed=args.seed,
            fixed_rollout_seed=args.fixed_rollout_seed,
            resume_checkpoint_path=args.resume_checkpoint,
            allow_resume_config_mismatch=args.allow_resume_config_mismatch,
            rollout_cities=rollout_cities,
            experiment_configuration=experiment_configuration,
            experiment_configuration_path=args.experiment_config,
            experiment_configuration_text=experiment_configuration_text(args.experiment_config),
            experiment_configuration_sha256=experiment_configuration_sha256(args.experiment_config),
            project_root=ROOT,
            action_samples_per_batch=ppo_settings.action_samples_per_batch,
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


def reward_objective(
    experiment_configuration: ExperimentConfiguration | None,
) -> PpoRewardObjective:
    if experiment_configuration is None:
        return PpoRewardObjective.MAXIMIZE
    return PpoRewardObjective(experiment_configuration.proximal_policy_optimization.reward_objective.value)


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
            action_samples_per_batch=None,
            update_batch_workers=args.update_batch_workers,
            warmup_steps=args.warmup_steps,
            decision_interval=args.decision_interval,
            yellow_duration=args.yellow_duration,
            min_green_steps=args.min_green_steps,
            time_to_teleport=args.time_to_teleport,
            initial_occupancy_min=args.initial_occupancy_min,
            initial_occupancy_max=args.initial_occupancy_max,
            eval_every=args.eval_every if args.eval_every is not None else DEFAULT_EVAL_EVERY,
            eval_workers=args.eval_workers,
            eval_learned_device=args.eval_learned_device,
            eval_learned_action_mode=LearnedEvaluationActionMode(args.eval_learned_action_mode),
            eval_learned_temperature=args.eval_learned_temperature,
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
        action_samples_per_batch=ppo.action_samples_per_batch,
        update_batch_workers=(
            ppo.update_batch_workers
            if args.update_batch_workers == DEFAULT_UPDATE_BATCH_WORKERS
            else args.update_batch_workers
        ),
        warmup_steps=(
            experiment_configuration.simulation.warmup_steps
            if args.warmup_steps == DEFAULT_WARMUP_STEPS
            else args.warmup_steps
        ),
        decision_interval=(
            experiment_configuration.simulation.decision_interval
            if args.decision_interval == DEFAULT_DECISION_INTERVAL
            else args.decision_interval
        ),
        yellow_duration=(
            experiment_configuration.simulation.yellow_duration
            if args.yellow_duration == DEFAULT_YELLOW_DURATION
            else args.yellow_duration
        ),
        min_green_steps=(
            experiment_configuration.simulation.minimum_green_steps
            if args.min_green_steps == DEFAULT_MIN_GREEN_STEPS
            else args.min_green_steps
        ),
        time_to_teleport=(
            experiment_configuration.simulation.time_to_teleport
            if args.time_to_teleport == DEFAULT_TIME_TO_TELEPORT
            else args.time_to_teleport
        ),
        initial_occupancy_min=(
            experiment_configuration.simulation.minimum_initial_occupancy
            if args.initial_occupancy_min == DEFAULT_INITIAL_OCCUPANCY_MIN
            else args.initial_occupancy_min
        ),
        initial_occupancy_max=(
            experiment_configuration.simulation.maximum_initial_occupancy
            if args.initial_occupancy_max == DEFAULT_INITIAL_OCCUPANCY_MAX
            else args.initial_occupancy_max
        ),
        eval_every=ppo.evaluate_every_iterations if args.eval_every is None else args.eval_every,
        eval_workers=ppo.evaluation_workers if args.eval_workers == DEFAULT_EVAL_WORKERS else args.eval_workers,
        eval_learned_device=(
            ppo.evaluation_learned_device
            if args.eval_learned_device == DEFAULT_EVAL_LEARNED_DEVICE
            else args.eval_learned_device
        ),
        eval_learned_action_mode=eval_learned_action_mode(args, experiment_configuration),
        eval_learned_temperature=eval_learned_temperature(args, experiment_configuration),
        save_every=ppo.save_every_iterations if args.save_every == DEFAULT_SAVE_EVERY else args.save_every,
    )


def eval_learned_action_mode(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration,
) -> LearnedEvaluationActionMode:
    if args.eval_learned_action_mode == DEFAULT_EVAL_LEARNED_ACTION_MODE.value:
        return LearnedEvaluationActionMode(
            experiment_configuration.proximal_policy_optimization.evaluation_learned_action_mode.value
        )
    return LearnedEvaluationActionMode(args.eval_learned_action_mode)


def eval_learned_temperature(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration,
) -> float:
    if args.eval_learned_temperature == DEFAULT_EVAL_LEARNED_TEMPERATURE:
        return experiment_configuration.proximal_policy_optimization.evaluation_learned_temperature
    return args.eval_learned_temperature


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
            rollout_workers=city.rollout_jobs_per_iteration,
            rollout_priority=city.rollout_priority,
            demand_scale_min=city.minimum_train_scale,
            demand_scale_max=city.maximum_train_scale,
        )
        for city in experiment_configuration.train_cities
    )


def rollouts_per_update(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if experiment_configuration is not None and not args.rollouts_per_update_explicit:
        return experiment_rollout_jobs_per_iteration(experiment_configuration)
    return args.rollouts_per_update


def experiment_rollout_jobs_per_iteration(experiment_configuration: ExperimentConfiguration) -> int:
    rollout_jobs = sum(city.rollout_jobs_per_iteration for city in experiment_configuration.train_cities)
    if rollout_jobs <= 0:
        raise ValueError('experiment train cities must define at least one rollout job per PPO iteration')
    return rollout_jobs


def reward_sample_interval(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if experiment_configuration is not None and args.reward_sample_interval == DEFAULT_REWARD_SAMPLE_INTERVAL:
        return experiment_configuration.proximal_policy_optimization.reward_sample_interval
    return args.reward_sample_interval


def reward_mode(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> PpoRewardMode:
    if experiment_configuration is not None and args.reward_mode == DEFAULT_REWARD_MODE.value:
        return PpoRewardMode(experiment_configuration.proximal_policy_optimization.reward_mode.value)
    return PpoRewardMode(args.reward_mode)


def global_reward_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.global_reward_weight == DEFAULT_GLOBAL_REWARD_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.global_reward_weight
    return args.global_reward_weight


def flow_reward_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.flow_reward_weight == DEFAULT_FLOW_REWARD_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.flow_reward_weight
    return args.flow_reward_weight


def throughput_reward_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.throughput_reward_weight == DEFAULT_THROUGHPUT_REWARD_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.throughput_reward_weight
    return args.throughput_reward_weight


def progress_reward_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.progress_reward_weight == DEFAULT_PROGRESS_REWARD_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.progress_reward_weight
    return args.progress_reward_weight


def discharge_reward_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.discharge_reward_weight == DEFAULT_DISCHARGE_REWARD_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.discharge_reward_weight
    return args.discharge_reward_weight


def gridlock_penalty_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.gridlock_penalty_weight == DEFAULT_GRIDLOCK_PENALTY_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.gridlock_penalty_weight
    return args.gridlock_penalty_weight


def speed_change_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.speed_change_weight == DEFAULT_SPEED_CHANGE_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.speed_change_weight
    return args.speed_change_weight


def speed_change_mode(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> PpoSpeedChangeMode:
    if experiment_configuration is not None and args.speed_change_mode == DEFAULT_SPEED_CHANGE_MODE.value:
        return PpoSpeedChangeMode(experiment_configuration.proximal_policy_optimization.speed_change_mode.value)
    return PpoSpeedChangeMode(args.speed_change_mode)


def switch_penalty_weight(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.switch_penalty_weight == DEFAULT_SWITCH_PENALTY_WEIGHT:
        return experiment_configuration.proximal_policy_optimization.switch_penalty_weight
    return args.switch_penalty_weight


def entropy_coefficient(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> float:
    if experiment_configuration is not None and args.entropy_coeff == DEFAULT_ENTROPY_COEFFICIENT:
        return experiment_configuration.proximal_policy_optimization.entropy_coefficient
    return args.entropy_coeff


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
    if experiment_configuration is not None and args.eval_demand_scale is None:
        return experiment_configuration.demand.evaluation_scales
    return (evaluation_demand_scale(args),)


def evaluation_demand_scale(args: argparse.Namespace) -> float:
    if args.eval_demand_scale is None:
        return 1.0
    return args.eval_demand_scale


def evaluation_steps(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if args.eval_steps is not None:
        return args.eval_steps
    if experiment_configuration is not None:
        return experiment_configuration.evaluation.steps
    return 600


def yellow_start_delay(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if args.yellow_start_delay is not None:
        return args.yellow_start_delay
    if experiment_configuration is not None:
        return experiment_configuration.simulation.yellow_start_delay
    return 0


def evaluation_seeds(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> tuple[int, ...]:
    if args.eval_seeds is not None:
        return tuple(args.eval_seeds)
    if experiment_configuration is not None:
        return experiment_configuration.evaluation.seeds
    return (42,)


def evaluation_fixed_time_phase_duration(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if args.eval_fixed_time_phase_duration is not None:
        return args.eval_fixed_time_phase_duration
    if experiment_configuration is not None:
        return experiment_configuration.evaluation.fixed_time_phase_duration
    return 10


def evaluation_queue_pressure_phase_duration(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> int:
    if args.eval_queue_pressure_phase_duration is not None:
        return args.eval_queue_pressure_phase_duration
    if experiment_configuration is not None:
        return experiment_configuration.evaluation.queue_pressure_phase_duration
    return 10


def evaluation_policies(
    args: argparse.Namespace,
    experiment_configuration: ExperimentConfiguration | None,
) -> tuple[EvaluationPolicy, ...]:
    if args.eval_policies is not None:
        return tuple(EvaluationPolicy(policy) for policy in args.eval_policies)
    if experiment_configuration is not None:
        return tuple(EvaluationPolicy(policy.value) for policy in experiment_configuration.evaluation.policies)
    return (
        EvaluationPolicy.MAX_PRESSURE,
        EvaluationPolicy.QUEUE,
        EvaluationPolicy.LEARNED,
    )


if __name__ == '__main__':
    main()
