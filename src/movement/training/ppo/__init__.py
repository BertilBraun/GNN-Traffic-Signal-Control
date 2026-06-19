"""PPO training for movement-score traffic signal policies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.tensorboard import SummaryWriter

from src.movement.training.ppo.checkpoint import (
    load_movement_ppo_checkpoint,
    save_actor_checkpoint,
    save_ppo_checkpoint,
)
from src.movement.training.ppo.evaluation import (
    run_training_evaluation,
    write_training_scalars,
)
from src.movement.training.ppo.rollout import (
    RolloutCollectionRequest,
    collect_computed_rollouts,
)
from src.movement.training.ppo.stats import combine_rollout_stats, training_diagnostics
from src.movement.training.ppo.state import PpoTrainingState, create_rollout_pool, initialize_training_state
from src.movement.training.ppo.types import (
    MovementPpoConfig,
    MovementPpoTrainingResult,
    RolloutStats,
    TrainingDiagnostics,
)
from src.movement.training.ppo.update import (
    PpoUpdateStats,
    set_actor_grad,
    set_value_grad,
    skipped_update_stats,
    update_ppo,
)
from src.movement.training.rollout import MovementRolloutBuffer

__all__ = [
    'MovementPpoConfig',
    'MovementPpoTrainingResult',
    'load_movement_ppo_checkpoint',
    'train_movement_ppo',
]


@dataclass(frozen=True)
class IterationTiming:
    iteration_started: float
    rollout_finished: float
    update_finished: float


def train_movement_ppo(config: MovementPpoConfig) -> MovementPpoTrainingResult:
    """Fine-tune a movement scorer with PPO."""
    validate_config(config)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    state = initialize_training_state(config)
    writer = SummaryWriter(log_dir=str(config.log_dir))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pool = create_rollout_pool(config)
    started = perf_counter()
    last_checkpoint_path = config.checkpoint_dir / 'movement_ppo_latest.pt'
    try:
        run_training_iterations(
            config=config,
            state=state,
            writer=writer,
            device=device,
            pool=pool,
            started=started,
        )
    finally:
        save_latest_outputs(
            config=config,
            state=state,
            checkpoint_path=last_checkpoint_path,
        )
        writer.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    return MovementPpoTrainingResult(
        checkpoint_path=last_checkpoint_path,
        iterations=state.completed_iteration,
    )


def validate_config(config: MovementPpoConfig) -> None:
    if config.demand_scale_min <= 0.0:
        raise ValueError('demand_scale_min must be positive.')
    if config.demand_scale_max <= 0.0:
        raise ValueError('demand_scale_max must be positive.')
    if config.demand_scale_min > config.demand_scale_max:
        raise ValueError('demand_scale_min must not exceed demand_scale_max.')
    if config.max_teleports_per_rollout < 0:
        raise ValueError('max_teleports_per_rollout must not be negative.')
    if config.target_kl <= 0.0:
        raise ValueError('target_kl must be positive.')
    if config.rollouts_per_update <= 0:
        raise ValueError('rollouts_per_update must be positive.')
    if config.num_workers <= 0:
        raise ValueError('num_workers must be positive.')
    if config.gui and config.num_workers > 1:
        raise ValueError('SUMO-GUI rollout collection is only supported with one worker.')


def run_training_iterations(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    writer: SummaryWriter,
    device: torch.device,
    pool: ProcessPoolExecutor | None,
    started: float,
) -> None:
    first_iteration = state.completed_iteration + 1
    validate_iteration_range(config=config, first_iteration=first_iteration)
    if config.resume_checkpoint_path is not None:
        print(f'Resuming PPO from iteration {state.completed_iteration}; target iteration={config.iterations}')
    for iteration in range(first_iteration, config.iterations + 1):
        run_training_iteration(
            config=config,
            state=state,
            writer=writer,
            device=device,
            pool=pool,
            started=started,
            iteration=iteration,
        )


def validate_iteration_range(config: MovementPpoConfig, first_iteration: int) -> None:
    if first_iteration <= config.iterations:
        return
    completed_iteration = first_iteration - 1
    raise ValueError(
        f'Resume checkpoint is already at iteration {completed_iteration}, '
        f'which is not below target iteration {config.iterations}.'
    )


def run_training_iteration(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    writer: SummaryWriter,
    device: torch.device,
    pool: ProcessPoolExecutor | None,
    started: float,
    iteration: int,
) -> None:
    iteration_started = perf_counter()
    warming_up = iteration <= config.value_warmup_iterations
    set_actor_grad(state.model, requires_grad=not warming_up)
    set_value_grad(state.model, requires_grad=True)
    buffer, rollout_stats = collect_iteration_rollouts(
        config=config,
        state=state,
        device=device,
        iteration=iteration,
        warming_up=warming_up,
        pool=pool,
    )
    rollout_finished = perf_counter()
    diagnostics = training_diagnostics(buffer)
    update_skipped = rollout_stats.teleport_count > config.max_teleports_per_rollout
    update_stats = update_iteration_model(
        config=config,
        state=state,
        buffer=buffer,
        device=device,
        warming_up=warming_up,
        update_skipped=update_skipped,
    )
    update_finished = perf_counter()
    timing = IterationTiming(
        iteration_started=iteration_started,
        rollout_finished=rollout_finished,
        update_finished=update_finished,
    )
    write_iteration_outputs(
        config=config,
        writer=writer,
        iteration=iteration,
        rollout_stats=rollout_stats,
        diagnostics=diagnostics,
        update_stats=update_stats,
        update_skipped=update_skipped,
        timing=timing,
        warming_up=warming_up,
        started=started,
    )
    state.completed_iteration = iteration
    maybe_evaluate_iteration(
        config=config,
        state=state,
        writer=writer,
        iteration=iteration,
    )
    maybe_save_numbered_checkpoint(config=config, state=state, iteration=iteration)


def collect_iteration_rollouts(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    device: torch.device,
    iteration: int,
    warming_up: bool,
    pool: ProcessPoolExecutor | None,
) -> tuple[MovementRolloutBuffer, RolloutStats]:
    collected_rollouts = collect_computed_rollouts(
        RolloutCollectionRequest(
            config=config,
            model=state.model,
            metadata=state.metadata,
            lane_normalizer=state.lane_normalizer,
            movement_normalizer=state.movement_normalizer,
            device=device,
            iteration=iteration,
            warming_up=warming_up,
            pool=pool,
        )
    )
    buffer = MovementRolloutBuffer.concatenate_computed(tuple(collected.buffer for collected in collected_rollouts))
    rollout_stats = combine_rollout_stats(tuple(collected.stats for collected in collected_rollouts))
    return buffer, rollout_stats


def update_iteration_model(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    buffer: MovementRolloutBuffer,
    device: torch.device,
    warming_up: bool,
    update_skipped: bool,
) -> PpoUpdateStats:
    if update_skipped:
        return skipped_update_stats()
    return update_ppo(
        model=state.model,
        optimizer=state.optimizer,
        buffer=buffer,
        lane_normalizer=state.lane_normalizer,
        movement_normalizer=state.movement_normalizer,
        device=device,
        config=config,
        warming_up=warming_up,
    )


def write_iteration_outputs(
    config: MovementPpoConfig,
    writer: SummaryWriter,
    iteration: int,
    rollout_stats: RolloutStats,
    diagnostics: TrainingDiagnostics,
    update_stats: PpoUpdateStats,
    update_skipped: bool,
    timing: IterationTiming,
    warming_up: bool,
    started: float,
) -> None:
    write_training_scalars(
        writer=writer,
        iteration=iteration,
        rollout_stats=rollout_stats,
        diagnostics=diagnostics,
        update_stats=update_stats,
    )
    writer.add_scalar('diagnostics/update_skipped', float(update_skipped), iteration)
    writer.add_scalar('timing/update_seconds', timing.update_finished - timing.rollout_finished, iteration)
    writer.add_scalar('timing/iteration_seconds', timing.update_finished - timing.iteration_started, iteration)
    if config.print_every > 0 and (iteration == 1 or iteration % config.print_every == 0):
        print_iteration_summary(
            config=config,
            iteration=iteration,
            rollout_stats=rollout_stats,
            diagnostics=diagnostics,
            update_stats=update_stats,
            update_skipped=update_skipped,
            warming_up=warming_up,
            timing=timing,
            started=started,
        )


def print_iteration_summary(
    config: MovementPpoConfig,
    iteration: int,
    rollout_stats: RolloutStats,
    diagnostics: TrainingDiagnostics,
    update_stats: PpoUpdateStats,
    update_skipped: bool,
    warming_up: bool,
    timing: IterationTiming,
    started: float,
) -> None:
    phase = 'skip' if update_skipped else ('value' if warming_up else 'ppo')
    print(
        f'[{phase}] iter={iteration}/{config.iterations} '
        f'reward={rollout_stats.mean_reward:+.4f} '
        f'return={diagnostics.mean_return:+.4f} '
        f'ev={diagnostics.explained_variance:+.3f} '
        f'policy_loss={update_stats.policy_loss:+.4f} '
        f'value_loss={update_stats.value_loss:.4f} '
        f'entropy={update_stats.entropy:.4f} '
        f'norm_entropy={rollout_stats.normalized_entropy:.3f} '
        f'top_p={rollout_stats.mean_top_action_probability:.3f} '
        f'demand={rollout_stats.mean_demand_scale:.2f}'
        f'[{rollout_stats.minimum_demand_scale:.2f}, {rollout_stats.maximum_demand_scale:.2f}] '
        f'clip={rollout_stats.reward_clip_fraction:.1%}/{update_stats.ratio_clip_fraction:.1%} '
        f'kl={update_stats.approximate_kl:.5f} '
        f'kl_stop={int(update_stats.early_stopped)} '
        f'grad={update_stats.backbone_gradient_norm:.3f}/'
        f'{update_stats.value_head_gradient_norm:.3f} '
        f'teleports={rollout_stats.teleport_count} '
        f'rollout={timing.rollout_finished - timing.iteration_started:.1f}s '
        f'update={timing.update_finished - timing.rollout_finished:.1f}s '
        f'elapsed={perf_counter() - started:.1f}s'
    )


def maybe_evaluate_iteration(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    writer: SummaryWriter,
    iteration: int,
) -> None:
    if config.eval_every <= 0 or iteration % config.eval_every != 0:
        return
    evaluation_started = perf_counter()
    evaluation_result = run_training_evaluation(
        config=config,
        model=state.model,
        metadata=state.metadata,
        iteration=iteration,
        writer=writer,
    )
    if is_best_checkpoint(evaluation_result.learned_checkpoint_score, state.best_checkpoint_score):
        assert evaluation_result.learned_checkpoint_score is not None
        state.best_checkpoint_score = evaluation_result.learned_checkpoint_score
        save_best_outputs(config=config, state=state, iteration=iteration)
        print(
            f'  new best completion-adjusted time-loss score={state.best_checkpoint_score:.3f} at iteration {iteration}'
        )
    print(f'  evaluation elapsed={perf_counter() - evaluation_started:.1f}s')


def is_best_checkpoint(score: float | None, best_checkpoint_score: float) -> bool:
    return score is not None and score < best_checkpoint_score


def save_best_outputs(config: MovementPpoConfig, state: PpoTrainingState, iteration: int) -> None:
    save_ppo_checkpoint(
        path=config.checkpoint_dir / 'movement_ppo_best.pt',
        model=state.model,
        optimizer=state.optimizer,
        metadata=state.metadata,
        iteration=iteration,
        best_checkpoint_score=state.best_checkpoint_score,
    )
    save_actor_checkpoint(
        path=config.checkpoint_dir / 'movement_policy_best.pt',
        model=state.model,
        metadata=state.metadata,
        loss=0.0,
    )


def maybe_save_numbered_checkpoint(config: MovementPpoConfig, state: PpoTrainingState, iteration: int) -> None:
    if config.save_every <= 0 or iteration % config.save_every != 0:
        return
    save_ppo_checkpoint(
        path=config.checkpoint_dir / f'movement_ppo_iter_{iteration:04d}.pt',
        model=state.model,
        optimizer=state.optimizer,
        metadata=state.metadata,
        iteration=iteration,
        best_checkpoint_score=state.best_checkpoint_score,
    )


def save_latest_outputs(
    config: MovementPpoConfig,
    state: PpoTrainingState,
    checkpoint_path: Path,
) -> None:
    save_ppo_checkpoint(
        path=checkpoint_path,
        model=state.model,
        optimizer=state.optimizer,
        metadata=state.metadata,
        iteration=state.completed_iteration,
        best_checkpoint_score=state.best_checkpoint_score,
    )
    save_actor_checkpoint(
        path=config.checkpoint_dir / 'movement_policy_latest.pt',
        model=state.model,
        metadata=state.metadata,
        loss=0.0,
    )
