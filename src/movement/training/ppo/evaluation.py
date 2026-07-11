"""Periodic evaluation and TensorBoard logging for movement PPO."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

from torch.utils.tensorboard import SummaryWriter

from src.movement.evaluation import (
    EvaluationAggregate,
    EvaluationMetrics,
    EvaluationPolicy,
    EvaluationRecord,
    LearnedPolicyConfig,
    aggregate_records,
    print_aggregate_metric_table,
    run_evaluation_episode,
    write_aggregate_json,
    write_records_csv,
)
from src.movement.evaluation.multi_city import (
    FileCachedEpisodeRunner,
    MultiCityEvaluationAggregate,
    MultiCityEvaluationResult,
    aggregate_multi_city_records,
    default_episode_runner,
    print_multi_city_summary,
    run_multi_city_evaluation,
    write_multi_city_csv,
    write_multi_city_json,
)
from src.movement.experiment_config import CitySplit
from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.sumo_backend import SumoBackendKind
from src.movement.training.il.checkpoint import MovementCheckpointMetadata
from src.movement.training.ppo.checkpoint import save_actor_checkpoint
from src.movement.training.ppo.types import (
    MovementPpoConfig,
    RolloutStats,
    TrainingDiagnostics,
    TrainingEvaluationResult,
)
from src.movement.training.ppo.update import PpoUpdateStats


def write_training_scalars(
    writer: SummaryWriter,
    iteration: int,
    rollout_stats: RolloutStats,
    diagnostics: TrainingDiagnostics,
    update_stats: PpoUpdateStats,
) -> None:
    writer.add_scalar('episode/mean_reward', rollout_stats.mean_reward, iteration)
    writer.add_scalar('episode/reward_standard_deviation', rollout_stats.reward_standard_deviation, iteration)
    writer.add_scalar('episode/minimum_reward', rollout_stats.minimum_reward, iteration)
    writer.add_scalar('episode/maximum_reward', rollout_stats.maximum_reward, iteration)
    writer.add_scalar('episode/raw_reward_standard_deviation', rollout_stats.raw_reward_standard_deviation, iteration)
    writer.add_scalar('episode/minimum_raw_reward', rollout_stats.minimum_raw_reward, iteration)
    writer.add_scalar('episode/maximum_raw_reward', rollout_stats.maximum_raw_reward, iteration)
    writer.add_scalar('episode/mean_local_delay_density', rollout_stats.mean_local_delay_density, iteration)
    writer.add_scalar('episode/mean_global_delay_density', rollout_stats.mean_global_delay_density, iteration)
    writer.add_scalar('episode/mean_flow_rate_per_signal', rollout_stats.mean_flow_rate_per_signal, iteration)
    writer.add_scalar('episode/mean_progress_density', rollout_stats.mean_progress_density, iteration)
    writer.add_scalar('episode/mean_speed_change_density', rollout_stats.mean_speed_change_density, iteration)
    writer.add_scalar('episode/mean_return', diagnostics.mean_return, iteration)
    writer.add_scalar('episode/return_standard_deviation', diagnostics.return_standard_deviation, iteration)
    writer.add_scalar('episode/mean_value', diagnostics.mean_value, iteration)
    writer.add_scalar('episode/value_standard_deviation', diagnostics.value_standard_deviation, iteration)
    writer.add_scalar('episode/advantage_standard_deviation', diagnostics.advantage_standard_deviation, iteration)
    writer.add_scalar('diagnostics/explained_variance', diagnostics.explained_variance, iteration)
    writer.add_scalar('diagnostics/teleports', rollout_stats.teleport_count, iteration)
    writer.add_scalar('diagnostics/reward_clip_fraction', rollout_stats.reward_clip_fraction, iteration)
    writer.add_scalar('diagnostics/normalized_entropy', rollout_stats.normalized_entropy, iteration)
    writer.add_scalar('diagnostics/mean_top_action_probability', rollout_stats.mean_top_action_probability, iteration)
    writer.add_scalar('diagnostics/policy_decision_fraction', rollout_stats.policy_decision_fraction, iteration)
    writer.add_scalar('diagnostics/mean_demand_scale', rollout_stats.mean_demand_scale, iteration)
    writer.add_scalar('diagnostics/minimum_demand_scale', rollout_stats.minimum_demand_scale, iteration)
    writer.add_scalar('diagnostics/maximum_demand_scale', rollout_stats.maximum_demand_scale, iteration)
    writer.add_scalar('diagnostics/approximate_kl', update_stats.approximate_kl, iteration)
    writer.add_scalar('diagnostics/ratio_clip_fraction', update_stats.ratio_clip_fraction, iteration)
    writer.add_scalar('diagnostics/backbone_gradient_norm', update_stats.backbone_gradient_norm, iteration)
    writer.add_scalar('diagnostics/value_head_gradient_norm', update_stats.value_head_gradient_norm, iteration)
    writer.add_scalar('diagnostics/kl_early_stop', float(update_stats.early_stopped), iteration)
    writer.add_scalar('timing/rollout_seconds', rollout_stats.simulation_elapsed_s, iteration)
    writer.add_scalar('timing/update_batch_count', update_stats.profile.batch_count, iteration)
    writer.add_scalar('timing/update_data_seconds', update_stats.profile.data_seconds, iteration)
    writer.add_scalar('timing/update_batch_loss_seconds', update_stats.profile.batch_loss_seconds, iteration)
    writer.add_scalar('timing/update_model_forward_seconds', update_stats.profile.model_forward_seconds, iteration)
    writer.add_scalar('timing/update_value_seconds', update_stats.profile.value_seconds, iteration)
    writer.add_scalar('timing/update_phase_log_prob_seconds', update_stats.profile.phase_log_prob_seconds, iteration)
    writer.add_scalar('timing/update_backward_seconds', update_stats.profile.backward_seconds, iteration)
    writer.add_scalar('timing/update_optimizer_seconds', update_stats.profile.optimizer_seconds, iteration)
    writer.add_scalar('train/policy_loss', update_stats.policy_loss, iteration)
    writer.add_scalar('train/value_loss', update_stats.value_loss, iteration)
    writer.add_scalar('train/entropy', update_stats.entropy, iteration)
    writer.add_scalar('train/loss', update_stats.total_loss, iteration)
    writer.add_scalar('train/approx_kl', update_stats.approximate_kl, iteration)
    writer.add_scalar('loss/policy', update_stats.policy_loss, iteration)
    writer.add_scalar('loss/value', update_stats.value_loss, iteration)
    writer.add_scalar('loss/entropy', update_stats.entropy, iteration)
    writer.add_scalar('loss/total', update_stats.total_loss, iteration)


def run_training_evaluation(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    metadata: MovementCheckpointMetadata,
    iteration: int,
    writer: SummaryWriter,
) -> TrainingEvaluationResult:
    with TemporaryDirectory(prefix='movement_ppo_eval_') as directory:
        checkpoint_path = Path(directory) / 'movement_policy.pt'
        save_actor_checkpoint(
            path=checkpoint_path,
            model=model,
            metadata=metadata,
            loss=0.0,
        )
        learned_policy_config = LearnedPolicyConfig(
            checkpoint_path=checkpoint_path,
            device=config.eval_learned_device,
            action_mode=config.eval_learned_action_mode,
            temperature=config.eval_learned_temperature,
        )
        if config.experiment_configuration is not None:
            return run_multi_city_training_evaluation(
                config=config,
                learned_policy_config=learned_policy_config,
                iteration=iteration,
                writer=writer,
            )
        records = _evaluation_records(
            config=config,
            learned_policy_config=learned_policy_config,
        )
        aggregates = aggregate_records(records)
        write_evaluation_scalars(
            writer=writer,
            iteration=iteration,
            evaluation_steps=config.eval_steps,
            aggregates=aggregates,
        )
        output_dir = config.checkpoint_dir / 'eval' / f'iter_{iteration:04d}'
        write_aggregate_json(output_dir / 'summary.json', records, aggregates)
        write_records_csv(output_dir / 'summary.csv', records, aggregates)
        print_aggregate_metric_table(f'PPO evaluation at iteration {iteration}', aggregates)
        learned_aggregate = next(
            (aggregate for aggregate in aggregates if aggregate.policy == EvaluationPolicy.LEARNED.value),
            None,
        )
        return TrainingEvaluationResult(
            learned_checkpoint_score=(
                checkpoint_selection_score(
                    metrics=learned_aggregate.mean,
                    evaluation_steps=config.eval_steps,
                )
                if learned_aggregate is not None
                else None
            ),
        )


def run_multi_city_training_evaluation(
    config: MovementPpoConfig,
    learned_policy_config: LearnedPolicyConfig,
    iteration: int,
    writer: SummaryWriter,
) -> TrainingEvaluationResult:
    if config.experiment_configuration is None:
        raise ValueError('experiment_configuration is required for multi-city PPO evaluation')
    result = run_multi_city_evaluation(
        configuration=config.experiment_configuration,
        project_root=config.project_root,
        policies=config.eval_policies,
        seeds=config.eval_seeds,
        steps=config.eval_steps,
        demand_scales=config.eval_demand_scales,
        learned_policy_config=learned_policy_config,
        episode_runner=FileCachedEpisodeRunner(
            cache_dir=config.project_root / '.cache' / 'evaluation',
            episode_runner=default_episode_runner,
        ),
        backend_kind=config.sumo_backend,
        worker_count=config.eval_worker_count,
    )
    write_multi_city_evaluation_scalars(
        writer=writer,
        iteration=iteration,
        evaluation_steps=config.eval_steps,
        aggregates=result.aggregates,
    )
    output_dir = config.checkpoint_dir / 'eval' / f'iter_{iteration:04d}'
    write_multi_city_json(output_dir / 'summary.json', result)
    write_multi_city_csv(output_dir / 'summary.csv', result)
    write_separated_multi_city_evaluation_outputs(output_dir=output_dir, result=result)
    print_multi_city_summary(result.aggregates)
    return TrainingEvaluationResult(
        learned_checkpoint_score=held_out_learned_checkpoint_score(
            aggregates=result.aggregates,
            evaluation_steps=config.eval_steps,
        )
    )


def write_evaluation_scalars(
    writer: SummaryWriter,
    iteration: int,
    evaluation_steps: int,
    aggregates: Sequence[EvaluationAggregate],
) -> None:
    for aggregate in aggregates:
        policy = aggregate.policy
        metrics = aggregate.mean
        writer.add_scalar(f'eval/{policy}/throughput_per_hour', metrics.throughput_per_hour, iteration)
        writer.add_scalar(f'eval/{policy}/completion_rate', metrics.completion_rate, iteration)
        writer.add_scalar(f'eval/{policy}/average_waiting_time_s', metrics.average_waiting_time_s, iteration)
        writer.add_scalar(f'eval/{policy}/average_travel_time_s', metrics.average_travel_time_s, iteration)
        writer.add_scalar(
            f'eval/{policy}/average_queue_length_vehicles', metrics.average_queue_length_vehicles, iteration
        )
        writer.add_scalar(
            f'eval/{policy}/average_wait_density_s_per_m', metrics.average_wait_density_s_per_m, iteration
        )
        writer.add_scalar(
            f'eval/{policy}/switch_frequency_per_junction_per_minute',
            metrics.phase_switch_frequency_per_junction_per_minute,
            iteration,
        )
        writer.add_scalar(f'eval/{policy}/teleports', metrics.teleport_count, iteration)
        writer.add_scalar(
            f'eval/{policy}/checkpoint_selection_score',
            checkpoint_selection_score(
                metrics=metrics,
                evaluation_steps=evaluation_steps,
            ),
            iteration,
        )
        writer.add_scalar(f'eval/{policy}/nonstop_tls_pass_rate', metrics.nonstop_tls_pass_rate, iteration)


def write_multi_city_evaluation_scalars(
    writer: SummaryWriter,
    iteration: int,
    evaluation_steps: int,
    aggregates: Sequence[MultiCityEvaluationAggregate],
) -> None:
    for aggregate in aggregates:
        demand_tag = f'demand_{aggregate.demand_scale:.3f}'.replace('.', '_')
        tag_prefix = f'eval/{aggregate.city_split.value}/{aggregate.city_name}/{aggregate.policy}/{demand_tag}'
        metrics = aggregate.mean
        writer.add_scalar(f'{tag_prefix}/throughput_per_hour', metrics.throughput_per_hour, iteration)
        writer.add_scalar(f'{tag_prefix}/completion_rate', metrics.completion_rate, iteration)
        writer.add_scalar(f'{tag_prefix}/average_wait_density_s_per_m', metrics.average_wait_density_s_per_m, iteration)
        writer.add_scalar(f'{tag_prefix}/average_time_loss_s', metrics.average_time_loss_s, iteration)
        writer.add_scalar(f'{tag_prefix}/teleport_count', metrics.teleport_count, iteration)
        writer.add_scalar(
            f'{tag_prefix}/checkpoint_selection_score',
            checkpoint_selection_score(
                metrics=metrics,
                evaluation_steps=evaluation_steps,
            ),
            iteration,
        )
    for aggregate in split_policy_aggregates(aggregates=aggregates):
        demand_tag = f'demand_{aggregate.demand_scale:.3f}'.replace('.', '_')
        tag_prefix = f'eval/aggregate/{aggregate.city_split.value}/{aggregate.policy}/{demand_tag}'
        metrics = aggregate.mean
        writer.add_scalar(f'{tag_prefix}/throughput_per_hour', metrics.throughput_per_hour, iteration)
        writer.add_scalar(f'{tag_prefix}/completion_rate', metrics.completion_rate, iteration)
        writer.add_scalar(f'{tag_prefix}/average_wait_density_s_per_m', metrics.average_wait_density_s_per_m, iteration)
        writer.add_scalar(f'{tag_prefix}/average_time_loss_s', metrics.average_time_loss_s, iteration)
        writer.add_scalar(f'{tag_prefix}/teleport_count', metrics.teleport_count, iteration)


def write_separated_multi_city_evaluation_outputs(
    output_dir: Path,
    result: MultiCityEvaluationResult,
) -> None:
    learned_result = filtered_multi_city_result(
        result=result,
        policies=(EvaluationPolicy.LEARNED.value,),
    )
    baseline_result = filtered_multi_city_result(
        result=result,
        policies=tuple(policy.value for policy in EvaluationPolicy if policy != EvaluationPolicy.LEARNED),
    )
    if learned_result.records:
        write_multi_city_json(output_dir / 'learned_summary.json', learned_result)
        write_multi_city_csv(output_dir / 'learned_summary.csv', learned_result)
    if baseline_result.records:
        write_multi_city_json(output_dir / 'baseline_summary.json', baseline_result)
        write_multi_city_csv(output_dir / 'baseline_summary.csv', baseline_result)


def filtered_multi_city_result(
    result: MultiCityEvaluationResult,
    policies: tuple[str, ...],
) -> MultiCityEvaluationResult:
    records = tuple(record for record in result.records if record.policy in policies)
    return MultiCityEvaluationResult(
        records=records,
        aggregates=aggregate_multi_city_records(records),
    )


def split_policy_aggregates(
    aggregates: Sequence[MultiCityEvaluationAggregate],
) -> tuple[MultiCityEvaluationAggregate, ...]:
    grouping_keys: list[tuple[CitySplit, str, float]] = []
    for aggregate in aggregates:
        grouping_key = (aggregate.city_split, aggregate.policy, aggregate.demand_scale)
        if grouping_key not in grouping_keys:
            grouping_keys.append(grouping_key)
    split_aggregates: list[MultiCityEvaluationAggregate] = []
    for city_split, policy, demand_scale in grouping_keys:
        group = tuple(
            aggregate
            for aggregate in aggregates
            if aggregate.city_split == city_split
            and aggregate.policy == policy
            and aggregate.demand_scale == demand_scale
        )
        records = tuple(
            EvaluationRecord(policy=aggregate.policy, seed=city_index, metrics=aggregate.mean)
            for city_index, aggregate in enumerate(group)
        )
        aggregate_record = aggregate_records(records)[0]
        split_aggregates.append(
            MultiCityEvaluationAggregate(
                city_name='aggregate',
                city_split=city_split,
                policy=policy,
                demand_scale=demand_scale,
                seeds=tuple(seed for aggregate in group for seed in aggregate.seeds),
                mean=aggregate_record.mean,
                standard_deviation=aggregate_record.standard_deviation,
            )
        )
    return tuple(split_aggregates)


def held_out_learned_checkpoint_score(
    aggregates: Sequence[MultiCityEvaluationAggregate],
    evaluation_steps: int,
) -> float | None:
    learned_held_out_aggregates = tuple(
        aggregate
        for aggregate in aggregates
        if aggregate.policy == EvaluationPolicy.LEARNED.value and aggregate.city_split == CitySplit.HELD_OUT
    )
    if not learned_held_out_aggregates:
        return None
    scores = tuple(
        checkpoint_selection_score(
            metrics=aggregate.mean,
            evaluation_steps=evaluation_steps,
        )
        for aggregate in learned_held_out_aggregates
    )
    return sum(scores) / len(scores)


def checkpoint_selection_score(
    metrics: EvaluationMetrics,
    evaluation_steps: int,
) -> float:
    if metrics.completion_rate <= 0.0:
        return float('inf')
    teleport_rate = metrics.teleport_count / max(1, metrics.departed_vehicles)
    return metrics.average_time_loss_s / metrics.completion_rate + evaluation_steps * teleport_rate


def _evaluation_records(
    config: MovementPpoConfig,
    learned_policy_config: LearnedPolicyConfig,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for policy in config.eval_policies:
        for seed in config.eval_seeds:
            if policy != EvaluationPolicy.LEARNED:
                records.append(cached_baseline_record(config=config, policy=policy, seed=seed))
                continue
            records.append(
                EvaluationRecord(
                    policy=policy.value,
                    seed=seed,
                    metrics=run_evaluation_episode(
                        cfg_path=config.cfg_path,
                        policy=policy,
                        seed=seed,
                        steps=config.eval_steps,
                        decision_interval=config.decision_interval,
                        yellow_duration=config.yellow_duration,
                        min_green_steps=config.min_green_steps,
                        learned_policy_config=learned_policy_config,
                        demand_scale=config.eval_demand_scale,
                        initial_occupancy_min=config.initial_occupancy_min,
                        initial_occupancy_max=config.initial_occupancy_max,
                        time_to_teleport=config.time_to_teleport,
                        backend_kind=config.sumo_backend,
                    ),
                )
            )
    return records


def cached_baseline_record(
    config: MovementPpoConfig,
    policy: EvaluationPolicy,
    seed: int,
) -> EvaluationRecord:
    return _cached_baseline_record(
        cfg_path=config.cfg_path,
        policy=policy,
        seed=seed,
        eval_steps=config.eval_steps,
        decision_interval=config.decision_interval,
        yellow_duration=config.yellow_duration,
        min_green_steps=config.min_green_steps,
        eval_demand_scale=config.eval_demand_scale,
        initial_occupancy_min=config.initial_occupancy_min,
        initial_occupancy_max=config.initial_occupancy_max,
        time_to_teleport=config.time_to_teleport,
        backend_kind=config.sumo_backend,
    )


@cache
def _cached_baseline_record(
    cfg_path: Path,
    policy: EvaluationPolicy,
    seed: int,
    eval_steps: int,
    decision_interval: int,
    yellow_duration: int,
    min_green_steps: int,
    eval_demand_scale: float,
    initial_occupancy_min: float,
    initial_occupancy_max: float,
    time_to_teleport: int | None,
    backend_kind: SumoBackendKind,
) -> EvaluationRecord:
    return EvaluationRecord(
        policy=policy.value,
        seed=seed,
        metrics=run_evaluation_episode(
            cfg_path=cfg_path,
            policy=policy,
            seed=seed,
            steps=eval_steps,
            decision_interval=decision_interval,
            yellow_duration=yellow_duration,
            min_green_steps=min_green_steps,
            learned_policy_config=None,
            demand_scale=eval_demand_scale,
            initial_occupancy_min=initial_occupancy_min,
            initial_occupancy_max=initial_occupancy_max,
            time_to_teleport=time_to_teleport,
            backend_kind=backend_kind,
        ),
    )
