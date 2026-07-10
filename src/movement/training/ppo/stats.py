"""Statistics helpers for movement PPO."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import torch

from src.movement.training.ppo.types import RolloutStats, TrainingDiagnostics
from src.movement.training.rollout import MovementRolloutBuffer


def standard_deviation(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def combine_rollout_stats(stats: Sequence[RolloutStats]) -> RolloutStats:
    if not stats:
        raise ValueError('Cannot combine an empty rollout stats list.')
    count = len(stats)
    return RolloutStats(
        mean_reward=sum(stat.mean_reward for stat in stats) / count,
        reward_standard_deviation=sum(stat.reward_standard_deviation for stat in stats) / count,
        minimum_reward=min(stat.minimum_reward for stat in stats),
        maximum_reward=max(stat.maximum_reward for stat in stats),
        raw_reward_standard_deviation=sum(stat.raw_reward_standard_deviation for stat in stats) / count,
        minimum_raw_reward=min(stat.minimum_raw_reward for stat in stats),
        maximum_raw_reward=max(stat.maximum_raw_reward for stat in stats),
        reward_clip_fraction=sum(stat.reward_clip_fraction for stat in stats) / count,
        mean_local_delay_density=sum(stat.mean_local_delay_density for stat in stats) / count,
        mean_global_delay_density=sum(stat.mean_global_delay_density for stat in stats) / count,
        mean_flow_rate_per_signal=sum(stat.mean_flow_rate_per_signal for stat in stats) / count,
        mean_progress_density=sum(stat.mean_progress_density for stat in stats) / count,
        mean_speed_change_density=sum(stat.mean_speed_change_density for stat in stats) / count,
        normalized_entropy=sum(stat.normalized_entropy for stat in stats) / count,
        mean_top_action_probability=sum(stat.mean_top_action_probability for stat in stats) / count,
        policy_decision_fraction=sum(stat.policy_decision_fraction for stat in stats) / count,
        teleport_count=sum(stat.teleport_count for stat in stats),
        mean_demand_scale=sum(stat.mean_demand_scale for stat in stats) / count,
        minimum_demand_scale=min(stat.minimum_demand_scale for stat in stats),
        maximum_demand_scale=max(stat.maximum_demand_scale for stat in stats),
        simulation_elapsed_s=max(stat.simulation_elapsed_s for stat in stats),
        initial_population_seconds=max(stat.initial_population_seconds for stat in stats),
        runtime_start_seconds=max(stat.runtime_start_seconds for stat in stats),
        context_build_seconds=max(stat.context_build_seconds for stat in stats),
        decision_sample_seconds=sum(stat.decision_sample_seconds for stat in stats),
        sample_capture_seconds=sum(stat.sample_capture_seconds for stat in stats),
        sample_index_seconds=sum(stat.sample_index_seconds for stat in stats),
        sample_flow_seconds=sum(stat.sample_flow_seconds for stat in stats),
        sample_feature_frame_seconds=sum(stat.sample_feature_frame_seconds for stat in stats),
        sample_dataset_seconds=sum(stat.sample_dataset_seconds for stat in stats),
        decision_model_seconds=sum(stat.decision_model_seconds for stat in stats),
        decision_action_seconds=sum(stat.decision_action_seconds for stat in stats),
        decision_apply_seconds=sum(stat.decision_apply_seconds for stat in stats),
        reward_seconds=sum(stat.reward_seconds for stat in stats),
        reward_sumo_step_seconds=sum(stat.reward_sumo_step_seconds for stat in stats),
        reward_lane_query_seconds=sum(stat.reward_lane_query_seconds for stat in stats),
        reward_vehicle_query_seconds=sum(stat.reward_vehicle_query_seconds for stat in stats),
        reward_aggregation_seconds=sum(stat.reward_aggregation_seconds for stat in stats),
        bootstrap_seconds=sum(stat.bootstrap_seconds for stat in stats),
        decision_step_count=sum(stat.decision_step_count for stat in stats),
        simulated_step_count=sum(stat.simulated_step_count for stat in stats),
    )


def training_diagnostics(buffer: MovementRolloutBuffer) -> TrainingDiagnostics:
    if buffer.returns is None or buffer.advantages is None:
        raise ValueError('Returns must be computed before training diagnostics.')
    values = torch.cat(tuple(torch.tensor(transition.values, dtype=torch.float32) for transition in buffer.transitions))
    returns = torch.cat(buffer.returns)
    advantages = torch.cat(buffer.advantages)
    return_variance = float(returns.var())
    residual_variance = float((returns - values).var())
    explained_variance = 1.0 - residual_variance / (return_variance + 1e-8)
    return TrainingDiagnostics(
        mean_return=float(returns.mean()),
        return_standard_deviation=float(returns.std()),
        mean_value=float(values.mean()),
        value_standard_deviation=float(values.std()),
        advantage_standard_deviation=float(advantages.std()),
        explained_variance=explained_variance,
    )
