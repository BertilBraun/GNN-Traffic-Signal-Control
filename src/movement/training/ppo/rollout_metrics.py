"""Rollout metric accumulation for movement PPO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log

from torch.distributions import Categorical

from src.movement.training.ppo.stats import standard_deviation
from src.movement.training.ppo.types import IntervalRewardResult, RolloutStats


@dataclass
class RolloutMetrics:
    rewards: list[float]
    raw_rewards: list[float]
    local_delay_densities: list[float]
    global_delay_densities: list[float]
    speed_change_densities: list[float]
    normalized_entropies: list[float]
    top_action_probabilities: list[float]
    policy_decision_count: int
    total_decision_count: int
    teleport_count: int
    initial_population_seconds: float
    runtime_start_seconds: float
    context_build_seconds: float
    decision_sample_seconds: float
    sample_capture_seconds: float
    sample_index_seconds: float
    sample_flow_seconds: float
    sample_feature_frame_seconds: float
    sample_dataset_seconds: float
    decision_model_seconds: float
    decision_action_seconds: float
    decision_apply_seconds: float
    reward_seconds: float
    reward_sumo_step_seconds: float
    reward_lane_query_seconds: float
    reward_vehicle_query_seconds: float
    reward_aggregation_seconds: float
    bootstrap_seconds: float
    decision_step_count: int
    simulated_step_count: int

    def __init__(self) -> None:
        self.rewards = []
        self.raw_rewards = []
        self.local_delay_densities = []
        self.global_delay_densities = []
        self.speed_change_densities = []
        self.normalized_entropies = []
        self.top_action_probabilities = []
        self.policy_decision_count = 0
        self.total_decision_count = 0
        self.teleport_count = 0
        self.initial_population_seconds = 0.0
        self.runtime_start_seconds = 0.0
        self.context_build_seconds = 0.0
        self.decision_sample_seconds = 0.0
        self.sample_capture_seconds = 0.0
        self.sample_index_seconds = 0.0
        self.sample_flow_seconds = 0.0
        self.sample_feature_frame_seconds = 0.0
        self.sample_dataset_seconds = 0.0
        self.decision_model_seconds = 0.0
        self.decision_action_seconds = 0.0
        self.decision_apply_seconds = 0.0
        self.reward_seconds = 0.0
        self.reward_sumo_step_seconds = 0.0
        self.reward_lane_query_seconds = 0.0
        self.reward_vehicle_query_seconds = 0.0
        self.reward_aggregation_seconds = 0.0
        self.bootstrap_seconds = 0.0
        self.decision_step_count = 0
        self.simulated_step_count = 0

    def observe_setup(
        self,
        initial_population_seconds: float,
        runtime_start_seconds: float,
        context_build_seconds: float,
    ) -> None:
        self.initial_population_seconds += initial_population_seconds
        self.runtime_start_seconds += runtime_start_seconds
        self.context_build_seconds += context_build_seconds

    def observe_policy(self, distributions: Sequence[Categorical], action_masks: Sequence[Sequence[bool]]) -> None:
        for distribution, action_mask in zip(distributions, action_masks):
            valid_action_count = sum(action_mask)
            self.total_decision_count += 1
            if valid_action_count > 1:
                self.policy_decision_count += 1
                self.normalized_entropies.append(float(distribution.entropy().detach().cpu()) / log(valid_action_count))
                self.top_action_probabilities.append(float(distribution.probs.max().detach().cpu()))

    def observe_reward(self, interval_reward: IntervalRewardResult) -> None:
        self.teleport_count += interval_reward.teleport_count
        self.rewards.extend(interval_reward.rewards)
        self.raw_rewards.extend(interval_reward.raw_rewards)
        self.local_delay_densities.extend(interval_reward.local_delay_densities)
        self.global_delay_densities.append(interval_reward.global_delay_density)
        self.speed_change_densities.extend(interval_reward.speed_change_densities)
        self.reward_sumo_step_seconds += interval_reward.reward_sumo_step_seconds
        self.reward_lane_query_seconds += interval_reward.reward_lane_query_seconds
        self.reward_vehicle_query_seconds += interval_reward.reward_vehicle_query_seconds
        self.reward_aggregation_seconds += interval_reward.reward_aggregation_seconds
        self.simulated_step_count += interval_reward.simulated_steps

    def observe_decision(
        self,
        sample_seconds: float,
        model_seconds: float,
        action_seconds: float,
        apply_seconds: float,
        reward_seconds: float,
    ) -> None:
        self.decision_sample_seconds += sample_seconds
        self.decision_model_seconds += model_seconds
        self.decision_action_seconds += action_seconds
        self.decision_apply_seconds += apply_seconds
        self.reward_seconds += reward_seconds
        self.decision_step_count += 1

    def observe_sample(
        self,
        capture_seconds: float,
        index_seconds: float,
        flow_seconds: float,
        feature_frame_seconds: float,
        dataset_sample_seconds: float,
    ) -> None:
        self.sample_capture_seconds += capture_seconds
        self.sample_index_seconds += index_seconds
        self.sample_flow_seconds += flow_seconds
        self.sample_feature_frame_seconds += feature_frame_seconds
        self.sample_dataset_seconds += dataset_sample_seconds

    def observe_bootstrap(self, bootstrap_seconds: float) -> None:
        self.bootstrap_seconds += bootstrap_seconds

    def stats(self, simulation_elapsed_s: float, demand_scale: float) -> RolloutStats:
        return RolloutStats(
            mean_reward=sum(self.rewards) / max(1, len(self.rewards)),
            reward_standard_deviation=standard_deviation(self.rewards),
            minimum_reward=min(self.rewards, default=0.0),
            maximum_reward=max(self.rewards, default=0.0),
            raw_reward_standard_deviation=standard_deviation(self.raw_rewards),
            minimum_raw_reward=min(self.raw_rewards, default=0.0),
            maximum_raw_reward=max(self.raw_rewards, default=0.0),
            reward_clip_fraction=(
                sum(clipped != raw for clipped, raw in zip(self.rewards, self.raw_rewards)) / max(1, len(self.rewards))
            ),
            mean_local_delay_density=sum(self.local_delay_densities) / max(1, len(self.local_delay_densities)),
            mean_global_delay_density=sum(self.global_delay_densities) / max(1, len(self.global_delay_densities)),
            mean_speed_change_density=sum(self.speed_change_densities) / max(1, len(self.speed_change_densities)),
            normalized_entropy=sum(self.normalized_entropies) / max(1, len(self.normalized_entropies)),
            mean_top_action_probability=sum(self.top_action_probabilities) / max(1, len(self.top_action_probabilities)),
            policy_decision_fraction=self.policy_decision_count / max(1, self.total_decision_count),
            teleport_count=self.teleport_count,
            mean_demand_scale=demand_scale,
            minimum_demand_scale=demand_scale,
            maximum_demand_scale=demand_scale,
            simulation_elapsed_s=simulation_elapsed_s,
            initial_population_seconds=self.initial_population_seconds,
            runtime_start_seconds=self.runtime_start_seconds,
            context_build_seconds=self.context_build_seconds,
            decision_sample_seconds=self.decision_sample_seconds,
            sample_capture_seconds=self.sample_capture_seconds,
            sample_index_seconds=self.sample_index_seconds,
            sample_flow_seconds=self.sample_flow_seconds,
            sample_feature_frame_seconds=self.sample_feature_frame_seconds,
            sample_dataset_seconds=self.sample_dataset_seconds,
            decision_model_seconds=self.decision_model_seconds,
            decision_action_seconds=self.decision_action_seconds,
            decision_apply_seconds=self.decision_apply_seconds,
            reward_seconds=self.reward_seconds,
            reward_sumo_step_seconds=self.reward_sumo_step_seconds,
            reward_lane_query_seconds=self.reward_lane_query_seconds,
            reward_vehicle_query_seconds=self.reward_vehicle_query_seconds,
            reward_aggregation_seconds=self.reward_aggregation_seconds,
            bootstrap_seconds=self.bootstrap_seconds,
            decision_step_count=self.decision_step_count,
            simulated_step_count=self.simulated_step_count,
        )
