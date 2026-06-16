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
    normalized_entropies: list[float]
    top_action_probabilities: list[float]
    policy_decision_count: int
    total_decision_count: int
    teleport_count: int

    def __init__(self) -> None:
        self.rewards = []
        self.raw_rewards = []
        self.local_delay_densities = []
        self.global_delay_densities = []
        self.normalized_entropies = []
        self.top_action_probabilities = []
        self.policy_decision_count = 0
        self.total_decision_count = 0
        self.teleport_count = 0

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

    def stats(self, simulation_elapsed_s: float) -> RolloutStats:
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
            normalized_entropy=sum(self.normalized_entropies) / max(1, len(self.normalized_entropies)),
            mean_top_action_probability=sum(self.top_action_probabilities) / max(1, len(self.top_action_probabilities)),
            policy_decision_fraction=self.policy_decision_count / max(1, self.total_decision_count),
            teleport_count=self.teleport_count,
            simulation_elapsed_s=simulation_elapsed_s,
        )
