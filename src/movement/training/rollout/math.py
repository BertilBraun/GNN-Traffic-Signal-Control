"""Tensor math for movement PPO rollout buffers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.movement.training.rollout.types import MovementTransition


@dataclass(frozen=True)
class RolloutStepTensors:
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class RolloutTargets:
    advantages: torch.Tensor
    returns: torch.Tensor


def compute_rollout_targets(
    transitions: Sequence[MovementTransition],
    traffic_light_count: int,
    gamma: float,
    lam: float,
    use_discounted_return_targets: bool,
    bootstrap_values: tuple[float, ...],
) -> RolloutTargets:
    if not transitions:
        raise ValueError('Cannot compute returns for an empty rollout buffer.')
    if len(bootstrap_values) != traffic_light_count:
        raise ValueError('bootstrap value count does not match traffic light count.')
    step_tensors = rollout_step_tensors(transitions)
    advantages = generalized_advantage_estimates(
        rewards=step_tensors.rewards,
        values=step_tensors.values,
        dones=step_tensors.dones,
        bootstrap_values=bootstrap_values,
        gamma=gamma,
        lam=lam,
        traffic_light_count=traffic_light_count,
    )
    returns = (
        discounted_returns(
            rewards=step_tensors.rewards,
            dones=step_tensors.dones,
            bootstrap_values=bootstrap_values,
            gamma=gamma,
            traffic_light_count=traffic_light_count,
        )
        if use_discounted_return_targets
        else advantages + step_tensors.values
    )
    return RolloutTargets(advantages=advantages, returns=returns)


def rollout_step_tensors(transitions: Sequence[MovementTransition]) -> RolloutStepTensors:
    return RolloutStepTensors(
        rewards=torch.tensor(
            tuple(transition.rewards for transition in transitions),
            dtype=torch.float32,
        ),
        values=torch.tensor(
            tuple(transition.values for transition in transitions),
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            tuple(float(transition.done) for transition in transitions),
            dtype=torch.float32,
        ),
    )


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: tuple[float, ...],
    gamma: float,
    lam: float,
    traffic_light_count: int,
) -> torch.Tensor:
    step_count = rewards.shape[0]
    advantages = torch.zeros((step_count, traffic_light_count), dtype=torch.float32)
    last_advantage = torch.zeros((traffic_light_count,), dtype=torch.float32)
    bootstrap = torch.tensor(bootstrap_values, dtype=torch.float32)
    for step in reversed(range(step_count)):
        next_value = bootstrap if step == step_count - 1 else values[step + 1]
        next_done = float(dones[step])
        delta = rewards[step] + gamma * next_value * (1.0 - next_done) - values[step]
        last_advantage = delta + gamma * lam * (1.0 - next_done) * last_advantage
        advantages[step] = last_advantage
    return advantages


def discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: tuple[float, ...],
    gamma: float,
    traffic_light_count: int,
) -> torch.Tensor:
    step_count = rewards.shape[0]
    returns = torch.zeros((step_count, traffic_light_count), dtype=torch.float32)
    running_returns = torch.tensor(bootstrap_values, dtype=torch.float32)
    for step in reversed(range(step_count)):
        next_done = float(dones[step])
        running_returns = rewards[step] + gamma * running_returns * (1.0 - next_done)
        returns[step] = running_returns
    return returns


def normalize_advantages(
    advantages: torch.Tensor,
    policy_mask: torch.Tensor,
) -> torch.Tensor:
    active_advantages = advantages[policy_mask]
    normalized = torch.zeros_like(advantages)
    if active_advantages.numel() <= 1:
        normalized[policy_mask] = active_advantages
        return normalized
    normalized[policy_mask] = (active_advantages - active_advantages.mean()) / (active_advantages.std() + 1e-8)
    return normalized
