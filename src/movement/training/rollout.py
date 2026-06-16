"""Rollout buffer for movement-based PPO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from src.movement.dataset import MovementDatasetSample


@dataclass(frozen=True)
class MovementTransition:
    sample: MovementDatasetSample
    actions: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    action_masks: tuple[tuple[bool, ...], ...]
    rewards: tuple[float, ...]
    values: tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class MovementPpoBatch:
    transitions: tuple[MovementTransition, ...]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    policy_mask: torch.Tensor


class MovementRolloutBuffer:
    """Store decision-step transitions and compute GAE advantages."""

    def __init__(
        self,
        traffic_light_count: int,
        gamma: float,
        lam: float,
    ) -> None:
        self.traffic_light_count = traffic_light_count
        self.gamma = gamma
        self.lam = lam
        self.transitions: list[MovementTransition] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(self, transition: MovementTransition) -> None:
        if len(transition.actions) != self.traffic_light_count:
            raise ValueError('action count does not match traffic light count.')
        self.transitions.append(transition)

    def compute_returns_and_advantages(
        self,
        use_discounted_return_targets: bool,
        bootstrap_values: tuple[float, ...],
    ) -> None:
        if not self.transitions:
            raise ValueError('Cannot compute returns for an empty rollout buffer.')
        if len(bootstrap_values) != self.traffic_light_count:
            raise ValueError('bootstrap value count does not match traffic light count.')
        rewards = torch.tensor(
            tuple(transition.rewards for transition in self.transitions),
            dtype=torch.float32,
        )
        values = torch.tensor(
            tuple(transition.values for transition in self.transitions),
            dtype=torch.float32,
        )
        dones = torch.tensor(
            tuple(float(transition.done) for transition in self.transitions),
            dtype=torch.float32,
        )
        step_count = len(self.transitions)
        advantages = torch.zeros((step_count, self.traffic_light_count), dtype=torch.float32)
        last_gae = torch.zeros((self.traffic_light_count,), dtype=torch.float32)
        bootstrap = torch.tensor(bootstrap_values, dtype=torch.float32)
        for step in reversed(range(step_count)):
            if step == step_count - 1:
                next_value = bootstrap
                next_done = float(dones[step])
            else:
                next_value = values[step + 1]
                next_done = float(dones[step])
            delta = rewards[step] + self.gamma * next_value * (1.0 - next_done) - values[step]
            last_gae = delta + self.gamma * self.lam * (1.0 - next_done) * last_gae
            advantages[step] = last_gae

        if use_discounted_return_targets:
            returns = torch.zeros((step_count, self.traffic_light_count), dtype=torch.float32)
            running = bootstrap
            for step in reversed(range(step_count)):
                next_done = float(dones[step])
                running = rewards[step] + self.gamma * running * (1.0 - next_done)
                returns[step] = running
        else:
            returns = advantages + values
        self.advantages = advantages
        self.returns = returns

    def iterate_minibatches(
        self,
        transitions_per_batch: int,
        device: torch.device | str,
    ) -> Iterator[MovementPpoBatch]:
        if self.advantages is None or self.returns is None:
            raise ValueError('compute_returns_and_advantages must be called before minibatching.')
        torch_device = torch.device(device)
        indices = torch.randperm(len(self.transitions))
        batch_size = max(1, transitions_per_batch)
        for start in range(0, len(self.transitions), batch_size):
            batch_indices = indices[start : start + batch_size]
            transitions = tuple(self.transitions[int(index)] for index in batch_indices)
            actions = torch.tensor(
                tuple(transition.actions for transition in transitions),
                dtype=torch.long,
                device=torch_device,
            )
            old_log_probs = torch.tensor(
                tuple(transition.old_log_probs for transition in transitions),
                dtype=torch.float32,
                device=torch_device,
            )
            advantages = self.advantages[batch_indices].to(torch_device)
            returns = self.returns[batch_indices].to(torch_device)
            policy_mask = torch.tensor(
                tuple(
                    tuple(sum(action_mask) > 1 for action_mask in transition.action_masks) for transition in transitions
                ),
                dtype=torch.bool,
                device=torch_device,
            )
            normalized_advantages = _normalize_advantages(
                advantages=advantages,
                policy_mask=policy_mask,
            )
            yield MovementPpoBatch(
                transitions=transitions,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=normalized_advantages,
                returns=returns,
                policy_mask=policy_mask,
            )

    def __len__(self) -> int:
        return len(self.transitions)


def _normalize_advantages(
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
