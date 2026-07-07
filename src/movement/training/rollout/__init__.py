"""Rollout buffer for movement-based PPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.movement.training.rollout.math import compute_rollout_targets
from src.movement.training.rollout.types import MovementTransition


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
        self.advantages: tuple[torch.Tensor, ...] | None = None
        self.returns: tuple[torch.Tensor, ...] | None = None

    def add(self, transition: MovementTransition) -> None:
        if len(transition.actions) != self.traffic_light_count:
            raise ValueError('action count does not match traffic light count.')
        self.transitions.append(transition)

    @classmethod
    def concatenate_computed(
        cls,
        buffers: Sequence[MovementRolloutBuffer],
    ) -> MovementRolloutBuffer:
        if not buffers:
            raise ValueError('Cannot concatenate an empty buffer list.')
        first = buffers[0]
        combined = cls(
            traffic_light_count=first.traffic_light_count,
            gamma=first.gamma,
            lam=first.lam,
        )
        advantages: list[torch.Tensor] = []
        returns: list[torch.Tensor] = []
        for buffer in buffers:
            if buffer.gamma != first.gamma or buffer.lam != first.lam:
                raise ValueError('Cannot concatenate rollout buffers with different discount settings.')
            if buffer.advantages is None or buffer.returns is None:
                raise ValueError('All rollout buffers must have computed returns before concatenation.')
            combined.transitions.extend(buffer.transitions)
            advantages.extend(buffer.advantages)
            returns.extend(buffer.returns)
        combined.advantages = tuple(advantages)
        combined.returns = tuple(returns)
        return combined

    def compute_returns_and_advantages(
        self,
        use_discounted_return_targets: bool,
        bootstrap_values: tuple[float, ...],
    ) -> None:
        targets = compute_rollout_targets(
            transitions=self.transitions,
            traffic_light_count=self.traffic_light_count,
            gamma=self.gamma,
            lam=self.lam,
            use_discounted_return_targets=use_discounted_return_targets,
            bootstrap_values=bootstrap_values,
        )
        self.advantages = tuple(targets.advantages[step_index] for step_index in range(targets.advantages.shape[0]))
        self.returns = tuple(targets.returns[step_index] for step_index in range(targets.returns.shape[0]))

    def __len__(self) -> int:
        return len(self.transitions)
