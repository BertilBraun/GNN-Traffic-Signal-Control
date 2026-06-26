"""Rollout buffer for movement-based PPO."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from src.movement.training.rollout.math import compute_rollout_targets, normalize_advantages
from src.movement.training.rollout.types import MovementPpoBatch, MovementTransition


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
            old_log_probs = torch.cat(
                tuple(
                    torch.tensor(transition.old_log_probs, dtype=torch.float32, device=torch_device)
                    for transition in transitions
                )
            )
            advantages = torch.cat(tuple(self.advantages[int(index)].to(torch_device) for index in batch_indices))
            returns = torch.cat(tuple(self.returns[int(index)].to(torch_device) for index in batch_indices))
            policy_mask = torch.cat(
                tuple(
                    torch.tensor(
                        tuple(sum(action_mask) > 1 for action_mask in transition.action_masks),
                        dtype=torch.bool,
                        device=torch_device,
                    )
                    for transition in transitions
                )
            )
            normalized_advantages = normalize_advantages(
                advantages=advantages,
                policy_mask=policy_mask,
            )
            yield MovementPpoBatch(
                transitions=transitions,
                old_log_probs=old_log_probs,
                advantages=normalized_advantages,
                returns=returns,
                policy_mask=policy_mask,
            )

    def __len__(self) -> int:
        return len(self.transitions)
