"""Shared movement PPO rollout data structures."""

from __future__ import annotations

from dataclasses import dataclass

from src.movement.training.movement_batch import MovementTensorSample


@dataclass(frozen=True)
class MovementTransition:
    tensor_sample: MovementTensorSample
    actions: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    action_masks: tuple[tuple[bool, ...], ...]
    rewards: tuple[float, ...]
    values: tuple[float, ...]
    done: bool
