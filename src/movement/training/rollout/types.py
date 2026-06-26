"""Shared movement PPO rollout data structures."""

from __future__ import annotations

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
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    policy_mask: torch.Tensor
