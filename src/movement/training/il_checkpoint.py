"""Checkpoint I/O for movement-score imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.training.il_types import MovementILTrainingConfig


@dataclass(frozen=True)
class NormalizerState:
    count: int
    mean: tuple[float, ...]
    squared_differences: tuple[float, ...]
    frozen: bool
    epsilon: float


@dataclass(frozen=True)
class MovementCheckpointPayload:
    model_state: dict[str, torch.Tensor]
    config: MovementILTrainingConfig
    lane_feature_dim: int
    movement_feature_dim: int
    hidden_dim: int
    num_hops: int
    lane_normalizer: NormalizerState
    movement_normalizer: NormalizerState
    loss: float


@dataclass(frozen=True)
class MovementCheckpointMetadata:
    lane_feature_dim: int
    movement_feature_dim: int
    hidden_dim: int
    num_hops: int
    lane_normalizer: NormalizerState
    movement_normalizer: NormalizerState
    config: MovementILTrainingConfig


def movement_checkpoint_payload(
    model_state: dict[str, torch.Tensor],
    config: MovementILTrainingConfig,
    lane_feature_dim: int,
    movement_feature_dim: int,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    loss: float,
) -> MovementCheckpointPayload:
    return MovementCheckpointPayload(
        model_state=model_state,
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
        lane_normalizer=normalizer_state(lane_normalizer),
        movement_normalizer=normalizer_state(movement_normalizer),
        loss=loss,
    )


def save_movement_checkpoint(
    checkpoint_path: Path | str,
    model: MovementScorer,
    config: MovementILTrainingConfig,
    lane_feature_dim: int,
    movement_feature_dim: int,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    loss: float,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = movement_checkpoint_payload(
        model_state=model.state_dict(),
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        loss=loss,
    )
    torch.save(checkpoint, path)


def load_movement_checkpoint(
    checkpoint_path: Path | str,
    device: str,
) -> tuple[MovementScorer, MovementCheckpointMetadata]:
    checkpoint = cast(
        MovementCheckpointPayload,
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    model = MovementScorer(
        lane_feature_dim=checkpoint.lane_feature_dim,
        movement_feature_dim=checkpoint.movement_feature_dim,
        hidden_dim=checkpoint.hidden_dim,
        num_hops=checkpoint.num_hops,
    )
    model.load_state_dict(checkpoint.model_state)
    model.to(torch.device(device))
    return model, MovementCheckpointMetadata(
        lane_feature_dim=checkpoint.lane_feature_dim,
        movement_feature_dim=checkpoint.movement_feature_dim,
        hidden_dim=checkpoint.hidden_dim,
        num_hops=checkpoint.num_hops,
        lane_normalizer=checkpoint.lane_normalizer,
        movement_normalizer=checkpoint.movement_normalizer,
        config=checkpoint.config,
    )


def normalizer_from_state(state: NormalizerState) -> RunningNormalizer:
    normalizer = RunningNormalizer(epsilon=state.epsilon)
    normalizer.count = state.count
    normalizer.mean = state.mean
    normalizer.m2 = state.squared_differences
    normalizer.frozen = state.frozen
    normalizer._dimension = len(normalizer.mean)
    return normalizer


def normalizer_state(normalizer: RunningNormalizer) -> NormalizerState:
    return NormalizerState(
        count=normalizer.count,
        mean=normalizer.mean,
        squared_differences=normalizer.m2,
        frozen=normalizer.frozen,
        epsilon=normalizer.epsilon,
    )
