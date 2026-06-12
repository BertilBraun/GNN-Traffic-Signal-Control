"""Offline movement-score imitation learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

from src.movement.dataset import MovementDatasetSample, load_jsonl_samples
from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer


class MovementILLoss(str, Enum):
    HUBER = 'huber'
    MEAN_SQUARED_ERROR = 'mse'


@dataclass(frozen=True)
class MovementILTrainingConfig:
    epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    checkpoint_dir: Path | str = Path('checkpoints/il')
    seed: int = 42
    loss: MovementILLoss = MovementILLoss.HUBER
    device: str = 'cpu'
    progress_every: int = 0
    num_hops: int = 0


@dataclass(frozen=True)
class MovementILTrainingResult:
    checkpoint_path: Path
    final_loss: float
    epochs: int


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


def train_movement_il(
    samples: Sequence[MovementDatasetSample],
    config: MovementILTrainingConfig,
) -> MovementILTrainingResult:
    """Train a movement scorer on stored movement-score samples."""
    if not samples:
        raise ValueError('At least one sample is required.')
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    lane_normalizer = _fit_normalizer(sample.x_lane for sample in samples)
    movement_normalizer = _fit_normalizer(sample.x_movement for sample in samples)
    lane_feature_dim = len(samples[0].x_lane[0])
    movement_feature_dim = len(samples[0].x_movement[0])
    model = MovementScorer(
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    final_loss = 0.0
    best_loss = float('inf')
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(config.epochs):
        total_loss = torch.zeros((), device=device)
        for sample in samples:
            x_lane, x_movement, target = tensors_from_sample(
                sample=sample,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
            )
            prediction = model(
                x_lane=x_lane,
                x_movement=x_movement,
                edge_index_dict=edge_tensors_from_sample(sample, device=device),
            )
            total_loss = total_loss + _loss(prediction, target, config.loss)
        loss = total_loss / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if _should_report_progress(epoch, config.epochs, config.progress_every):
            print(f'epoch={epoch + 1}/{config.epochs} loss={final_loss:.6f}')

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = _checkpoint_payload(
        model_state=model.state_dict(),
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        loss=final_loss,
    )
    best_checkpoint = _checkpoint_payload(
        model_state=best_state,
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        loss=best_loss,
    )
    last_path = checkpoint_dir / 'movement_policy_last.pt'
    best_path = checkpoint_dir / 'movement_policy_best.pt'
    torch.save(last_checkpoint, last_path)
    torch.save(best_checkpoint, best_path)
    return MovementILTrainingResult(
        checkpoint_path=last_path,
        final_loss=final_loss,
        epochs=config.epochs,
    )


def _checkpoint_payload(
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
        lane_normalizer=_normalizer_state(lane_normalizer),
        movement_normalizer=_normalizer_state(movement_normalizer),
        loss=loss,
    )


def train_movement_il_from_jsonl(
    dataset_path: Path | str,
    config: MovementILTrainingConfig,
) -> MovementILTrainingResult:
    """Load JSONL samples and train the movement scorer."""
    return train_movement_il(load_jsonl_samples(dataset_path), config)


def tensors_from_sample(
    sample: MovementDatasetSample,
    lane_normalizer: RunningNormalizer | None = None,
    movement_normalizer: RunningNormalizer | None = None,
    device: torch.device | str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one dataset sample to tensors."""
    x_lane_rows = sample.x_lane
    x_movement_rows = sample.x_movement
    if lane_normalizer is not None:
        x_lane_rows = tuple(lane_normalizer.transform_row(row) for row in x_lane_rows)
    if movement_normalizer is not None:
        x_movement_rows = tuple(_normalize_movement_row(row, movement_normalizer) for row in x_movement_rows)
    torch_device = torch.device(device)
    return (
        torch.tensor(x_lane_rows, dtype=torch.float32, device=torch_device),
        torch.tensor(x_movement_rows, dtype=torch.float32, device=torch_device),
        torch.tensor(sample.teacher_movement_scores, dtype=torch.float32, device=torch_device),
    )


def edge_tensors_from_sample(
    sample: MovementDatasetSample,
    device: torch.device | str = 'cpu',
) -> dict[str, torch.Tensor]:
    """Convert stored typed edge lists to 2 x E long tensors."""
    torch_device = torch.device(device)
    return {
        relation: torch.tensor(edges, dtype=torch.long, device=torch_device).t().contiguous()
        for relation, edges in sample.edge_index_dict.items()
    }


def load_movement_checkpoint(
    checkpoint_path: Path | str,
    device: str = 'cpu',
) -> tuple[MovementScorer, MovementCheckpointMetadata]:
    """Load a movement IL checkpoint."""
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
    """Reconstruct a running normalizer from checkpoint metadata."""
    normalizer = RunningNormalizer(epsilon=state.epsilon)
    normalizer.count = state.count
    normalizer.mean = state.mean
    normalizer.m2 = state.squared_differences
    normalizer.frozen = state.frozen
    normalizer._dimension = len(normalizer.mean)
    return normalizer


def _fit_normalizer(
    batches: Sequence[Sequence[Sequence[float]]],
) -> RunningNormalizer:
    normalizer = RunningNormalizer()
    for rows in batches:
        normalizer.update_rows(rows)
    normalizer.freeze()
    return normalizer


def _normalizer_state(normalizer: RunningNormalizer) -> NormalizerState:
    return NormalizerState(
        count=normalizer.count,
        mean=normalizer.mean,
        squared_differences=normalizer.m2,
        frozen=normalizer.frozen,
        epsilon=normalizer.epsilon,
    )


def _normalize_movement_row(
    row: Sequence[float],
    normalizer: RunningNormalizer,
) -> tuple[float, ...]:
    normalized = list(normalizer.transform_row(row))
    # Columns 3 and 4 are graph lane-group IDs used as tensor indices.
    normalized[3] = float(row[3])
    normalized[4] = float(row[4])
    return tuple(normalized)


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: MovementILLoss,
) -> torch.Tensor:
    match loss_name:
        case MovementILLoss.HUBER:
            return F.smooth_l1_loss(prediction, target)
        case MovementILLoss.MEAN_SQUARED_ERROR:
            return F.mse_loss(prediction, target)


def _should_report_progress(epoch: int, epochs: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return (epoch + 1) % progress_every == 0 or epoch == epochs - 1
