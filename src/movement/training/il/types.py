"""Shared imitation-learning training data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

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
    phase_loss_coefficient: float = 1.0
    samples_per_batch: int = 16
    log_dir: Path | str | None = None
    checkpoint_every_epochs: int = 1
    progress_every_batches: int = 0
    progress_every_seconds: int = 60
    validation_every_epochs: int = 1
    cache_workers: int = 1
    preload_cache: bool = False
    train_workers: int = 1
    prefetch_batches: int = 2
    gradient_workers: int = 1


@dataclass(frozen=True)
class MovementILTrainingResult:
    checkpoint_path: Path
    final_loss: float
    epochs: int


@dataclass(frozen=True)
class MovementILTrainingSnapshot:
    epoch: int
    epochs: int
    loss: float
    regression_loss: float
    phase_loss: float
    phase_match_rate: float
    best_loss: float
    model: MovementScorer
    config: MovementILTrainingConfig
    lane_feature_dim: int
    movement_feature_dim: int
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer


class MovementILTrainingObserver(Protocol):
    def on_epoch_completed(self, snapshot: MovementILTrainingSnapshot) -> None: ...
