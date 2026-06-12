"""Offline movement-score imitation learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from src.movement.dataset import MovementDatasetSample, load_jsonl_samples
from src.movement.models.zero_hop import ZeroHopMovementScorer
from src.movement.normalization import RunningNormalizer


@dataclass(frozen=True)
class ZeroHopTrainingConfig:
    epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    checkpoint_dir: Path | str = Path("checkpoints/il/zero_hop")
    seed: int = 42
    loss: str = "huber"
    device: str = "cpu"


@dataclass(frozen=True)
class ZeroHopTrainingResult:
    checkpoint_path: Path
    final_loss: float
    epochs: int


def train_zero_hop_il(
    samples: Sequence[MovementDatasetSample],
    config: ZeroHopTrainingConfig,
) -> ZeroHopTrainingResult:
    """Train a zero-hop scorer on stored movement-score samples."""
    if not samples:
        raise ValueError("At least one sample is required.")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    lane_normalizer = _fit_normalizer(sample.x_lane for sample in samples)
    movement_normalizer = _fit_normalizer(sample.x_movement for sample in samples)
    lane_feature_dim = len(samples[0].x_lane[0])
    movement_feature_dim = len(samples[0].x_movement[0])
    model = ZeroHopMovementScorer(
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    final_loss = 0.0
    for _ in range(config.epochs):
        total_loss = torch.zeros((), device=device)
        for sample in samples:
            x_lane, x_movement, target = tensors_from_sample(
                sample=sample,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
            )
            prediction = model(x_lane=x_lane, x_movement=x_movement)
            total_loss = total_loss + _loss(prediction, target, config.loss)
        loss = total_loss / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "zero_hop_il.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config) | {"checkpoint_dir": str(config.checkpoint_dir)},
            "lane_feature_dim": lane_feature_dim,
            "movement_feature_dim": movement_feature_dim,
            "hidden_dim": config.hidden_dim,
            "lane_normalizer": _normalizer_state(lane_normalizer),
            "movement_normalizer": _normalizer_state(movement_normalizer),
        },
        checkpoint_path,
    )
    return ZeroHopTrainingResult(
        checkpoint_path=checkpoint_path,
        final_loss=final_loss,
        epochs=config.epochs,
    )


def train_zero_hop_il_from_jsonl(
    dataset_path: Path | str,
    config: ZeroHopTrainingConfig,
) -> ZeroHopTrainingResult:
    """Load JSONL samples and train the zero-hop model."""
    return train_zero_hop_il(load_jsonl_samples(dataset_path), config)


def tensors_from_sample(
    sample: MovementDatasetSample,
    lane_normalizer: RunningNormalizer | None = None,
    movement_normalizer: RunningNormalizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one dataset sample to tensors."""
    x_lane_rows = sample.x_lane
    x_movement_rows = sample.x_movement
    if lane_normalizer is not None:
        x_lane_rows = tuple(lane_normalizer.transform_row(row) for row in x_lane_rows)
    if movement_normalizer is not None:
        x_movement_rows = tuple(
            _normalize_movement_row(row, movement_normalizer)
            for row in x_movement_rows
        )
    dev = torch.device(device)
    return (
        torch.tensor(x_lane_rows, dtype=torch.float32, device=dev),
        torch.tensor(x_movement_rows, dtype=torch.float32, device=dev),
        torch.tensor(sample.teacher_movement_scores, dtype=torch.float32, device=dev),
    )


def load_zero_hop_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> tuple[ZeroHopMovementScorer, dict[str, Any]]:
    """Load a zero-hop IL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ZeroHopMovementScorer(
        lane_feature_dim=int(checkpoint["lane_feature_dim"]),
        movement_feature_dim=int(checkpoint["movement_feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device(device))
    return model, {
        "lane_feature_dim": int(checkpoint["lane_feature_dim"]),
        "movement_feature_dim": int(checkpoint["movement_feature_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "lane_normalizer": checkpoint["lane_normalizer"],
        "movement_normalizer": checkpoint["movement_normalizer"],
        "config": checkpoint.get("config", {}),
    }


def normalizer_from_state(state: dict[str, Any]) -> RunningNormalizer:
    """Reconstruct a running normalizer from checkpoint metadata."""
    normalizer = RunningNormalizer(epsilon=float(state.get("epsilon", 1e-8)))
    normalizer.count = int(state["count"])
    normalizer.mean = tuple(float(value) for value in state["mean"])
    normalizer.m2 = tuple(float(value) for value in state["m2"])
    normalizer.frozen = bool(state.get("frozen", True))
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


def _normalizer_state(normalizer: RunningNormalizer) -> dict[str, Any]:
    return {
        "count": normalizer.count,
        "mean": normalizer.mean,
        "m2": normalizer.m2,
        "frozen": normalizer.frozen,
        "epsilon": normalizer.epsilon,
    }


def _normalize_movement_row(
    row: Sequence[float],
    normalizer: RunningNormalizer,
) -> tuple[float, ...]:
    normalized = list(normalizer.transform_row(row))
    # Columns 3 and 4 are graph lane-group IDs used as tensor indices.
    normalized[3] = float(row[3])
    normalized[4] = float(row[4])
    return tuple(normalized)


def _loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "huber":
        return F.smooth_l1_loss(prediction, target)
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {loss_name}")
