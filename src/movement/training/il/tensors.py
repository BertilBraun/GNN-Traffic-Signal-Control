"""Tensor adapters for movement imitation samples."""

from __future__ import annotations

import torch

from src.movement.dataset import MovementDatasetSample
from src.movement.normalization import RunningNormalizer


def tensors_from_sample(
    sample: MovementDatasetSample,
    lane_normalizer: RunningNormalizer | None,
    movement_normalizer: RunningNormalizer | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_lane_rows = sample.x_lane
    x_movement_rows = sample.x_movement
    if lane_normalizer is not None:
        x_lane_rows = tuple(lane_normalizer.transform_row(row) for row in x_lane_rows)
    if movement_normalizer is not None:
        x_movement_rows = tuple(movement_normalizer.transform_row(row) for row in x_movement_rows)
    torch_device = torch.device(device)
    return (
        torch.tensor(x_lane_rows, dtype=torch.float32, device=torch_device),
        torch.tensor(x_movement_rows, dtype=torch.float32, device=torch_device),
        torch.tensor(sample.teacher_movement_scores, dtype=torch.float32, device=torch_device),
    )


def edge_tensors_from_sample(
    sample: MovementDatasetSample,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    torch_device = torch.device(device)
    return {
        'input_lane_to_movement': edge_tensor(sample.edge_indices.input_lane_to_movement, torch_device),
        'output_lane_to_movement': edge_tensor(sample.edge_indices.output_lane_to_movement, torch_device),
        'movement_to_input_lane': edge_tensor(sample.edge_indices.movement_to_input_lane, torch_device),
        'movement_to_output_lane': edge_tensor(sample.edge_indices.movement_to_output_lane, torch_device),
        'lane_to_lane': edge_tensor(sample.edge_indices.lane_to_lane, torch_device),
        'lane_to_lane_weight': torch.tensor(
            sample.edge_indices.lane_to_lane_weight,
            dtype=torch.float32,
            device=torch_device,
        ),
    }


def edge_tensor(edges: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
