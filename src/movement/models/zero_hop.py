"""Zero-hop movement scorer for offline imitation learning."""
from __future__ import annotations

import torch
from torch import nn


class ZeroHopMovementScorer(nn.Module):
    """Score each movement from its own features and adjacent lane groups."""

    def __init__(
        self,
        lane_feature_dim: int,
        movement_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if lane_feature_dim <= 0:
            raise ValueError("lane_feature_dim must be positive.")
        if movement_feature_dim <= 0:
            raise ValueError("movement_feature_dim must be positive.")
        self.lane_feature_dim = lane_feature_dim
        self.movement_feature_dim = movement_feature_dim
        input_dim = movement_feature_dim + 2 * lane_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_lane: torch.Tensor, x_movement: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per movement."""
        input_lane_ids = x_movement[:, 3].long()
        output_lane_ids = x_movement[:, 4].long()
        input_lane = x_lane[input_lane_ids]
        output_lane = x_lane[output_lane_ids]
        features = torch.cat((x_movement, input_lane, output_lane), dim=-1)
        return self.net(features).squeeze(-1)
