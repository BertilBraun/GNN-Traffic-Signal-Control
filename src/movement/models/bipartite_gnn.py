"""Bipartite LaneGroup/Movement scorer with configurable macro-hops."""
from __future__ import annotations

import torch
from torch import nn


class MovementScorer(nn.Module):
    """Score movements from local lane context plus optional bipartite hops."""

    def __init__(
        self,
        lane_feature_dim: int,
        movement_feature_dim: int,
        hidden_dim: int = 64,
        num_hops: int = 1,
    ) -> None:
        super().__init__()
        if lane_feature_dim <= 0:
            raise ValueError("lane_feature_dim must be positive.")
        if movement_feature_dim <= 0:
            raise ValueError("movement_feature_dim must be positive.")
        if num_hops < 0:
            raise ValueError("num_hops must be non-negative.")
        self.lane_feature_dim = lane_feature_dim
        self.movement_feature_dim = movement_feature_dim
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.movement_encoder = nn.Sequential(
            nn.Linear(movement_feature_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hops = nn.ModuleList(
            MovementLaneHop(hidden_dim)
            for _ in range(num_hops)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_lane: torch.Tensor,
        x_movement: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return one scalar score per movement."""
        if self.num_hops > 0 and edge_index_dict is None:
            raise ValueError("edge_index_dict is required when num_hops > 0.")
        h_lane = self.lane_encoder(x_lane)
        input_lane_ids = x_movement[:, 3].long()
        output_lane_ids = x_movement[:, 4].long()
        h_move = self.movement_encoder(
            torch.cat(
                (
                    x_movement,
                    h_lane[input_lane_ids],
                    h_lane[output_lane_ids],
                ),
                dim=-1,
            )
        )
        for hop in self.hops:
            h_lane, h_move = hop(h_lane, h_move, edge_index_dict or {})
        return self.score_head(h_move).squeeze(-1)


class MovementLaneHop(nn.Module):
    """One Movement-to-LaneGroup-to-Movement macro-hop."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.move_to_input = nn.Linear(hidden_dim, hidden_dim)
        self.move_to_output = nn.Linear(hidden_dim, hidden_dim)
        self.lane_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.input_to_move = nn.Linear(hidden_dim, hidden_dim)
        self.output_to_move = nn.Linear(hidden_dim, hidden_dim)
        self.move_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_lane: torch.Tensor,
        h_move: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        move_to_input = _aggregate(
            edge_index_dict["movement_to_input_lane"],
            self.move_to_input(h_move),
            num_targets=h_lane.shape[0],
        )
        move_to_output = _aggregate(
            edge_index_dict["movement_to_output_lane"],
            self.move_to_output(h_move),
            num_targets=h_lane.shape[0],
        )
        updated_lane = self.lane_update(
            torch.cat((h_lane, move_to_input, move_to_output), dim=-1)
        )
        input_to_move = _aggregate(
            edge_index_dict["input_lane_to_movement"],
            self.input_to_move(updated_lane),
            num_targets=h_move.shape[0],
        )
        output_to_move = _aggregate(
            edge_index_dict["output_lane_to_movement"],
            self.output_to_move(updated_lane),
            num_targets=h_move.shape[0],
        )
        updated_move = self.move_update(
            torch.cat((h_move, input_to_move, output_to_move), dim=-1)
        )
        return updated_lane, updated_move


def _aggregate(
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """Mean-aggregate source features along a 2 x E edge index."""
    if edge_index.numel() == 0:
        return source_features.new_zeros((num_targets, source_features.shape[-1]))
    src = edge_index[0].long()
    dst = edge_index[1].long()
    out = source_features.new_zeros((num_targets, source_features.shape[-1]))
    out.index_add_(0, dst, source_features[src])
    counts = source_features.new_zeros((num_targets, 1))
    counts.index_add_(0, dst, torch.ones((dst.numel(), 1), device=source_features.device))
    return out / counts.clamp_min(1.0)
