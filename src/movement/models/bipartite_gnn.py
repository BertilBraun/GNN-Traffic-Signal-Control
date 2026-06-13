"""Bipartite LaneGroup/Movement scorer with configurable macro-hops."""

from __future__ import annotations

from collections.abc import Sequence

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
            raise ValueError('lane_feature_dim must be positive.')
        if movement_feature_dim <= 0:
            raise ValueError('movement_feature_dim must be positive.')
        if num_hops < 0:
            raise ValueError('num_hops must be non-negative.')
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
        self.hops = nn.ModuleList(MovementLaneHop(hidden_dim) for _ in range(num_hops))
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def movement_embeddings(
        self,
        x_lane: torch.Tensor,
        x_movement: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return one embedding per movement."""
        if x_lane.ndim != 2 or x_lane.shape[1] != self.lane_feature_dim:
            raise ValueError(
                f'Expected lane features with shape [N, {self.lane_feature_dim}], '
                f'got {tuple(x_lane.shape)}. Regenerate the IL dataset and checkpoint '
                'for the current feature schema.'
            )
        if x_movement.ndim != 2 or x_movement.shape[1] != self.movement_feature_dim:
            raise ValueError(
                f'Expected movement features with shape [N, {self.movement_feature_dim}], '
                f'got {tuple(x_movement.shape)}.'
            )
        if self.num_hops > 0 and edge_index_dict is None:
            raise ValueError('edge_index_dict is required when num_hops > 0.')
        lane_embeddings = self.lane_encoder(x_lane)
        input_lane_ids = x_movement[:, 3].long()
        output_lane_ids = x_movement[:, 4].long()
        movement_embeddings = self.movement_encoder(
            torch.cat(
                (
                    x_movement,
                    lane_embeddings[input_lane_ids],
                    lane_embeddings[output_lane_ids],
                ),
                dim=-1,
            )
        )
        assert edge_index_dict is not None or self.num_hops == 0
        for hop in self.hops:
            lane_embeddings, movement_embeddings = hop(
                lane_embeddings,
                movement_embeddings,
                edge_index_dict,
            )
        return movement_embeddings

    def forward(
        self,
        x_lane: torch.Tensor,
        x_movement: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return one scalar score per movement."""
        return self.score_head(
            self.movement_embeddings(
                x_lane=x_lane,
                x_movement=x_movement,
                edge_index_dict=edge_index_dict,
            )
        ).squeeze(-1)


class MovementActorCritic(MovementScorer):
    """Movement-score actor plus per-traffic-light value critic."""

    def __init__(
        self,
        lane_feature_dim: int,
        movement_feature_dim: int,
        hidden_dim: int = 64,
        num_hops: int = 1,
    ) -> None:
        super().__init__(
            lane_feature_dim=lane_feature_dim,
            movement_feature_dim=movement_feature_dim,
            hidden_dim=hidden_dim,
            num_hops=num_hops,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor_critic(
        self,
        x_lane: torch.Tensor,
        x_movement: torch.Tensor,
        movement_ids_by_traffic_light: Sequence[Sequence[int]],
        edge_index_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return movement scores and one value per traffic light."""
        movement_embeddings = self.movement_embeddings(
            x_lane=x_lane,
            x_movement=x_movement,
            edge_index_dict=edge_index_dict,
        )
        movement_scores = self.score_head(movement_embeddings).squeeze(-1)
        traffic_light_embeddings = torch.stack(
            tuple(
                movement_embeddings[
                    torch.tensor(tuple(movement_ids), dtype=torch.long, device=movement_embeddings.device)
                ].mean(dim=0)
                for movement_ids in movement_ids_by_traffic_light
            )
        )
        values = self.value_head(traffic_light_embeddings).squeeze(-1)
        return movement_scores, values


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
        lane_embeddings: torch.Tensor,
        movement_embeddings: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        move_to_input = _aggregate(
            edge_index_dict['movement_to_input_lane'],
            self.move_to_input(movement_embeddings),
            num_targets=lane_embeddings.shape[0],
        )
        move_to_output = _aggregate(
            edge_index_dict['movement_to_output_lane'],
            self.move_to_output(movement_embeddings),
            num_targets=lane_embeddings.shape[0],
        )
        updated_lane = self.lane_update(torch.cat((lane_embeddings, move_to_input, move_to_output), dim=-1))
        input_to_move = _aggregate(
            edge_index_dict['input_lane_to_movement'],
            self.input_to_move(updated_lane),
            num_targets=movement_embeddings.shape[0],
        )
        output_to_move = _aggregate(
            edge_index_dict['output_lane_to_movement'],
            self.output_to_move(updated_lane),
            num_targets=movement_embeddings.shape[0],
        )
        updated_move = self.move_update(torch.cat((movement_embeddings, input_to_move, output_to_move), dim=-1))
        return updated_lane, updated_move


def _aggregate(
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """Mean-aggregate source features along a 2 x E edge index."""
    if edge_index.numel() == 0:
        return source_features.new_zeros((num_targets, source_features.shape[-1]))
    source_indices = edge_index[0].long()
    target_indices = edge_index[1].long()
    aggregated_features = source_features.new_zeros((num_targets, source_features.shape[-1]))
    aggregated_features.index_add_(
        0,
        target_indices,
        source_features[source_indices],
    )
    counts = source_features.new_zeros((num_targets, 1))
    counts.index_add_(
        0,
        target_indices,
        torch.ones((target_indices.numel(), 1), device=source_features.device),
    )
    return aggregated_features / counts.clamp_min(1.0)
