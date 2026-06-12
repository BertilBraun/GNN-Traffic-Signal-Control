"""Backward-compatible zero-hop movement scorer."""
from __future__ import annotations

from .bipartite_gnn import MovementScorer


class ZeroHopMovementScorer(MovementScorer):
    """Compatibility wrapper for checkpoints/tests that name the zero-hop model."""

    def __init__(
        self,
        lane_feature_dim: int,
        movement_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            lane_feature_dim=lane_feature_dim,
            movement_feature_dim=movement_feature_dim,
            hidden_dim=hidden_dim,
            num_hops=0,
        )
