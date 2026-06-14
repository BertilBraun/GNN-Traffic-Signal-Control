from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.movement.models.bipartite_gnn import MovementScorer


def _edges() -> dict[str, torch.Tensor]:
    return {
        'input_lane_to_movement': torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        'output_lane_to_movement': torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
        'movement_to_input_lane': torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        'movement_to_output_lane': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }


def test_movement_scorer_supports_zero_and_one_hop_outputs() -> None:
    x_lane = torch.randn(3, 4)
    x_movement = torch.tensor(
        [
            [1.0, 1.0, 0.1, 0.0],
            [1.0, 2.0, 0.2, 1.0],
        ],
        dtype=torch.float32,
    )

    local_model = MovementScorer(
        lane_feature_dim=4,
        movement_feature_dim=4,
        hidden_dim=8,
        num_hops=0,
    )
    one_hop = MovementScorer(
        lane_feature_dim=4,
        movement_feature_dim=4,
        hidden_dim=8,
        num_hops=1,
    )

    assert tuple(local_model(x_lane, x_movement, _edges()).shape) == (2,)
    assert tuple(one_hop(x_lane, x_movement, _edges()).shape) == (2,)


def test_one_hop_scorer_requires_edges() -> None:
    model = MovementScorer(
        lane_feature_dim=4,
        movement_feature_dim=4,
        hidden_dim=8,
        num_hops=1,
    )

    try:
        model(torch.randn(3, 4), torch.randn(2, 4))
    except ValueError as exc:
        assert 'edge_index_dict' in str(exc)
    else:
        raise AssertionError('expected one-hop model to require graph edges')
