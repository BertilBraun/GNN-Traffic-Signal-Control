from pathlib import Path
import sys

import pytest
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.training.ppo import _masked_phase_logits


def test_actor_critic_returns_movement_scores_and_tls_values() -> None:
    model = MovementActorCritic(
        lane_feature_dim=2,
        movement_feature_dim=4,
        hidden_dim=8,
        num_hops=0,
    )

    movement_scores, values = model.forward_actor_critic(
        x_lane=torch.tensor(((10.0, 1.0), (20.0, 2.0), (30.0, 3.0))),
        x_movement=torch.tensor(
            (
                (1.0, 1.0, 0.1, 0.0),
                (1.0, 2.0, 0.2, 1.0),
                (1.0, 3.0, 0.3, 0.0),
            )
        ),
        movement_ids_by_traffic_light=((0, 1), (2,)),
        edge_index_dict={
            'input_lane_to_movement': torch.tensor(((0, 1, 2), (0, 1, 2))),
            'output_lane_to_movement': torch.tensor(((1, 2, 0), (0, 1, 2))),
        },
    )

    assert tuple(movement_scores.shape) == (3,)
    assert tuple(values.shape) == (2,)


def test_actor_critic_rejects_stale_lane_feature_schema() -> None:
    model = MovementActorCritic(
        lane_feature_dim=4,
        movement_feature_dim=4,
        hidden_dim=8,
        num_hops=0,
    )

    with pytest.raises(ValueError, match='Regenerate the IL dataset and checkpoint'):
        model.forward_actor_critic(
            x_lane=torch.zeros((2, 6)),
            x_movement=torch.zeros((1, 4)),
            movement_ids_by_traffic_light=((0,),),
            edge_index_dict={
                'input_lane_to_movement': torch.tensor(((0,), (0,))),
                'output_lane_to_movement': torch.tensor(((1,), (0,))),
            },
        )


def test_action_mask_prevents_sampling_rejected_phase() -> None:
    logits = (torch.tensor((100.0, 0.0, -100.0)),)

    masked_logits = _masked_phase_logits(logits, ((False, True, False),))
    distribution = Categorical(logits=masked_logits[0])

    assert {int(distribution.sample()) for _sample in range(20)} == {1}
