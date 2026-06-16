"""Checkpoint I/O for movement PPO."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from src.movement.models.bipartite_gnn import MovementActorCritic, MovementScorer
from src.movement.training.il_checkpoint import (
    MovementCheckpointMetadata,
    load_movement_checkpoint,
    normalizer_from_state,
    save_movement_checkpoint,
)
from src.movement.training.ppo.types import MovementPpoCheckpoint


def load_actor_critic(
    checkpoint_path: Path,
    device: str,
) -> tuple[MovementActorCritic, MovementCheckpointMetadata]:
    scorer, metadata = load_movement_checkpoint(checkpoint_path, device=device)
    model = MovementActorCritic(
        lane_feature_dim=metadata.lane_feature_dim,
        movement_feature_dim=metadata.movement_feature_dim,
        hidden_dim=metadata.hidden_dim,
        num_hops=metadata.num_hops,
    )
    missing, unexpected = model.load_state_dict(scorer.state_dict(), strict=False)
    allowed_missing = {key for key in model.state_dict() if key.startswith('value_head.')}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f'Unexpected actor-critic checkpoint keys: missing={missing}, unexpected={unexpected}')
    model.to(torch.device(device))
    return model, metadata


def save_actor_checkpoint(
    path: Path,
    model: MovementActorCritic,
    metadata: MovementCheckpointMetadata,
    loss: float,
) -> None:
    actor = MovementScorer(
        lane_feature_dim=metadata.lane_feature_dim,
        movement_feature_dim=metadata.movement_feature_dim,
        hidden_dim=metadata.hidden_dim,
        num_hops=metadata.num_hops,
    )
    actor.load_state_dict(
        {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith('value_head.')}
    )
    save_movement_checkpoint(
        checkpoint_path=path,
        model=actor,
        config=metadata.config,
        lane_feature_dim=metadata.lane_feature_dim,
        movement_feature_dim=metadata.movement_feature_dim,
        lane_normalizer=normalizer_from_state(metadata.lane_normalizer),
        movement_normalizer=normalizer_from_state(metadata.movement_normalizer),
        loss=loss,
    )


def save_ppo_checkpoint(
    path: Path,
    model: MovementActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: MovementCheckpointMetadata,
    iteration: int,
    best_checkpoint_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = MovementPpoCheckpoint(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        lane_feature_dim=metadata.lane_feature_dim,
        movement_feature_dim=metadata.movement_feature_dim,
        hidden_dim=metadata.hidden_dim,
        num_hops=metadata.num_hops,
        lane_normalizer=metadata.lane_normalizer,
        movement_normalizer=metadata.movement_normalizer,
        il_config=metadata.config,
        iteration=iteration,
        best_checkpoint_score=best_checkpoint_score,
        torch_random_state=torch.get_rng_state(),
        cuda_random_states=tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )
    torch.save(checkpoint, path)


def load_movement_ppo_checkpoint(
    checkpoint_path: Path | str,
    device: str,
) -> MovementActorCritic:
    checkpoint = load_ppo_checkpoint_payload(checkpoint_path=checkpoint_path, device=device)
    model, _metadata = model_and_metadata_from_ppo_checkpoint(checkpoint=checkpoint, device=device)
    return model


def load_ppo_checkpoint_payload(
    checkpoint_path: Path | str,
    device: str,
) -> MovementPpoCheckpoint:
    return cast(
        MovementPpoCheckpoint,
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )


def model_and_metadata_from_ppo_checkpoint(
    checkpoint: MovementPpoCheckpoint,
    device: str,
) -> tuple[MovementActorCritic, MovementCheckpointMetadata]:
    model = MovementActorCritic(
        lane_feature_dim=checkpoint.lane_feature_dim,
        movement_feature_dim=checkpoint.movement_feature_dim,
        hidden_dim=checkpoint.hidden_dim,
        num_hops=checkpoint.num_hops,
    )
    model.load_state_dict(checkpoint.model_state)
    model.to(torch.device(device))
    return (
        model,
        MovementCheckpointMetadata(
            lane_feature_dim=checkpoint.lane_feature_dim,
            movement_feature_dim=checkpoint.movement_feature_dim,
            hidden_dim=checkpoint.hidden_dim,
            num_hops=checkpoint.num_hops,
            lane_normalizer=checkpoint.lane_normalizer,
            movement_normalizer=checkpoint.movement_normalizer,
            config=checkpoint.il_config,
        ),
    )


def zero_value_output(model: MovementActorCritic) -> None:
    with torch.no_grad():
        last_layer = model.value_head[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)
