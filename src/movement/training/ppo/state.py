"""PPO training state initialization."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import torch

from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.normalization import RunningNormalizer
from src.movement.training.il.checkpoint import MovementCheckpointMetadata
from src.movement.training.normalizer_state import normalizer_from_state
from src.movement.training.ppo.checkpoint import (
    load_actor_critic,
    load_ppo_checkpoint_payload,
    model_and_metadata_from_ppo_checkpoint,
    zero_value_output,
)
from src.movement.training.ppo.types import MovementPpoCheckpoint, MovementPpoConfig


@dataclass
class PpoTrainingState:
    model: MovementActorCritic
    metadata: MovementCheckpointMetadata
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer
    optimizer: torch.optim.Optimizer
    completed_iteration: int
    best_checkpoint_score: float


def initialize_training_state(config: MovementPpoConfig) -> PpoTrainingState:
    if config.resume_checkpoint_path is None:
        model, metadata = initialize_from_il_checkpoint(config)
        completed_iteration = 0
        best_checkpoint_score = float('inf')
    else:
        model, metadata, completed_iteration, best_checkpoint_score = initialize_from_ppo_checkpoint(config)
    lane_normalizer = normalizer_from_state(metadata.lane_normalizer)
    movement_normalizer = normalizer_from_state(metadata.movement_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.resume_checkpoint_path is not None:
        resume_checkpoint = load_ppo_checkpoint_payload(
            checkpoint_path=config.resume_checkpoint_path,
            device=config.device,
        )
        validate_resume_experiment_configuration(config=config, checkpoint=resume_checkpoint)
        optimizer.load_state_dict(resume_checkpoint.optimizer_state)
    return PpoTrainingState(
        model=model,
        metadata=metadata,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        optimizer=optimizer,
        completed_iteration=completed_iteration,
        best_checkpoint_score=best_checkpoint_score,
    )


def initialize_from_il_checkpoint(
    config: MovementPpoConfig,
) -> tuple[MovementActorCritic, MovementCheckpointMetadata]:
    if config.il_checkpoint_path is None:
        raise ValueError('il_checkpoint_path is required when resume_checkpoint_path is not set.')
    model, metadata = load_actor_critic(config.il_checkpoint_path, device=config.device)
    zero_value_output(model)
    return model, metadata


def initialize_from_ppo_checkpoint(
    config: MovementPpoConfig,
) -> tuple[MovementActorCritic, MovementCheckpointMetadata, int, float]:
    assert config.resume_checkpoint_path is not None
    resume_checkpoint = load_ppo_checkpoint_payload(
        checkpoint_path=config.resume_checkpoint_path,
        device=config.device,
    )
    validate_resume_experiment_configuration(config=config, checkpoint=resume_checkpoint)
    model, metadata = model_and_metadata_from_ppo_checkpoint(
        checkpoint=resume_checkpoint,
        device=config.device,
    )
    torch.set_rng_state(resume_checkpoint.torch_random_state.cpu())
    if torch.cuda.is_available() and resume_checkpoint.cuda_random_states:
        torch.cuda.set_rng_state_all(cuda_random_states_for_restore(resume_checkpoint.cuda_random_states))
    return model, metadata, resume_checkpoint.iteration, resume_checkpoint.best_checkpoint_score


def cuda_random_states_for_restore(cuda_random_states: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [state.detach().cpu().to(dtype=torch.uint8) for state in cuda_random_states]


def validate_resume_experiment_configuration(
    config: MovementPpoConfig,
    checkpoint: MovementPpoCheckpoint,
) -> None:
    expected_sha256 = config.experiment_configuration_sha256
    checkpoint_sha256 = checkpoint.experiment_configuration_sha256
    if checkpoint_sha256 is None and expected_sha256 is None:
        return
    if checkpoint_sha256 is None:
        raise ValueError('resume checkpoint does not contain experiment configuration metadata')
    if expected_sha256 is None:
        raise ValueError('resume checkpoint requires the original experiment configuration')
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f'resume experiment configuration hash mismatch: checkpoint={checkpoint_sha256} current={expected_sha256}'
        )


def create_rollout_pool(config: MovementPpoConfig) -> ProcessPoolExecutor | None:
    worker_count = min(config.num_workers, config.rollouts_per_update)
    if worker_count <= 1:
        return None
    return ProcessPoolExecutor(max_workers=worker_count)
