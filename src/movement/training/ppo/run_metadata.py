"""Machine-readable PPO run metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.movement.training.ppo.types import MovementPpoConfig, RolloutCity


class PpoRunCityMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    city_split: str
    sumo_config_path: str
    rollout_jobs_per_iteration: int
    rollout_priority: int
    rollout_workers: int
    demand_scale_min: float | None
    demand_scale_max: float | None


class PpoRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at_utc: str
    experiment_name: str | None
    experiment_configuration_path: str | None
    experiment_configuration_sha256: str | None
    resume_checkpoint_path: str | None
    scratch_initialization: bool
    checkpoint_dir: str
    log_dir: str
    iterations: int
    completed_iteration_at_start: int
    rollout_jobs_per_update: int
    rollouts_per_update: int
    rollout_process_workers: int
    rollout_workers: int
    sumo_backend: str
    steps_per_rollout: int
    demand_scale_min: float
    demand_scale_max: float
    reward_mode: str
    global_reward_weight: float
    flow_reward_weight: float
    throughput_reward_weight: float
    progress_reward_weight: float
    gridlock_penalty_weight: float
    speed_change_weight: float
    evaluation_every_iterations: int
    evaluation_steps: int
    evaluation_seeds: tuple[int, ...]
    evaluation_demand_scales: tuple[float, ...]
    evaluation_policies: tuple[str, ...]
    rollout_cities: tuple[PpoRunCityMetadata, ...]


def build_run_metadata(
    config: MovementPpoConfig,
    completed_iteration_at_start: int,
) -> PpoRunMetadata:
    return PpoRunMetadata(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment_name=config.experiment_configuration.name if config.experiment_configuration is not None else None,
        experiment_configuration_path=(
            str(config.experiment_configuration_path) if config.experiment_configuration_path is not None else None
        ),
        experiment_configuration_sha256=config.experiment_configuration_sha256,
        resume_checkpoint_path=str(config.resume_checkpoint_path)
        if config.resume_checkpoint_path is not None
        else None,
        scratch_initialization=config.scratch_initialization,
        checkpoint_dir=str(config.checkpoint_dir),
        log_dir=str(config.log_dir),
        iterations=config.iterations,
        completed_iteration_at_start=completed_iteration_at_start,
        rollout_jobs_per_update=config.rollouts_per_update,
        rollouts_per_update=config.rollouts_per_update,
        rollout_process_workers=config.num_workers,
        rollout_workers=config.num_workers,
        sumo_backend=config.sumo_backend.value,
        steps_per_rollout=config.steps_per_rollout,
        demand_scale_min=config.demand_scale_min,
        demand_scale_max=config.demand_scale_max,
        reward_mode=config.reward_mode.value,
        global_reward_weight=config.global_reward_weight,
        flow_reward_weight=config.flow_reward_weight,
        throughput_reward_weight=config.throughput_reward_weight,
        progress_reward_weight=config.progress_reward_weight,
        gridlock_penalty_weight=config.gridlock_penalty_weight,
        speed_change_weight=config.speed_change_weight,
        evaluation_every_iterations=config.eval_every,
        evaluation_steps=config.eval_steps,
        evaluation_seeds=config.eval_seeds,
        evaluation_demand_scales=config.eval_demand_scales,
        evaluation_policies=tuple(policy.value for policy in config.eval_policies),
        rollout_cities=tuple(city_metadata(city=city) for city in config.rollout_cities),
    )


def city_metadata(city: RolloutCity) -> PpoRunCityMetadata:
    return PpoRunCityMetadata(
        city_name=city.city_name,
        city_split=city.city_split.value,
        sumo_config_path=str(city.sumo_config_path),
        rollout_jobs_per_iteration=city.rollout_jobs_per_iteration,
        rollout_priority=city.rollout_priority,
        rollout_workers=city.rollout_workers,
        demand_scale_min=city.demand_scale_min,
        demand_scale_max=city.demand_scale_max,
    )


def write_run_metadata(
    checkpoint_dir: Path,
    log_dir: Path,
    metadata: PpoRunMetadata,
) -> None:
    payload = metadata.model_dump_json(indent=2)
    for directory in (checkpoint_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'run_metadata.json').write_text(payload, encoding='utf-8')
