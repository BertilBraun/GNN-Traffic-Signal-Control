"""Shared PPO training data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.optim.optimizer import StateDict

from src.movement.evaluation import EvaluationPolicy
from src.movement.experiment_config import CitySplit, ExperimentConfiguration
from src.movement.features import LaneGroupGeometry
from src.movement.graph_schema import MovementGraph
from src.movement.sumo_backend import SumoBackendKind
from src.movement.training.il.types import MovementILTrainingConfig
from src.movement.training.normalizer_state import NormalizerState
from src.movement.training.rollout import MovementRolloutBuffer


class PpoRewardMode(str, Enum):
    DELAY_DENSITY = 'delay-density'
    THROUGHPUT = 'throughput'


@dataclass(frozen=True)
class RolloutCity:
    city_name: str
    city_split: CitySplit
    sumo_config_path: Path
    rollout_workers: int
    rollout_priority: int = 0
    demand_scale_min: float | None = None
    demand_scale_max: float | None = None

    @property
    def rollout_jobs_per_iteration(self) -> int:
        return self.rollout_workers


@dataclass(frozen=True)
class MovementPpoConfig:
    cfg_path: Path
    il_checkpoint_path: Path | None
    scratch_initialization: bool
    iterations: int
    steps_per_rollout: int
    rollouts_per_update: int
    num_workers: int
    decision_interval: int
    learning_rate: float
    gamma: float
    lam: float
    clip_epsilon: float
    update_epochs: int
    value_warmup_iterations: int
    warmup_epochs: int
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    transitions_per_batch: int
    update_batch_workers: int
    yellow_duration: int
    min_green_steps: int
    demand_scale_min: float
    demand_scale_max: float
    global_reward_weight: float
    flow_reward_weight: float
    reward_mode: PpoRewardMode
    throughput_reward_weight: float
    progress_reward_weight: float
    gridlock_penalty_weight: float
    speed_change_weight: float
    reward_sample_interval: int
    reward_clip: float
    teleport_penalty: float
    max_teleports_per_rollout: int
    time_to_teleport: int | None
    target_kl: float
    gui: bool
    sumo_backend: SumoBackendKind
    initial_occupancy_min: float
    initial_occupancy_max: float
    eval_every: int
    eval_steps: int
    eval_seeds: tuple[int, ...]
    eval_policies: tuple[EvaluationPolicy, ...]
    eval_demand_scale: float
    eval_demand_scales: tuple[float, ...]
    save_every: int
    print_every: int
    checkpoint_dir: Path
    log_dir: Path
    device: str
    seed: int
    fixed_rollout_seed: int | None
    resume_checkpoint_path: Path | None
    allow_resume_config_mismatch: bool
    rollout_cities: tuple[RolloutCity, ...]
    experiment_configuration: ExperimentConfiguration | None
    experiment_configuration_path: Path | None
    experiment_configuration_text: str | None
    experiment_configuration_sha256: str | None
    project_root: Path


@dataclass(frozen=True)
class MovementPpoTrainingResult:
    checkpoint_path: Path
    iterations: int


@dataclass(frozen=True)
class MovementPpoCheckpoint:
    model_state: dict[str, torch.Tensor]
    optimizer_state: StateDict
    lane_feature_dim: int
    movement_feature_dim: int
    hidden_dim: int
    num_hops: int
    lane_normalizer: NormalizerState
    movement_normalizer: NormalizerState
    il_config: MovementILTrainingConfig
    iteration: int
    best_checkpoint_score: float
    experiment_configuration_sha256: str | None
    experiment_configuration_text: str | None
    torch_random_state: torch.Tensor
    cuda_random_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class RolloutContext:
    graph: MovementGraph
    traffic_light_ids: tuple[str, ...]
    movement_ids_by_traffic_light: tuple[tuple[int, ...], ...]
    lane_ids_by_edge: dict[str, tuple[str, ...]]
    lane_geometries: dict[str, LaneGroupGeometry]
    incoming_lanes_by_traffic_light: dict[str, tuple[str, ...]]
    incoming_lane_length_by_traffic_light: dict[str, float]
    speed_limit_by_lane: dict[str, float]
    all_incoming_lane_ids: tuple[str, ...]
    all_incoming_lane_length_m: float


@dataclass(frozen=True)
class PolicyContext:
    traffic_light_ids: tuple[str, ...]
    movement_ids_by_traffic_light: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RolloutStats:
    mean_reward: float
    reward_standard_deviation: float
    minimum_reward: float
    maximum_reward: float
    raw_reward_standard_deviation: float
    minimum_raw_reward: float
    maximum_raw_reward: float
    reward_clip_fraction: float
    mean_local_delay_density: float
    mean_global_delay_density: float
    mean_flow_rate_per_signal: float
    mean_progress_density: float
    mean_speed_change_density: float
    normalized_entropy: float
    mean_top_action_probability: float
    policy_decision_fraction: float
    teleport_count: int
    mean_demand_scale: float
    minimum_demand_scale: float
    maximum_demand_scale: float
    simulation_elapsed_s: float
    initial_population_seconds: float = 0.0
    runtime_start_seconds: float = 0.0
    context_build_seconds: float = 0.0
    decision_sample_seconds: float = 0.0
    sample_capture_seconds: float = 0.0
    sample_index_seconds: float = 0.0
    sample_flow_seconds: float = 0.0
    sample_feature_frame_seconds: float = 0.0
    sample_dataset_seconds: float = 0.0
    decision_model_seconds: float = 0.0
    decision_action_seconds: float = 0.0
    decision_apply_seconds: float = 0.0
    reward_seconds: float = 0.0
    reward_sumo_step_seconds: float = 0.0
    reward_lane_query_seconds: float = 0.0
    reward_vehicle_query_seconds: float = 0.0
    reward_aggregation_seconds: float = 0.0
    bootstrap_seconds: float = 0.0
    decision_step_count: int = 0
    simulated_step_count: int = 0


@dataclass(frozen=True)
class CollectedRollout:
    buffer: MovementRolloutBuffer
    stats: RolloutStats
    seed: int
    city_name: str
    city_split: CitySplit


@dataclass(frozen=True)
class CityRolloutStats:
    city_name: str
    city_split: CitySplit
    rollout_count: int
    stats: RolloutStats


@dataclass(frozen=True)
class TrainingDiagnostics:
    mean_return: float
    return_standard_deviation: float
    mean_value: float
    value_standard_deviation: float
    advantage_standard_deviation: float
    explained_variance: float


@dataclass(frozen=True)
class TrainingEvaluationResult:
    learned_checkpoint_score: float | None


@dataclass(frozen=True)
class IntervalRewardResult:
    rewards: tuple[float, ...]
    raw_rewards: tuple[float, ...]
    local_delay_densities: tuple[float, ...]
    global_delay_density: float
    flow_rate_per_signal: float
    progress_densities: tuple[float, ...]
    speed_change_densities: tuple[float, ...]
    teleport_count: int
    simulated_steps: int = 0
    reward_sumo_step_seconds: float = 0.0
    reward_lane_query_seconds: float = 0.0
    reward_vehicle_query_seconds: float = 0.0
    reward_aggregation_seconds: float = 0.0
