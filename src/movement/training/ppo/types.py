"""Shared PPO training data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim.optimizer import StateDict

from src.movement.evaluation import EvaluationPolicy
from src.movement.features import LaneGroupGeometry
from src.movement.graph_schema import MovementGraph
from src.movement.training.il.checkpoint import NormalizerState
from src.movement.training.il.types import MovementILTrainingConfig
from src.movement.training.rollout import MovementRolloutBuffer


@dataclass(frozen=True)
class MovementPpoConfig:
    cfg_path: Path
    il_checkpoint_path: Path | None
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
    yellow_duration: int
    min_green_steps: int
    demand_scale_min: float
    demand_scale_max: float
    global_reward_weight: float
    speed_change_weight: float
    reward_clip: float
    teleport_penalty: float
    max_teleports_per_rollout: int
    time_to_teleport: int | None
    target_kl: float
    gui: bool
    initial_occupancy_min: float
    initial_occupancy_max: float
    eval_every: int
    eval_steps: int
    eval_seeds: tuple[int, ...]
    eval_policies: tuple[EvaluationPolicy, ...]
    eval_demand_scale: float
    save_every: int
    print_every: int
    checkpoint_dir: Path
    log_dir: Path
    device: str
    seed: int
    fixed_rollout_seed: int | None
    resume_checkpoint_path: Path | None


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
    mean_speed_change_density: float
    normalized_entropy: float
    mean_top_action_probability: float
    policy_decision_fraction: float
    teleport_count: int
    mean_demand_scale: float
    minimum_demand_scale: float
    maximum_demand_scale: float
    simulation_elapsed_s: float


@dataclass(frozen=True)
class CollectedRollout:
    buffer: MovementRolloutBuffer
    stats: RolloutStats
    seed: int


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
    speed_change_densities: tuple[float, ...]
    teleport_count: int
