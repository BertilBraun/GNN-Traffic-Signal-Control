"""Rollout collection for movement PPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter

import torch
from torch.distributions import Categorical
import traci

from scripts.collect_il_data import resolve_sumocfg_net_path
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.features import (
    LaneGroupFlowTracker,
    MovementControlState,
    VehicleSnapshotCollector,
    movement_control_state_from_targets,
)
from src.movement.initial_traffic import (
    InitialTrafficPopulation,
    generate_initial_traffic_population,
    sample_target_occupancy,
)
from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.normalization import RunningNormalizer
from src.movement.runtime import MovementControlRuntime
from src.movement.training.il.checkpoint import MovementCheckpointMetadata, NormalizerState, normalizer_from_state
from src.movement.training.ppo.policy import (
    actions_from_states,
    allowed_action_masks,
    current_sample,
    forward_policy,
    masked_phase_logits,
    rollout_context,
    states_from_actions,
)
from src.movement.training.ppo.reward import advance_and_reward
from src.movement.training.ppo.rollout_metrics import RolloutMetrics
from src.movement.training.ppo.types import (
    CollectedRollout,
    MovementPpoConfig,
    RolloutContext,
    RolloutStats,
)
from src.movement.training.rollout import MovementRolloutBuffer
from src.movement.training.rollout_types import MovementTransition


@dataclass(frozen=True)
class RolloutCollectionRequest:
    config: MovementPpoConfig
    model: MovementActorCritic
    metadata: MovementCheckpointMetadata
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer
    device: torch.device
    iteration: int
    warming_up: bool
    pool: ProcessPoolExecutor | None


@dataclass(frozen=True)
class RolloutWorkerRequest:
    config: MovementPpoConfig
    model_state: dict[str, torch.Tensor]
    lane_feature_dim: int
    movement_feature_dim: int
    hidden_dim: int
    num_hops: int
    lane_normalizer: NormalizerState
    movement_normalizer: NormalizerState
    rollout_seed: int
    warming_up: bool


@dataclass(frozen=True)
class RolloutRunResult:
    buffer: MovementRolloutBuffer
    stats: RolloutStats
    bootstrap_values: tuple[float, ...]


def collect_computed_rollouts(request: RolloutCollectionRequest) -> tuple[CollectedRollout, ...]:
    seeds = tuple(
        rollout_seed(
            training_seed=request.config.seed,
            iteration=request.iteration,
            rollout_index=rollout_index,
            rollouts_per_update=request.config.rollouts_per_update,
            fixed_rollout_seed=request.config.fixed_rollout_seed,
        )
        for rollout_index in range(request.config.rollouts_per_update)
    )
    if request.pool is None:
        return tuple(
            collect_computed_rollout(
                config=request.config,
                model=request.model,
                lane_normalizer=request.lane_normalizer,
                movement_normalizer=request.movement_normalizer,
                device=request.device,
                rollout_seed=seed,
                warming_up=request.warming_up,
            )
            for seed in seeds
        )
    worker_requests = tuple(worker_request(request=request, rollout_seed=seed) for seed in seeds)
    futures = tuple(request.pool.submit(collect_computed_rollout_worker, worker) for worker in worker_requests)
    return tuple(future.result() for future in as_completed(futures))


def rollout_seed(
    training_seed: int,
    iteration: int,
    rollout_index: int,
    rollouts_per_update: int,
    fixed_rollout_seed: int | None,
) -> int:
    if fixed_rollout_seed is not None:
        return fixed_rollout_seed
    return training_seed + iteration * rollouts_per_update + rollout_index


def collect_computed_rollout_worker(request: RolloutWorkerRequest) -> CollectedRollout:
    model = MovementActorCritic(
        lane_feature_dim=request.lane_feature_dim,
        movement_feature_dim=request.movement_feature_dim,
        hidden_dim=request.hidden_dim,
        num_hops=request.num_hops,
    )
    model.load_state_dict(request.model_state)
    return collect_computed_rollout(
        config=request.config,
        model=model,
        lane_normalizer=normalizer_from_state(request.lane_normalizer),
        movement_normalizer=normalizer_from_state(request.movement_normalizer),
        device=torch.device('cpu'),
        rollout_seed=request.rollout_seed,
        warming_up=request.warming_up,
    )


def collect_computed_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
    warming_up: bool,
) -> CollectedRollout:
    rollout = collect_rollout(
        config=config,
        model=model,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        rollout_seed=rollout_seed,
    )
    rollout.buffer.compute_returns_and_advantages(
        use_discounted_return_targets=warming_up,
        bootstrap_values=rollout.bootstrap_values,
    )
    return CollectedRollout(
        buffer=rollout.buffer,
        stats=rollout.stats,
        seed=rollout_seed,
    )


def collect_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
) -> RolloutRunResult:
    net_path = resolve_sumocfg_net_path(config.cfg_path)
    demand_route_files = route_files_for_demand_scale(
        cfg_path=config.cfg_path,
        demand_scale=config.demand_scale,
    )
    target_occupancy = sample_target_occupancy(
        minimum_occupancy=config.initial_occupancy_min,
        maximum_occupancy=config.initial_occupancy_max,
        seed=rollout_seed,
    )
    initial_population = generate_initial_traffic_population(
        cfg_path=config.cfg_path,
        net_path=net_path,
        target_occupancy=target_occupancy,
        seed=rollout_seed,
    )
    runtime = MovementControlRuntime(
        cfg_path=config.cfg_path,
        gui=config.gui,
        seed=rollout_seed,
        yellow_duration=config.yellow_duration,
        min_green_steps=config.min_green_steps,
        time_to_teleport=config.time_to_teleport,
        additional_sumo_args=route_file_sumo_args((*demand_route_files.route_files, initial_population.route_file)),
    )
    simulation_started = perf_counter()
    try:
        runtime.start()
        runtime.step()
        context = rollout_context(
            cfg_path=config.cfg_path,
            programs=runtime.programs,
        )
        print_initial_population(rollout_seed=rollout_seed, initial_population=initial_population)
        rollout = collect_runtime_rollout(
            config=config,
            runtime=runtime,
            context=context,
            model=model,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
            simulation_started=simulation_started,
        )
        return rollout
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()


def collect_runtime_rollout(
    config: MovementPpoConfig,
    runtime: MovementControlRuntime,
    context: RolloutContext,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    simulation_started: float,
) -> RolloutRunResult:
    vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
    lane_flow_tracker = LaneGroupFlowTracker(
        graph=context.graph,
        lane_ids_by_edge=context.lane_ids_by_edge,
        lane_geometries=context.lane_geometries,
        decision_interval_s=config.decision_interval,
    )
    control_state = MovementControlState()
    buffer = MovementRolloutBuffer(
        traffic_light_count=len(context.traffic_light_ids),
        gamma=config.gamma,
        lam=config.lam,
    )
    metrics = RolloutMetrics()
    model.eval()
    for _decision_step in range(config.steps_per_rollout):
        if not runtime.is_running():
            break
        control_state = collect_decision_transition(
            config=config,
            runtime=runtime,
            context=context,
            model=model,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
            vehicle_snapshot_collector=vehicle_snapshot_collector,
            lane_flow_tracker=lane_flow_tracker,
            control_state=control_state,
            buffer=buffer,
            metrics=metrics,
        )
    next_values = bootstrap_values(
        runtime=runtime,
        context=context,
        control_state=control_state,
        vehicle_snapshot_collector=vehicle_snapshot_collector,
        lane_flow_tracker=lane_flow_tracker,
        model=model,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
    return RolloutRunResult(
        buffer=buffer,
        stats=metrics.stats(simulation_elapsed_s=perf_counter() - simulation_started),
        bootstrap_values=next_values,
    )


def collect_decision_transition(
    config: MovementPpoConfig,
    runtime: MovementControlRuntime,
    context: RolloutContext,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    vehicle_snapshot_collector: VehicleSnapshotCollector,
    lane_flow_tracker: LaneGroupFlowTracker,
    control_state: MovementControlState,
    buffer: MovementRolloutBuffer,
    metrics: RolloutMetrics,
) -> MovementControlState:
    sample = current_sample(
        context=context,
        programs=runtime.programs,
        vehicle_snapshot_collector=vehicle_snapshot_collector,
        lane_flow_tracker=lane_flow_tracker,
        control_state=control_state,
    )
    with torch.no_grad():
        _movement_scores, values, phase_logits = forward_policy(
            model=model,
            sample=sample,
            context=context,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
        )
        action_masks = allowed_action_masks(
            runtime=runtime,
            context=context,
            programs=runtime.programs,
        )
        masked_logits = masked_phase_logits(phase_logits, action_masks)
        distributions = tuple(Categorical(logits=logits) for logits in masked_logits)
        metrics.observe_policy(distributions=distributions, action_masks=action_masks)
        actions = tuple(int(distribution.sample().item()) for distribution in distributions)
        old_log_probs = tuple(
            float(distribution.log_prob(torch.tensor(action, device=device)).detach().cpu())
            for distribution, action in zip(distributions, actions)
        )
    accepted_states = request_runtime_targets(
        runtime=runtime,
        context=context,
        actions=actions,
    )
    next_control_state = movement_control_state_from_targets(
        graph=context.graph,
        programs=runtime.programs,
        target_states=accepted_states,
    )
    interval_reward = advance_and_reward(
        runtime=runtime,
        context=context,
        decision_interval=config.decision_interval,
        global_reward_weight=config.global_reward_weight,
        reward_clip=config.reward_clip,
        teleport_penalty=config.teleport_penalty,
    )
    metrics.observe_reward(interval_reward)
    buffer.add(
        MovementTransition(
            sample=sample,
            actions=actions,
            old_log_probs=old_log_probs,
            action_masks=action_masks,
            rewards=interval_reward.rewards,
            values=tuple(float(value) for value in values.detach().cpu()),
            done=not runtime.is_running(),
        )
    )
    return next_control_state


def request_runtime_targets(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    actions: Sequence[int],
) -> Mapping[str, str]:
    desired_states = states_from_actions(
        programs=runtime.programs,
        traffic_light_ids=context.traffic_light_ids,
        actions=actions,
    )
    accepted_states = runtime.request_targets(desired_states)
    accepted_actions = actions_from_states(
        programs=runtime.programs,
        traffic_light_ids=context.traffic_light_ids,
        states=accepted_states,
    )
    if accepted_actions != tuple(actions):
        raise RuntimeError('Runtime rejected an action permitted by the PPO action mask.')
    return accepted_states


def bootstrap_values(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    control_state: MovementControlState,
    vehicle_snapshot_collector: VehicleSnapshotCollector,
    lane_flow_tracker: LaneGroupFlowTracker,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> tuple[float, ...]:
    if not runtime.is_running():
        return tuple(0.0 for _traffic_light_id in context.traffic_light_ids)
    sample = current_sample(
        context=context,
        programs=runtime.programs,
        vehicle_snapshot_collector=vehicle_snapshot_collector,
        lane_flow_tracker=lane_flow_tracker,
        control_state=control_state,
    )
    with torch.no_grad():
        _movement_scores, values, _phase_logits = forward_policy(
            model=model,
            sample=sample,
            context=context,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
        )
    return tuple(float(value) for value in values.detach().cpu())


def worker_request(request: RolloutCollectionRequest, rollout_seed: int) -> RolloutWorkerRequest:
    return RolloutWorkerRequest(
        config=request.config,
        model_state={key: value.detach().cpu() for key, value in request.model.state_dict().items()},
        lane_feature_dim=request.metadata.lane_feature_dim,
        movement_feature_dim=request.metadata.movement_feature_dim,
        hidden_dim=request.metadata.hidden_dim,
        num_hops=request.metadata.num_hops,
        lane_normalizer=request.metadata.lane_normalizer,
        movement_normalizer=request.metadata.movement_normalizer,
        rollout_seed=rollout_seed,
        warming_up=request.warming_up,
    )


def print_initial_population(rollout_seed: int, initial_population: InitialTrafficPopulation) -> None:
    print(
        f'  rollout seed={rollout_seed} '
        f'initial_occupancy={initial_population.target_occupancy:.3f} '
        f'initial_vehicles={initial_population.generated_vehicle_count}/'
        f'{initial_population.requested_vehicle_count}'
    )
