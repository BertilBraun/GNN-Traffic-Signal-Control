"""Rollout collection for movement PPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from random import Random
from time import perf_counter

import torch
from torch.distributions import Categorical
import traci

from scripts.collect_il_data import resolve_sumocfg_net_path
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.experiment_config import CitySplit
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
from src.movement.training.il.checkpoint import MovementCheckpointMetadata
from src.movement.training.normalizer_state import NormalizerState, normalizer_from_state
from src.movement.training.ppo.policy import (
    allowed_action_masks,
    cached_policy_tensor_sample,
    current_sample,
    forward_tensor_policy,
    masked_phase_logits,
    rollout_context,
    states_from_actions,
    timed_current_sample,
)
from src.movement.training.ppo.reward import SpeedChangeTracker, advance_and_reward
from src.movement.training.ppo.rollout_metrics import RolloutMetrics
from src.movement.training.ppo.types import (
    CollectedRollout,
    MovementPpoConfig,
    RolloutContext,
    RolloutCity,
    RolloutStats,
)
from src.movement.training.rollout import MovementRolloutBuffer
from src.movement.training.rollout.types import MovementTransition


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
    rollout_index: int
    rollout_seed: int
    rollout_city: RolloutCity
    warming_up: bool


@dataclass(frozen=True)
class RolloutRunResult:
    buffer: MovementRolloutBuffer
    stats: RolloutStats
    bootstrap_values: tuple[float, ...]


def collect_computed_rollouts(request: RolloutCollectionRequest) -> tuple[CollectedRollout, ...]:
    assignments = rollout_schedule(
        config=request.config,
        iteration=request.iteration,
    )
    if request.pool is None:
        return tuple(
            collect_computed_rollout(
                config=request.config,
                model=request.model,
                lane_normalizer=request.lane_normalizer,
                movement_normalizer=request.movement_normalizer,
                device=request.device,
                rollout_seed=assignment.rollout_seed,
                rollout_city=assignment.rollout_city,
                warming_up=request.warming_up,
            )
            for assignment in assignments
        )
    worker_requests = tuple(worker_request(request=request, assignment=assignment) for assignment in assignments)
    futures = {request.pool.submit(collect_computed_rollout_worker, worker): worker for worker in worker_requests}
    collected_rollouts: dict[int, CollectedRollout] = {}
    for future in as_completed(futures):
        worker = futures[future]
        try:
            collected_rollouts[worker.rollout_index] = future.result()
        except Exception as exc:
            raise RuntimeError(
                f'PPO rollout failed for city={worker.rollout_city.city_name} seed={worker.rollout_seed}'
            ) from exc
    return tuple(collected_rollouts[rollout_index] for rollout_index in range(len(worker_requests)))


@dataclass(frozen=True)
class ScheduledRollout:
    rollout_index: int
    rollout_seed: int
    rollout_city: RolloutCity


def rollout_schedule(
    config: MovementPpoConfig,
    iteration: int,
) -> tuple[ScheduledRollout, ...]:
    rollout_cities = effective_rollout_cities(config)
    expanded_cities = tuple(city for city in rollout_cities for _worker_index in range(city.rollout_workers))
    if len(expanded_cities) != config.rollouts_per_update:
        raise ValueError(
            'rollout city worker counts must equal rollouts_per_update: '
            f'{len(expanded_cities)} != {config.rollouts_per_update}'
        )
    return tuple(
        ScheduledRollout(
            rollout_index=rollout_index,
            rollout_seed=rollout_seed(
                training_seed=config.seed,
                iteration=iteration,
                rollout_index=rollout_index,
                rollouts_per_update=config.rollouts_per_update,
                fixed_rollout_seed=config.fixed_rollout_seed,
            ),
            rollout_city=expanded_cities[rollout_index],
        )
        for rollout_index in range(config.rollouts_per_update)
    )


def effective_rollout_cities(config: MovementPpoConfig) -> tuple[RolloutCity, ...]:
    if config.rollout_cities:
        return config.rollout_cities
    return (
        RolloutCity(
            city_name=config.cfg_path.stem,
            city_split=CitySplit.TRAIN,
            sumo_config_path=config.cfg_path,
            rollout_workers=config.rollouts_per_update,
        ),
    )


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
        rollout_city=request.rollout_city,
        warming_up=request.warming_up,
    )


def collect_computed_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
    rollout_city: RolloutCity,
    warming_up: bool,
) -> CollectedRollout:
    rollout = collect_rollout(
        config=config,
        model=model,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        rollout_seed=rollout_seed,
        rollout_city=rollout_city,
    )
    rollout.buffer.compute_returns_and_advantages(
        use_discounted_return_targets=warming_up,
        bootstrap_values=rollout.bootstrap_values,
    )
    return CollectedRollout(
        buffer=rollout.buffer,
        stats=rollout.stats,
        seed=rollout_seed,
        city_name=rollout_city.city_name,
        city_split=rollout_city.city_split,
    )


def collect_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
    rollout_city: RolloutCity,
) -> RolloutRunResult:
    net_path = resolve_sumocfg_net_path(rollout_city.sumo_config_path)
    demand_scale = sample_demand_scale(
        demand_scale_min=config.demand_scale_min,
        demand_scale_max=config.demand_scale_max,
        seed=rollout_seed,
    )
    demand_route_files = route_files_for_demand_scale(
        cfg_path=rollout_city.sumo_config_path,
        demand_scale=demand_scale,
    )
    target_occupancy = sample_target_occupancy(
        minimum_occupancy=config.initial_occupancy_min,
        maximum_occupancy=config.initial_occupancy_max,
        seed=rollout_seed,
    )
    initial_population_started = perf_counter()
    initial_population = generate_initial_traffic_population(
        cfg_path=rollout_city.sumo_config_path,
        net_path=net_path,
        target_occupancy=target_occupancy,
        seed=rollout_seed,
    )
    initial_population_seconds = perf_counter() - initial_population_started
    runtime = MovementControlRuntime(
        cfg_path=rollout_city.sumo_config_path,
        gui=config.gui,
        seed=rollout_seed,
        yellow_duration=config.yellow_duration,
        min_green_steps=config.min_green_steps,
        time_to_teleport=config.time_to_teleport,
        additional_sumo_args=route_file_sumo_args((*demand_route_files.route_files, initial_population.route_file)),
    )
    simulation_started = perf_counter()
    try:
        runtime_start_started = perf_counter()
        runtime.start()
        runtime.step()
        runtime_start_seconds = perf_counter() - runtime_start_started
        context_started = perf_counter()
        context = rollout_context(
            cfg_path=rollout_city.sumo_config_path,
            programs=runtime.programs,
        )
        context_build_seconds = perf_counter() - context_started
        print_initial_population(
            rollout_seed=rollout_seed,
            city_name=rollout_city.city_name,
            initial_population=initial_population,
            demand_scale=demand_scale,
        )
        rollout = collect_runtime_rollout(
            config=config,
            runtime=runtime,
            context=context,
            model=model,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
            simulation_started=simulation_started,
            demand_scale=demand_scale,
            initial_population_seconds=initial_population_seconds,
            runtime_start_seconds=runtime_start_seconds,
            context_build_seconds=context_build_seconds,
            city_name=rollout_city.city_name,
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
    demand_scale: float,
    initial_population_seconds: float,
    runtime_start_seconds: float,
    context_build_seconds: float,
    city_name: str,
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
    metrics.observe_setup(
        initial_population_seconds=initial_population_seconds,
        runtime_start_seconds=runtime_start_seconds,
        context_build_seconds=context_build_seconds,
    )
    speed_change_tracker = SpeedChangeTracker()
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
            speed_change_tracker=speed_change_tracker,
            city_name=city_name,
        )
    bootstrap_started = perf_counter()
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
        city_name=city_name,
    )
    metrics.observe_bootstrap(perf_counter() - bootstrap_started)
    return RolloutRunResult(
        buffer=buffer,
        stats=metrics.stats(
            simulation_elapsed_s=perf_counter() - simulation_started,
            demand_scale=demand_scale,
        ),
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
    speed_change_tracker: SpeedChangeTracker,
    city_name: str,
) -> MovementControlState:
    sample_started = perf_counter()
    sample_result = timed_current_sample(
        context=context,
        programs=runtime.programs,
        vehicle_snapshot_collector=vehicle_snapshot_collector,
        lane_flow_tracker=lane_flow_tracker,
        control_state=control_state,
    )
    sample = sample_result.sample
    sample_seconds = perf_counter() - sample_started
    metrics.observe_sample(
        capture_seconds=sample_result.capture_seconds,
        index_seconds=sample_result.index_seconds,
        flow_seconds=sample_result.flow_seconds,
        feature_frame_seconds=sample_result.feature_frame_seconds,
        dataset_sample_seconds=sample_result.dataset_sample_seconds,
    )
    tensor_started = perf_counter()
    tensor_sample = cached_policy_tensor_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        city_name=city_name,
    )
    tensor_seconds = perf_counter() - tensor_started
    with torch.no_grad():
        model_started = perf_counter()
        _movement_scores, values, phase_logits = forward_tensor_policy(
            model=model,
            tensor_sample=tensor_sample,
            context=context,
            device=device,
        )
        model_seconds = perf_counter() - model_started + tensor_seconds
        action_started = perf_counter()
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
        action_seconds = perf_counter() - action_started
    apply_started = perf_counter()
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
    apply_seconds = perf_counter() - apply_started
    reward_started = perf_counter()
    interval_reward = advance_and_reward(
        runtime=runtime,
        context=context,
        decision_interval=config.decision_interval,
        global_reward_weight=config.global_reward_weight,
        speed_change_weight=config.speed_change_weight,
        reward_sample_interval=config.reward_sample_interval,
        reward_clip=config.reward_clip,
        teleport_penalty=config.teleport_penalty,
        speed_change_tracker=speed_change_tracker,
    )
    reward_seconds = perf_counter() - reward_started
    metrics.observe_reward(interval_reward)
    metrics.observe_decision(
        sample_seconds=sample_seconds,
        model_seconds=model_seconds,
        action_seconds=action_seconds,
        apply_seconds=apply_seconds,
        reward_seconds=reward_seconds,
    )
    buffer.add(
        MovementTransition(
            tensor_sample=tensor_sample,
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
    if any(
        accepted_states[traffic_light_id] != desired_states[traffic_light_id] for traffic_light_id in desired_states
    ):
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
    city_name: str,
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
    tensor_sample = cached_policy_tensor_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        city_name=city_name,
    )
    with torch.no_grad():
        _movement_scores, values, _phase_logits = forward_tensor_policy(
            model=model,
            tensor_sample=tensor_sample,
            context=context,
            device=device,
        )
    return tuple(float(value) for value in values.detach().cpu())


def worker_request(request: RolloutCollectionRequest, assignment: ScheduledRollout) -> RolloutWorkerRequest:
    return RolloutWorkerRequest(
        config=request.config,
        model_state={key: value.detach().cpu() for key, value in request.model.state_dict().items()},
        lane_feature_dim=request.metadata.lane_feature_dim,
        movement_feature_dim=request.metadata.movement_feature_dim,
        hidden_dim=request.metadata.hidden_dim,
        num_hops=request.metadata.num_hops,
        lane_normalizer=request.metadata.lane_normalizer,
        movement_normalizer=request.metadata.movement_normalizer,
        rollout_index=assignment.rollout_index,
        rollout_seed=assignment.rollout_seed,
        rollout_city=assignment.rollout_city,
        warming_up=request.warming_up,
    )


def sample_demand_scale(demand_scale_min: float, demand_scale_max: float, seed: int) -> float:
    if demand_scale_min == demand_scale_max:
        return demand_scale_min
    return Random(seed).uniform(demand_scale_min, demand_scale_max)


def print_initial_population(
    rollout_seed: int,
    city_name: str,
    initial_population: InitialTrafficPopulation,
    demand_scale: float,
) -> None:
    print(
        f'  rollout city={city_name} seed={rollout_seed} '
        f'demand_scale={demand_scale:.3f} '
        f'initial_occupancy={initial_population.target_occupancy:.3f} '
        f'initial_vehicles={initial_population.generated_vehicle_count}/'
        f'{initial_population.requested_vehicle_count}'
    )
