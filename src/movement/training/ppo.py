"""PPO training for movement-score traffic signal policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import cast

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.optimizer import StateDict
from torch.utils.tensorboard import SummaryWriter
import traci

from scripts.collect_il_data import (
    lane_inputs_from_net,
    resolve_sumocfg_net_path,
)
from src.movement.dataset import MovementDatasetSample, build_dataset_sample
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.evaluation import (
    EvaluationAggregate,
    EvaluationMetrics,
    EvaluationPolicy,
    EvaluationRecord,
    LearnedPolicyConfig,
    aggregate_records,
    print_aggregate_metric_table,
    run_evaluation_episode,
    write_aggregate_json,
    write_records_csv,
)
from src.movement.features import (
    LaneGroupGeometry,
    LaneGroupFlowTracker,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
    movement_control_state_from_targets,
)
from src.movement.graph import build_movement_graph
from src.movement.graph_schema import MovementGraph
from src.movement.initial_traffic import (
    generate_initial_traffic_population,
    sample_target_occupancy,
)
from src.movement.models.bipartite_gnn import MovementActorCritic, MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.runtime import MovementControlRuntime
from src.movement.schema import TrafficLightProgram
from src.movement.training.il import (
    MovementCheckpointMetadata,
    MovementILTrainingConfig,
    NormalizerState,
    edge_tensors_from_sample,
    load_movement_checkpoint,
    normalizer_from_state,
    save_movement_checkpoint,
    tensors_from_sample,
)
from src.movement.training.rollout import MovementRolloutBuffer, MovementTransition


@dataclass(frozen=True)
class MovementPpoConfig:
    cfg_path: Path
    il_checkpoint_path: Path | None
    iterations: int
    steps_per_rollout: int
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
    demand_scale: float
    global_reward_weight: float
    reward_clip: float
    teleport_penalty: float
    max_teleports_per_rollout: int
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
    normalized_entropy: float
    mean_top_action_probability: float
    policy_decision_fraction: float
    teleport_count: int
    simulation_elapsed_s: float


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
    baseline_records: tuple[EvaluationRecord, ...]
    learned_checkpoint_score: float | None


@dataclass(frozen=True)
class IntervalRewardResult:
    rewards: tuple[float, ...]
    raw_rewards: tuple[float, ...]
    local_delay_densities: tuple[float, ...]
    global_delay_density: float
    teleport_count: int


def train_movement_ppo(config: MovementPpoConfig) -> MovementPpoTrainingResult:
    """Fine-tune a movement scorer with PPO."""
    if config.max_teleports_per_rollout < 0:
        raise ValueError('max_teleports_per_rollout must not be negative.')
    if config.target_kl <= 0.0:
        raise ValueError('target_kl must be positive.')
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if config.resume_checkpoint_path is None:
        if config.il_checkpoint_path is None:
            raise ValueError('il_checkpoint_path is required when resume_checkpoint_path is not set.')
        model, metadata = _load_actor_critic(config.il_checkpoint_path, device=config.device)
        _zero_value_output(model)
        completed_iteration = 0
        best_checkpoint_score = float('inf')
    else:
        resume_checkpoint = _load_ppo_checkpoint_payload(
            checkpoint_path=config.resume_checkpoint_path,
            device=config.device,
        )
        model, metadata = _model_and_metadata_from_ppo_checkpoint(
            checkpoint=resume_checkpoint,
            device=config.device,
        )
        completed_iteration = resume_checkpoint.iteration
        best_checkpoint_score = resume_checkpoint.best_checkpoint_score
        torch.set_rng_state(resume_checkpoint.torch_random_state.cpu())
        if torch.cuda.is_available() and resume_checkpoint.cuda_random_states:
            torch.cuda.set_rng_state_all(list(resume_checkpoint.cuda_random_states))
    lane_normalizer = normalizer_from_state(metadata.lane_normalizer)
    movement_normalizer = normalizer_from_state(metadata.movement_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.resume_checkpoint_path is not None:
        optimizer.load_state_dict(resume_checkpoint.optimizer_state)
    writer = SummaryWriter(log_dir=str(config.log_dir))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = config.checkpoint_dir / 'movement_ppo_latest.pt'
    started = perf_counter()
    baseline_evaluation_records: tuple[EvaluationRecord, ...] = ()
    first_iteration = completed_iteration + 1
    if first_iteration > config.iterations:
        raise ValueError(
            f'Resume checkpoint is already at iteration {completed_iteration}, '
            f'which is not below target iteration {config.iterations}.'
        )
    if config.resume_checkpoint_path is not None:
        print(f'Resuming PPO from iteration {completed_iteration}; target iteration={config.iterations}')
    try:
        for iteration in range(first_iteration, config.iterations + 1):
            iteration_started = perf_counter()
            warming_up = iteration <= config.value_warmup_iterations
            _set_actor_grad(model, requires_grad=not warming_up)
            _set_value_grad(model, requires_grad=True)
            buffer, rollout_stats, bootstrap_values = _collect_rollout(
                config=config,
                model=model,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
                rollout_seed=_rollout_seed(
                    training_seed=config.seed,
                    iteration=iteration,
                    fixed_rollout_seed=config.fixed_rollout_seed,
                ),
            )
            rollout_finished = perf_counter()
            buffer.compute_returns_and_advantages(
                use_discounted_return_targets=warming_up,
                bootstrap_values=bootstrap_values,
            )
            diagnostics = _training_diagnostics(buffer)
            update_skipped = rollout_stats.teleport_count > config.max_teleports_per_rollout
            update_stats = (
                _skipped_update_stats()
                if update_skipped
                else _update_ppo(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    device=device,
                    config=config,
                    warming_up=warming_up,
                )
            )
            update_finished = perf_counter()
            _write_training_scalars(
                writer=writer,
                iteration=iteration,
                rollout_stats=rollout_stats,
                diagnostics=diagnostics,
                update_stats=update_stats,
            )
            writer.add_scalar('diagnostics/update_skipped', float(update_skipped), iteration)
            writer.add_scalar('timing/update_seconds', update_finished - rollout_finished, iteration)
            writer.add_scalar('timing/iteration_seconds', update_finished - iteration_started, iteration)
            if config.print_every > 0 and (iteration == 1 or iteration % config.print_every == 0):
                phase = 'skip' if update_skipped else ('value' if warming_up else 'ppo')
                print(
                    f'[{phase}] iter={iteration}/{config.iterations} '
                    f'reward={rollout_stats.mean_reward:+.4f} '
                    f'return={diagnostics.mean_return:+.4f} '
                    f'ev={diagnostics.explained_variance:+.3f} '
                    f'policy_loss={update_stats.policy_loss:+.4f} '
                    f'value_loss={update_stats.value_loss:.4f} '
                    f'entropy={update_stats.entropy:.4f} '
                    f'norm_entropy={rollout_stats.normalized_entropy:.3f} '
                    f'top_p={rollout_stats.mean_top_action_probability:.3f} '
                    f'clip={rollout_stats.reward_clip_fraction:.1%}/{update_stats.ratio_clip_fraction:.1%} '
                    f'kl={update_stats.approximate_kl:.5f} '
                    f'kl_stop={int(update_stats.early_stopped)} '
                    f'grad={update_stats.backbone_gradient_norm:.3f}/'
                    f'{update_stats.value_head_gradient_norm:.3f} '
                    f'teleports={rollout_stats.teleport_count} '
                    f'rollout={rollout_finished - iteration_started:.1f}s '
                    f'update={update_finished - rollout_finished:.1f}s '
                    f'elapsed={perf_counter() - started:.1f}s'
                )
            completed_iteration = iteration
            if config.eval_every > 0 and iteration % config.eval_every == 0:
                evaluation_started = perf_counter()
                evaluation_result = _run_training_evaluation(
                    config=config,
                    model=model,
                    metadata=metadata,
                    iteration=iteration,
                    writer=writer,
                    baseline_records=baseline_evaluation_records,
                )
                baseline_evaluation_records = evaluation_result.baseline_records
                if (
                    evaluation_result.learned_checkpoint_score is not None
                    and evaluation_result.learned_checkpoint_score < best_checkpoint_score
                ):
                    best_checkpoint_score = evaluation_result.learned_checkpoint_score
                    _save_ppo_checkpoint(
                        path=config.checkpoint_dir / 'movement_ppo_best.pt',
                        model=model,
                        optimizer=optimizer,
                        metadata=metadata,
                        iteration=iteration,
                        best_checkpoint_score=best_checkpoint_score,
                    )
                    _save_actor_checkpoint(
                        path=config.checkpoint_dir / 'movement_policy_best.pt',
                        model=model,
                        metadata=metadata,
                        loss=0.0,
                    )
                    print(
                        f'  new best completion-adjusted time-loss score='
                        f'{best_checkpoint_score:.3f} at iteration {iteration}'
                    )
                print(f'  evaluation elapsed={perf_counter() - evaluation_started:.1f}s')
            if config.save_every > 0 and iteration % config.save_every == 0:
                _save_ppo_checkpoint(
                    path=config.checkpoint_dir / f'movement_ppo_iter_{iteration:04d}.pt',
                    model=model,
                    optimizer=optimizer,
                    metadata=metadata,
                    iteration=iteration,
                    best_checkpoint_score=best_checkpoint_score,
                )
    finally:
        _save_ppo_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            metadata=metadata,
            iteration=completed_iteration,
            best_checkpoint_score=best_checkpoint_score,
        )
        _save_actor_checkpoint(
            path=config.checkpoint_dir / 'movement_policy_latest.pt',
            model=model,
            metadata=metadata,
            loss=0.0,
        )
        writer.close()
    return MovementPpoTrainingResult(
        checkpoint_path=last_checkpoint_path,
        iterations=completed_iteration,
    )


@dataclass(frozen=True)
class PpoUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approximate_kl: float
    ratio_clip_fraction: float
    backbone_gradient_norm: float
    value_head_gradient_norm: float
    early_stopped: bool


def _rollout_seed(
    training_seed: int,
    iteration: int,
    fixed_rollout_seed: int | None,
) -> int:
    if fixed_rollout_seed is not None:
        return fixed_rollout_seed
    return training_seed + iteration


def _collect_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
) -> tuple[MovementRolloutBuffer, RolloutStats, tuple[float, ...]]:
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
        additional_sumo_args=route_file_sumo_args(
            (
                *demand_route_files.route_files,
                initial_population.route_file,
            )
        ),
    )
    rewards: list[float] = []
    raw_rewards: list[float] = []
    local_delay_densities: list[float] = []
    global_delay_densities: list[float] = []
    normalized_entropies: list[float] = []
    top_action_probabilities: list[float] = []
    policy_decision_count = 0
    total_decision_count = 0
    teleport_count = 0
    simulation_started = perf_counter()
    try:
        runtime.start()
        runtime.step()
        context = _rollout_context(
            cfg_path=config.cfg_path,
            programs=runtime.programs,
        )
        vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
        lane_flow_tracker = LaneGroupFlowTracker(
            graph=context.graph,
            lane_ids_by_edge=context.lane_ids_by_edge,
            lane_geometries=context.lane_geometries,
            decision_interval_s=config.decision_interval,
        )
        control_state = MovementControlState()
        print(
            f'  rollout seed={rollout_seed} '
            f'initial_occupancy={initial_population.target_occupancy:.3f} '
            f'initial_vehicles={initial_population.generated_vehicle_count}/'
            f'{initial_population.requested_vehicle_count}'
        )
        buffer = MovementRolloutBuffer(
            traffic_light_count=len(context.traffic_light_ids),
            gamma=config.gamma,
            lam=config.lam,
        )
        model.eval()
        for _decision_step in range(config.steps_per_rollout):
            if not runtime.is_running():
                break
            sample = _current_sample(
                context=context,
                programs=runtime.programs,
                vehicle_snapshot_collector=vehicle_snapshot_collector,
                lane_flow_tracker=lane_flow_tracker,
                control_state=control_state,
            )
            with torch.no_grad():
                movement_scores, values, phase_logits = _forward_policy(
                    model=model,
                    sample=sample,
                    context=context,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    device=device,
                )
                del movement_scores
                action_masks = _allowed_action_masks(
                    runtime=runtime,
                    context=context,
                    programs=runtime.programs,
                )
                masked_phase_logits = _masked_phase_logits(phase_logits, action_masks)
                distributions = tuple(Categorical(logits=logits) for logits in masked_phase_logits)
                for distribution, action_mask in zip(distributions, action_masks):
                    valid_action_count = sum(action_mask)
                    total_decision_count += 1
                    if valid_action_count > 1:
                        policy_decision_count += 1
                        normalized_entropies.append(
                            float(distribution.entropy().detach().cpu()) / log(valid_action_count)
                        )
                        top_action_probabilities.append(float(distribution.probs.max().detach().cpu()))
                actions = tuple(int(distribution.sample().item()) for distribution in distributions)
                old_log_probs = tuple(
                    float(distribution.log_prob(torch.tensor(action, device=device)).detach().cpu())
                    for distribution, action in zip(distributions, actions)
                )
            desired_states = _states_from_actions(
                programs=runtime.programs,
                traffic_light_ids=context.traffic_light_ids,
                actions=actions,
            )
            accepted_states = runtime.request_targets(desired_states)
            accepted_actions = _actions_from_states(
                programs=runtime.programs,
                traffic_light_ids=context.traffic_light_ids,
                states=accepted_states,
            )
            if accepted_actions != actions:
                raise RuntimeError('Runtime rejected an action permitted by the PPO action mask.')
            control_state = movement_control_state_from_targets(
                graph=context.graph,
                programs=runtime.programs,
                target_states=accepted_states,
            )
            interval_reward = _advance_and_reward(
                runtime=runtime,
                context=context,
                decision_interval=config.decision_interval,
                global_reward_weight=config.global_reward_weight,
                reward_clip=config.reward_clip,
                teleport_penalty=config.teleport_penalty,
            )
            teleport_count += interval_reward.teleport_count
            rewards.extend(interval_reward.rewards)
            raw_rewards.extend(interval_reward.raw_rewards)
            local_delay_densities.extend(interval_reward.local_delay_densities)
            global_delay_densities.append(interval_reward.global_delay_density)
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
        bootstrap_values = _bootstrap_values(
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
        return (
            buffer,
            RolloutStats(
                mean_reward=sum(rewards) / max(1, len(rewards)),
                reward_standard_deviation=_standard_deviation(rewards),
                minimum_reward=min(rewards, default=0.0),
                maximum_reward=max(rewards, default=0.0),
                raw_reward_standard_deviation=_standard_deviation(raw_rewards),
                minimum_raw_reward=min(raw_rewards, default=0.0),
                maximum_raw_reward=max(raw_rewards, default=0.0),
                reward_clip_fraction=(
                    sum(clipped != raw for clipped, raw in zip(rewards, raw_rewards)) / max(1, len(rewards))
                ),
                mean_local_delay_density=sum(local_delay_densities) / max(1, len(local_delay_densities)),
                mean_global_delay_density=sum(global_delay_densities) / max(1, len(global_delay_densities)),
                normalized_entropy=sum(normalized_entropies) / max(1, len(normalized_entropies)),
                mean_top_action_probability=(sum(top_action_probabilities) / max(1, len(top_action_probabilities))),
                policy_decision_fraction=policy_decision_count / max(1, total_decision_count),
                teleport_count=teleport_count,
                simulation_elapsed_s=perf_counter() - simulation_started,
            ),
            bootstrap_values,
        )
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()


def _bootstrap_values(
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
    sample = _current_sample(
        context=context,
        programs=runtime.programs,
        vehicle_snapshot_collector=vehicle_snapshot_collector,
        lane_flow_tracker=lane_flow_tracker,
        control_state=control_state,
    )
    with torch.no_grad():
        _movement_scores, values, _phase_logits = _forward_policy(
            model=model,
            sample=sample,
            context=context,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
        )
    return tuple(float(value) for value in values.detach().cpu())


def _update_ppo(
    model: MovementActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: MovementRolloutBuffer,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    config: MovementPpoConfig,
    warming_up: bool,
) -> PpoUpdateStats:
    model.train()
    epochs = config.warmup_epochs if warming_up else config.update_epochs
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    approximate_kls: list[float] = []
    ratio_clip_fractions: list[float] = []
    backbone_gradient_norms: list[float] = []
    value_head_gradient_norms: list[float] = []
    early_stopped = False
    for _epoch in range(epochs):
        for batch in buffer.iterate_minibatches(config.transitions_per_batch, device=device):
            batch_log_probs = []
            batch_entropies = []
            batch_values = []
            for transition in batch.transitions:
                _movement_scores, values, phase_logits = _forward_policy(
                    model=model,
                    sample=transition.sample,
                    context=_policy_context_from_sample(transition.sample),
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    device=device,
                )
                distributions = tuple(
                    Categorical(logits=logits)
                    for logits in _masked_phase_logits(
                        phase_logits,
                        transition.action_masks,
                    )
                )
                action_tensor = torch.tensor(transition.actions, dtype=torch.long, device=device)
                batch_log_probs.append(
                    torch.stack(
                        tuple(
                            distribution.log_prob(action) for distribution, action in zip(distributions, action_tensor)
                        )
                    )
                )
                batch_entropies.append(torch.stack(tuple(distribution.entropy() for distribution in distributions)))
                batch_values.append(values)
            new_log_probs = torch.stack(batch_log_probs)
            entropy_values = torch.stack(batch_entropies)
            entropy = (
                entropy_values[batch.policy_mask].mean()
                if bool(batch.policy_mask.any())
                else entropy_values.new_zeros(())
            )
            values = torch.stack(batch_values)
            value_loss = F.mse_loss(values, batch.returns)
            if warming_up:
                policy_loss = values.new_zeros(())
                loss = config.value_coefficient * value_loss
                approximate_kl = values.new_zeros(())
                ratio_clip_fraction = values.new_zeros(())
            else:
                log_ratio = new_log_probs - batch.old_log_probs
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                policy_objective = torch.min(
                    ratio * batch.advantages,
                    clipped_ratio * batch.advantages,
                )
                policy_loss = (
                    -policy_objective[batch.policy_mask].mean()
                    if bool(batch.policy_mask.any())
                    else policy_objective.new_zeros(())
                )
                approximate_kl = (
                    ((ratio - 1.0) - log_ratio)[batch.policy_mask].mean()
                    if bool(batch.policy_mask.any())
                    else ratio.new_zeros(())
                )
                ratio_clip_fraction = (
                    ((ratio - 1.0).abs() > config.clip_epsilon)[batch.policy_mask].float().mean()
                    if bool(batch.policy_mask.any())
                    else ratio.new_zeros(())
                )
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                if float(approximate_kl.detach().cpu()) > config.target_kl:
                    early_stopped = True
                    break
            optimizer.zero_grad()
            loss.backward()
            backbone_gradient_norms.append(
                _gradient_norm(
                    tuple(
                        parameter
                        for module in (model.lane_encoder, model.movement_encoder, model.hops, model.score_head)
                        for parameter in module.parameters()
                    )
                )
            )
            value_head_gradient_norms.append(_gradient_norm(tuple(model.value_head.parameters())))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
            approximate_kls.append(float(approximate_kl.detach().cpu()))
            ratio_clip_fractions.append(float(ratio_clip_fraction.detach().cpu()))
        if early_stopped:
            break
    return PpoUpdateStats(
        policy_loss=sum(policy_losses) / max(1, len(policy_losses)),
        value_loss=sum(value_losses) / max(1, len(value_losses)),
        entropy=sum(entropies) / max(1, len(entropies)),
        total_loss=sum(total_losses) / max(1, len(total_losses)),
        approximate_kl=sum(approximate_kls) / max(1, len(approximate_kls)),
        ratio_clip_fraction=sum(ratio_clip_fractions) / max(1, len(ratio_clip_fractions)),
        backbone_gradient_norm=sum(backbone_gradient_norms) / max(1, len(backbone_gradient_norms)),
        value_head_gradient_norm=sum(value_head_gradient_norms) / max(1, len(value_head_gradient_norms)),
        early_stopped=early_stopped,
    )


def _skipped_update_stats() -> PpoUpdateStats:
    return PpoUpdateStats(
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        total_loss=0.0,
        approximate_kl=0.0,
        ratio_clip_fraction=0.0,
        backbone_gradient_norm=0.0,
        value_head_gradient_norm=0.0,
        early_stopped=False,
    )


def _rollout_context(
    cfg_path: Path,
    programs: Mapping[str, TrafficLightProgram],
) -> RolloutContext:
    net_path = resolve_sumocfg_net_path(cfg_path)
    graph = build_movement_graph(programs, net_path=net_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    traffic_light_ids = tuple(sorted((str(traffic_light_id) for traffic_light_id in programs), key=str))
    movement_ids_by_traffic_light = tuple(
        tuple(int(value) for value in graph.phase_incidences[programs[traffic_light_id].traffic_light_id].movement_ids)
        for traffic_light_id in traffic_light_ids
    )
    incoming_lanes_by_traffic_light = {
        traffic_light_id: tuple(
            dict.fromkeys(str(movement.incoming_lane_id) for movement in programs[traffic_light_id].movements)
        )
        for traffic_light_id in traffic_light_ids
    }
    incoming_lane_length_by_traffic_light = {
        traffic_light_id: sum(float(lane_geometries[_edge_id_from_lane_id(lane_id)].length_m) for lane_id in lane_ids)
        for traffic_light_id, lane_ids in incoming_lanes_by_traffic_light.items()
    }
    speed_limit_by_lane = {
        lane_id: float(lane_geometries[_edge_id_from_lane_id(lane_id)].speed_limit_mps)
        for lane_ids in incoming_lanes_by_traffic_light.values()
        for lane_id in lane_ids
    }
    all_incoming_lane_ids = tuple(
        dict.fromkeys(
            lane_id
            for traffic_light_id in traffic_light_ids
            for lane_id in incoming_lanes_by_traffic_light[traffic_light_id]
        )
    )
    all_incoming_lane_length_m = sum(
        float(lane_geometries[_edge_id_from_lane_id(lane_id)].length_m) for lane_id in all_incoming_lane_ids
    )
    return RolloutContext(
        graph=graph,
        traffic_light_ids=traffic_light_ids,
        movement_ids_by_traffic_light=movement_ids_by_traffic_light,
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        incoming_lanes_by_traffic_light=incoming_lanes_by_traffic_light,
        incoming_lane_length_by_traffic_light=incoming_lane_length_by_traffic_light,
        speed_limit_by_lane=speed_limit_by_lane,
        all_incoming_lane_ids=all_incoming_lane_ids,
        all_incoming_lane_length_m=all_incoming_lane_length_m,
    )


def _policy_context_from_sample(sample: MovementDatasetSample) -> PolicyContext:
    traffic_light_ids = tuple(sorted(sample.phase_incidences.keys(), key=str))
    movement_ids_by_traffic_light = tuple(
        tuple(int(value) for value in sample.phase_incidences[traffic_light_id]['movement_ids'])
        for traffic_light_id in traffic_light_ids
    )
    return PolicyContext(
        traffic_light_ids=traffic_light_ids,
        movement_ids_by_traffic_light=movement_ids_by_traffic_light,
    )


def _current_sample(
    context: RolloutContext,
    programs: Mapping[str, TrafficLightProgram],
    vehicle_snapshot_collector: VehicleSnapshotCollector,
    lane_flow_tracker: LaneGroupFlowTracker,
    control_state: MovementControlState,
) -> MovementDatasetSample:
    vehicles = vehicle_snapshot_collector.capture()
    feature_frame = build_feature_frame(
        graph=context.graph,
        lane_ids_by_edge=context.lane_ids_by_edge,
        lane_geometries=context.lane_geometries,
        control_state=control_state,
        vehicles=vehicles,
        lane_flow_rates=lane_flow_tracker.observe(vehicles),
    )
    return build_dataset_sample(
        graph=context.graph,
        feature_frame=feature_frame,
        programs=programs,
        teacher_controlled_scores={traffic_light_id: {} for traffic_light_id in programs},
        metadata={},
    )


def _forward_policy(
    model: MovementActorCritic,
    sample: MovementDatasetSample,
    context: RolloutContext | PolicyContext,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    x_lane, x_movement, _target = tensors_from_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
    movement_scores, values = model.forward_actor_critic(
        x_lane=x_lane,
        x_movement=x_movement,
        edge_index_dict=edge_tensors_from_sample(sample, device=device),
        movement_ids_by_traffic_light=context.movement_ids_by_traffic_light,
    )
    return (
        movement_scores,
        values,
        _phase_logits(
            sample=sample,
            traffic_light_ids=context.traffic_light_ids,
            movement_scores=movement_scores,
        ),
    )


def _phase_logits(
    sample: MovementDatasetSample,
    traffic_light_ids: Sequence[str],
    movement_scores: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    logits = []
    for traffic_light_id in traffic_light_ids:
        incidence = sample.phase_incidences[traffic_light_id]
        movement_ids = tuple(int(value) for value in incidence['movement_ids'])
        phase_scores = []
        for row in incidence['rows']:
            enabled_scores = [
                movement_scores[movement_id] for enabled, movement_id in zip(row, movement_ids) if int(enabled) == 1
            ]
            phase_scores.append(torch.stack(tuple(enabled_scores)).sum())
        logits.append(torch.stack(tuple(phase_scores)))
    return tuple(logits)


def _states_from_actions(
    programs: Mapping[str, TrafficLightProgram],
    traffic_light_ids: Sequence[str],
    actions: Sequence[int],
) -> dict[str, str]:
    return {
        traffic_light_id: str(programs[traffic_light_id].selectable_phases[action].state)
        for traffic_light_id, action in zip(traffic_light_ids, actions)
    }


def _actions_from_states(
    programs: Mapping[str, TrafficLightProgram],
    traffic_light_ids: Sequence[str],
    states: Mapping[str, str],
) -> tuple[int, ...]:
    return tuple(
        next(
            local_phase_index
            for local_phase_index, phase in enumerate(programs[traffic_light_id].selectable_phases)
            if str(phase.state) == states[traffic_light_id]
        )
        for traffic_light_id in traffic_light_ids
    )


def _allowed_action_masks(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    programs: Mapping[str, TrafficLightProgram],
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            str(phase.state) in set(runtime.allowed_target_states(traffic_light_id))
            for phase in programs[traffic_light_id].selectable_phases
        )
        for traffic_light_id in context.traffic_light_ids
    )


def _masked_phase_logits(
    phase_logits: Sequence[torch.Tensor],
    action_masks: Sequence[Sequence[bool]],
) -> tuple[torch.Tensor, ...]:
    if len(phase_logits) != len(action_masks):
        raise ValueError('Phase-logit count does not match action-mask count.')
    masked_logits = []
    for logits, mask in zip(phase_logits, action_masks):
        if len(mask) != len(logits):
            raise ValueError('Action-mask width does not match phase-logit width.')
        if not any(mask):
            raise ValueError('Each traffic light must permit at least one action.')
        mask_tensor = torch.tensor(tuple(mask), dtype=torch.bool, device=logits.device)
        masked_logits.append(logits.masked_fill(~mask_tensor, float('-inf')))
    return tuple(masked_logits)


def _advance_and_reward(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    decision_interval: int,
    global_reward_weight: float,
    reward_clip: float,
    teleport_penalty: float,
) -> IntervalRewardResult:
    if teleport_penalty < 0.0:
        raise ValueError('teleport_penalty must not be negative.')
    local_delay_sums = {traffic_light_id: 0.0 for traffic_light_id in context.traffic_light_ids}
    global_delay_sum = 0.0
    teleport_count = 0
    simulated_steps = 0
    for _step in range(decision_interval):
        runtime.step()
        simulated_steps += 1
        teleport_count += int(traci.simulation.getStartingTeleportNumber())
        for traffic_light_id in context.traffic_light_ids:
            local_delay_sums[traffic_light_id] += _speed_deficit_density(
                lane_ids=context.incoming_lanes_by_traffic_light[traffic_light_id],
                speed_limit_by_lane=context.speed_limit_by_lane,
                total_lane_length_m=context.incoming_lane_length_by_traffic_light[traffic_light_id],
            )
        global_delay_sum += _speed_deficit_density(
            lane_ids=context.all_incoming_lane_ids,
            speed_limit_by_lane=context.speed_limit_by_lane,
            total_lane_length_m=context.all_incoming_lane_length_m,
        )
        if not runtime.is_running():
            break
    local_delay_densities = tuple(
        local_delay_sums[traffic_light_id] / max(1, simulated_steps) for traffic_light_id in context.traffic_light_ids
    )
    global_delay_density = global_delay_sum / max(1, simulated_steps)
    raw_rewards = tuple(
        _delay_density_reward(
            local_delay_density=local_delay_density,
            global_delay_density=global_delay_density,
            global_reward_weight=global_reward_weight,
            teleport_penalty=teleport_penalty,
            teleport_count=teleport_count,
        )
        for local_delay_density in local_delay_densities
    )
    rewards = tuple(_clip_reward(reward, reward_clip=reward_clip) for reward in raw_rewards)
    return IntervalRewardResult(
        rewards=rewards,
        raw_rewards=raw_rewards,
        local_delay_densities=local_delay_densities,
        global_delay_density=global_delay_density,
        teleport_count=teleport_count,
    )


def _delay_density_reward(
    local_delay_density: float,
    global_delay_density: float,
    global_reward_weight: float,
    teleport_penalty: float,
    teleport_count: int,
) -> float:
    return -local_delay_density - global_reward_weight * global_delay_density - teleport_penalty * teleport_count


def _speed_deficit_density(
    lane_ids: Sequence[str],
    speed_limit_by_lane: Mapping[str, float],
    total_lane_length_m: float,
) -> float:
    if total_lane_length_m <= 0.0:
        return 0.0
    delayed_vehicle_equivalents = 0.0
    for lane_id in lane_ids:
        vehicle_count = int(traci.lane.getLastStepVehicleNumber(lane_id))
        if vehicle_count <= 0:
            continue
        speed_limit = speed_limit_by_lane[lane_id]
        if speed_limit <= 0.0:
            continue
        mean_speed = max(0.0, float(traci.lane.getLastStepMeanSpeed(lane_id)))
        speed_deficit = max(0.0, 1.0 - min(mean_speed / speed_limit, 1.0))
        delayed_vehicle_equivalents += vehicle_count * speed_deficit
    return delayed_vehicle_equivalents / total_lane_length_m


def _clip_reward(reward: float, reward_clip: float) -> float:
    if reward_clip <= 0.0:
        raise ValueError('reward_clip must be positive.')
    return max(-reward_clip, min(reward_clip, reward))


def _standard_deviation(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = sum(
        float(parameter.grad.detach().pow(2).sum().cpu()) for parameter in parameters if parameter.grad is not None
    )
    return sqrt(squared_norm)


def _edge_id_from_lane_id(lane_id: str) -> str:
    edge_id, separator, lane_index = lane_id.rpartition('_')
    if separator and lane_index.isdigit() and edge_id:
        return edge_id
    return lane_id


def _load_actor_critic(
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


def _zero_value_output(model: MovementActorCritic) -> None:
    with torch.no_grad():
        last_layer = model.value_head[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)


def _set_actor_grad(model: MovementActorCritic, requires_grad: bool) -> None:
    for module in (model.lane_encoder, model.movement_encoder, model.hops, model.score_head):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def _set_value_grad(model: MovementActorCritic, requires_grad: bool) -> None:
    for parameter in model.value_head.parameters():
        parameter.requires_grad = requires_grad


def _write_training_scalars(
    writer: SummaryWriter,
    iteration: int,
    rollout_stats: RolloutStats,
    diagnostics: TrainingDiagnostics,
    update_stats: PpoUpdateStats,
) -> None:
    writer.add_scalar('episode/mean_reward', rollout_stats.mean_reward, iteration)
    writer.add_scalar('episode/reward_standard_deviation', rollout_stats.reward_standard_deviation, iteration)
    writer.add_scalar('episode/minimum_reward', rollout_stats.minimum_reward, iteration)
    writer.add_scalar('episode/maximum_reward', rollout_stats.maximum_reward, iteration)
    writer.add_scalar(
        'episode/raw_reward_standard_deviation',
        rollout_stats.raw_reward_standard_deviation,
        iteration,
    )
    writer.add_scalar('episode/minimum_raw_reward', rollout_stats.minimum_raw_reward, iteration)
    writer.add_scalar('episode/maximum_raw_reward', rollout_stats.maximum_raw_reward, iteration)
    writer.add_scalar(
        'episode/mean_local_delay_density',
        rollout_stats.mean_local_delay_density,
        iteration,
    )
    writer.add_scalar(
        'episode/mean_global_delay_density',
        rollout_stats.mean_global_delay_density,
        iteration,
    )
    writer.add_scalar('episode/mean_return', diagnostics.mean_return, iteration)
    writer.add_scalar('episode/return_standard_deviation', diagnostics.return_standard_deviation, iteration)
    writer.add_scalar('episode/mean_value', diagnostics.mean_value, iteration)
    writer.add_scalar('episode/value_standard_deviation', diagnostics.value_standard_deviation, iteration)
    writer.add_scalar('episode/advantage_standard_deviation', diagnostics.advantage_standard_deviation, iteration)
    writer.add_scalar('diagnostics/explained_variance', diagnostics.explained_variance, iteration)
    writer.add_scalar('diagnostics/teleports', rollout_stats.teleport_count, iteration)
    writer.add_scalar('diagnostics/reward_clip_fraction', rollout_stats.reward_clip_fraction, iteration)
    writer.add_scalar('diagnostics/normalized_entropy', rollout_stats.normalized_entropy, iteration)
    writer.add_scalar(
        'diagnostics/mean_top_action_probability',
        rollout_stats.mean_top_action_probability,
        iteration,
    )
    writer.add_scalar('diagnostics/policy_decision_fraction', rollout_stats.policy_decision_fraction, iteration)
    writer.add_scalar('diagnostics/approximate_kl', update_stats.approximate_kl, iteration)
    writer.add_scalar('diagnostics/ratio_clip_fraction', update_stats.ratio_clip_fraction, iteration)
    writer.add_scalar('diagnostics/backbone_gradient_norm', update_stats.backbone_gradient_norm, iteration)
    writer.add_scalar('diagnostics/value_head_gradient_norm', update_stats.value_head_gradient_norm, iteration)
    writer.add_scalar('diagnostics/kl_early_stop', float(update_stats.early_stopped), iteration)
    writer.add_scalar('timing/rollout_seconds', rollout_stats.simulation_elapsed_s, iteration)
    writer.add_scalar('loss/policy', update_stats.policy_loss, iteration)
    writer.add_scalar('loss/value', update_stats.value_loss, iteration)
    writer.add_scalar('loss/entropy', update_stats.entropy, iteration)
    writer.add_scalar('loss/total', update_stats.total_loss, iteration)


def _run_training_evaluation(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    metadata: MovementCheckpointMetadata,
    iteration: int,
    writer: SummaryWriter,
    baseline_records: tuple[EvaluationRecord, ...],
) -> TrainingEvaluationResult:
    with TemporaryDirectory(prefix='movement_ppo_eval_') as directory:
        checkpoint_path = Path(directory) / 'movement_policy.pt'
        _save_actor_checkpoint(
            path=checkpoint_path,
            model=model,
            metadata=metadata,
            loss=0.0,
        )
        learned_policy_config = LearnedPolicyConfig(
            checkpoint_path=checkpoint_path,
            device=config.device,
        )
        records: list[EvaluationRecord] = list(baseline_records)
        cached_baseline_keys = {(record.policy, record.seed) for record in baseline_records}
        for policy in config.eval_policies:
            for seed in config.eval_seeds:
                if policy != EvaluationPolicy.LEARNED and (policy.value, seed) in cached_baseline_keys:
                    continue
                records.append(
                    EvaluationRecord(
                        policy=policy.value,
                        seed=seed,
                        metrics=run_evaluation_episode(
                            cfg_path=config.cfg_path,
                            policy=policy,
                            seed=seed,
                            steps=config.eval_steps,
                            decision_interval=config.decision_interval,
                            yellow_duration=config.yellow_duration,
                            min_green_steps=config.min_green_steps,
                            learned_policy_config=learned_policy_config,
                            demand_scale=config.eval_demand_scale,
                            initial_occupancy_min=config.initial_occupancy_min,
                            initial_occupancy_max=config.initial_occupancy_max,
                        ),
                    )
                )
        aggregates = aggregate_records(records)
        _write_evaluation_scalars(
            writer=writer,
            iteration=iteration,
            evaluation_steps=config.eval_steps,
            aggregates=aggregates,
        )
        output_dir = config.checkpoint_dir / 'eval' / f'iter_{iteration:04d}'
        write_aggregate_json(output_dir / 'summary.json', records, aggregates)
        write_records_csv(output_dir / 'summary.csv', records, aggregates)
        print_aggregate_metric_table(f'PPO evaluation at iteration {iteration}', aggregates)
        learned_aggregate = next(
            (aggregate for aggregate in aggregates if aggregate.policy == EvaluationPolicy.LEARNED.value),
            None,
        )
        return TrainingEvaluationResult(
            baseline_records=tuple(record for record in records if record.policy != EvaluationPolicy.LEARNED.value),
            learned_checkpoint_score=(
                _checkpoint_selection_score(
                    metrics=learned_aggregate.mean,
                    evaluation_steps=config.eval_steps,
                )
                if learned_aggregate is not None
                else None
            ),
        )


def _training_diagnostics(buffer: MovementRolloutBuffer) -> TrainingDiagnostics:
    if buffer.returns is None or buffer.advantages is None:
        raise ValueError('Returns must be computed before training diagnostics.')
    values = torch.tensor(
        tuple(transition.values for transition in buffer.transitions),
        dtype=torch.float32,
    )
    returns = buffer.returns
    return_variance = float(returns.var())
    residual_variance = float((returns - values).var())
    explained_variance = 1.0 - residual_variance / (return_variance + 1e-8)
    return TrainingDiagnostics(
        mean_return=float(returns.mean()),
        return_standard_deviation=float(returns.std()),
        mean_value=float(values.mean()),
        value_standard_deviation=float(values.std()),
        advantage_standard_deviation=float(buffer.advantages.std()),
        explained_variance=explained_variance,
    )


def _write_evaluation_scalars(
    writer: SummaryWriter,
    iteration: int,
    evaluation_steps: int,
    aggregates: Sequence[EvaluationAggregate],
) -> None:
    for aggregate in aggregates:
        policy = aggregate.policy
        metrics = aggregate.mean
        writer.add_scalar(f'eval/{policy}/throughput_per_hour', metrics.throughput_per_hour, iteration)
        writer.add_scalar(f'eval/{policy}/completion_rate', metrics.completion_rate, iteration)
        writer.add_scalar(f'eval/{policy}/average_waiting_time_s', metrics.average_waiting_time_s, iteration)
        writer.add_scalar(f'eval/{policy}/average_travel_time_s', metrics.average_travel_time_s, iteration)
        writer.add_scalar(
            f'eval/{policy}/average_queue_length_vehicles',
            metrics.average_queue_length_vehicles,
            iteration,
        )
        writer.add_scalar(
            f'eval/{policy}/average_wait_density_s_per_m',
            metrics.average_wait_density_s_per_m,
            iteration,
        )
        writer.add_scalar(
            f'eval/{policy}/switch_frequency_per_junction_per_minute',
            metrics.phase_switch_frequency_per_junction_per_minute,
            iteration,
        )
        writer.add_scalar(f'eval/{policy}/teleports', metrics.teleport_count, iteration)
        writer.add_scalar(
            f'eval/{policy}/checkpoint_selection_score',
            _checkpoint_selection_score(
                metrics=metrics,
                evaluation_steps=evaluation_steps,
            ),
            iteration,
        )
        writer.add_scalar(
            f'eval/{policy}/nonstop_tls_pass_rate',
            metrics.nonstop_tls_pass_rate,
            iteration,
        )


def _checkpoint_selection_score(
    metrics: EvaluationMetrics,
    evaluation_steps: int,
) -> float:
    if metrics.completion_rate <= 0.0:
        return float('inf')
    teleport_rate = metrics.teleport_count / max(1, metrics.departed_vehicles)
    return metrics.average_time_loss_s / metrics.completion_rate + evaluation_steps * teleport_rate


def _save_actor_checkpoint(
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


def _save_ppo_checkpoint(
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
    checkpoint = _load_ppo_checkpoint_payload(checkpoint_path=checkpoint_path, device=device)
    model, _metadata = _model_and_metadata_from_ppo_checkpoint(checkpoint=checkpoint, device=device)
    return model


def _load_ppo_checkpoint_payload(
    checkpoint_path: Path | str,
    device: str,
) -> MovementPpoCheckpoint:
    return cast(
        MovementPpoCheckpoint,
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )


def _model_and_metadata_from_ppo_checkpoint(
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
