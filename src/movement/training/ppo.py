"""PPO training for movement-score traffic signal policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import cast

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import traci

from scripts.collect_il_data import (
    lane_inputs_from_net,
    resolve_sumocfg_net_path,
)
from src.movement.dataset import MovementDatasetSample, build_dataset_sample
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.evaluation import (
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
    MovementControlState,
    build_feature_frame,
    vehicle_snapshots_from_api,
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
    il_checkpoint_path: Path
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


@dataclass(frozen=True)
class MovementPpoTrainingResult:
    checkpoint_path: Path
    iterations: int


@dataclass(frozen=True)
class MovementPpoCheckpoint:
    model_state: dict[str, torch.Tensor]
    lane_feature_dim: int
    movement_feature_dim: int
    hidden_dim: int
    num_hops: int
    lane_normalizer: NormalizerState
    movement_normalizer: NormalizerState
    iteration: int


@dataclass(frozen=True)
class RolloutContext:
    graph: MovementGraph
    traffic_light_ids: tuple[str, ...]
    movement_ids_by_traffic_light: tuple[tuple[int, ...], ...]
    lane_ids_by_edge: dict[str, tuple[str, ...]]
    lane_geometries: dict[str, LaneGroupGeometry]
    incoming_lanes_by_traffic_light: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PolicyContext:
    traffic_light_ids: tuple[str, ...]
    movement_ids_by_traffic_light: tuple[tuple[int, ...], ...]


def train_movement_ppo(config: MovementPpoConfig) -> MovementPpoTrainingResult:
    """Fine-tune a movement scorer with PPO."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model, metadata = _load_actor_critic(config.il_checkpoint_path, device=config.device)
    _zero_value_output(model)
    lane_normalizer = normalizer_from_state(metadata.lane_normalizer)
    movement_normalizer = normalizer_from_state(metadata.movement_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=str(config.log_dir))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = config.checkpoint_dir / 'movement_ppo_latest.pt'
    started = perf_counter()
    completed_iteration = 0
    try:
        for iteration in range(1, config.iterations + 1):
            warming_up = iteration <= config.value_warmup_iterations
            _set_actor_grad(model, requires_grad=not warming_up)
            _set_value_grad(model, requires_grad=True)
            buffer, mean_reward = _collect_rollout(
                config=config,
                model=model,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
                rollout_seed=config.seed + iteration,
            )
            buffer.compute_returns_and_advantages(use_mc_targets=warming_up)
            update_stats = _update_ppo(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
                config=config,
                warming_up=warming_up,
            )
            _write_training_scalars(
                writer=writer,
                iteration=iteration,
                mean_reward=mean_reward,
                update_stats=update_stats,
            )
            if config.print_every > 0 and (iteration == 1 or iteration % config.print_every == 0):
                phase = 'value' if warming_up else 'ppo'
                print(
                    f'[{phase}] iter={iteration}/{config.iterations} '
                    f'reward={mean_reward:+.4f} '
                    f'policy_loss={update_stats.policy_loss:+.4f} '
                    f'value_loss={update_stats.value_loss:.4f} '
                    f'entropy={update_stats.entropy:.4f} '
                    f'elapsed={perf_counter() - started:.1f}s'
                )
            if config.save_every > 0 and iteration % config.save_every == 0:
                _save_ppo_checkpoint(
                    path=config.checkpoint_dir / f'movement_ppo_iter_{iteration:04d}.pt',
                    model=model,
                    metadata=metadata,
                    iteration=iteration,
                )
            if config.eval_every > 0 and iteration % config.eval_every == 0:
                _run_training_evaluation(
                    config=config,
                    model=model,
                    metadata=metadata,
                    iteration=iteration,
                )
            completed_iteration = iteration
    finally:
        _save_ppo_checkpoint(
            path=last_checkpoint_path,
            model=model,
            metadata=metadata,
            iteration=completed_iteration,
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


def _collect_rollout(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    rollout_seed: int,
) -> tuple[MovementRolloutBuffer, float]:
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
    try:
        runtime.start()
        context = _rollout_context(
            cfg_path=config.cfg_path,
            programs=runtime.programs,
        )
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
                distributions = tuple(Categorical(logits=logits) for logits in phase_logits)
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
            runtime.request_targets(desired_states)
            reward = _advance_and_reward(
                runtime=runtime,
                context=context,
                decision_interval=config.decision_interval,
            )
            rewards.extend(reward)
            buffer.add(
                MovementTransition(
                    sample=sample,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    rewards=reward,
                    values=tuple(float(value) for value in values.detach().cpu()),
                    done=not runtime.is_running(),
                )
            )
        return buffer, sum(rewards) / max(1, len(rewards))
    finally:
        runtime.close()
        initial_population.cleanup()
        demand_route_files.cleanup()


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
                distributions = tuple(Categorical(logits=logits) for logits in phase_logits)
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
            entropy = torch.stack(batch_entropies).mean()
            values = torch.stack(batch_values)
            value_loss = F.mse_loss(values, batch.returns)
            if warming_up:
                policy_loss = values.new_zeros(())
                loss = config.value_coefficient * value_loss
            else:
                ratio = (new_log_probs - batch.old_log_probs).exp()
                clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                policy_loss = -torch.min(
                    ratio * batch.advantages,
                    clipped_ratio * batch.advantages,
                ).mean()
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
    return PpoUpdateStats(
        policy_loss=sum(policy_losses) / max(1, len(policy_losses)),
        value_loss=sum(value_losses) / max(1, len(value_losses)),
        entropy=sum(entropies) / max(1, len(entropies)),
        total_loss=sum(total_losses) / max(1, len(total_losses)),
    )


def _rollout_context(
    cfg_path: Path,
    programs: Mapping[str, TrafficLightProgram],
) -> RolloutContext:
    graph = build_movement_graph(programs)
    net_path = resolve_sumocfg_net_path(cfg_path)
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
    return RolloutContext(
        graph=graph,
        traffic_light_ids=traffic_light_ids,
        movement_ids_by_traffic_light=movement_ids_by_traffic_light,
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        incoming_lanes_by_traffic_light=incoming_lanes_by_traffic_light,
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
) -> MovementDatasetSample:
    feature_frame = build_feature_frame(
        graph=context.graph,
        lane_ids_by_edge=context.lane_ids_by_edge,
        lane_geometries=context.lane_geometries,
        control_state=MovementControlState(),
        vehicles=vehicle_snapshots_from_api(traci.vehicle),
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


def _advance_and_reward(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    decision_interval: int,
) -> tuple[float, ...]:
    reward_sum = [0.0 for _traffic_light_id in context.traffic_light_ids]
    actual_steps = 0
    for _step in range(decision_interval):
        runtime.step()
        actual_steps += 1
        for index, traffic_light_id in enumerate(context.traffic_light_ids):
            reward_sum[index] += _traffic_light_reward(context.incoming_lanes_by_traffic_light[traffic_light_id])
        if not runtime.is_running():
            break
    return tuple(value / max(1, actual_steps) for value in reward_sum)


def _traffic_light_reward(lane_ids: Sequence[str]) -> float:
    lane_length = sum(float(traci.lane.getLength(lane_id)) for lane_id in lane_ids)
    if lane_length <= 0.0:
        return 0.0
    waiting_time = sum(float(traci.lane.getWaitingTime(lane_id)) for lane_id in lane_ids)
    return -(waiting_time / lane_length)


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
    mean_reward: float,
    update_stats: PpoUpdateStats,
) -> None:
    writer.add_scalar('episode/mean_reward', mean_reward, iteration)
    writer.add_scalar('loss/policy', update_stats.policy_loss, iteration)
    writer.add_scalar('loss/value', update_stats.value_loss, iteration)
    writer.add_scalar('loss/entropy', update_stats.entropy, iteration)
    writer.add_scalar('loss/total', update_stats.total_loss, iteration)


def _run_training_evaluation(
    config: MovementPpoConfig,
    model: MovementActorCritic,
    metadata: MovementCheckpointMetadata,
    iteration: int,
) -> None:
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
        records: list[EvaluationRecord] = []
        for policy in config.eval_policies:
            for seed in config.eval_seeds:
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
                        ),
                    )
                )
        aggregates = aggregate_records(records)
        output_dir = config.checkpoint_dir / 'eval' / f'iter_{iteration:04d}'
        write_aggregate_json(output_dir / 'summary.json', records, aggregates)
        write_records_csv(output_dir / 'summary.csv', records, aggregates)
        print_aggregate_metric_table(f'PPO evaluation at iteration {iteration}', aggregates)


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
    metadata: MovementCheckpointMetadata,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = MovementPpoCheckpoint(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        lane_feature_dim=metadata.lane_feature_dim,
        movement_feature_dim=metadata.movement_feature_dim,
        hidden_dim=metadata.hidden_dim,
        num_hops=metadata.num_hops,
        lane_normalizer=metadata.lane_normalizer,
        movement_normalizer=metadata.movement_normalizer,
        iteration=iteration,
    )
    torch.save(checkpoint, path)


def load_movement_ppo_checkpoint(
    checkpoint_path: Path | str,
    device: str,
) -> MovementActorCritic:
    checkpoint = cast(
        MovementPpoCheckpoint,
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    model = MovementActorCritic(
        lane_feature_dim=checkpoint.lane_feature_dim,
        movement_feature_dim=checkpoint.movement_feature_dim,
        hidden_dim=checkpoint.hidden_dim,
        num_hops=checkpoint.num_hops,
    )
    model.load_state_dict(checkpoint.model_state)
    model.to(torch.device(device))
    return model
