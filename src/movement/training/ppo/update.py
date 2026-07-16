"""PPO optimizer update step."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import torch
import torch.nn.functional as F

from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.training.ppo.batch import (
    PackedMovementPpoBatch,
    PackedPpoPhaseLogitGroup,
    PackedPpoValueGroup,
    move_packed_movement_ppo_batch,
    ppo_batch_data_loader,
)
from src.movement.training.ppo.types import MovementPpoConfig
from src.movement.training.rollout import MovementRolloutBuffer


@dataclass(frozen=True)
class PpoUpdateProfileStats:
    batch_count: int
    data_seconds: float
    batch_loss_seconds: float
    model_forward_seconds: float
    value_seconds: float
    phase_log_prob_seconds: float
    backward_seconds: float
    optimizer_seconds: float


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
    profile: PpoUpdateProfileStats


def update_ppo(
    model: MovementActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: MovementRolloutBuffer,
    device: torch.device,
    config: MovementPpoConfig,
    warming_up: bool,
) -> PpoUpdateStats:
    if buffer.advantages is None or buffer.returns is None:
        raise ValueError('compute_returns_and_advantages must be called before PPO update.')
    model.train()
    epochs = config.warmup_epochs if warming_up else config.update_epochs
    history = PpoUpdateHistory()
    for _epoch in range(epochs):
        early_stopped = _run_update_epoch(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            device=device,
            config=config,
            warming_up=warming_up,
            history=history,
        )
        if early_stopped:
            return history.stats(early_stopped=True)
    return history.stats(early_stopped=False)


def skipped_update_stats() -> PpoUpdateStats:
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
        profile=PpoUpdateProfileStats(
            batch_count=0,
            data_seconds=0.0,
            batch_loss_seconds=0.0,
            model_forward_seconds=0.0,
            value_seconds=0.0,
            phase_log_prob_seconds=0.0,
            backward_seconds=0.0,
            optimizer_seconds=0.0,
        ),
    )


def set_actor_grad(model: MovementActorCritic, requires_grad: bool) -> None:
    for module in (model.lane_encoder, model.movement_encoder, model.hops, model.score_head):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def set_value_grad(model: MovementActorCritic, requires_grad: bool) -> None:
    for parameter in model.value_head.parameters():
        parameter.requires_grad = requires_grad


@dataclass
class PpoUpdateHistory:
    policy_losses: list[float]
    value_losses: list[float]
    entropies: list[float]
    total_losses: list[float]
    approximate_kls: list[float]
    ratio_clip_fractions: list[float]
    backbone_gradient_norms: list[float]
    value_head_gradient_norms: list[float]
    profile: PpoUpdateProfileAccumulator

    def __init__(self) -> None:
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.total_losses = []
        self.approximate_kls = []
        self.ratio_clip_fractions = []
        self.backbone_gradient_norms = []
        self.value_head_gradient_norms = []
        self.profile = PpoUpdateProfileAccumulator()

    def stats(self, early_stopped: bool) -> PpoUpdateStats:
        return PpoUpdateStats(
            policy_loss=_mean(self.policy_losses),
            value_loss=_mean(self.value_losses),
            entropy=_mean(self.entropies),
            total_loss=_mean(self.total_losses),
            approximate_kl=_mean(self.approximate_kls),
            ratio_clip_fraction=_mean(self.ratio_clip_fractions),
            backbone_gradient_norm=_mean(self.backbone_gradient_norms),
            value_head_gradient_norm=_mean(self.value_head_gradient_norms),
            early_stopped=early_stopped,
            profile=self.profile.stats(),
        )


@dataclass
class PpoUpdateProfileAccumulator:
    batch_count: int = 0
    data_seconds: float = 0.0
    batch_loss_seconds: float = 0.0
    model_forward_seconds: float = 0.0
    value_seconds: float = 0.0
    phase_log_prob_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0

    def observe_evaluation(self, timing: PpoBatchEvaluationTiming) -> None:
        self.model_forward_seconds += timing.model_forward_seconds
        self.value_seconds += timing.value_seconds
        self.phase_log_prob_seconds += timing.phase_log_prob_seconds

    def stats(self) -> PpoUpdateProfileStats:
        return PpoUpdateProfileStats(
            batch_count=self.batch_count,
            data_seconds=self.data_seconds,
            batch_loss_seconds=self.batch_loss_seconds,
            model_forward_seconds=self.model_forward_seconds,
            value_seconds=self.value_seconds,
            phase_log_prob_seconds=self.phase_log_prob_seconds,
            backward_seconds=self.backward_seconds,
            optimizer_seconds=self.optimizer_seconds,
        )


@dataclass(frozen=True)
class PpoBatchLoss:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor
    approximate_kl: torch.Tensor
    ratio_clip_fraction: torch.Tensor
    timing: PpoBatchEvaluationTiming


@dataclass(frozen=True)
class PpoBatchPolicyEvaluation:
    new_log_probs: torch.Tensor
    entropy_values: torch.Tensor
    values: torch.Tensor
    timing: PpoBatchEvaluationTiming


@dataclass(frozen=True)
class PpoBatchEvaluationTiming:
    model_forward_seconds: float
    value_seconds: float
    phase_log_prob_seconds: float


def _run_update_epoch(
    model: MovementActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: MovementRolloutBuffer,
    device: torch.device,
    config: MovementPpoConfig,
    warming_up: bool,
    history: PpoUpdateHistory,
) -> bool:
    assert buffer.advantages is not None
    assert buffer.returns is not None
    loader = ppo_batch_data_loader(
        transitions=buffer.transitions,
        advantages=buffer.advantages,
        returns=buffer.returns,
        transitions_per_batch=config.transitions_per_batch,
        update_batch_workers=config.update_batch_workers,
        action_samples_per_batch=config.action_samples_per_batch,
    )
    loader_iterator = iter(loader)
    while True:
        data_started = perf_counter()
        try:
            cpu_batch = next(loader_iterator)
        except StopIteration:
            break
        batch = move_packed_movement_ppo_batch(cpu_batch=cpu_batch, device=device)
        synchronize_if_cuda(device=device)
        history.profile.data_seconds += perf_counter() - data_started
        loss_started = perf_counter()
        loss = _batch_loss(
            model=model,
            config=config,
            warming_up=warming_up,
            batch=batch,
        )
        synchronize_if_cuda(device=device)
        history.profile.batch_loss_seconds += perf_counter() - loss_started
        history.profile.observe_evaluation(loss.timing)
        if not warming_up and float(loss.approximate_kl.detach().cpu()) > config.target_kl:
            return True
        optimizer_started = perf_counter()
        optimizer.zero_grad()
        synchronize_if_cuda(device=device)
        history.profile.optimizer_seconds += perf_counter() - optimizer_started
        backward_started = perf_counter()
        loss.total_loss.backward()
        synchronize_if_cuda(device=device)
        history.profile.backward_seconds += perf_counter() - backward_started
        optimizer_started = perf_counter()
        history.backbone_gradient_norms.append(gradient_norm(tuple(_actor_parameters(model))))
        history.value_head_gradient_norms.append(gradient_norm(tuple(model.value_head.parameters())))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        synchronize_if_cuda(device=device)
        history.profile.optimizer_seconds += perf_counter() - optimizer_started
        history.profile.batch_count += 1
        history.policy_losses.append(float(loss.policy_loss.detach().cpu()))
        history.value_losses.append(float(loss.value_loss.detach().cpu()))
        history.entropies.append(float(loss.entropy.detach().cpu()))
        history.total_losses.append(float(loss.total_loss.detach().cpu()))
        history.approximate_kls.append(float(loss.approximate_kl.detach().cpu()))
        history.ratio_clip_fractions.append(float(loss.ratio_clip_fraction.detach().cpu()))
    return False


def _batch_loss(
    model: MovementActorCritic,
    config: MovementPpoConfig,
    warming_up: bool,
    batch: PackedMovementPpoBatch,
) -> PpoBatchLoss:
    evaluation = _evaluate_packed_batch_policy(
        model=model,
        batch=batch,
    )
    entropy = (
        evaluation.entropy_values[batch.policy_mask].mean()
        if bool(batch.policy_mask.any())
        else evaluation.entropy_values.new_zeros(())
    )
    value_loss = F.mse_loss(evaluation.values, batch.returns)
    if warming_up:
        zero = evaluation.values.new_zeros(())
        return PpoBatchLoss(
            policy_loss=zero,
            value_loss=value_loss,
            entropy=entropy,
            total_loss=config.value_coefficient * value_loss,
            approximate_kl=zero,
            ratio_clip_fraction=zero,
            timing=evaluation.timing,
        )
    log_ratio = evaluation.new_log_probs - batch.old_log_probs
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_objective = torch.min(
        ratio * batch.advantages,
        clipped_ratio * batch.advantages,
    )
    policy_loss = (
        -policy_objective[batch.policy_mask].mean() if bool(batch.policy_mask.any()) else policy_objective.new_zeros(())
    )
    approximate_kl = (
        ((ratio - 1.0) - log_ratio)[batch.policy_mask].mean() if bool(batch.policy_mask.any()) else ratio.new_zeros(())
    )
    ratio_clip_fraction = (
        ((ratio - 1.0).abs() > config.clip_epsilon)[batch.policy_mask].float().mean()
        if bool(batch.policy_mask.any())
        else ratio.new_zeros(())
    )
    return PpoBatchLoss(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        total_loss=policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy,
        approximate_kl=approximate_kl,
        ratio_clip_fraction=ratio_clip_fraction,
        timing=evaluation.timing,
    )


def _evaluate_packed_batch_policy(
    model: MovementActorCritic,
    batch: PackedMovementPpoBatch,
) -> PpoBatchPolicyEvaluation:
    device = batch.movement_batch.x_lane.device
    forward_started = perf_counter()
    movement_embeddings = model.movement_embeddings(
        x_lane=batch.movement_batch.x_lane,
        x_movement=batch.movement_batch.x_movement,
        edge_index_dict=batch.movement_batch.edge_index_dict,
    )
    movement_scores = model.score_head(movement_embeddings).squeeze(-1)
    synchronize_if_cuda(device=device)
    model_forward_seconds = perf_counter() - forward_started
    value_started = perf_counter()
    values = _packed_values(
        model=model,
        movement_embeddings=movement_embeddings,
        value_groups=batch.value_groups,
        value_count=batch.policy_value_count,
    )
    synchronize_if_cuda(device=device)
    value_seconds = perf_counter() - value_started
    phase_started = perf_counter()
    new_log_probs, entropy_values = _packed_action_log_probs_and_entropy(
        movement_scores=movement_scores,
        phase_logit_groups=batch.phase_logit_groups,
        policy_value_count=batch.policy_value_count,
    )
    synchronize_if_cuda(device=device)
    phase_log_prob_seconds = perf_counter() - phase_started
    return PpoBatchPolicyEvaluation(
        new_log_probs=new_log_probs,
        entropy_values=entropy_values,
        values=values,
        timing=PpoBatchEvaluationTiming(
            model_forward_seconds=model_forward_seconds,
            value_seconds=value_seconds,
            phase_log_prob_seconds=phase_log_prob_seconds,
        ),
    )


def _packed_values(
    model: MovementActorCritic,
    movement_embeddings: torch.Tensor,
    value_groups: Sequence[PackedPpoValueGroup],
    value_count: int,
) -> torch.Tensor:
    values = movement_embeddings.new_empty((value_count,))
    for group in value_groups:
        traffic_light_embeddings = movement_embeddings[group.movement_ids].mean(dim=1)
        group_values = model.value_head(traffic_light_embeddings).squeeze(-1)
        values.index_copy_(0, group.flat_value_indices, group_values)
    return values


def _packed_action_log_probs_and_entropy(
    movement_scores: torch.Tensor,
    phase_logit_groups: Sequence[PackedPpoPhaseLogitGroup],
    policy_value_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = movement_scores.new_empty((policy_value_count,))
    entropy_values = movement_scores.new_empty((policy_value_count,))
    for group in phase_logit_groups:
        logits = _packed_phase_logits(movement_scores=movement_scores, group=group)
        masked_logits = logits.masked_fill(~group.action_masks, float('-inf'))
        group_log_probs = F.log_softmax(masked_logits, dim=1)
        selected_log_probs = group_log_probs.gather(1, group.actions.view(-1, 1)).squeeze(1)
        probabilities = group_log_probs.exp()
        entropy = -(probabilities * group_log_probs.masked_fill(~group.action_masks, 0.0)).sum(dim=1)
        log_probs.index_copy_(0, group.flat_policy_indices, selected_log_probs)
        entropy_values.index_copy_(0, group.flat_policy_indices, entropy)
    return log_probs, entropy_values


def _packed_phase_logits(
    movement_scores: torch.Tensor,
    group: PackedPpoPhaseLogitGroup,
) -> torch.Tensor:
    group_movement_scores = movement_scores[group.movement_ids]
    return torch.bmm(group.incidence_matrices, group_movement_scores.unsqueeze(-1)).squeeze(-1)


def _actor_parameters(model: MovementActorCritic) -> Iterator[torch.nn.Parameter]:
    for module in (model.lane_encoder, model.movement_encoder, model.hops, model.score_head):
        yield from module.parameters()


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = sum(
        float(parameter.grad.detach().pow(2).sum().cpu()) for parameter in parameters if parameter.grad is not None
    )
    return sqrt(squared_norm)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))
