"""PPO optimizer update step."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.normalization import RunningNormalizer
from src.movement.training.ppo.policy import forward_policy, masked_phase_logits, policy_context_from_sample
from src.movement.training.ppo.types import MovementPpoConfig
from src.movement.training.rollout import MovementRolloutBuffer
from src.movement.training.rollout.types import MovementPpoBatch


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


def update_ppo(
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
    history = PpoUpdateHistory()
    for _epoch in range(epochs):
        early_stopped = _run_update_epoch(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
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

    def __init__(self) -> None:
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.total_losses = []
        self.approximate_kls = []
        self.ratio_clip_fractions = []
        self.backbone_gradient_norms = []
        self.value_head_gradient_norms = []

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
        )


@dataclass(frozen=True)
class PpoBatchLoss:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor
    approximate_kl: torch.Tensor
    ratio_clip_fraction: torch.Tensor


def _run_update_epoch(
    model: MovementActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: MovementRolloutBuffer,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    config: MovementPpoConfig,
    warming_up: bool,
    history: PpoUpdateHistory,
) -> bool:
    for batch in buffer.iterate_minibatches(config.transitions_per_batch, device=device):
        loss = _batch_loss(
            model=model,
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
            config=config,
            warming_up=warming_up,
            batch=batch,
        )
        if not warming_up and float(loss.approximate_kl.detach().cpu()) > config.target_kl:
            return True
        optimizer.zero_grad()
        loss.total_loss.backward()
        history.backbone_gradient_norms.append(gradient_norm(tuple(_actor_parameters(model))))
        history.value_head_gradient_norms.append(gradient_norm(tuple(model.value_head.parameters())))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        history.policy_losses.append(float(loss.policy_loss.detach().cpu()))
        history.value_losses.append(float(loss.value_loss.detach().cpu()))
        history.entropies.append(float(loss.entropy.detach().cpu()))
        history.total_losses.append(float(loss.total_loss.detach().cpu()))
        history.approximate_kls.append(float(loss.approximate_kl.detach().cpu()))
        history.ratio_clip_fractions.append(float(loss.ratio_clip_fraction.detach().cpu()))
    return False


def _batch_loss(
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    config: MovementPpoConfig,
    warming_up: bool,
    batch: MovementPpoBatch,
) -> PpoBatchLoss:
    new_log_probs, entropy_values, values = _evaluate_batch_policy(
        model=model,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        batch=batch,
    )
    entropy = (
        entropy_values[batch.policy_mask].mean() if bool(batch.policy_mask.any()) else entropy_values.new_zeros(())
    )
    value_loss = F.mse_loss(values, batch.returns)
    if warming_up:
        zero = values.new_zeros(())
        return PpoBatchLoss(
            policy_loss=zero,
            value_loss=value_loss,
            entropy=entropy,
            total_loss=config.value_coefficient * value_loss,
            approximate_kl=zero,
            ratio_clip_fraction=zero,
        )
    log_ratio = new_log_probs - batch.old_log_probs
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
    )


def _evaluate_batch_policy(
    model: MovementActorCritic,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    batch: MovementPpoBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_log_probs = []
    batch_entropies = []
    batch_values = []
    for transition in batch.transitions:
        _movement_scores, values, phase_logits = forward_policy(
            model=model,
            sample=transition.sample,
            context=policy_context_from_sample(transition.sample),
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
            device=device,
        )
        distributions = tuple(
            Categorical(logits=logits)
            for logits in masked_phase_logits(
                phase_logits,
                transition.action_masks,
            )
        )
        batch_log_probs.append(
            torch.stack(
                tuple(
                    distribution.log_prob(torch.tensor(action, dtype=torch.long, device=device))
                    for distribution, action in zip(distributions, transition.actions)
                )
            )
        )
        batch_entropies.append(torch.stack(tuple(distribution.entropy() for distribution in distributions)))
        batch_values.append(values)
    return torch.cat(batch_log_probs), torch.cat(batch_entropies), torch.cat(batch_values)


def _actor_parameters(model: MovementActorCritic) -> Iterator[torch.nn.Parameter]:
    for module in (model.lane_encoder, model.movement_encoder, model.hops, model.score_head):
        yield from module.parameters()


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = sum(
        float(parameter.grad.detach().pow(2).sum().cpu()) for parameter in parameters if parameter.grad is not None
    )
    return sqrt(squared_norm)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))
