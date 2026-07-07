"""Packed PPO minibatches for movement actor-critic updates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from src.movement.training.movement_batch import (
    MovementTensorSample,
    PackedMovementTensorBatch,
    batched_edge_index_dict,
    movement_sample_indices,
)
from src.movement.training.rollout.math import normalize_advantages
from src.movement.training.rollout.types import MovementTransition


@dataclass(frozen=True)
class MovementPpoBatchRow:
    transition: MovementTransition
    advantage: torch.Tensor
    return_value: torch.Tensor


@dataclass(frozen=True)
class PackedPpoPhaseLogitGroup:
    incidence_matrices: torch.Tensor
    movement_ids: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    flat_policy_indices: torch.Tensor


@dataclass(frozen=True)
class PackedPpoValueGroup:
    movement_ids: torch.Tensor
    flat_value_indices: torch.Tensor


@dataclass(frozen=True)
class PackedMovementPpoBatch:
    movement_batch: PackedMovementTensorBatch
    phase_logit_groups: tuple[PackedPpoPhaseLogitGroup, ...]
    value_groups: tuple[PackedPpoValueGroup, ...]
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    policy_mask: torch.Tensor
    transition_count: int
    policy_value_count: int


@dataclass
class PpoPhaseLogitGroupRows:
    incidence_matrices: list[torch.Tensor]
    movement_ids: list[torch.Tensor]
    actions: list[int]
    action_masks: list[torch.Tensor]
    flat_policy_indices: list[int]


@dataclass
class PpoValueGroupRows:
    movement_ids: list[torch.Tensor]
    flat_value_indices: list[int]


class MovementPpoDataset(Dataset[MovementPpoBatchRow]):
    def __init__(
        self,
        transitions: Sequence[MovementTransition],
        advantages: Sequence[torch.Tensor],
        returns: Sequence[torch.Tensor],
    ) -> None:
        if len(transitions) != len(advantages) or len(transitions) != len(returns):
            raise ValueError('transitions, advantages, and returns must have the same length.')
        self.transitions = tuple(transitions)
        self.advantages = tuple(advantages)
        self.returns = tuple(returns)

    def __getitem__(self, index: int) -> MovementPpoBatchRow:
        return MovementPpoBatchRow(
            transition=self.transitions[index],
            advantage=self.advantages[index],
            return_value=self.returns[index],
        )

    def __len__(self) -> int:
        return len(self.transitions)


def ppo_batch_data_loader(
    transitions: Sequence[MovementTransition],
    advantages: Sequence[torch.Tensor],
    returns: Sequence[torch.Tensor],
    transitions_per_batch: int,
    update_batch_workers: int,
) -> DataLoader[MovementPpoBatchRow]:
    return DataLoader(
        MovementPpoDataset(
            transitions=transitions,
            advantages=advantages,
            returns=returns,
        ),
        batch_size=max(1, transitions_per_batch),
        shuffle=True,
        num_workers=update_batch_workers,
        collate_fn=collate_movement_ppo_batch,
    )


def collate_movement_ppo_batch(rows: Sequence[MovementPpoBatchRow]) -> PackedMovementPpoBatch:
    if not rows:
        raise ValueError('Cannot collate an empty PPO batch.')
    transitions = tuple(row.transition for row in rows)
    tensor_samples = tuple(transition.tensor_sample for transition in transitions)
    old_log_probs = torch.cat(
        tuple(torch.tensor(transition.old_log_probs, dtype=torch.float32) for transition in transitions)
    )
    advantages = torch.cat(tuple(row.advantage.to(torch.device('cpu')) for row in rows))
    returns = torch.cat(tuple(row.return_value.to(torch.device('cpu')) for row in rows))
    policy_mask = torch.cat(tuple(policy_mask_from_transition(transition) for transition in transitions))
    return PackedMovementPpoBatch(
        movement_batch=pack_ppo_movement_tensor_samples(tensor_samples),
        phase_logit_groups=ppo_phase_logit_groups(transitions=transitions),
        value_groups=ppo_value_groups(transitions=transitions),
        old_log_probs=old_log_probs,
        advantages=normalize_advantages(advantages=advantages, policy_mask=policy_mask),
        returns=returns,
        policy_mask=policy_mask,
        transition_count=len(transitions),
        policy_value_count=old_log_probs.numel(),
    )


def pack_ppo_movement_tensor_samples(samples: Sequence[MovementTensorSample]) -> PackedMovementTensorBatch:
    lane_counts = tuple(sample.x_lane.shape[0] for sample in samples)
    movement_counts = tuple(sample.x_movement.shape[0] for sample in samples)
    return PackedMovementTensorBatch(
        x_lane=torch.cat(tuple(sample.x_lane for sample in samples), dim=0),
        x_movement=torch.cat(tuple(sample.x_movement for sample in samples), dim=0),
        target=torch.empty((0,), dtype=torch.float32),
        edge_index_dict=batched_edge_index_dict(samples=samples),
        movement_sample_indices=movement_sample_indices(movement_counts),
        lane_counts=lane_counts,
        movement_counts=movement_counts,
        phase_logit_groups=(),
        city_names=tuple(sample.city_name for sample in samples),
    )


def ppo_phase_logit_groups(
    transitions: Sequence[MovementTransition],
) -> tuple[PackedPpoPhaseLogitGroup, ...]:
    groups: dict[tuple[int, int], PpoPhaseLogitGroupRows] = {}
    movement_offset = 0
    flat_policy_index = 0
    for transition in transitions:
        traffic_light_ids = tuple(sorted(transition.tensor_sample.phase_incidences.keys(), key=str))
        if len(traffic_light_ids) != len(transition.actions):
            raise ValueError('PPO transition action count does not match traffic-light count.')
        for traffic_light_index, traffic_light_id in enumerate(traffic_light_ids):
            incidence = transition.tensor_sample.phase_incidences[traffic_light_id]
            action_mask = transition.action_masks[traffic_light_index]
            group_key = (len(incidence.rows), len(incidence.movement_ids))
            group = groups.setdefault(
                group_key,
                PpoPhaseLogitGroupRows(
                    incidence_matrices=[],
                    movement_ids=[],
                    actions=[],
                    action_masks=[],
                    flat_policy_indices=[],
                ),
            )
            group.incidence_matrices.append(torch.tensor(incidence.rows, dtype=torch.float32))
            group.movement_ids.append(torch.tensor(incidence.movement_ids, dtype=torch.long) + movement_offset)
            group.actions.append(transition.actions[traffic_light_index])
            group.action_masks.append(torch.tensor(action_mask, dtype=torch.bool))
            group.flat_policy_indices.append(flat_policy_index)
            flat_policy_index += 1
        movement_offset += transition.tensor_sample.x_movement.shape[0]
    return tuple(
        PackedPpoPhaseLogitGroup(
            incidence_matrices=torch.stack(tuple(group.incidence_matrices)),
            movement_ids=torch.stack(tuple(group.movement_ids)),
            actions=torch.tensor(group.actions, dtype=torch.long),
            action_masks=torch.stack(tuple(group.action_masks)),
            flat_policy_indices=torch.tensor(group.flat_policy_indices, dtype=torch.long),
        )
        for group in groups.values()
    )


def ppo_value_groups(transitions: Sequence[MovementTransition]) -> tuple[PackedPpoValueGroup, ...]:
    groups: dict[int, PpoValueGroupRows] = {}
    movement_offset = 0
    flat_value_index = 0
    for transition in transitions:
        traffic_light_ids = tuple(sorted(transition.tensor_sample.phase_incidences.keys(), key=str))
        for traffic_light_id in traffic_light_ids:
            incidence = transition.tensor_sample.phase_incidences[traffic_light_id]
            movement_count = len(incidence.movement_ids)
            group = groups.setdefault(
                movement_count,
                PpoValueGroupRows(
                    movement_ids=[],
                    flat_value_indices=[],
                ),
            )
            group.movement_ids.append(torch.tensor(incidence.movement_ids, dtype=torch.long) + movement_offset)
            group.flat_value_indices.append(flat_value_index)
            flat_value_index += 1
        movement_offset += transition.tensor_sample.x_movement.shape[0]
    return tuple(
        PackedPpoValueGroup(
            movement_ids=torch.stack(tuple(group.movement_ids)),
            flat_value_indices=torch.tensor(group.flat_value_indices, dtype=torch.long),
        )
        for group in groups.values()
    )


def policy_mask_from_transition(transition: MovementTransition) -> torch.Tensor:
    return torch.tensor(
        tuple(sum(action_mask) > 1 for action_mask in transition.action_masks),
        dtype=torch.bool,
    )


def move_packed_movement_ppo_batch(
    cpu_batch: PackedMovementPpoBatch,
    device: torch.device,
) -> PackedMovementPpoBatch:
    movement_batch = cpu_batch.movement_batch
    return PackedMovementPpoBatch(
        movement_batch=PackedMovementTensorBatch(
            x_lane=movement_batch.x_lane.to(device),
            x_movement=movement_batch.x_movement.to(device),
            target=movement_batch.target.to(device),
            edge_index_dict={key: value.to(device) for key, value in movement_batch.edge_index_dict.items()},
            movement_sample_indices=movement_batch.movement_sample_indices.to(device),
            lane_counts=movement_batch.lane_counts,
            movement_counts=movement_batch.movement_counts,
            phase_logit_groups=(),
            city_names=movement_batch.city_names,
        ),
        phase_logit_groups=tuple(
            move_ppo_phase_logit_group(group=group, device=device) for group in cpu_batch.phase_logit_groups
        ),
        value_groups=tuple(move_ppo_value_group(group=group, device=device) for group in cpu_batch.value_groups),
        old_log_probs=cpu_batch.old_log_probs.to(device),
        advantages=cpu_batch.advantages.to(device),
        returns=cpu_batch.returns.to(device),
        policy_mask=cpu_batch.policy_mask.to(device),
        transition_count=cpu_batch.transition_count,
        policy_value_count=cpu_batch.policy_value_count,
    )


def move_ppo_phase_logit_group(
    group: PackedPpoPhaseLogitGroup,
    device: torch.device,
) -> PackedPpoPhaseLogitGroup:
    return PackedPpoPhaseLogitGroup(
        incidence_matrices=group.incidence_matrices.to(device),
        movement_ids=group.movement_ids.to(device),
        actions=group.actions.to(device),
        action_masks=group.action_masks.to(device),
        flat_policy_indices=group.flat_policy_indices.to(device),
    )


def move_ppo_value_group(
    group: PackedPpoValueGroup,
    device: torch.device,
) -> PackedPpoValueGroup:
    return PackedPpoValueGroup(
        movement_ids=group.movement_ids.to(device),
        flat_value_indices=group.flat_value_indices.to(device),
    )
