"""Movement tensor samples and packed graph batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch

from src.movement.dataset import MovementDatasetSample, StoredPhaseIncidence
from src.movement.normalization import RunningNormalizer
from src.movement.training.il.tensors import edge_tensors_from_sample


@dataclass(frozen=True)
class MovementPhaseTensor:
    incidence_matrix: torch.Tensor
    movement_ids: torch.Tensor
    target_phase: int


@dataclass(frozen=True)
class PackedPhaseLogitGroupBatch:
    incidence_matrices: torch.Tensor
    movement_ids: torch.Tensor
    targets: torch.Tensor
    sample_indices: torch.Tensor


@dataclass(frozen=True)
class MovementTensorSample:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    phase_incidences: dict[str, StoredPhaseIncidence]
    teacher_selected_phase_by_tls: dict[str, int]
    city_name: str
    phase_tensors: dict[str, MovementPhaseTensor] | None = None


@dataclass(frozen=True)
class PackedMovementTensorBatch:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    movement_sample_indices: torch.Tensor
    lane_counts: tuple[int, ...]
    movement_counts: tuple[int, ...]
    phase_logit_groups: tuple[PackedPhaseLogitGroupBatch, ...]
    city_names: tuple[str, ...]


class PackedPhaseLogitGroupPayload(TypedDict):
    incidence_matrices: torch.Tensor
    movement_ids: torch.Tensor
    targets: torch.Tensor
    sample_indices: torch.Tensor


class PackedMovementTensorBatchPayload(TypedDict):
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    movement_sample_indices: torch.Tensor
    lane_counts: tuple[int, ...]
    movement_counts: tuple[int, ...]
    phase_logit_groups: tuple[PackedPhaseLogitGroupPayload, ...]
    city_names: tuple[str, ...]


@dataclass(frozen=True)
class PhaseLogitGroupRows:
    incidence_matrices: list[torch.Tensor]
    movement_ids: list[torch.Tensor]
    targets: list[int]
    sample_indices: list[int]


def movement_tensor_sample_from_dataset_sample(
    sample: MovementDatasetSample,
    city_name: str,
    device: torch.device,
) -> MovementTensorSample:
    return MovementTensorSample(
        x_lane=torch.tensor(sample.x_lane, dtype=torch.float32, device=device),
        x_movement=torch.tensor(sample.x_movement, dtype=torch.float32, device=device),
        target=torch.tensor(sample.teacher_movement_scores, dtype=torch.float32, device=device),
        edge_index_dict=edge_tensors_from_sample(sample, device=device),
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=city_name,
    )


def normalise_movement_tensor_sample(
    sample: MovementTensorSample,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> MovementTensorSample:
    return MovementTensorSample(
        x_lane=normalise_tensor(sample.x_lane, lane_normalizer).to(device),
        x_movement=normalise_tensor(sample.x_movement, movement_normalizer).to(device),
        target=sample.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in sample.edge_index_dict.items()},
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=sample.city_name,
        phase_tensors=phase_tensors_from_sample(sample=sample, device=device),
    )


def move_movement_tensor_sample(sample: MovementTensorSample, device: torch.device) -> MovementTensorSample:
    assert sample.phase_tensors is not None
    return MovementTensorSample(
        x_lane=sample.x_lane.to(device),
        x_movement=sample.x_movement.to(device),
        target=sample.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in sample.edge_index_dict.items()},
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=sample.city_name,
        phase_tensors={
            traffic_light_id: MovementPhaseTensor(
                incidence_matrix=phase_tensor.incidence_matrix.to(device),
                movement_ids=phase_tensor.movement_ids.to(device),
                target_phase=phase_tensor.target_phase,
            )
            for traffic_light_id, phase_tensor in sample.phase_tensors.items()
        },
    )


def pack_movement_tensor_samples(samples: Sequence[MovementTensorSample]) -> PackedMovementTensorBatch:
    lane_counts = tuple(sample.x_lane.shape[0] for sample in samples)
    movement_counts = tuple(sample.x_movement.shape[0] for sample in samples)
    return PackedMovementTensorBatch(
        x_lane=torch.cat(tuple(sample.x_lane for sample in samples), dim=0),
        x_movement=torch.cat(tuple(sample.x_movement for sample in samples), dim=0),
        target=torch.cat(tuple(sample.target for sample in samples), dim=0),
        edge_index_dict=batched_edge_index_dict(samples=samples),
        movement_sample_indices=movement_sample_indices(movement_counts),
        lane_counts=lane_counts,
        movement_counts=movement_counts,
        phase_logit_groups=phase_logit_group_batches(samples=samples),
        city_names=tuple(sample.city_name for sample in samples),
    )


def packed_batch_payload(batch: PackedMovementTensorBatch) -> PackedMovementTensorBatchPayload:
    return PackedMovementTensorBatchPayload(
        x_lane=batch.x_lane,
        x_movement=batch.x_movement,
        target=batch.target,
        edge_index_dict=batch.edge_index_dict,
        movement_sample_indices=batch.movement_sample_indices,
        lane_counts=batch.lane_counts,
        movement_counts=batch.movement_counts,
        phase_logit_groups=tuple(phase_group_payload(group) for group in batch.phase_logit_groups),
        city_names=batch.city_names,
    )


def packed_batch_from_payload(payload: PackedMovementTensorBatchPayload) -> PackedMovementTensorBatch:
    return PackedMovementTensorBatch(
        x_lane=payload['x_lane'],
        x_movement=payload['x_movement'],
        target=payload['target'],
        edge_index_dict=payload['edge_index_dict'],
        movement_sample_indices=payload['movement_sample_indices'],
        lane_counts=payload['lane_counts'],
        movement_counts=payload['movement_counts'],
        phase_logit_groups=tuple(phase_group_from_payload(group) for group in payload['phase_logit_groups']),
        city_names=payload['city_names'],
    )


def move_packed_movement_tensor_batch(
    cpu_batch: PackedMovementTensorBatch,
    device: torch.device,
) -> PackedMovementTensorBatch:
    return PackedMovementTensorBatch(
        x_lane=cpu_batch.x_lane.to(device),
        x_movement=cpu_batch.x_movement.to(device),
        target=cpu_batch.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in cpu_batch.edge_index_dict.items()},
        movement_sample_indices=cpu_batch.movement_sample_indices.to(device),
        lane_counts=cpu_batch.lane_counts,
        movement_counts=cpu_batch.movement_counts,
        phase_logit_groups=tuple(
            move_phase_logit_group(group=group, device=device) for group in cpu_batch.phase_logit_groups
        ),
        city_names=cpu_batch.city_names,
    )


def phase_group_payload(group: PackedPhaseLogitGroupBatch) -> PackedPhaseLogitGroupPayload:
    return PackedPhaseLogitGroupPayload(
        incidence_matrices=group.incidence_matrices,
        movement_ids=group.movement_ids,
        targets=group.targets,
        sample_indices=group.sample_indices,
    )


def phase_group_from_payload(payload: PackedPhaseLogitGroupPayload) -> PackedPhaseLogitGroupBatch:
    return PackedPhaseLogitGroupBatch(
        incidence_matrices=payload['incidence_matrices'],
        movement_ids=payload['movement_ids'],
        targets=payload['targets'],
        sample_indices=payload['sample_indices'],
    )


def movement_sample_indices(movement_counts: Sequence[int]) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.full((movement_count,), sample_index, dtype=torch.long)
            for sample_index, movement_count in enumerate(movement_counts)
        ),
        dim=0,
    )


def phase_logit_group_batches(
    samples: Sequence[MovementTensorSample],
) -> tuple[PackedPhaseLogitGroupBatch, ...]:
    groups: dict[tuple[int, int], PhaseLogitGroupRows] = {}
    movement_offset = 0
    for sample_index, sample in enumerate(samples):
        assert sample.phase_tensors is not None
        for phase_tensor in sample.phase_tensors.values():
            group_key = (phase_tensor.incidence_matrix.shape[0], phase_tensor.incidence_matrix.shape[1])
            group = groups.setdefault(
                group_key,
                PhaseLogitGroupRows(
                    incidence_matrices=[],
                    movement_ids=[],
                    targets=[],
                    sample_indices=[],
                ),
            )
            group.incidence_matrices.append(phase_tensor.incidence_matrix)
            group.movement_ids.append(phase_tensor.movement_ids + movement_offset)
            group.targets.append(phase_tensor.target_phase)
            group.sample_indices.append(sample_index)
        movement_offset += sample.x_movement.shape[0]
    return tuple(
        PackedPhaseLogitGroupBatch(
            incidence_matrices=torch.stack(tuple(group.incidence_matrices)),
            movement_ids=torch.stack(tuple(group.movement_ids)),
            targets=torch.tensor(group.targets, dtype=torch.long),
            sample_indices=torch.tensor(group.sample_indices, dtype=torch.long),
        )
        for group in groups.values()
    )


def move_phase_logit_group(
    group: PackedPhaseLogitGroupBatch,
    device: torch.device,
) -> PackedPhaseLogitGroupBatch:
    return PackedPhaseLogitGroupBatch(
        incidence_matrices=group.incidence_matrices.to(device),
        movement_ids=group.movement_ids.to(device),
        targets=group.targets.to(device),
        sample_indices=group.sample_indices.to(device),
    )


def phase_tensors_from_sample(
    sample: MovementTensorSample,
    device: torch.device,
) -> dict[str, MovementPhaseTensor]:
    return {
        traffic_light_id: MovementPhaseTensor(
            incidence_matrix=torch.tensor(incidence.rows, dtype=torch.float32, device=device),
            movement_ids=torch.tensor(incidence.movement_ids, dtype=torch.long, device=device),
            target_phase=sample.teacher_selected_phase_by_tls[traffic_light_id],
        )
        for traffic_light_id, incidence in sample.phase_incidences.items()
    }


def normalise_tensor(tensor: torch.Tensor, normalizer: RunningNormalizer) -> torch.Tensor:
    if normalizer.count == 0:
        return torch.zeros_like(tensor)
    mean = torch.tensor(normalizer.mean, dtype=torch.float32)
    std = torch.tensor(normalizer.std, dtype=torch.float32).clamp_min(normalizer.epsilon)
    return torch.round((tensor - mean) / std, decimals=6)


def batched_edge_index_dict(
    samples: Sequence[MovementTensorSample],
) -> dict[str, torch.Tensor]:
    edge_parts: dict[str, list[torch.Tensor]] = {
        'input_lane_to_movement': [],
        'output_lane_to_movement': [],
        'movement_to_input_lane': [],
        'movement_to_output_lane': [],
        'lane_to_lane': [],
    }
    lane_to_lane_weight_parts: list[torch.Tensor] = []
    lane_offset = 0
    movement_offset = 0
    for sample in samples:
        edge_parts['input_lane_to_movement'].append(
            offset_edge_index(sample.edge_index_dict['input_lane_to_movement'], lane_offset, movement_offset)
        )
        edge_parts['output_lane_to_movement'].append(
            offset_edge_index(sample.edge_index_dict['output_lane_to_movement'], lane_offset, movement_offset)
        )
        edge_parts['movement_to_input_lane'].append(
            offset_edge_index(sample.edge_index_dict['movement_to_input_lane'], movement_offset, lane_offset)
        )
        edge_parts['movement_to_output_lane'].append(
            offset_edge_index(sample.edge_index_dict['movement_to_output_lane'], movement_offset, lane_offset)
        )
        edge_parts['lane_to_lane'].append(
            offset_edge_index(sample.edge_index_dict['lane_to_lane'], lane_offset, lane_offset)
        )
        lane_to_lane_weight_parts.append(sample.edge_index_dict['lane_to_lane_weight'])
        lane_offset += sample.x_lane.shape[0]
        movement_offset += sample.x_movement.shape[0]
    return {
        'input_lane_to_movement': cat_edge_parts(edge_parts['input_lane_to_movement']),
        'output_lane_to_movement': cat_edge_parts(edge_parts['output_lane_to_movement']),
        'movement_to_input_lane': cat_edge_parts(edge_parts['movement_to_input_lane']),
        'movement_to_output_lane': cat_edge_parts(edge_parts['movement_to_output_lane']),
        'lane_to_lane': cat_edge_parts(edge_parts['lane_to_lane']),
        'lane_to_lane_weight': torch.cat(tuple(lane_to_lane_weight_parts), dim=0),
    }


def offset_edge_index(edge_index: torch.Tensor, source_offset: int, target_offset: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    offsets = torch.tensor(
        ((source_offset,), (target_offset,)),
        dtype=edge_index.dtype,
        device=edge_index.device,
    )
    return edge_index + offsets


def cat_edge_parts(edge_parts: Sequence[torch.Tensor]) -> torch.Tensor:
    non_empty_parts = tuple(edge_part for edge_part in edge_parts if edge_part.numel() > 0)
    if not non_empty_parts:
        device = edge_parts[0].device
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(non_empty_parts, dim=1)
