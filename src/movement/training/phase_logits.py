"""Phase-logit construction from movement scores."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.movement.dataset import MovementDatasetSample, StoredPhaseIncidence


def phase_logits_from_incidence(
    incidence: StoredPhaseIncidence,
    movement_scores: torch.Tensor,
) -> torch.Tensor:
    phase_scores = []
    for row in incidence.rows:
        enabled_scores = tuple(
            movement_scores[movement_id] for enabled, movement_id in zip(row, incidence.movement_ids) if enabled == 1
        )
        phase_scores.append(torch.stack(enabled_scores).sum())
    return torch.stack(tuple(phase_scores))


def phase_logits_from_sample(
    sample: MovementDatasetSample,
    traffic_light_ids: Sequence[str],
    movement_scores: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        phase_logits_from_incidence(
            incidence=sample.phase_incidences[traffic_light_id],
            movement_scores=movement_scores,
        )
        for traffic_light_id in traffic_light_ids
    )
