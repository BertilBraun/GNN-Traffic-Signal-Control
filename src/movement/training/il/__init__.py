"""Offline movement-score imitation learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from src.movement.dataset import MovementDatasetSample, StoredPhaseIncidence, load_jsonl_samples
from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.training.il.checkpoint import movement_checkpoint_payload
from src.movement.training.il.batching import MovementILBatchPlanner, RandomBatchPlanner
from src.movement.training.il.tensors import edge_tensors_from_sample, tensors_from_sample
from src.movement.training.il.types import (
    MovementILLoss,
    MovementILTrainingConfig,
    MovementILTrainingObserver,
    MovementILTrainingResult,
    MovementILTrainingSnapshot,
)


def train_movement_il(
    samples: Sequence[MovementDatasetSample],
    config: MovementILTrainingConfig,
    observer: MovementILTrainingObserver | None,
    batch_planner: MovementILBatchPlanner,
    validation_samples: Sequence[MovementDatasetSample],
) -> MovementILTrainingResult:
    """Train a movement scorer on stored movement-score samples."""
    if not samples:
        raise ValueError('At least one sample is required.')
    if config.samples_per_batch <= 0:
        raise ValueError('samples_per_batch must be positive.')
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    lane_normalizer = _fit_normalizer(sample.x_lane for sample in samples)
    movement_normalizer = _fit_normalizer(sample.x_movement for sample in samples)
    lane_feature_dim = len(samples[0].x_lane[0])
    movement_feature_dim = len(samples[0].x_movement[0])
    model = MovementScorer(
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    writer = SummaryWriter(log_dir=str(config.log_dir)) if config.log_dir is not None else None
    final_loss = 0.0
    best_loss = float('inf')
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(config.epochs):
        total_loss_value = 0.0
        total_regression_loss_value = 0.0
        total_phase_loss_value = 0.0
        total_phase_matches = 0
        total_phase_decisions = 0
        total_trained_samples = 0
        for sample_indices in batch_planner.epoch_batches(samples=samples, epoch=epoch):
            batch_losses = []
            for sample_index in sample_indices:
                sample = samples[sample_index]
                x_lane, x_movement, target = tensors_from_sample(
                    sample=sample,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    device=device,
                )
                prediction = model(
                    x_lane=x_lane,
                    x_movement=x_movement,
                    edge_index_dict=edge_tensors_from_sample(sample, device=device),
                )
                regression_loss = _loss(prediction, target, config.loss)
                phase_loss = _phase_classification_loss(
                    sample=sample,
                    movement_scores=prediction,
                )
                sample_loss = regression_loss + config.phase_loss_coefficient * phase_loss
                batch_losses.append(sample_loss)
                total_loss_value += float(sample_loss.detach().cpu())
                total_regression_loss_value += float(regression_loss.detach().cpu())
                total_phase_loss_value += float(phase_loss.detach().cpu())
                matches, decisions = _phase_match_counts(
                    sample=sample,
                    movement_scores=prediction,
                )
                total_phase_matches += matches
                total_phase_decisions += decisions
                total_trained_samples += 1
            loss = torch.stack(tuple(batch_losses)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = total_loss_value / total_trained_samples
        regression_loss_value = total_regression_loss_value / total_trained_samples
        phase_loss_value = total_phase_loss_value / total_trained_samples
        phase_match_rate = total_phase_matches / max(1, total_phase_decisions)
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if _should_report_progress(epoch, config.epochs, config.progress_every):
            print(
                f'epoch={epoch + 1}/{config.epochs} '
                f'loss={final_loss:.6f} regression={regression_loss_value:.6f} '
                f'phase={phase_loss_value:.6f} match={phase_match_rate:.3f}'
            )
        if writer is not None:
            writer.add_scalar('loss/total', final_loss, epoch + 1)
            writer.add_scalar('loss/regression', regression_loss_value, epoch + 1)
            writer.add_scalar('loss/phase', phase_loss_value, epoch + 1)
            writer.add_scalar('accuracy/phase_match', phase_match_rate, epoch + 1)
            _write_validation_losses(
                writer=writer,
                epoch=epoch + 1,
                validation_samples=validation_samples,
                model=model,
                lane_normalizer=lane_normalizer,
                movement_normalizer=movement_normalizer,
                device=device,
                loss_name=config.loss,
                phase_loss_coefficient=config.phase_loss_coefficient,
            )
        if observer is not None:
            observer.on_epoch_completed(
                MovementILTrainingSnapshot(
                    epoch=epoch + 1,
                    epochs=config.epochs,
                    loss=final_loss,
                    regression_loss=regression_loss_value,
                    phase_loss=phase_loss_value,
                    phase_match_rate=phase_match_rate,
                    best_loss=best_loss,
                    model=model,
                    config=config,
                    lane_feature_dim=lane_feature_dim,
                    movement_feature_dim=movement_feature_dim,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                )
            )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = movement_checkpoint_payload(
        model_state=model.state_dict(),
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        loss=final_loss,
    )
    best_checkpoint = movement_checkpoint_payload(
        model_state=best_state,
        config=config,
        lane_feature_dim=lane_feature_dim,
        movement_feature_dim=movement_feature_dim,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        loss=best_loss,
    )
    last_path = checkpoint_dir / 'movement_policy_last.pt'
    best_path = checkpoint_dir / 'movement_policy_best.pt'
    torch.save(last_checkpoint, last_path)
    torch.save(best_checkpoint, best_path)
    if writer is not None:
        writer.close()
    return MovementILTrainingResult(
        checkpoint_path=last_path,
        final_loss=final_loss,
        epochs=config.epochs,
    )


def train_movement_il_from_jsonl(
    dataset_path: Path | str,
    config: MovementILTrainingConfig,
    observer: MovementILTrainingObserver | None,
    batch_planner: MovementILBatchPlanner,
    validation_samples: Sequence[MovementDatasetSample],
) -> MovementILTrainingResult:
    """Load JSONL samples and train the movement scorer."""
    return train_movement_il(load_jsonl_samples(dataset_path), config, observer, batch_planner, validation_samples)


def random_batch_planner(config: MovementILTrainingConfig) -> RandomBatchPlanner:
    return RandomBatchPlanner(
        samples_per_batch=config.samples_per_batch,
        seed=config.seed,
    )


def _fit_normalizer(
    batches: Iterable[Sequence[Sequence[float]]],
) -> RunningNormalizer:
    normalizer = RunningNormalizer()
    for rows in batches:
        normalizer.update_rows(rows)
    normalizer.freeze()
    return normalizer


def _write_validation_losses(
    writer: SummaryWriter,
    epoch: int,
    validation_samples: Sequence[MovementDatasetSample],
    model: MovementScorer,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    loss_name: MovementILLoss,
    phase_loss_coefficient: float,
) -> None:
    if not validation_samples:
        return
    losses_by_city = _validation_losses_by_city(
        validation_samples=validation_samples,
        model=model,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        loss_name=loss_name,
        phase_loss_coefficient=phase_loss_coefficient,
    )
    all_losses = tuple(loss for city_losses in losses_by_city for loss in city_losses.losses)
    writer.add_scalar('validation/loss', sum(all_losses) / len(all_losses), epoch)
    for city_losses in losses_by_city:
        writer.add_scalar(
            f'validation/{city_losses.city_name}/loss',
            sum(city_losses.losses) / len(city_losses.losses),
            epoch,
        )


@dataclass(frozen=True)
class CityValidationLosses:
    city_name: str
    losses: tuple[float, ...]


class ValidationSampleMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str


def _validation_losses_by_city(
    validation_samples: Sequence[MovementDatasetSample],
    model: MovementScorer,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
    loss_name: MovementILLoss,
    phase_loss_coefficient: float,
) -> tuple[CityValidationLosses, ...]:
    city_names: list[str] = []
    for sample in validation_samples:
        city_name = ValidationSampleMetadata.model_validate(sample.metadata).city_name
        if city_name not in city_names:
            city_names.append(city_name)
    losses_by_city: list[CityValidationLosses] = []
    model.eval()
    with torch.no_grad():
        for city_name in city_names:
            losses: list[float] = []
            for sample in validation_samples:
                if ValidationSampleMetadata.model_validate(sample.metadata).city_name != city_name:
                    continue
                x_lane, x_movement, target = tensors_from_sample(
                    sample=sample,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    device=device,
                )
                prediction = model(
                    x_lane=x_lane,
                    x_movement=x_movement,
                    edge_index_dict=edge_tensors_from_sample(sample, device=device),
                )
                regression_loss = _loss(prediction, target, loss_name)
                phase_loss = _phase_classification_loss(
                    sample=sample,
                    movement_scores=prediction,
                )
                losses.append(float((regression_loss + phase_loss_coefficient * phase_loss).detach().cpu()))
            losses_by_city.append(CityValidationLosses(city_name=city_name, losses=tuple(losses)))
    model.train()
    return tuple(losses_by_city)


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: MovementILLoss,
) -> torch.Tensor:
    match loss_name:
        case MovementILLoss.HUBER:
            return F.smooth_l1_loss(prediction, target)
        case MovementILLoss.MEAN_SQUARED_ERROR:
            return F.mse_loss(prediction, target)


def _phase_classification_loss(
    sample: MovementDatasetSample,
    movement_scores: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for traffic_light_id, incidence in sample.phase_incidences.items():
        logits = _phase_logits(
            incidence=incidence,
            movement_scores=movement_scores,
        )
        target = torch.tensor(
            (sample.teacher_selected_phase_by_tls[traffic_light_id],),
            dtype=torch.long,
            device=movement_scores.device,
        )
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
    return torch.stack(tuple(losses)).mean()


def _phase_match_counts(
    sample: MovementDatasetSample,
    movement_scores: torch.Tensor,
) -> tuple[int, int]:
    matches = 0
    for traffic_light_id, incidence in sample.phase_incidences.items():
        predicted_phase = int(
            _phase_logits(
                incidence=incidence,
                movement_scores=movement_scores,
            )
            .argmax()
            .detach()
            .cpu()
        )
        matches += int(predicted_phase == sample.teacher_selected_phase_by_tls[traffic_light_id])
    return matches, len(sample.phase_incidences)


def _phase_logits(
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


def _should_report_progress(epoch: int, epochs: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return (epoch + 1) % progress_every == 0 or epoch == epochs - 1
