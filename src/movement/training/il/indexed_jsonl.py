"""Indexed JSONL training path for large imitation-learning datasets."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
import time
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from src.movement.dataset import MovementDatasetSample, StoredPhaseIncidence, load_jsonl_sample_line
from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.training.il import (
    _loss,
    _phase_classification_loss,
    _phase_match_counts,
    _save_training_checkpoints,
)
from src.movement.training.il.tensors import edge_tensors_from_sample, tensors_from_sample
from src.movement.training.il.types import (
    MovementILLoss,
    MovementILTrainingConfig,
    MovementILTrainingObserver,
    MovementILTrainingResult,
    MovementILTrainingSnapshot,
)


class IndexedSampleMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    collection_seed: int


@dataclass(frozen=True)
class IndexedJsonlRecord:
    sample_index: int
    byte_offset: int
    byte_length: int
    city_name: str
    collection_seed: int


@dataclass(frozen=True)
class CitySeedIndexedGroup:
    city_name: str
    collection_seed: int
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class IndexedJsonlStats:
    dataset_path: Path
    file_size_bytes: int
    sample_count: int
    train_count: int
    validation_count: int
    groups: tuple[CitySeedIndexedGroup, ...]


@dataclass(frozen=True)
class IndexedTrainValidationSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class CachedMovementTensorSample:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    phase_incidences: dict[str, StoredPhaseIncidence]
    teacher_selected_phase_by_tls: dict[str, int]
    city_name: str


class IndexedJsonlDataset:
    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)
        self.records = _index_jsonl_records(self.dataset_path)
        if not self.records:
            raise ValueError('At least one sample is required.')

    def sample(self, sample_index: int) -> MovementDatasetSample:
        record = self.records[sample_index]
        with self.dataset_path.open('rb') as handle:
            handle.seek(record.byte_offset)
            line = handle.read(record.byte_length).decode('utf-8')
        return load_jsonl_sample_line(line)

    def split_train_validation(
        self,
        validation_fraction: float,
        seed: int,
        max_train_samples: int | None,
    ) -> IndexedTrainValidationSplit:
        if validation_fraction < 0.0 or validation_fraction >= 1.0:
            raise ValueError('validation_fraction must be in [0.0, 1.0).')
        validation_indices: set[int] = set()
        if validation_fraction > 0.0:
            groups = _groups_from_records(self.records)
            city_names = _ordered_city_names(groups)
            for city_index, city_name in enumerate(city_names):
                city_groups = tuple(group for group in groups if group.city_name == city_name)
                if len(city_groups) < 2:
                    continue
                shuffled_groups = list(city_groups)
                Random(seed + city_index).shuffle(shuffled_groups)
                validation_group_count = max(1, round(len(shuffled_groups) * validation_fraction))
                for group in shuffled_groups[:validation_group_count]:
                    validation_indices.update(group.sample_indices)
        train_indices = tuple(
            record.sample_index for record in self.records if record.sample_index not in validation_indices
        )
        if max_train_samples is not None:
            if max_train_samples <= 0:
                raise ValueError('max_train_samples must be positive when set.')
            limited_train_indices = train_indices[:max_train_samples]
        else:
            limited_train_indices = train_indices
        if not limited_train_indices:
            raise ValueError('validation split left no training samples.')
        return IndexedTrainValidationSplit(
            train_indices=limited_train_indices,
            validation_indices=tuple(
                record.sample_index for record in self.records if record.sample_index in validation_indices
            ),
        )

    def stats(self, split: IndexedTrainValidationSplit) -> IndexedJsonlStats:
        split_indices = set(split.train_indices).union(split.validation_indices)
        groups = tuple(
            CitySeedIndexedGroup(
                city_name=group.city_name,
                collection_seed=group.collection_seed,
                sample_indices=tuple(index for index in group.sample_indices if index in split_indices),
            )
            for group in _groups_from_records(self.records)
        )
        return IndexedJsonlStats(
            dataset_path=self.dataset_path,
            file_size_bytes=self.dataset_path.stat().st_size,
            sample_count=len(self.records),
            train_count=len(split.train_indices),
            validation_count=len(split.validation_indices),
            groups=tuple(group for group in groups if group.sample_indices),
        )


class MovementTensorCache:
    def __init__(
        self,
        dataset: IndexedJsonlDataset,
        cache_dir: Path,
        lane_normalizer: RunningNormalizer,
        movement_normalizer: RunningNormalizer,
        device: torch.device,
        max_cached_samples: int,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.lane_normalizer = lane_normalizer
        self.movement_normalizer = movement_normalizer
        self.device = device
        self.max_cached_samples = max_cached_samples
        self.memory_cache: OrderedDict[int, CachedMovementTensorSample] = OrderedDict()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self, sample_indices: Sequence[int]) -> None:
        missing_indices = tuple(index for index in sample_indices if not self._cache_path(index).exists())
        if not missing_indices:
            return
        started_s = time.monotonic()
        for position, sample_index in enumerate(missing_indices, start=1):
            sample = self.dataset.sample(sample_index)
            cached_sample = _cached_sample_from_sample(
                sample=sample,
                lane_normalizer=self.lane_normalizer,
                movement_normalizer=self.movement_normalizer,
                device=torch.device('cpu'),
            )
            torch.save(cached_sample, self._cache_path(sample_index))
            if position % 1000 == 0 or position == len(missing_indices):
                elapsed_s = time.monotonic() - started_s
                print(f'tensor_cache samples={position}/{len(missing_indices)} elapsed={_format_duration(elapsed_s)}')

    def get(self, sample_index: int) -> CachedMovementTensorSample:
        cached = self.memory_cache.get(sample_index)
        if cached is not None:
            self.memory_cache.move_to_end(sample_index)
            return _move_cached_sample(cached, self.device)
        loaded = cast(
            CachedMovementTensorSample,
            torch.load(self._cache_path(sample_index), map_location='cpu', weights_only=False),
        )
        self.memory_cache[sample_index] = loaded
        if len(self.memory_cache) > self.max_cached_samples:
            self.memory_cache.popitem(last=False)
        return _move_cached_sample(loaded, self.device)

    def _cache_path(self, sample_index: int) -> Path:
        return self.cache_dir / f'sample_{sample_index:08d}.pt'


def train_movement_il_from_indexed_jsonl(
    dataset_path: Path | str,
    config: MovementILTrainingConfig,
    observer: MovementILTrainingObserver | None,
    validation_fraction: float,
    max_train_samples: int | None,
) -> MovementILTrainingResult:
    dataset = IndexedJsonlDataset(dataset_path)
    split = dataset.split_train_validation(
        validation_fraction=validation_fraction,
        seed=config.seed,
        max_train_samples=max_train_samples,
    )
    stats = dataset.stats(split)
    print_indexed_dataset_stats(stats)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    lane_normalizer = _fit_indexed_normalizer(dataset=dataset, sample_indices=split.train_indices, lane_rows=True)
    movement_normalizer = _fit_indexed_normalizer(
        dataset=dataset,
        sample_indices=split.train_indices,
        lane_rows=False,
    )
    first_sample = dataset.sample(split.train_indices[0])
    lane_feature_dim = len(first_sample.x_lane[0])
    movement_feature_dim = len(first_sample.x_movement[0])
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensor_cache = MovementTensorCache(
        dataset=dataset,
        cache_dir=checkpoint_dir / 'tensor_cache',
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        max_cached_samples=4096,
    )
    tensor_cache.build((*split.train_indices, *split.validation_indices))
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
    try:
        for epoch in range(config.epochs):
            print(f'epoch_start={epoch + 1}/{config.epochs} train_samples={len(split.train_indices)}')
            epoch_metrics = _train_indexed_epoch(
                model=model,
                optimizer=optimizer,
                tensor_cache=tensor_cache,
                sample_indices=split.train_indices,
                epoch=epoch,
                config=config,
            )
            final_loss = epoch_metrics.loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f'epoch_end={epoch + 1}/{config.epochs} loss={epoch_metrics.loss:.6f} '
                f'regression={epoch_metrics.regression_loss:.6f} phase={epoch_metrics.phase_loss:.6f} '
                f'match={epoch_metrics.phase_match_rate:.3f}'
            )
            if writer is not None:
                writer.add_scalar('loss/total', epoch_metrics.loss, epoch + 1)
                writer.add_scalar('loss/regression', epoch_metrics.regression_loss, epoch + 1)
                writer.add_scalar('loss/phase', epoch_metrics.phase_loss, epoch + 1)
                writer.add_scalar('accuracy/phase_match', epoch_metrics.phase_match_rate, epoch + 1)
                _write_indexed_validation_losses(
                    writer=writer,
                    epoch=epoch + 1,
                    model=model,
                    tensor_cache=tensor_cache,
                    validation_indices=split.validation_indices,
                    loss_name=config.loss,
                    phase_loss_coefficient=config.phase_loss_coefficient,
                )
            if _should_save_checkpoint(epoch=epoch, config=config):
                _save_training_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    model_state=model.state_dict(),
                    best_state=best_state,
                    config=config,
                    lane_feature_dim=lane_feature_dim,
                    movement_feature_dim=movement_feature_dim,
                    lane_normalizer=lane_normalizer,
                    movement_normalizer=movement_normalizer,
                    final_loss=final_loss,
                    best_loss=best_loss,
                )
            if observer is not None:
                observer.on_epoch_completed(
                    MovementILTrainingSnapshot(
                        epoch=epoch + 1,
                        epochs=config.epochs,
                        loss=epoch_metrics.loss,
                        regression_loss=epoch_metrics.regression_loss,
                        phase_loss=epoch_metrics.phase_loss,
                        phase_match_rate=epoch_metrics.phase_match_rate,
                        best_loss=best_loss,
                        model=model,
                        config=config,
                        lane_feature_dim=lane_feature_dim,
                        movement_feature_dim=movement_feature_dim,
                        lane_normalizer=lane_normalizer,
                        movement_normalizer=movement_normalizer,
                    )
                )
    finally:
        if writer is not None:
            writer.close()
    return MovementILTrainingResult(
        checkpoint_path=checkpoint_dir / 'movement_policy_last.pt',
        final_loss=final_loss,
        epochs=config.epochs,
    )


@dataclass(frozen=True)
class IndexedEpochMetrics:
    loss: float
    regression_loss: float
    phase_loss: float
    phase_match_rate: float


def _train_indexed_epoch(
    model: MovementScorer,
    optimizer: torch.optim.Optimizer,
    tensor_cache: MovementTensorCache,
    sample_indices: Sequence[int],
    epoch: int,
    config: MovementILTrainingConfig,
) -> IndexedEpochMetrics:
    total_loss_value = 0.0
    total_regression_loss_value = 0.0
    total_phase_loss_value = 0.0
    total_phase_matches = 0
    total_phase_decisions = 0
    total_trained_samples = 0
    batch_indices = _epoch_batches(
        sample_indices=sample_indices,
        samples_per_batch=config.samples_per_batch,
        seed=config.seed,
        epoch=epoch,
    )
    started_s = time.monotonic()
    last_progress_s = started_s
    moving_losses: list[float] = []
    for batch_number, batch_sample_indices in enumerate(batch_indices, start=1):
        batch_losses: list[torch.Tensor] = []
        for sample_index in batch_sample_indices:
            sample = tensor_cache.get(sample_index)
            prediction = model(
                x_lane=sample.x_lane,
                x_movement=sample.x_movement,
                edge_index_dict=sample.edge_index_dict,
            )
            regression_loss = _loss(prediction, sample.target, config.loss)
            phase_loss = _cached_phase_classification_loss(sample=sample, movement_scores=prediction)
            sample_loss = regression_loss + config.phase_loss_coefficient * phase_loss
            batch_losses.append(sample_loss)
            sample_loss_value = float(sample_loss.detach().cpu())
            moving_losses.append(sample_loss_value)
            if len(moving_losses) > 100:
                moving_losses.pop(0)
            total_loss_value += sample_loss_value
            total_regression_loss_value += float(regression_loss.detach().cpu())
            total_phase_loss_value += float(phase_loss.detach().cpu())
            matches, decisions = _cached_phase_match_counts(sample=sample, movement_scores=prediction)
            total_phase_matches += matches
            total_phase_decisions += decisions
            total_trained_samples += 1
        loss = torch.stack(tuple(batch_losses)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_s = time.monotonic()
        if _should_print_batch_progress(
            batch_number=batch_number,
            total_batches=len(batch_indices),
            current_s=current_s,
            last_progress_s=last_progress_s,
            config=config,
        ):
            last_progress_s = current_s
            _print_batch_progress(
                epoch=epoch + 1,
                epochs=config.epochs,
                batch_number=batch_number,
                total_batches=len(batch_indices),
                samples_processed=total_trained_samples,
                total_samples=len(sample_indices),
                started_s=started_s,
                moving_average_loss=sum(moving_losses) / len(moving_losses),
                device=torch.device(config.device),
            )
    return IndexedEpochMetrics(
        loss=total_loss_value / total_trained_samples,
        regression_loss=total_regression_loss_value / total_trained_samples,
        phase_loss=total_phase_loss_value / total_trained_samples,
        phase_match_rate=total_phase_matches / max(1, total_phase_decisions),
    )


def print_indexed_dataset_stats(stats: IndexedJsonlStats) -> None:
    print(
        f'dataset path={stats.dataset_path} size={stats.file_size_bytes} bytes '
        f'samples={stats.sample_count} train={stats.train_count} validation={stats.validation_count}'
    )
    for group in stats.groups:
        print(f'dataset_group city={group.city_name} seed={group.collection_seed} samples={len(group.sample_indices)}')


def _write_indexed_validation_losses(
    writer: SummaryWriter,
    epoch: int,
    model: MovementScorer,
    tensor_cache: MovementTensorCache,
    validation_indices: Sequence[int],
    loss_name: MovementILLoss,
    phase_loss_coefficient: float,
) -> None:
    if not validation_indices:
        return
    losses_by_city: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for sample_index in validation_indices:
            sample = tensor_cache.get(sample_index)
            prediction = model(
                x_lane=sample.x_lane,
                x_movement=sample.x_movement,
                edge_index_dict=sample.edge_index_dict,
            )
            regression_loss = _loss(prediction, sample.target, loss_name)
            phase_loss = _cached_phase_classification_loss(sample=sample, movement_scores=prediction)
            loss_value = float((regression_loss + phase_loss_coefficient * phase_loss).detach().cpu())
            city_losses = losses_by_city.setdefault(sample.city_name, [])
            city_losses.append(loss_value)
    model.train()
    all_losses = tuple(loss for city_losses in losses_by_city.values() for loss in city_losses)
    writer.add_scalar('validation/loss', sum(all_losses) / len(all_losses), epoch)
    for city_name, city_losses in losses_by_city.items():
        writer.add_scalar(f'validation/{city_name}/loss', sum(city_losses) / len(city_losses), epoch)


def _index_jsonl_records(dataset_path: Path) -> tuple[IndexedJsonlRecord, ...]:
    records: list[IndexedJsonlRecord] = []
    byte_offset = 0
    with dataset_path.open('rb') as handle:
        for sample_index, raw_line in enumerate(handle):
            byte_length = len(raw_line)
            if raw_line.strip():
                sample = load_jsonl_sample_line(raw_line.decode('utf-8'))
                metadata = IndexedSampleMetadata.model_validate(sample.metadata)
                records.append(
                    IndexedJsonlRecord(
                        sample_index=len(records),
                        byte_offset=byte_offset,
                        byte_length=byte_length,
                        city_name=metadata.city_name,
                        collection_seed=metadata.collection_seed,
                    )
                )
            byte_offset += byte_length
    return tuple(records)


def _groups_from_records(records: Sequence[IndexedJsonlRecord]) -> tuple[CitySeedIndexedGroup, ...]:
    group_keys: list[tuple[str, int]] = []
    for record in records:
        group_key = (record.city_name, record.collection_seed)
        if group_key not in group_keys:
            group_keys.append(group_key)
    return tuple(
        CitySeedIndexedGroup(
            city_name=city_name,
            collection_seed=collection_seed,
            sample_indices=tuple(
                record.sample_index
                for record in records
                if record.city_name == city_name and record.collection_seed == collection_seed
            ),
        )
        for city_name, collection_seed in group_keys
    )


def _ordered_city_names(groups: Sequence[CitySeedIndexedGroup]) -> tuple[str, ...]:
    city_names: list[str] = []
    for group in groups:
        if group.city_name not in city_names:
            city_names.append(group.city_name)
    return tuple(city_names)


def _fit_indexed_normalizer(
    dataset: IndexedJsonlDataset,
    sample_indices: Iterable[int],
    lane_rows: bool,
) -> RunningNormalizer:
    normalizer = RunningNormalizer()
    for sample_index in sample_indices:
        sample = dataset.sample(sample_index)
        normalizer.update_rows(sample.x_lane if lane_rows else sample.x_movement)
    normalizer.freeze()
    return normalizer


def _cached_sample_from_sample(
    sample: MovementDatasetSample,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> CachedMovementTensorSample:
    x_lane, x_movement, target = tensors_from_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
    city_name = IndexedSampleMetadata.model_validate(sample.metadata).city_name
    return CachedMovementTensorSample(
        x_lane=x_lane,
        x_movement=x_movement,
        target=target,
        edge_index_dict=edge_tensors_from_sample(sample, device=device),
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=city_name,
    )


def _move_cached_sample(sample: CachedMovementTensorSample, device: torch.device) -> CachedMovementTensorSample:
    return CachedMovementTensorSample(
        x_lane=sample.x_lane.to(device),
        x_movement=sample.x_movement.to(device),
        target=sample.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in sample.edge_index_dict.items()},
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=sample.city_name,
    )


def _epoch_batches(
    sample_indices: Sequence[int],
    samples_per_batch: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, ...], ...]:
    shuffled_indices = list(sample_indices)
    Random(seed + epoch).shuffle(shuffled_indices)
    return tuple(
        tuple(shuffled_indices[batch_start : batch_start + samples_per_batch])
        for batch_start in range(0, len(shuffled_indices), samples_per_batch)
    )


def _cached_phase_classification_loss(
    sample: CachedMovementTensorSample,
    movement_scores: torch.Tensor,
) -> torch.Tensor:
    return _phase_classification_loss(
        sample=cast(MovementDatasetSample, sample),
        movement_scores=movement_scores,
    )


def _cached_phase_match_counts(
    sample: CachedMovementTensorSample,
    movement_scores: torch.Tensor,
) -> tuple[int, int]:
    return _phase_match_counts(
        sample=cast(MovementDatasetSample, sample),
        movement_scores=movement_scores,
    )


def _should_save_checkpoint(epoch: int, config: MovementILTrainingConfig) -> bool:
    if config.checkpoint_every_epochs <= 0:
        return epoch == config.epochs - 1
    return (epoch + 1) % config.checkpoint_every_epochs == 0 or epoch == config.epochs - 1


def _should_print_batch_progress(
    batch_number: int,
    total_batches: int,
    current_s: float,
    last_progress_s: float,
    config: MovementILTrainingConfig,
) -> bool:
    if batch_number == total_batches:
        return True
    if config.progress_every_batches > 0 and batch_number % config.progress_every_batches == 0:
        return True
    return config.progress_every_seconds > 0 and current_s - last_progress_s >= config.progress_every_seconds


def _print_batch_progress(
    epoch: int,
    epochs: int,
    batch_number: int,
    total_batches: int,
    samples_processed: int,
    total_samples: int,
    started_s: float,
    moving_average_loss: float,
    device: torch.device,
) -> None:
    elapsed_s = time.monotonic() - started_s
    percent = samples_processed / total_samples * 100.0
    samples_per_second = samples_processed / max(elapsed_s, 1e-9)
    eta_s = (total_samples - samples_processed) / max(samples_per_second, 1e-9)
    print(
        f'epoch={epoch}/{epochs} batch={batch_number}/{total_batches} '
        f'samples={samples_processed}/{total_samples} percent={percent:.1f} '
        f'elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)} '
        f'loss_ma={moving_average_loss:.6f} {_memory_summary(device)}'
    )


def _memory_summary(device: torch.device) -> str:
    cpu_memory = ''
    if device.type == 'cuda' and torch.cuda.is_available():
        allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
        reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
        return f'gpu_alloc_gib={allocated_gib:.2f} gpu_reserved_gib={reserved_gib:.2f}'
    return cpu_memory


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = total_seconds % 3600 // 60
    remaining_seconds = total_seconds % 60
    if hours > 0:
        return f'{hours}h{minutes:02d}m{remaining_seconds:02d}s'
    if minutes > 0:
        return f'{minutes}m{remaining_seconds:02d}s'
    return f'{remaining_seconds}s'
