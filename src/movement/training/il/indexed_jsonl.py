"""Indexed JSONL training path for large imitation-learning datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from random import Random
import time
from typing import cast
from typing import TypedDict

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.movement.dataset import MovementDatasetSample, StoredPhaseIncidence, load_jsonl_sample_line
from src.movement.models.bipartite_gnn import MovementScorer
from src.movement.normalization import RunningNormalizer
from src.movement.training.il import _save_training_checkpoints
from src.movement.training.il.tensors import edge_tensors_from_sample
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


class IndexedSampleEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: IndexedSampleMetadata


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
class CachedPhaseTensor:
    incidence_matrix: torch.Tensor
    movement_ids: torch.Tensor
    target_phase: int


@dataclass(frozen=True)
class CachedPhaseLogitGroupBatch:
    incidence_matrices: torch.Tensor
    movement_ids: torch.Tensor
    targets: torch.Tensor
    sample_indices: torch.Tensor


@dataclass(frozen=True)
class CachedMovementTensorSample:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    phase_incidences: dict[str, StoredPhaseIncidence]
    teacher_selected_phase_by_tls: dict[str, int]
    city_name: str
    phase_tensors: dict[str, CachedPhaseTensor] | None = None


@dataclass(frozen=True)
class CachedMovementTensorBatch:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    movement_sample_indices: torch.Tensor
    lane_counts: tuple[int, ...]
    movement_counts: tuple[int, ...]
    phase_logit_groups: tuple[CachedPhaseLogitGroupBatch, ...]
    city_names: tuple[str, ...]


class CachedPhaseLogitGroupPayload(TypedDict):
    incidence_matrices: torch.Tensor
    movement_ids: torch.Tensor
    targets: torch.Tensor
    sample_indices: torch.Tensor


class CachedMovementTensorBatchPayload(TypedDict):
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    movement_sample_indices: torch.Tensor
    lane_counts: tuple[int, ...]
    movement_counts: tuple[int, ...]
    phase_logit_groups: tuple[CachedPhaseLogitGroupPayload, ...]
    city_names: tuple[str, ...]


@dataclass(frozen=True)
class JsonlIndexState:
    dataset_path: str
    file_size_bytes: int
    modified_time_ns: int
    records: tuple[IndexedJsonlRecord, ...]


@dataclass(frozen=True)
class RawTensorPreparationState:
    dataset_path: str
    file_size_bytes: int
    train_indices: tuple[int, ...]
    cache_indices: tuple[int, ...]
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer


@dataclass(frozen=True)
class RawTensorCacheChunk:
    dataset_path: str
    cache_dir: str
    records: tuple[IndexedJsonlRecord, ...]
    train_indices: tuple[int, ...]


@dataclass(frozen=True)
class RawTensorCacheChunkResult:
    processed_count: int
    lane_normalizer: RunningNormalizer
    movement_normalizer: RunningNormalizer


class IndexedJsonlDataset:
    def __init__(self, dataset_path: Path | str, index_cache_path: Path | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.records = _index_jsonl_records(self.dataset_path, index_cache_path=index_cache_path)
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
        cache_dir: Path,
        lane_normalizer: RunningNormalizer,
        movement_normalizer: RunningNormalizer,
        device: torch.device,
    ) -> None:
        self.cache_dir = cache_dir
        self.lane_normalizer = lane_normalizer
        self.movement_normalizer = movement_normalizer
        self.device = device
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, sample_index: int) -> CachedMovementTensorSample:
        return _move_cached_sample(self.get_cpu(sample_index), self.device)

    def get_cpu(self, sample_index: int) -> CachedMovementTensorSample:
        raw_sample = cast(
            CachedMovementTensorSample,
            _load_torch_cpu(self._cache_path(sample_index)),
        )
        loaded = _normalised_cached_sample(
            sample=raw_sample,
            lane_normalizer=self.lane_normalizer,
            movement_normalizer=self.movement_normalizer,
            device=torch.device('cpu'),
        )
        return loaded

    def _cache_path(self, sample_index: int) -> Path:
        return self.cache_dir / f'sample_{sample_index:08d}.pt'


class IndexedBatchTensorCacheDataset(Dataset[CachedMovementTensorBatch]):
    def __init__(
        self,
        raw_cache_dir: Path,
        batch_cache_dir: Path,
        batch_indices: Sequence[Sequence[int]],
        lane_normalizer: RunningNormalizer,
        movement_normalizer: RunningNormalizer,
    ) -> None:
        self.raw_cache_dir = raw_cache_dir
        self.batch_cache_dir = batch_cache_dir
        self.batch_indices = tuple(tuple(batch) for batch in batch_indices)
        self.lane_normalizer = lane_normalizer
        self.movement_normalizer = movement_normalizer
        self.batch_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.batch_indices)

    def status(self) -> BatchTensorCacheStatus:
        ready_count = sum(1 for sample_indices in self.batch_indices if self._batch_path(sample_indices).exists())
        return BatchTensorCacheStatus(
            batch_count=len(self.batch_indices),
            ready_count=ready_count,
            cache_dir=self.batch_cache_dir,
        )

    def __getitem__(self, index: int) -> CachedMovementTensorBatch:
        sample_indices = self.batch_indices[index]
        batch_path = self._batch_path(sample_indices)
        if batch_path.exists():
            return _batch_from_payload(cast(CachedMovementTensorBatchPayload, _load_torch_cpu(batch_path)))
        batch = _collate_cached_samples(tuple(self._sample(sample_index) for sample_index in sample_indices))
        torch.save(_batch_payload(batch), batch_path)
        return batch

    def _sample(self, sample_index: int) -> CachedMovementTensorSample:
        raw_sample = cast(
            CachedMovementTensorSample,
            _load_torch_cpu(_raw_cache_path(cache_dir=self.raw_cache_dir, sample_index=sample_index)),
        )
        return _normalised_cached_sample(
            sample=raw_sample,
            lane_normalizer=self.lane_normalizer,
            movement_normalizer=self.movement_normalizer,
            device=torch.device('cpu'),
        )

    def _batch_path(self, sample_indices: Sequence[int]) -> Path:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b'indexed-il-batch-tensor-payload-v1')
        digest.update(_normalizer_digest_bytes(self.lane_normalizer))
        digest.update(_normalizer_digest_bytes(self.movement_normalizer))
        for sample_index in sample_indices:
            digest.update(int(sample_index).to_bytes(8, byteorder='little', signed=False))
        return self.batch_cache_dir / f'batch_{digest.hexdigest()}.pt'


@dataclass(frozen=True)
class BatchTensorCacheStatus:
    batch_count: int
    ready_count: int
    cache_dir: Path


def _collate_cached_samples(samples: Sequence[CachedMovementTensorSample]) -> CachedMovementTensorBatch:
    return _cached_sample_batch(samples=samples)


def _load_torch_cpu(path: Path) -> object:
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def train_movement_il_from_indexed_jsonl(
    dataset_path: Path | str,
    config: MovementILTrainingConfig,
    observer: MovementILTrainingObserver | None,
    validation_fraction: float,
    max_train_samples: int | None,
) -> MovementILTrainingResult:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = checkpoint_dir / 'raw_tensor_cache'
    dataset = IndexedJsonlDataset(dataset_path, index_cache_path=raw_cache_dir / 'jsonl_index.pt')
    split = dataset.split_train_validation(
        validation_fraction=validation_fraction,
        seed=config.seed,
        max_train_samples=max_train_samples,
    )
    stats = dataset.stats(split)
    print_indexed_dataset_stats(stats)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    lane_normalizer, movement_normalizer = _prepare_raw_tensor_cache(
        dataset=dataset,
        cache_dir=raw_cache_dir,
        train_indices=split.train_indices,
        cache_indices=(*split.train_indices, *split.validation_indices),
        cache_workers=config.cache_workers,
    )
    first_sample = dataset.sample(split.train_indices[0])
    lane_feature_dim = len(first_sample.x_lane[0])
    movement_feature_dim = len(first_sample.x_movement[0])
    tensor_cache = MovementTensorCache(
        cache_dir=raw_cache_dir,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
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
            validation_started_s = time.monotonic()
            if writer is not None and _should_validate(epoch=epoch, config=config):
                _write_indexed_validation_losses(
                    writer=writer,
                    epoch=epoch + 1,
                    model=model,
                    tensor_cache=tensor_cache,
                    validation_indices=split.validation_indices,
                    loss_name=config.loss,
                    phase_loss_coefficient=config.phase_loss_coefficient,
                )
            validation_elapsed_s = time.monotonic() - validation_started_s
            checkpoint_started_s = time.monotonic()
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
            checkpoint_elapsed_s = time.monotonic() - checkpoint_started_s
            observer_started_s = time.monotonic()
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
            observer_elapsed_s = time.monotonic() - observer_started_s
            if validation_elapsed_s > 0.05 or checkpoint_elapsed_s > 0.05 or observer_elapsed_s > 0.05:
                print(
                    f'epoch_post={epoch + 1}/{config.epochs} '
                    f't_validation={validation_elapsed_s:.1f}s '
                    f't_checkpoint={checkpoint_elapsed_s:.1f}s '
                    f't_observer={observer_elapsed_s:.1f}s',
                    flush=True,
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


@dataclass(frozen=True)
class IndexedBatchLossMetrics:
    loss: torch.Tensor
    regression_loss: torch.Tensor
    phase_loss: torch.Tensor
    phase_matches: int
    phase_decisions: int
    sample_count: int


@dataclass(frozen=True)
class TimedIndexedBatchLossMetrics:
    metrics: IndexedBatchLossMetrics
    forward_elapsed_s: float
    loss_elapsed_s: float


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
    load_elapsed_s = 0.0
    forward_elapsed_s = 0.0
    loss_elapsed_s = 0.0
    backward_elapsed_s = 0.0
    train_loader = _train_data_loader(
        tensor_cache=tensor_cache,
        batch_indices=batch_indices,
        config=config,
    )
    train_loader_iterator = iter(train_loader)
    for batch_number in range(1, len(batch_indices) + 1):
        load_started_s = time.monotonic()
        batch = _move_cached_batch(cpu_batch=next(train_loader_iterator), device=torch.device(config.device))
        load_elapsed_s += time.monotonic() - load_started_s
        timed_batch_metrics = _timed_indexed_cached_batch_loss_metrics(
            model=model,
            batch=batch,
            loss_name=config.loss,
            phase_loss_coefficient=config.phase_loss_coefficient,
        )
        batch_metrics = timed_batch_metrics.metrics
        forward_elapsed_s += timed_batch_metrics.forward_elapsed_s
        loss_elapsed_s += timed_batch_metrics.loss_elapsed_s
        batch_sample_count = len(batch.city_names)
        batch_loss_value = float(batch_metrics.loss.detach().cpu())
        moving_losses.append(batch_loss_value)
        if len(moving_losses) > 100:
            moving_losses.pop(0)
        total_loss_value += batch_loss_value * batch_sample_count
        total_regression_loss_value += float(batch_metrics.regression_loss.detach().cpu()) * batch_sample_count
        total_phase_loss_value += float(batch_metrics.phase_loss.detach().cpu()) * batch_sample_count
        total_phase_matches += batch_metrics.phase_matches
        total_phase_decisions += batch_metrics.phase_decisions
        total_trained_samples += batch_sample_count
        backward_started_s = time.monotonic()
        optimizer.zero_grad()
        batch_metrics.loss.backward()
        optimizer.step()
        backward_elapsed_s += time.monotonic() - backward_started_s
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
                load_elapsed_s=load_elapsed_s,
                forward_elapsed_s=forward_elapsed_s,
                loss_elapsed_s=loss_elapsed_s,
                backward_elapsed_s=backward_elapsed_s,
            )
    return IndexedEpochMetrics(
        loss=total_loss_value / total_trained_samples,
        regression_loss=total_regression_loss_value / total_trained_samples,
        phase_loss=total_phase_loss_value / total_trained_samples,
        phase_match_rate=total_phase_matches / max(1, total_phase_decisions),
    )


def _train_worker_count(train_workers: int, samples_per_batch: int) -> int:
    if train_workers == -1:
        cpu_count = os.cpu_count()
        resolved_workers = cpu_count if cpu_count is not None else 1
    elif train_workers <= 0:
        raise ValueError('train_workers must be positive or -1.')
    else:
        resolved_workers = train_workers
    return min(resolved_workers, samples_per_batch)


def _prefetch_factor(prefetch_batches: int, train_workers: int) -> int:
    if prefetch_batches <= 0:
        raise ValueError('prefetch_batches must be positive.')
    return max(1, (prefetch_batches + train_workers - 1) // train_workers)


def _train_data_loader(
    tensor_cache: MovementTensorCache,
    batch_indices: Sequence[Sequence[int]],
    config: MovementILTrainingConfig,
) -> DataLoader[CachedMovementTensorBatch]:
    train_workers = _train_worker_count(config.train_workers, config.samples_per_batch)
    dataset = _batch_train_dataset(tensor_cache=tensor_cache, batch_indices=batch_indices)
    if train_workers == 1:
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=train_workers,
        prefetch_factor=_prefetch_factor(
            prefetch_batches=config.prefetch_batches,
            train_workers=train_workers,
        ),
        persistent_workers=True,
    )


def _batch_train_dataset(
    tensor_cache: MovementTensorCache,
    batch_indices: Sequence[Sequence[int]],
) -> Dataset[CachedMovementTensorBatch]:
    dataset = IndexedBatchTensorCacheDataset(
        raw_cache_dir=tensor_cache.cache_dir,
        batch_cache_dir=tensor_cache.cache_dir / 'batch_tensor_cache',
        batch_indices=batch_indices,
        lane_normalizer=tensor_cache.lane_normalizer,
        movement_normalizer=tensor_cache.movement_normalizer,
    )
    _print_batch_tensor_cache_status(dataset.status())
    return dataset


def _print_batch_tensor_cache_status(status: BatchTensorCacheStatus) -> None:
    missing_count = status.batch_count - status.ready_count
    if missing_count == 0:
        print(
            f'batch_tensor_cache_ready batches={status.batch_count} cache_dir={status.cache_dir}',
            flush=True,
        )
        return
    print(
        f'batch_tensor_cache_pending batches={status.batch_count} ready={status.ready_count} '
        f'missing={missing_count} cache_dir={status.cache_dir}',
        flush=True,
    )


def _timed_indexed_cached_batch_loss_metrics(
    model: MovementScorer,
    batch: CachedMovementTensorBatch,
    loss_name: MovementILLoss,
    phase_loss_coefficient: float,
) -> TimedIndexedBatchLossMetrics:
    forward_started_s = time.monotonic()
    flat_predictions = model(
        x_lane=batch.x_lane,
        x_movement=batch.x_movement,
        edge_index_dict=batch.edge_index_dict,
    )
    forward_elapsed_s = time.monotonic() - forward_started_s
    loss_started_s = time.monotonic()
    metrics = _indexed_cached_batch_loss_metrics(
        batch=batch,
        flat_predictions=flat_predictions,
        loss_name=loss_name,
        phase_loss_coefficient=phase_loss_coefficient,
    )
    return TimedIndexedBatchLossMetrics(
        metrics=metrics,
        forward_elapsed_s=forward_elapsed_s,
        loss_elapsed_s=time.monotonic() - loss_started_s,
    )


def _indexed_cached_batch_loss_metrics(
    batch: CachedMovementTensorBatch,
    flat_predictions: torch.Tensor,
    loss_name: MovementILLoss,
    phase_loss_coefficient: float,
) -> IndexedBatchLossMetrics:
    regression_loss = _packed_sample_weighted_regression_loss(
        prediction=flat_predictions,
        target=batch.target,
        movement_sample_indices=batch.movement_sample_indices,
        sample_count=len(batch.city_names),
        loss_name=loss_name,
    )
    phase_loss = _packed_sample_weighted_phase_loss(
        flat_predictions=flat_predictions,
        phase_logit_groups=batch.phase_logit_groups,
        sample_count=len(batch.city_names),
    )
    with torch.no_grad():
        phase_matches, phase_decisions = _packed_phase_match_counts(
            flat_predictions=flat_predictions,
            phase_logit_groups=batch.phase_logit_groups,
        )
    return IndexedBatchLossMetrics(
        loss=regression_loss + phase_loss_coefficient * phase_loss,
        regression_loss=regression_loss,
        phase_loss=phase_loss,
        phase_matches=phase_matches,
        phase_decisions=phase_decisions,
        sample_count=len(batch.city_names),
    )


def _packed_sample_weighted_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    movement_sample_indices: torch.Tensor,
    sample_count: int,
    loss_name: MovementILLoss,
) -> torch.Tensor:
    elementwise_loss = _elementwise_regression_loss(prediction=prediction, target=target, loss_name=loss_name)
    sample_loss_sums = prediction.new_zeros((sample_count,))
    sample_loss_counts = prediction.new_zeros((sample_count,))
    sample_loss_sums.index_add_(0, movement_sample_indices, elementwise_loss)
    sample_loss_counts.index_add_(0, movement_sample_indices, torch.ones_like(elementwise_loss))
    return (sample_loss_sums / sample_loss_counts.clamp_min(1.0)).mean()


def _packed_sample_weighted_phase_loss(
    flat_predictions: torch.Tensor,
    phase_logit_groups: Sequence[CachedPhaseLogitGroupBatch],
    sample_count: int,
) -> torch.Tensor:
    sample_loss_sums = flat_predictions.new_zeros((sample_count,))
    sample_loss_counts = flat_predictions.new_zeros((sample_count,))
    for group in phase_logit_groups:
        logits = _packed_phase_logits(flat_predictions=flat_predictions, group=group)
        losses = F.cross_entropy(logits, group.targets, reduction='none')
        sample_loss_sums.index_add_(0, group.sample_indices, losses)
        sample_loss_counts.index_add_(0, group.sample_indices, torch.ones_like(losses))
    return (sample_loss_sums / sample_loss_counts.clamp_min(1.0)).mean()


def _packed_phase_match_counts(
    flat_predictions: torch.Tensor,
    phase_logit_groups: Sequence[CachedPhaseLogitGroupBatch],
) -> tuple[int, int]:
    matches = 0
    decisions = 0
    for group in phase_logit_groups:
        logits = _packed_phase_logits(flat_predictions=flat_predictions, group=group)
        matches += int((logits.argmax(dim=1) == group.targets).sum().detach().cpu())
        decisions += group.targets.numel()
    return matches, decisions


def _packed_phase_logits(
    flat_predictions: torch.Tensor,
    group: CachedPhaseLogitGroupBatch,
) -> torch.Tensor:
    movement_scores = flat_predictions[group.movement_ids]
    return torch.bmm(group.incidence_matrices, movement_scores.unsqueeze(-1)).squeeze(-1)


def _elementwise_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: MovementILLoss,
) -> torch.Tensor:
    match loss_name:
        case MovementILLoss.HUBER:
            return F.smooth_l1_loss(prediction, target, reduction='none')
        case MovementILLoss.MEAN_SQUARED_ERROR:
            return F.mse_loss(prediction, target, reduction='none')


def _batched_edge_index_dict(
    samples: Sequence[CachedMovementTensorSample],
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
            _offset_edge_index(sample.edge_index_dict['input_lane_to_movement'], lane_offset, movement_offset)
        )
        edge_parts['output_lane_to_movement'].append(
            _offset_edge_index(sample.edge_index_dict['output_lane_to_movement'], lane_offset, movement_offset)
        )
        edge_parts['movement_to_input_lane'].append(
            _offset_edge_index(sample.edge_index_dict['movement_to_input_lane'], movement_offset, lane_offset)
        )
        edge_parts['movement_to_output_lane'].append(
            _offset_edge_index(sample.edge_index_dict['movement_to_output_lane'], movement_offset, lane_offset)
        )
        edge_parts['lane_to_lane'].append(
            _offset_edge_index(sample.edge_index_dict['lane_to_lane'], lane_offset, lane_offset)
        )
        lane_to_lane_weight_parts.append(sample.edge_index_dict['lane_to_lane_weight'])
        lane_offset += sample.x_lane.shape[0]
        movement_offset += sample.x_movement.shape[0]
    return {
        'input_lane_to_movement': _cat_edge_parts(edge_parts['input_lane_to_movement']),
        'output_lane_to_movement': _cat_edge_parts(edge_parts['output_lane_to_movement']),
        'movement_to_input_lane': _cat_edge_parts(edge_parts['movement_to_input_lane']),
        'movement_to_output_lane': _cat_edge_parts(edge_parts['movement_to_output_lane']),
        'lane_to_lane': _cat_edge_parts(edge_parts['lane_to_lane']),
        'lane_to_lane_weight': torch.cat(tuple(lane_to_lane_weight_parts), dim=0),
    }


def _offset_edge_index(edge_index: torch.Tensor, source_offset: int, target_offset: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    offsets = torch.tensor(
        ((source_offset,), (target_offset,)),
        dtype=edge_index.dtype,
        device=edge_index.device,
    )
    return edge_index + offsets


def _cat_edge_parts(edge_parts: Sequence[torch.Tensor]) -> torch.Tensor:
    non_empty_parts = tuple(edge_part for edge_part in edge_parts if edge_part.numel() > 0)
    if not non_empty_parts:
        device = edge_parts[0].device
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(non_empty_parts, dim=1)


def print_indexed_dataset_stats(stats: IndexedJsonlStats) -> None:
    print(
        f'dataset path={stats.dataset_path} size={stats.file_size_bytes} bytes '
        f'samples={stats.sample_count} train={stats.train_count} validation={stats.validation_count}'
    )
    print(f'dataset_groups count={len(stats.groups)}')
    city_names = tuple(dict.fromkeys(group.city_name for group in stats.groups))
    for city_name in city_names:
        city_groups = tuple(group for group in stats.groups if group.city_name == city_name)
        city_samples = sum(len(group.sample_indices) for group in city_groups)
        city_seeds = tuple(group.collection_seed for group in city_groups)
        print(
            f'dataset_city city={city_name} groups={len(city_groups)} samples={city_samples} '
            f'seed_min={min(city_seeds)} seed_max={max(city_seeds)}'
        )


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
            batch = _cached_sample_batch(samples=(sample,))
            prediction = model(
                x_lane=batch.x_lane,
                x_movement=batch.x_movement,
                edge_index_dict=batch.edge_index_dict,
            )
            metrics = _indexed_cached_batch_loss_metrics(
                batch=batch,
                flat_predictions=prediction,
                loss_name=loss_name,
                phase_loss_coefficient=phase_loss_coefficient,
            )
            loss_value = float(metrics.loss.detach().cpu())
            city_losses = losses_by_city.setdefault(sample.city_name, [])
            city_losses.append(loss_value)
    model.train()
    all_losses = tuple(loss for city_losses in losses_by_city.values() for loss in city_losses)
    writer.add_scalar('validation/loss', sum(all_losses) / len(all_losses), epoch)
    for city_name, city_losses in losses_by_city.items():
        writer.add_scalar(f'validation/{city_name}/loss', sum(city_losses) / len(city_losses), epoch)


def _index_jsonl_records(dataset_path: Path, index_cache_path: Path | None) -> tuple[IndexedJsonlRecord, ...]:
    cached_state = _load_jsonl_index_state(dataset_path=dataset_path, index_cache_path=index_cache_path)
    if cached_state is not None:
        print(
            f'jsonl_index_ready path={dataset_path} records={len(cached_state.records)} cache={index_cache_path}',
            flush=True,
        )
        return cached_state.records
    records: list[IndexedJsonlRecord] = []
    byte_offset = 0
    file_size_bytes = dataset_path.stat().st_size
    started_s = time.monotonic()
    last_progress_s = started_s
    print(f'jsonl_index_start path={dataset_path} size={file_size_bytes} bytes', flush=True)
    with dataset_path.open('rb') as handle:
        for sample_index, raw_line in enumerate(handle):
            byte_length = len(raw_line)
            if raw_line.strip():
                envelope = IndexedSampleEnvelope.model_validate_json(raw_line)
                records.append(
                    IndexedJsonlRecord(
                        sample_index=len(records),
                        byte_offset=byte_offset,
                        byte_length=byte_length,
                        city_name=envelope.metadata.city_name,
                        collection_seed=envelope.metadata.collection_seed,
                    )
                )
            byte_offset += byte_length
            current_s = time.monotonic()
            if current_s - last_progress_s >= 60.0:
                last_progress_s = current_s
                print(
                    f'jsonl_index bytes={byte_offset}/{file_size_bytes} '
                    f'percent={_percent(byte_offset, file_size_bytes):.1f} records={len(records)} '
                    f'elapsed={_format_duration(current_s - started_s)}',
                    flush=True,
                )
    print(
        f'jsonl_index_done bytes={byte_offset}/{file_size_bytes} records={len(records)} '
        f'elapsed={_format_duration(time.monotonic() - started_s)}',
        flush=True,
    )
    indexed_records = tuple(records)
    if index_cache_path is not None:
        index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            JsonlIndexState(
                dataset_path=str(dataset_path),
                file_size_bytes=file_size_bytes,
                modified_time_ns=dataset_path.stat().st_mtime_ns,
                records=indexed_records,
            ),
            index_cache_path,
        )
    return indexed_records


def _load_jsonl_index_state(dataset_path: Path, index_cache_path: Path | None) -> JsonlIndexState | None:
    if index_cache_path is None or not index_cache_path.exists():
        return None
    state = cast(JsonlIndexState, _load_torch_cpu(index_cache_path))
    file_stat = dataset_path.stat()
    if state.dataset_path != str(dataset_path):
        return None
    if state.file_size_bytes != file_stat.st_size:
        return None
    return state


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


def _prepare_raw_tensor_cache(
    dataset: IndexedJsonlDataset,
    cache_dir: Path,
    train_indices: Sequence[int],
    cache_indices: Sequence[int],
    cache_workers: int,
) -> tuple[RunningNormalizer, RunningNormalizer]:
    preparation_path = cache_dir / 'preparation_state.pt'
    cached_state = _load_preparation_state(
        preparation_path=preparation_path,
        dataset=dataset,
        train_indices=train_indices,
        cache_indices=cache_indices,
    )
    if cached_state is not None:
        print(f'raw_tensor_cache_ready samples={len(cache_indices)} cache_dir={cache_dir}', flush=True)
        return cached_state.lane_normalizer, cached_state.movement_normalizer
    lane_normalizer, movement_normalizer = _build_raw_tensor_cache_and_fit_normalizers(
        dataset=dataset,
        cache_dir=cache_dir,
        train_indices=train_indices,
        cache_indices=cache_indices,
        cache_workers=cache_workers,
    )
    torch.save(
        RawTensorPreparationState(
            dataset_path=str(dataset.dataset_path),
            file_size_bytes=dataset.dataset_path.stat().st_size,
            train_indices=tuple(train_indices),
            cache_indices=tuple(cache_indices),
            lane_normalizer=lane_normalizer,
            movement_normalizer=movement_normalizer,
        ),
        preparation_path,
    )
    return lane_normalizer, movement_normalizer


def _load_preparation_state(
    preparation_path: Path,
    dataset: IndexedJsonlDataset,
    train_indices: Sequence[int],
    cache_indices: Sequence[int],
) -> RawTensorPreparationState | None:
    if not preparation_path.exists():
        return None
    state = cast(
        RawTensorPreparationState,
        _load_torch_cpu(preparation_path),
    )
    if state.dataset_path != str(dataset.dataset_path):
        return None
    if state.file_size_bytes != dataset.dataset_path.stat().st_size:
        return None
    if state.train_indices != tuple(train_indices):
        return None
    if state.cache_indices != tuple(cache_indices):
        return None
    if not all(
        _raw_cache_path(cache_dir=preparation_path.parent, sample_index=index).exists() for index in cache_indices
    ):
        return None
    return state


def _build_raw_tensor_cache_and_fit_normalizers(
    dataset: IndexedJsonlDataset,
    cache_dir: Path,
    train_indices: Sequence[int],
    cache_indices: Sequence[int],
    cache_workers: int,
) -> tuple[RunningNormalizer, RunningNormalizer]:
    if cache_workers == -1:
        cpu_count = os.cpu_count()
        resolved_cache_workers = cpu_count if cpu_count is not None else 1
    elif cache_workers <= 0:
        raise ValueError('cache_workers must be positive or -1.')
    else:
        resolved_cache_workers = cache_workers
    cache_dir.mkdir(parents=True, exist_ok=True)
    lane_normalizer = RunningNormalizer()
    movement_normalizer = RunningNormalizer()
    started_s = time.monotonic()
    last_progress_s = started_s
    worker_count = min(resolved_cache_workers, len(cache_indices))
    print(
        f'raw_tensor_cache_start samples={len(cache_indices)} train_samples={len(train_indices)} '
        f'workers={worker_count} cache_dir={cache_dir}',
        flush=True,
    )
    records_by_index = {record.sample_index: record for record in dataset.records}
    records = tuple(records_by_index[sample_index] for sample_index in cache_indices)
    chunks = _raw_cache_chunks(
        dataset_path=dataset.dataset_path,
        cache_dir=cache_dir,
        records=records,
        train_indices=tuple(train_indices),
        worker_count=worker_count,
    )
    processed_count = 0
    for result in _raw_cache_chunk_results(chunks=chunks, worker_count=worker_count):
        processed_count += result.processed_count
        _merge_normalizer(lane_normalizer, result.lane_normalizer)
        _merge_normalizer(movement_normalizer, result.movement_normalizer)
        current_s = time.monotonic()
        if processed_count == len(cache_indices) or current_s - last_progress_s >= 30.0:
            last_progress_s = current_s
            percent = _percent(processed_count, len(cache_indices))
            print(
                f'raw_tensor_cache samples={processed_count}/{len(cache_indices)} percent={percent:.1f} '
                f'elapsed={_format_duration(current_s - started_s)}',
                flush=True,
            )
    lane_normalizer.freeze()
    movement_normalizer.freeze()
    return lane_normalizer, movement_normalizer


def _raw_cache_chunks(
    dataset_path: Path,
    cache_dir: Path,
    records: Sequence[IndexedJsonlRecord],
    train_indices: tuple[int, ...],
    worker_count: int,
) -> tuple[RawTensorCacheChunk, ...]:
    chunk_size = max(1, min(512, (len(records) + worker_count * 8 - 1) // (worker_count * 8)))
    train_index_set = set(train_indices)
    return tuple(
        _raw_cache_chunk(
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            records=tuple(records[start : start + chunk_size]),
            train_index_set=train_index_set,
        )
        for start in range(0, len(records), chunk_size)
    )


def _raw_cache_chunk(
    dataset_path: Path,
    cache_dir: Path,
    records: Sequence[IndexedJsonlRecord],
    train_index_set: set[int],
) -> RawTensorCacheChunk:
    return RawTensorCacheChunk(
        dataset_path=str(dataset_path),
        cache_dir=str(cache_dir),
        records=tuple(records),
        train_indices=tuple(record.sample_index for record in records if record.sample_index in train_index_set),
    )


def _raw_cache_chunk_results(
    chunks: Sequence[RawTensorCacheChunk],
    worker_count: int,
) -> Iterator[RawTensorCacheChunkResult]:
    if worker_count == 1:
        for chunk in chunks:
            yield _build_raw_tensor_cache_chunk(chunk)
        return
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = tuple(executor.submit(_build_raw_tensor_cache_chunk, chunk) for chunk in chunks)
        for future in as_completed(futures):
            yield future.result()


def _build_raw_tensor_cache_chunk(chunk: RawTensorCacheChunk) -> RawTensorCacheChunkResult:
    train_index_set = set(chunk.train_indices)
    lane_normalizer = RunningNormalizer()
    movement_normalizer = RunningNormalizer()
    dataset_path = Path(chunk.dataset_path)
    cache_dir = Path(chunk.cache_dir)
    with dataset_path.open('rb') as handle:
        for record in chunk.records:
            cache_path = _raw_cache_path(cache_dir=cache_dir, sample_index=record.sample_index)
            if cache_path.exists():
                cached_sample = cast(
                    CachedMovementTensorSample,
                    _load_torch_cpu(cache_path),
                )
            else:
                handle.seek(record.byte_offset)
                line = handle.read(record.byte_length).decode('utf-8')
                cached_sample = _cached_sample_from_sample(
                    sample=load_jsonl_sample_line(line),
                    device=torch.device('cpu'),
                )
                torch.save(cached_sample, cache_path)
            if record.sample_index in train_index_set:
                lane_normalizer.update_rows(_tensor_rows(cached_sample.x_lane))
                movement_normalizer.update_rows(_tensor_rows(cached_sample.x_movement))
    return RawTensorCacheChunkResult(
        processed_count=len(chunk.records),
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
    )


def _merge_normalizer(target: RunningNormalizer, source: RunningNormalizer) -> None:
    if source.count == 0:
        return
    if target.count == 0:
        target.count = source.count
        target.mean = source.mean
        target.m2 = source.m2
        target._dimension = len(source.mean)
        return
    combined_count = target.count + source.count
    means: list[float] = []
    squared_differences: list[float] = []
    for target_mean, source_mean, target_m2, source_m2 in zip(target.mean, source.mean, target.m2, source.m2):
        delta = source_mean - target_mean
        means.append(target_mean + delta * source.count / combined_count)
        squared_differences.append(target_m2 + source_m2 + delta * delta * target.count * source.count / combined_count)
    target.count = combined_count
    target.mean = tuple(means)
    target.m2 = tuple(squared_differences)
    target._dimension = len(target.mean)


def _cached_sample_from_sample(
    sample: MovementDatasetSample,
    device: torch.device,
) -> CachedMovementTensorSample:
    x_lane = torch.tensor(sample.x_lane, dtype=torch.float32, device=device)
    x_movement = torch.tensor(sample.x_movement, dtype=torch.float32, device=device)
    target = torch.tensor(sample.teacher_movement_scores, dtype=torch.float32, device=device)
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


def _normalised_cached_sample(
    sample: CachedMovementTensorSample,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> CachedMovementTensorSample:
    return CachedMovementTensorSample(
        x_lane=_normalise_tensor(sample.x_lane, lane_normalizer).to(device),
        x_movement=_normalise_tensor(sample.x_movement, movement_normalizer).to(device),
        target=sample.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in sample.edge_index_dict.items()},
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=sample.city_name,
        phase_tensors=_phase_tensors_from_sample(sample=sample, device=device),
    )


def _move_cached_sample(sample: CachedMovementTensorSample, device: torch.device) -> CachedMovementTensorSample:
    assert sample.phase_tensors is not None
    return CachedMovementTensorSample(
        x_lane=sample.x_lane.to(device),
        x_movement=sample.x_movement.to(device),
        target=sample.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in sample.edge_index_dict.items()},
        phase_incidences=sample.phase_incidences,
        teacher_selected_phase_by_tls=sample.teacher_selected_phase_by_tls,
        city_name=sample.city_name,
        phase_tensors={
            traffic_light_id: CachedPhaseTensor(
                incidence_matrix=phase_tensor.incidence_matrix.to(device),
                movement_ids=phase_tensor.movement_ids.to(device),
                target_phase=phase_tensor.target_phase,
            )
            for traffic_light_id, phase_tensor in sample.phase_tensors.items()
        },
    )


def _cached_sample_batch(samples: Sequence[CachedMovementTensorSample]) -> CachedMovementTensorBatch:
    lane_counts = tuple(sample.x_lane.shape[0] for sample in samples)
    movement_counts = tuple(sample.x_movement.shape[0] for sample in samples)
    return CachedMovementTensorBatch(
        x_lane=torch.cat(tuple(sample.x_lane for sample in samples), dim=0),
        x_movement=torch.cat(tuple(sample.x_movement for sample in samples), dim=0),
        target=torch.cat(tuple(sample.target for sample in samples), dim=0),
        edge_index_dict=_batched_edge_index_dict(samples=samples),
        movement_sample_indices=_movement_sample_indices(movement_counts),
        lane_counts=lane_counts,
        movement_counts=movement_counts,
        phase_logit_groups=_phase_logit_group_batches(samples=samples),
        city_names=tuple(sample.city_name for sample in samples),
    )


def _batch_payload(batch: CachedMovementTensorBatch) -> CachedMovementTensorBatchPayload:
    return CachedMovementTensorBatchPayload(
        x_lane=batch.x_lane,
        x_movement=batch.x_movement,
        target=batch.target,
        edge_index_dict=batch.edge_index_dict,
        movement_sample_indices=batch.movement_sample_indices,
        lane_counts=batch.lane_counts,
        movement_counts=batch.movement_counts,
        phase_logit_groups=tuple(_phase_group_payload(group) for group in batch.phase_logit_groups),
        city_names=batch.city_names,
    )


def _phase_group_payload(group: CachedPhaseLogitGroupBatch) -> CachedPhaseLogitGroupPayload:
    return CachedPhaseLogitGroupPayload(
        incidence_matrices=group.incidence_matrices,
        movement_ids=group.movement_ids,
        targets=group.targets,
        sample_indices=group.sample_indices,
    )


def _batch_from_payload(payload: CachedMovementTensorBatchPayload) -> CachedMovementTensorBatch:
    return CachedMovementTensorBatch(
        x_lane=payload['x_lane'],
        x_movement=payload['x_movement'],
        target=payload['target'],
        edge_index_dict=payload['edge_index_dict'],
        movement_sample_indices=payload['movement_sample_indices'],
        lane_counts=payload['lane_counts'],
        movement_counts=payload['movement_counts'],
        phase_logit_groups=tuple(_phase_group_from_payload(group) for group in payload['phase_logit_groups']),
        city_names=payload['city_names'],
    )


def _phase_group_from_payload(payload: CachedPhaseLogitGroupPayload) -> CachedPhaseLogitGroupBatch:
    return CachedPhaseLogitGroupBatch(
        incidence_matrices=payload['incidence_matrices'],
        movement_ids=payload['movement_ids'],
        targets=payload['targets'],
        sample_indices=payload['sample_indices'],
    )


def _move_cached_batch(cpu_batch: CachedMovementTensorBatch, device: torch.device) -> CachedMovementTensorBatch:
    return CachedMovementTensorBatch(
        x_lane=cpu_batch.x_lane.to(device),
        x_movement=cpu_batch.x_movement.to(device),
        target=cpu_batch.target.to(device),
        edge_index_dict={key: value.to(device) for key, value in cpu_batch.edge_index_dict.items()},
        movement_sample_indices=cpu_batch.movement_sample_indices.to(device),
        lane_counts=cpu_batch.lane_counts,
        movement_counts=cpu_batch.movement_counts,
        phase_logit_groups=tuple(
            _move_phase_logit_group(group=group, device=device) for group in cpu_batch.phase_logit_groups
        ),
        city_names=cpu_batch.city_names,
    )


def _movement_sample_indices(movement_counts: Sequence[int]) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.full((movement_count,), sample_index, dtype=torch.long)
            for sample_index, movement_count in enumerate(movement_counts)
        ),
        dim=0,
    )


@dataclass(frozen=True)
class PhaseLogitGroupRows:
    incidence_matrices: list[torch.Tensor]
    movement_ids: list[torch.Tensor]
    targets: list[int]
    sample_indices: list[int]


def _phase_logit_group_batches(
    samples: Sequence[CachedMovementTensorSample],
) -> tuple[CachedPhaseLogitGroupBatch, ...]:
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
        CachedPhaseLogitGroupBatch(
            incidence_matrices=torch.stack(tuple(group.incidence_matrices)),
            movement_ids=torch.stack(tuple(group.movement_ids)),
            targets=torch.tensor(group.targets, dtype=torch.long),
            sample_indices=torch.tensor(group.sample_indices, dtype=torch.long),
        )
        for group in groups.values()
    )


def _move_phase_logit_group(
    group: CachedPhaseLogitGroupBatch,
    device: torch.device,
) -> CachedPhaseLogitGroupBatch:
    return CachedPhaseLogitGroupBatch(
        incidence_matrices=group.incidence_matrices.to(device),
        movement_ids=group.movement_ids.to(device),
        targets=group.targets.to(device),
        sample_indices=group.sample_indices.to(device),
    )


def _phase_tensors_from_sample(
    sample: CachedMovementTensorSample,
    device: torch.device,
) -> dict[str, CachedPhaseTensor]:
    return {
        traffic_light_id: CachedPhaseTensor(
            incidence_matrix=torch.tensor(incidence.rows, dtype=torch.float32, device=device),
            movement_ids=torch.tensor(incidence.movement_ids, dtype=torch.long, device=device),
            target_phase=sample.teacher_selected_phase_by_tls[traffic_light_id],
        )
        for traffic_light_id, incidence in sample.phase_incidences.items()
    }


def _normalise_tensor(tensor: torch.Tensor, normalizer: RunningNormalizer) -> torch.Tensor:
    if normalizer.count == 0:
        return torch.zeros_like(tensor)
    mean = torch.tensor(normalizer.mean, dtype=torch.float32)
    std = torch.tensor(normalizer.std, dtype=torch.float32).clamp_min(normalizer.epsilon)
    return torch.round((tensor - mean) / std, decimals=6)


def _normalizer_digest_bytes(normalizer: RunningNormalizer) -> bytes:
    return (
        int(normalizer.count).to_bytes(8, byteorder='little', signed=False)
        + repr(normalizer.mean).encode('utf-8')
        + repr(normalizer.std).encode('utf-8')
    )


def _tensor_rows(tensor: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in tensor.tolist())


def _raw_cache_path(cache_dir: Path, sample_index: int) -> Path:
    return cache_dir / f'sample_{sample_index:08d}.pt'


def _epoch_batches(
    sample_indices: Sequence[int],
    samples_per_batch: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, ...], ...]:
    shuffled_indices = list(sample_indices)
    Random(seed).shuffle(shuffled_indices)
    batches = list(
        tuple(shuffled_indices[batch_start : batch_start + samples_per_batch])
        for batch_start in range(0, len(shuffled_indices), samples_per_batch)
    )
    Random(seed + epoch + 1_000_003).shuffle(batches)
    return tuple(batches)


def _should_save_checkpoint(epoch: int, config: MovementILTrainingConfig) -> bool:
    if config.checkpoint_every_epochs <= 0:
        return epoch == config.epochs - 1
    return (epoch + 1) % config.checkpoint_every_epochs == 0 or epoch == config.epochs - 1


def _should_validate(epoch: int, config: MovementILTrainingConfig) -> bool:
    if config.validation_every_epochs <= 0:
        return False
    return (epoch + 1) % config.validation_every_epochs == 0 or epoch == config.epochs - 1


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
    load_elapsed_s: float,
    forward_elapsed_s: float,
    loss_elapsed_s: float,
    backward_elapsed_s: float,
) -> None:
    elapsed_s = time.monotonic() - started_s
    percent = samples_processed / total_samples * 100.0
    samples_per_second = samples_processed / max(elapsed_s, 1e-9)
    eta_s = (total_samples - samples_processed) / max(samples_per_second, 1e-9)
    print(
        f'epoch={epoch}/{epochs} batch={batch_number}/{total_batches} '
        f'samples={samples_processed}/{total_samples} percent={percent:.1f} '
        f'elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)} '
        f'loss_ma={moving_average_loss:.6f} '
        f't_load={load_elapsed_s:.1f}s t_forward={forward_elapsed_s:.1f}s '
        f't_loss={loss_elapsed_s:.1f}s t_backward={backward_elapsed_s:.1f}s '
        f'{_memory_summary(device)}'
    )


def _memory_summary(device: torch.device) -> str:
    cpu_memory = ''
    if device.type == 'cuda' and torch.cuda.is_available():
        allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
        reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
        return f'gpu_alloc_gib={allocated_gib:.2f} gpu_reserved_gib={reserved_gib:.2f}'
    return cpu_memory


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


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
