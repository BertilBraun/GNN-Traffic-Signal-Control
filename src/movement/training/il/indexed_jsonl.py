"""Indexed JSONL training path for large imitation-learning datasets."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
class CachedMovementTensorSample:
    x_lane: torch.Tensor
    x_movement: torch.Tensor
    target: torch.Tensor
    edge_index_dict: dict[str, torch.Tensor]
    phase_incidences: dict[str, StoredPhaseIncidence]
    teacher_selected_phase_by_tls: dict[str, int]
    city_name: str


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

    def get(self, sample_index: int) -> CachedMovementTensorSample:
        cached = self.memory_cache.get(sample_index)
        if cached is not None:
            self.memory_cache.move_to_end(sample_index)
            return _normalised_cached_sample(
                sample=cached,
                lane_normalizer=self.lane_normalizer,
                movement_normalizer=self.movement_normalizer,
                device=self.device,
            )
        loaded = cast(
            CachedMovementTensorSample,
            torch.load(self._cache_path(sample_index), map_location='cpu', weights_only=False),
        )
        self.memory_cache[sample_index] = loaded
        if len(self.memory_cache) > self.max_cached_samples:
            self.memory_cache.popitem(last=False)
        return _normalised_cached_sample(
            sample=loaded,
            lane_normalizer=self.lane_normalizer,
            movement_normalizer=self.movement_normalizer,
            device=self.device,
        )

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
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = checkpoint_dir / 'raw_tensor_cache'
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
        dataset=dataset,
        cache_dir=raw_cache_dir,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
        max_cached_samples=4096,
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
        samples = tuple(tensor_cache.get(sample_index) for sample_index in batch_sample_indices)
        predictions = _batched_predictions(model=model, samples=samples)
        batch_losses: list[torch.Tensor] = []
        for sample, prediction in zip(samples, predictions):
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


def _batched_predictions(
    model: MovementScorer,
    samples: Sequence[CachedMovementTensorSample],
) -> tuple[torch.Tensor, ...]:
    movement_counts = tuple(sample.x_movement.shape[0] for sample in samples)
    x_lane = torch.cat(tuple(sample.x_lane for sample in samples), dim=0)
    x_movement = torch.cat(tuple(sample.x_movement for sample in samples), dim=0)
    predictions = model(
        x_lane=x_lane,
        x_movement=x_movement,
        edge_index_dict=_batched_edge_index_dict(samples=samples),
    )
    movement_ends: list[int] = []
    movement_offset = 0
    for movement_count in movement_counts:
        movement_offset += movement_count
        movement_ends.append(movement_offset)
    movement_starts = (0, *movement_ends[:-1])
    return tuple(predictions[start:end] for start, end in zip(movement_starts, movement_ends))


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
        torch.load(preparation_path, map_location='cpu', weights_only=False),
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
    if cache_workers <= 0:
        raise ValueError('cache_workers must be positive.')
    cache_dir.mkdir(parents=True, exist_ok=True)
    lane_normalizer = RunningNormalizer()
    movement_normalizer = RunningNormalizer()
    started_s = time.monotonic()
    last_progress_s = started_s
    worker_count = min(cache_workers, len(cache_indices))
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
    return tuple(
        RawTensorCacheChunk(
            dataset_path=str(dataset_path),
            cache_dir=str(cache_dir),
            records=tuple(records[start : start + chunk_size]),
            train_indices=train_indices,
        )
        for start in range(0, len(records), chunk_size)
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
                    torch.load(cache_path, map_location='cpu', weights_only=False),
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
    )


def _normalise_tensor(tensor: torch.Tensor, normalizer: RunningNormalizer) -> torch.Tensor:
    if normalizer.count == 0:
        return torch.zeros_like(tensor)
    mean = torch.tensor(normalizer.mean, dtype=torch.float32)
    std = torch.tensor(normalizer.std, dtype=torch.float32).clamp_min(normalizer.epsilon)
    return torch.round((tensor - mean) / std, decimals=6)


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
