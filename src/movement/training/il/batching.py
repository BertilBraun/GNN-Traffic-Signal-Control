"""Batch planning for movement imitation learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.movement.dataset import MovementDatasetSample


class MovementILBatchPlanner(Protocol):
    def epoch_batches(
        self,
        samples: Sequence[MovementDatasetSample],
        epoch: int,
    ) -> tuple[tuple[int, ...], ...]: ...


@dataclass(frozen=True)
class RandomBatchPlanner:
    samples_per_batch: int
    seed: int

    def epoch_batches(
        self,
        samples: Sequence[MovementDatasetSample],
        epoch: int,
    ) -> tuple[tuple[int, ...], ...]:
        _validate_batch_inputs(samples=samples, samples_per_batch=self.samples_per_batch)
        sample_indices = list(range(len(samples)))
        Random(self.seed + epoch).shuffle(sample_indices)
        return _batches(sample_indices=tuple(sample_indices), samples_per_batch=self.samples_per_batch)


@dataclass(frozen=True)
class CityBalancedBatchPlanner:
    samples_per_batch: int
    seed: int

    def epoch_batches(
        self,
        samples: Sequence[MovementDatasetSample],
        epoch: int,
    ) -> tuple[tuple[int, ...], ...]:
        _validate_batch_inputs(samples=samples, samples_per_batch=self.samples_per_batch)
        city_groups = _sample_groups_by_city(samples)
        if len(city_groups) < 2:
            raise ValueError('city-balanced batching requires samples from at least two cities')
        target_samples_per_city = max(len(group.sample_indices) for group in city_groups)
        balanced_indices: list[int] = []
        repeated_groups = tuple(
            _repeat_group_indices(
                group=group,
                target_samples_per_city=target_samples_per_city,
                seed=self.seed,
                epoch=epoch,
                city_index=city_index,
            )
            for city_index, group in enumerate(city_groups)
        )
        for sample_offset in range(target_samples_per_city):
            for group_indices in repeated_groups:
                balanced_indices.append(group_indices[sample_offset])
        Random(self.seed + epoch + 65_537).shuffle(balanced_indices)
        return _batches(sample_indices=tuple(balanced_indices), samples_per_batch=self.samples_per_batch)


@dataclass(frozen=True)
class CitySampleGroup:
    city_name: str
    sample_indices: tuple[int, ...]


class SampleCityMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str


@dataclass(frozen=True)
class CitySeedSampleGroup:
    city_name: str
    collection_seed: int
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrainValidationSamples:
    training_samples: tuple[MovementDatasetSample, ...]
    validation_samples: tuple[MovementDatasetSample, ...]


class SampleSplitMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    collection_seed: int


def split_train_validation_by_city_seed(
    samples: Sequence[MovementDatasetSample],
    validation_fraction: float,
    seed: int,
) -> TrainValidationSamples:
    if not samples:
        raise ValueError('At least one sample is required.')
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError('validation_fraction must be in [0.0, 1.0).')
    if validation_fraction == 0.0:
        return TrainValidationSamples(
            training_samples=tuple(samples),
            validation_samples=(),
        )
    groups = _sample_groups_by_city_seed(samples)
    validation_indices: set[int] = set()
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
    training_samples = tuple(sample for index, sample in enumerate(samples) if index not in validation_indices)
    validation_samples = tuple(sample for index, sample in enumerate(samples) if index in validation_indices)
    if not training_samples:
        raise ValueError('validation split left no training samples.')
    return TrainValidationSamples(
        training_samples=training_samples,
        validation_samples=validation_samples,
    )


def _sample_groups_by_city(samples: Sequence[MovementDatasetSample]) -> tuple[CitySampleGroup, ...]:
    city_names: list[str] = []
    sample_city_names = tuple(_sample_city_name(sample) for sample in samples)
    for city_name in sample_city_names:
        if city_name not in city_names:
            city_names.append(city_name)
    return tuple(
        CitySampleGroup(
            city_name=city_name,
            sample_indices=tuple(
                index for index, sample_city_name in enumerate(sample_city_names) if sample_city_name == city_name
            ),
        )
        for city_name in city_names
    )


def _sample_city_name(sample: MovementDatasetSample) -> str:
    return SampleCityMetadata.model_validate(sample.metadata).city_name


def _sample_groups_by_city_seed(samples: Sequence[MovementDatasetSample]) -> tuple[CitySeedSampleGroup, ...]:
    group_keys: list[tuple[str, int]] = []
    sample_metadata = tuple(SampleSplitMetadata.model_validate(sample.metadata) for sample in samples)
    for metadata in sample_metadata:
        group_key = (metadata.city_name, metadata.collection_seed)
        if group_key not in group_keys:
            group_keys.append(group_key)
    return tuple(
        CitySeedSampleGroup(
            city_name=city_name,
            collection_seed=collection_seed,
            sample_indices=tuple(
                index
                for index, metadata in enumerate(sample_metadata)
                if metadata.city_name == city_name and metadata.collection_seed == collection_seed
            ),
        )
        for city_name, collection_seed in group_keys
    )


def _ordered_city_names(groups: Sequence[CitySeedSampleGroup]) -> tuple[str, ...]:
    city_names: list[str] = []
    for group in groups:
        if group.city_name not in city_names:
            city_names.append(group.city_name)
    return tuple(city_names)


def _repeat_group_indices(
    group: CitySampleGroup,
    target_samples_per_city: int,
    seed: int,
    epoch: int,
    city_index: int,
) -> tuple[int, ...]:
    shuffled_indices = list(group.sample_indices)
    Random(seed + epoch * 997 + city_index).shuffle(shuffled_indices)
    repeated_indices: list[int] = []
    while len(repeated_indices) < target_samples_per_city:
        repeated_indices.extend(shuffled_indices)
    return tuple(repeated_indices[:target_samples_per_city])


def _batches(
    sample_indices: tuple[int, ...],
    samples_per_batch: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sample_indices[batch_start : batch_start + samples_per_batch]
        for batch_start in range(0, len(sample_indices), samples_per_batch)
    )


def _validate_batch_inputs(
    samples: Sequence[MovementDatasetSample],
    samples_per_batch: int,
) -> None:
    if not samples:
        raise ValueError('At least one sample is required.')
    if samples_per_batch <= 0:
        raise ValueError('samples_per_batch must be positive.')
