"""Serialization helpers for training normalizers."""

from __future__ import annotations

from dataclasses import dataclass

from src.movement.normalization import RunningNormalizer


@dataclass(frozen=True)
class NormalizerState:
    count: int
    mean: tuple[float, ...]
    squared_differences: tuple[float, ...]
    frozen: bool
    epsilon: float


def normalizer_from_state(state: NormalizerState) -> RunningNormalizer:
    normalizer = RunningNormalizer(epsilon=state.epsilon)
    normalizer.count = state.count
    normalizer.mean = state.mean
    normalizer.m2 = state.squared_differences
    normalizer.frozen = state.frozen
    normalizer._dimension = len(normalizer.mean)
    return normalizer


def normalizer_state(normalizer: RunningNormalizer) -> NormalizerState:
    return NormalizerState(
        count=normalizer.count,
        mean=normalizer.mean,
        squared_differences=normalizer.m2,
        frozen=normalizer.frozen,
        epsilon=normalizer.epsilon,
    )
