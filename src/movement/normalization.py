"""Normalization utilities for movement learning features."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Sequence


@dataclass
class RunningNormalizer:
    """Column-wise running mean/std normalizer with a freeze switch."""

    epsilon: float = 1e-8
    count: int = 0
    mean: tuple[float, ...] = ()
    m2: tuple[float, ...] = ()
    frozen: bool = False
    _dimension: int | None = field(default=None, init=False, repr=False)

    def update_rows(self, rows: Sequence[Sequence[float]]) -> None:
        """Update running statistics from feature rows unless frozen."""
        if self.frozen:
            return
        for row in rows:
            self.update_row(row)

    def update_row(self, row: Sequence[float]) -> None:
        """Update running statistics from one feature row unless frozen."""
        if self.frozen:
            return
        values = tuple(float(value) for value in row)
        self._ensure_dimension(values)
        if self.count == 0:
            self.count = 1
            self.mean = values
            self.m2 = tuple(0.0 for _ in values)
            return

        self.count += 1
        new_mean = []
        new_m2 = []
        for value, mean, m2 in zip(values, self.mean, self.m2):
            delta = value - mean
            updated_mean = mean + delta / self.count
            delta2 = value - updated_mean
            new_mean.append(updated_mean)
            new_m2.append(m2 + delta * delta2)
        self.mean = tuple(new_mean)
        self.m2 = tuple(new_m2)

    def transform_row(self, row: Sequence[float]) -> tuple[float, ...]:
        """Normalize one row using current statistics."""
        values = tuple(float(value) for value in row)
        self._ensure_dimension(values)
        if self.count == 0:
            return tuple(0.0 for _ in values)
        return tuple(
            round((value - mean) / max(std, self.epsilon), 6) for value, mean, std in zip(values, self.mean, self.std)
        )

    def freeze(self) -> None:
        """Prevent later updates from changing fitted statistics."""
        self.frozen = True

    @property
    def variance(self) -> tuple[float, ...]:
        if self.count == 0:
            return ()
        return tuple(m2 / self.count for m2 in self.m2)

    @property
    def std(self) -> tuple[float, ...]:
        return tuple(math.sqrt(value) for value in self.variance)

    def _ensure_dimension(self, values: tuple[float, ...]) -> None:
        if self._dimension is None:
            self._dimension = len(values)
            return
        if len(values) != self._dimension:
            raise ValueError(f'Expected {self._dimension} features, got {len(values)}.')
