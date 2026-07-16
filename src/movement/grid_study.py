"""Controlled synthetic grid scenarios and PPO sample allocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random


class GridStudyRole(str, Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    EVALUATION_ONLY = 'evaluation_only'


@dataclass(frozen=True)
class GridScenarioSpec:
    name: str
    rows: int
    cols: int
    role: GridStudyRole

    @property
    def internal_junction_count(self) -> int:
        return self.rows * self.cols

    @property
    def controller_count(self) -> int:
        return len(controllable_node_ids(rows=self.rows, cols=self.cols))


@dataclass(frozen=True)
class GridCoverageSpec:
    name: str
    rows: int
    cols: int
    signalized_controller_count: int

    @property
    def signalized_node_ids(self) -> frozenset[str]:
        return evenly_spaced_signalized_node_ids(
            rows=self.rows,
            cols=self.cols,
            signal_count=self.signalized_controller_count,
        )

    @property
    def unsignalized_node_ids(self) -> frozenset[str]:
        return controllable_node_ids(rows=self.rows, cols=self.cols) - self.signalized_node_ids


@dataclass(frozen=True)
class GridCoverageVariantSpec:
    name: str
    rows: int
    cols: int
    signalized_controller_count: int
    mask_seed: int
    role: GridStudyRole

    @property
    def signalized_node_ids(self) -> frozenset[str]:
        return spatially_distributed_signalized_node_ids(
            rows=self.rows,
            cols=self.cols,
            signal_count=self.signalized_controller_count,
            mask_seed=self.mask_seed,
        )

    @property
    def unsignalized_node_ids(self) -> frozenset[str]:
        return controllable_node_ids(rows=self.rows, cols=self.cols) - self.signalized_node_ids


@dataclass(frozen=True)
class GridRolloutAllocation:
    scenario_name: str
    controller_count: int
    rollout_jobs: int
    decisions_per_rollout: int
    action_sample_count: int


MATCHED_GRID_SCENARIOS: tuple[GridScenarioSpec, ...] = (
    GridScenarioSpec('matched_grid_2x3_wide', 2, 3, GridStudyRole.EVALUATION_ONLY),
    GridScenarioSpec('matched_grid_3x2_tall', 3, 2, GridStudyRole.EVALUATION_ONLY),
    GridScenarioSpec('matched_grid_2x5_wide', 2, 5, GridStudyRole.EVALUATION_ONLY),
    GridScenarioSpec('matched_grid_5x2_tall', 5, 2, GridStudyRole.TRAIN),
    GridScenarioSpec('matched_grid_3x3_square', 3, 3, GridStudyRole.TRAIN),
    GridScenarioSpec('matched_grid_3x5_wide', 3, 5, GridStudyRole.EVALUATION_ONLY),
    GridScenarioSpec('matched_grid_5x3_tall', 5, 3, GridStudyRole.TRAIN),
    GridScenarioSpec('matched_grid_4x4_square', 4, 4, GridStudyRole.TRAIN),
    GridScenarioSpec('matched_grid_5x5_square', 5, 5, GridStudyRole.TRAIN),
    GridScenarioSpec('matched_grid_6x6_square_validation', 6, 6, GridStudyRole.VALIDATION),
)


GRID_COVERAGE_SCENARIOS: tuple[GridCoverageSpec, ...] = (
    GridCoverageSpec('coverage_grid_6x6_eligible_signals_32_of_32', 6, 6, 32),
    GridCoverageSpec('coverage_grid_6x6_eligible_signals_24_of_32', 6, 6, 24),
    GridCoverageSpec('coverage_grid_6x6_eligible_signals_16_of_32', 6, 6, 16),
    GridCoverageSpec('coverage_grid_6x6_eligible_signals_08_of_32', 6, 6, 8),
)

COVERAGE_GENERALIZATION_4X4_TRAINING_SCENARIOS: tuple[GridCoverageVariantSpec, ...] = tuple(
    GridCoverageVariantSpec(
        name=f'coverage_generalization_grid_4x4_train_signals_06_of_12_mask_{mask_index:02d}',
        rows=4,
        cols=4,
        signalized_controller_count=6,
        mask_seed=mask_seed,
        role=GridStudyRole.TRAIN,
    )
    for mask_index, mask_seed in enumerate((1_074, 1_198, 1_415, 1_102, 1_051), start=1)
)

COVERAGE_GENERALIZATION_4X4_VALIDATION_SCENARIO = GridCoverageVariantSpec(
    name='coverage_generalization_grid_4x4_validation_signals_06_of_12_mask_01',
    rows=4,
    cols=4,
    signalized_controller_count=6,
    mask_seed=2_000,
    role=GridStudyRole.VALIDATION,
)

COVERAGE_GENERALIZATION_4X4_EVALUATION_SCENARIOS: tuple[GridCoverageVariantSpec, ...] = (
    *tuple(
        GridCoverageVariantSpec(
            name=f'coverage_generalization_grid_4x4_eval_signals_03_of_12_mask_{mask_index:02d}',
            rows=4,
            cols=4,
            signalized_controller_count=3,
            mask_seed=mask_seed,
            role=GridStudyRole.EVALUATION_ONLY,
        )
        for mask_index, mask_seed in ((1, 3_101), (2, 3_102), (4, 3_104), (5, 3_105), (6, 3_107))
    ),
    *tuple(
        GridCoverageVariantSpec(
            name=f'coverage_generalization_grid_4x4_eval_signals_06_of_12_mask_{mask_index:02d}',
            rows=4,
            cols=4,
            signalized_controller_count=6,
            mask_seed=3_200 + mask_index,
            role=GridStudyRole.EVALUATION_ONLY,
        )
        for mask_index in range(1, 6)
    ),
    *tuple(
        GridCoverageVariantSpec(
            name=f'coverage_generalization_grid_4x4_eval_signals_09_of_12_mask_{mask_index:02d}',
            rows=4,
            cols=4,
            signalized_controller_count=9,
            mask_seed=3_300 + mask_index,
            role=GridStudyRole.EVALUATION_ONLY,
        )
        for mask_index in range(1, 6)
    ),
    GridCoverageVariantSpec(
        name='coverage_generalization_grid_4x4_eval_signals_12_of_12_mask_01',
        rows=4,
        cols=4,
        signalized_controller_count=12,
        mask_seed=3_401,
        role=GridStudyRole.EVALUATION_ONLY,
    ),
)

COVERAGE_GENERALIZATION_4X4_SCENARIOS: tuple[GridCoverageVariantSpec, ...] = (
    *COVERAGE_GENERALIZATION_4X4_TRAINING_SCENARIOS,
    COVERAGE_GENERALIZATION_4X4_VALIDATION_SCENARIO,
    *COVERAGE_GENERALIZATION_4X4_EVALUATION_SCENARIOS,
)


def controllable_node_ids(rows: int, cols: int) -> frozenset[str]:
    _validate_grid_dimensions(rows=rows, cols=cols)
    return frozenset(
        f'N{row}_{col}'
        for row in range(rows)
        for col in range(cols)
        if _grid_degree(row=row, col=col, rows=rows, cols=cols) >= 3
    )


def evenly_spaced_signalized_node_ids(
    rows: int,
    cols: int,
    signal_count: int,
) -> frozenset[str]:
    candidates = tuple(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if _grid_degree(row=row, col=col, rows=rows, cols=cols) >= 3
    )
    if signal_count <= 0 or signal_count > len(candidates):
        raise ValueError(f'signal_count must be between 1 and {len(candidates)}.')
    selected: list[tuple[int, int]] = []
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    while len(selected) < signal_count:
        remaining = tuple(candidate for candidate in candidates if candidate not in selected)
        next_node = max(
            remaining,
            key=lambda candidate: (
                _minimum_distance(candidate=candidate, selected=selected, center=center),
                -abs(candidate[0] - center[0]) - abs(candidate[1] - center[1]),
                -candidate[0],
                -candidate[1],
            ),
        )
        selected.append(next_node)
    return frozenset(f'N{row}_{col}' for row, col in selected)


def spatially_distributed_signalized_node_ids(
    rows: int,
    cols: int,
    signal_count: int,
    mask_seed: int,
) -> frozenset[str]:
    candidates = tuple(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if _grid_degree(row=row, col=col, rows=rows, cols=cols) >= 3
    )
    if signal_count <= 0 or signal_count > len(candidates):
        raise ValueError(f'signal_count must be between 1 and {len(candidates)}.')
    shuffled_candidates = list(candidates)
    random.Random(mask_seed).shuffle(shuffled_candidates)
    tie_break_rank = {candidate: rank for rank, candidate in enumerate(shuffled_candidates)}
    selected = [shuffled_candidates[0]]
    while len(selected) < signal_count:
        remaining = tuple(candidate for candidate in candidates if candidate not in selected)
        next_node = max(
            remaining,
            key=lambda candidate: (
                min(math.hypot(candidate[0] - row, candidate[1] - col) for row, col in selected),
                -tie_break_rank[candidate],
            ),
        )
        selected.append(next_node)
    return frozenset(f'N{row}_{col}' for row, col in selected)


def balanced_rollout_allocation(
    scenarios: tuple[GridScenarioSpec, ...],
    target_action_samples_per_scenario: int,
    decisions_per_rollout: int,
) -> tuple[GridRolloutAllocation, ...]:
    if target_action_samples_per_scenario <= 0:
        raise ValueError('target_action_samples_per_scenario must be positive.')
    if decisions_per_rollout <= 0:
        raise ValueError('decisions_per_rollout must be positive.')
    allocations: list[GridRolloutAllocation] = []
    for scenario in scenarios:
        if scenario.role is not GridStudyRole.TRAIN:
            continue
        action_samples_per_rollout = scenario.controller_count * decisions_per_rollout
        rollout_jobs = math.ceil(target_action_samples_per_scenario / action_samples_per_rollout)
        allocations.append(
            GridRolloutAllocation(
                scenario_name=scenario.name,
                controller_count=scenario.controller_count,
                rollout_jobs=rollout_jobs,
                decisions_per_rollout=decisions_per_rollout,
                action_sample_count=rollout_jobs * action_samples_per_rollout,
            )
        )
    return tuple(allocations)


def _minimum_distance(
    candidate: tuple[int, int],
    selected: list[tuple[int, int]],
    center: tuple[float, float],
) -> float:
    if not selected:
        return -math.hypot(candidate[0] - center[0], candidate[1] - center[1])
    return min(math.hypot(candidate[0] - row, candidate[1] - col) for row, col in selected)


def _grid_degree(row: int, col: int, rows: int, cols: int) -> int:
    return int(row > 0) + int(row < rows - 1) + int(col > 0) + int(col < cols - 1)


def _validate_grid_dimensions(rows: int, cols: int) -> None:
    if rows < 2 or cols < 2:
        raise ValueError('Grid dimensions must be at least 2x2.')
    if rows == 2 and cols == 2:
        raise ValueError('A 2x2 grid has no controllable degree-three junctions.')
