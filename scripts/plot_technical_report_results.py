"""Create the compact evaluation figure used by the technical report."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class EvaluationRow:
    city_name: str
    policy: str
    demand_scale: float
    throughput_per_hour: float
    completion_rate: float


@dataclass(frozen=True)
class MeanMetrics:
    throughput_per_hour: float
    completion_rate: float


def _read_evaluation_rows(path: Path) -> tuple[EvaluationRow, ...]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        city_index = header.index('city_name')
        policy_index = header.index('policy')
        demand_index = header.index('demand_scale')
        row_type_index = header.index('row_type')
        throughput_index = header.index('throughput_per_hour')
        completion_index = header.index('completion_rate')
        return tuple(
            EvaluationRow(
                city_name=row[city_index],
                policy=row[policy_index],
                demand_scale=float(row[demand_index]),
                throughput_per_hour=float(row[throughput_index]),
                completion_rate=float(row[completion_index]),
            )
            for row in reader
            if row[row_type_index] == 'seed'
        )


def _mean_metrics(rows: Iterable[EvaluationRow]) -> MeanMetrics:
    materialized_rows = tuple(rows)
    if not materialized_rows:
        raise ValueError('At least one evaluation row is required.')
    return MeanMetrics(
        throughput_per_hour=float(np.mean(tuple(row.throughput_per_hour for row in materialized_rows))),
        completion_rate=float(np.mean(tuple(row.completion_rate for row in materialized_rows))),
    )


def _grid_means(results_root: Path) -> tuple[tuple[float, MeanMetrics, MeanMetrics], ...]:
    learned_directories = (
        'train_mixed_seed5101_best0060',
        'train_mixed_seed5102_best0060',
        'train_mixed_seed5103_best0060',
    )
    learned_rows = tuple(
        row
        for directory in learned_directories
        for row in _read_evaluation_rows(results_root / 'grid_generalization_final' / directory / 'summary.csv')
        if row.city_name == 'matched_grid_6x6_square_validation' and row.policy == 'learned'
    )
    baseline_rows = tuple(
        row
        for row in _read_evaluation_rows(
            results_root / 'grid_generalization_final' / 'common_baselines_fresh7101_7106' / 'summary.csv'
        )
        if row.city_name == 'matched_grid_6x6_square_validation' and row.policy == 'max-pressure'
    )
    return tuple(
        (
            demand_scale,
            _mean_metrics(row for row in learned_rows if row.demand_scale == demand_scale),
            _mean_metrics(row for row in baseline_rows if row.demand_scale == demand_scale),
        )
        for demand_scale in (0.6, 0.7, 0.8)
    )


def _city_ratios(results_root: Path) -> tuple[tuple[str, float, float], ...]:
    path = (
        results_root
        / 'city_stuttgart_visible_validation_local_reward_2hop_60_seed_9702'
        / 'eval'
        / 'iter_0060'
        / 'summary.csv'
    )
    rows = _read_evaluation_rows(path)
    baseline_policies = {'max-pressure', 'queue', 'fixed-time', 'uniform-random'}
    city_labels = {
        'karlsruhe_oststadt': 'Karlsruhe',
        'mannheim_innenstadt': 'Mannheim',
        'stuttgart_mitte': 'Stuttgart',
        'heidelberg_bergheim': 'Heidelberg',
        'freiburg_altstadt': 'Freiburg',
    }
    ratios = []
    for city_name, label in city_labels.items():
        learned = _mean_metrics(row for row in rows if row.city_name == city_name and row.policy == 'learned')
        baseline_means = tuple(
            _mean_metrics(row for row in rows if row.city_name == city_name and row.policy == policy)
            for policy in baseline_policies
        )
        ratios.append(
            (
                label,
                learned.throughput_per_hour / max(item.throughput_per_hour for item in baseline_means),
                learned.completion_rate / max(item.completion_rate for item in baseline_means),
            )
        )
    return tuple(ratios)


def _plot_grid_panel(axis: plt.Axes, grid_means: tuple[tuple[float, MeanMetrics, MeanMetrics], ...]) -> None:
    demand = np.array(tuple(item[0] for item in grid_means))
    learned = np.array(tuple(item[1].throughput_per_hour for item in grid_means))
    baseline = np.array(tuple(item[2].throughput_per_hour for item in grid_means))
    axis.plot(demand, learned, marker='o', linewidth=2.2, color='#087f8c', label='learned (sampled)')
    axis.plot(demand, baseline, marker='s', linewidth=2.0, color='#6b7280', label='max pressure')
    axis.set_title('(a) Full-coverage $6\\times6$ transfer', loc='left', fontweight='bold')
    axis.set_xlabel('Demand scale')
    axis.set_ylabel('Throughput (vehicles/hour)')
    axis.set_xticks(demand)
    axis.grid(axis='y', color='#d7dde2', linewidth=0.8)
    axis.legend(frameon=False, loc='upper left')


def _plot_city_panel(axis: plt.Axes, city_ratios: tuple[tuple[str, float, float], ...]) -> None:
    colors = ('#087f8c', '#d97706', '#2563eb', '#7c3aed', '#dc2626')
    offsets = {
        'Karlsruhe': (5, 5),
        'Mannheim': (5, 6),
        'Stuttgart': (5, 5),
        'Heidelberg': (5, -13),
        'Freiburg': (5, 5),
    }
    for (label, throughput_ratio, completion_ratio), color in zip(city_ratios, colors, strict=True):
        axis.scatter(throughput_ratio, completion_ratio, s=58, color=color, edgecolor='white', linewidth=0.8)
        axis.annotate(
            label,
            (throughput_ratio, completion_ratio),
            xytext=offsets[label],
            textcoords='offset points',
            fontsize=8.5,
        )
    axis.axvline(1.0, color='#9ca3af', linewidth=1.0)
    axis.axhline(1.0, color='#9ca3af', linewidth=1.0)
    axis.set_title('(b) City result relative to best baseline', loc='left', fontweight='bold')
    axis.set_xlabel('Throughput ratio')
    axis.set_ylabel('Completion ratio')
    axis.set_xlim(0.935, 1.165)
    axis.set_ylim(0.93, 1.17)
    axis.grid(color='#e5e7eb', linewidth=0.7)
    axis.text(
        1.005,
        1.006,
        'learned higher on both',
        ha='left',
        va='bottom',
        color='#4b5563',
        fontsize=8,
    )


def create_figure(results_root: Path, output_path: Path) -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.3), constrained_layout=True)
    _plot_grid_panel(axes[0], _grid_means(results_root))
    _plot_city_panel(axes[1], _city_ratios(results_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches='tight')
    figure.savefig(output_path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    create_figure(results_root=arguments.results_root, output_path=arguments.output)


if __name__ == '__main__':
    main()
