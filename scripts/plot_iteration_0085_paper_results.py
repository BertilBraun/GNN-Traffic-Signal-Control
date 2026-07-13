"""Plot publication figures from the frozen iteration-85 evaluation export."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / 'artifacts'
    / 'ppo_runs'
    / 'city_first_pass_throughput_progress_025_sample_eval_v3'
    / 'selected_iteration_0085'
    / 'evaluation'
    / 'summary.csv'
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / 'docs' / 'results' / 'assets'


@dataclass(frozen=True)
class City:
    key: str
    label: str
    split: str
    color: str


@dataclass(frozen=True)
class Evaluation:
    city: str
    policy: str
    seed: int
    throughput_per_hour: float
    completion_rate: float


@dataclass(frozen=True)
class PolicyStyle:
    key: str
    label: str
    color: str
    marker: str


CITIES = (
    City('karlsruhe_oststadt', 'Karlsruhe', 'train', '#0072B2'),
    City('mannheim_innenstadt', 'Mannheim', 'train', '#D55E00'),
    City('stuttgart_mitte', 'Stuttgart', 'train', '#009E73'),
    City('heidelberg_bergheim', 'Heidelberg', 'train', '#CC79A7'),
    City('freiburg_altstadt', 'Freiburg', 'no_rollouts', '#E69F00'),
)
POLICIES = (
    PolicyStyle('learned', 'Learned', '#0072B2', 'o'),
    PolicyStyle('max-pressure', 'Max pressure', '#D55E00', 's'),
    PolicyStyle('queue', 'Longest queue', '#009E73', '^'),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def load_evaluations(summary_path: Path) -> tuple[Evaluation, ...]:
    evaluations: list[Evaluation] = []
    with summary_path.open(encoding='utf-8', newline='') as summary_file:
        for row in csv.DictReader(summary_file):
            if row['row_type'] != 'seed':
                continue
            evaluations.append(
                Evaluation(
                    city=row['city_name'],
                    policy=row['policy'],
                    seed=int(row['seed']),
                    throughput_per_hour=float(row['throughput_per_hour']),
                    completion_rate=float(row['completion_rate']),
                )
            )
    expected_rows = len(CITIES) * len(POLICIES) * 6
    if len(evaluations) != expected_rows:
        raise ValueError(f'Expected {expected_rows} seed rows, found {len(evaluations)}.')
    return tuple(evaluations)


def group_evaluations(
    evaluations: tuple[Evaluation, ...],
) -> dict[tuple[str, str], tuple[Evaluation, ...]]:
    grouped: defaultdict[tuple[str, str], list[Evaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[(evaluation.city, evaluation.policy)].append(evaluation)
    return {key: tuple(sorted(group, key=lambda evaluation: evaluation.seed)) for key, group in grouped.items()}


def configure_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 140,
            'savefig.dpi': 300,
            'font.family': 'DejaVu Sans',
            'font.size': 9.5,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.frameon': False,
            'pdf.fonttype': 42,
            'svg.fonttype': 'none',
        }
    )


def add_freiburg_region(axis: Axes) -> None:
    axis.axvspan(3.5, 4.5, color='#E8F1F5', zorder=0)
    axis.axvline(3.5, color='#78909C', linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    axis.text(
        4.0,
        0.985,
        'No PPO rollouts',
        transform=axis.get_xaxis_transform(),
        ha='center',
        va='top',
        color='#455A64',
        fontsize=8.5,
        fontweight='bold',
    )


def save_figure(figure: Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix('.png'), bbox_inches='tight', facecolor='white')
    figure.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(figure)


def plot_throughput_comparison(grouped: dict[tuple[str, str], tuple[Evaluation, ...]], output_directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.7))
    city_positions = np.arange(len(CITIES), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    add_freiburg_region(axis)

    for policy, offset in zip(POLICIES, offsets, strict=True):
        means: list[float] = []
        standard_deviations: list[float] = []
        for city in CITIES:
            values = [evaluation.throughput_per_hour for evaluation in grouped[(city.key, policy.key)]]
            means.append(fmean(values))
            standard_deviations.append(stdev(values))
        positions = city_positions + offset
        axis.errorbar(
            positions,
            means,
            yerr=standard_deviations,
            fmt=policy.marker,
            markersize=7.0,
            markeredgecolor='white',
            markeredgewidth=0.8,
            color=policy.color,
            ecolor=policy.color,
            elinewidth=1.5,
            capsize=3.0,
            capthick=1.5,
            label=policy.label,
            zorder=4,
        )
        for position, city in zip(positions, CITIES, strict=True):
            seed_values = [evaluation.throughput_per_hour for evaluation in grouped[(city.key, policy.key)]]
            axis.scatter(
                np.full(len(seed_values), position),
                seed_values,
                s=10,
                color=policy.color,
                alpha=0.24,
                linewidths=0,
                zorder=2,
            )

    axis.set_xticks(city_positions, [city.label for city in CITIES])
    axis.set_xlim(-0.55, 4.55)
    axis.set_ylim(1900, 4500)
    axis.set_ylabel('Throughput (vehicles/hour)')
    axis.set_title('Iteration 85: throughput across OSM-derived city networks', loc='left', fontweight='bold')
    axis.legend(ncols=3, loc='upper left')
    axis.grid(axis='x', visible=False)
    axis.text(
        0.0,
        -0.18,
        'Points show six evaluation seeds; error bars show mean ± one seed standard deviation.',
        transform=axis.transAxes,
        color='#4F4F4F',
        fontsize=8.5,
    )
    save_figure(figure, output_directory / 'iteration-0085-throughput-comparison')


def plot_throughput_completion(grouped: dict[tuple[str, str], tuple[Evaluation, ...]], output_directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for policy in POLICIES:
        for city in CITIES:
            rows = grouped[(city.key, policy.key)]
            mean_completion = fmean(row.completion_rate for row in rows) * 100.0
            mean_throughput = fmean(row.throughput_per_hour for row in rows)
            axis.scatter(
                mean_completion,
                mean_throughput,
                s=58,
                color=city.color,
                marker=policy.marker,
                edgecolor='white',
                linewidth=0.8,
                zorder=3,
            )
    axis.set(xlabel='Completion rate (%)', ylabel='Throughput (vehicles/hour)')
    axis.set_ylim(2300, 4100)
    policy_handles = [
        Line2D(
            [0],
            [0],
            marker=policy.marker,
            color='none',
            markerfacecolor='#616161',
            markeredgecolor='white',
            markersize=7,
            label=policy.label,
        )
        for policy in POLICIES
    ]
    city_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=city.color, markersize=7, label=city.label)
        for city in CITIES
    ]
    policy_legend = axis.legend(handles=policy_handles, ncols=3, loc='lower left', bbox_to_anchor=(0.0, 1.01))
    axis.add_artist(policy_legend)
    axis.legend(handles=city_handles, ncols=5, loc='lower right', bbox_to_anchor=(1.0, 1.11), fontsize=8)
    axis.grid(alpha=0.35)
    save_figure(figure, output_directory / 'iteration-0085-throughput-completion')


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    grouped = group_evaluations(load_evaluations(arguments.summary))
    plot_throughput_comparison(grouped, arguments.output_dir)
    plot_throughput_completion(grouped, arguments.output_dir)
    print(f'Wrote iteration-85 paper figures to {arguments.output_dir}')


if __name__ == '__main__':
    main()
