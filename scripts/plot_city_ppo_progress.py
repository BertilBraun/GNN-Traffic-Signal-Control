"""Plot multi-city PPO evaluation progress across checkpoint iterations."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

LEARNED_POLICIES = ('learned', 'learned-greedy')
BASELINE_POLICIES = ('fixed-time', 'max-pressure', 'queue', 'uniform-random')
POLICY_LABELS = {
    'learned': 'Learned sampled',
    'learned-greedy': 'Learned greedy',
    'fixed-time': 'Fixed-time',
    'max-pressure': 'Max-pressure',
    'queue': 'Queue',
    'uniform-random': 'Uniform-random',
}
POLICY_COLORS = {
    'learned': '#0072B2',
    'learned-greedy': '#D55E00',
    'fixed-time': '#009E73',
    'max-pressure': '#CC79A7',
    'queue': '#E69F00',
    'uniform-random': '#7A7A7A',
}
CITY_LABELS = {
    'karlsruhe_oststadt': 'Karlsruhe',
    'mannheim_innenstadt': 'Mannheim',
    'stuttgart_mitte': 'Stuttgart',
    'heidelberg_bergheim': 'Heidelberg',
    'freiburg_altstadt': 'Freiburg',
}


@dataclass(frozen=True)
class EvaluationPoint:
    iteration: int
    city_name: str
    policy: str
    throughput_per_hour: float
    completion_percent: float
    wait_density_s_per_m: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_evaluation_points(evaluation_root: Path) -> tuple[EvaluationPoint, ...]:
    points: list[EvaluationPoint] = []
    for iteration_directory in sorted(evaluation_root.glob('iter_*')):
        summary_path = iteration_directory / 'summary.csv'
        if not summary_path.is_file():
            continue
        iteration = int(iteration_directory.name.removeprefix('iter_'))
        with summary_path.open(encoding='utf-8', newline='') as summary_file:
            rows = csv.DictReader(summary_file)
            points.extend(
                EvaluationPoint(
                    iteration=iteration,
                    city_name=row['city_name'],
                    policy=row['policy'],
                    throughput_per_hour=float(row['throughput_per_hour']),
                    completion_percent=100.0 * float(row['completion_rate']),
                    wait_density_s_per_m=float(row['average_wait_density_s_per_m']),
                )
                for row in rows
                if row['row_type'] == 'mean'
            )
    if not points:
        raise ValueError(f'No evaluation summaries found under {evaluation_root}')
    return tuple(points)


def aggregate_city_means(points: tuple[EvaluationPoint, ...]) -> tuple[EvaluationPoint, ...]:
    grouped: dict[tuple[int, str], list[EvaluationPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.iteration, point.policy)].append(point)
    return tuple(
        EvaluationPoint(
            iteration=iteration,
            city_name='five_city_mean',
            policy=policy,
            throughput_per_hour=sum(point.throughput_per_hour for point in group) / len(group),
            completion_percent=sum(point.completion_percent for point in group) / len(group),
            wait_density_s_per_m=sum(point.wait_density_s_per_m for point in group) / len(group),
        )
        for (iteration, policy), group in sorted(grouped.items())
    )


def policy_points(
    points: tuple[EvaluationPoint, ...],
    policy: str,
    city_name: str,
) -> tuple[EvaluationPoint, ...]:
    return tuple(point for point in points if point.policy == policy and point.city_name == city_name)


def metric_value(point: EvaluationPoint, metric: str) -> float:
    match metric:
        case 'throughput':
            return point.throughput_per_hour
        case 'completion':
            return point.completion_percent
        case 'wait_density':
            return point.wait_density_s_per_m
        case _:
            raise ValueError(f'Unsupported metric: {metric}')


def draw_metric(
    axis: Axes,
    points: tuple[EvaluationPoint, ...],
    city_name: str,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    for policy in LEARNED_POLICIES:
        series = policy_points(points, policy=policy, city_name=city_name)
        axis.plot(
            tuple(point.iteration for point in series),
            tuple(metric_value(point, metric) for point in series),
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            marker='o' if policy == 'learned' else 's',
            linewidth=2.2,
            markersize=4.5,
        )
    for policy in BASELINE_POLICIES:
        series = policy_points(points, policy=policy, city_name=city_name)
        if not series:
            continue
        axis.axhline(
            metric_value(series[0], metric),
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            linestyle='--',
            linewidth=1.15,
            alpha=0.85,
        )
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(axis='y', alpha=0.22)
    if metric == 'wait_density':
        axis.set_yscale('log')


def save_aggregate_plot(points: tuple[EvaluationPoint, ...], output_path: Path) -> None:
    aggregate_points = aggregate_city_means(points)
    iterations = sorted({point.iteration for point in aggregate_points})
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    draw_metric(axes[0], aggregate_points, 'five_city_mean', 'throughput', 'Throughput', 'vehicles/hour')
    draw_metric(axes[1], aggregate_points, 'five_city_mean', 'completion', 'Completion', '% completed')
    draw_metric(
        axes[2],
        aggregate_points,
        'five_city_mean',
        'wait_density',
        'Wait density (log scale)',
        'seconds/meter',
    )
    axes[2].set_xlabel('PPO iteration')
    axes[2].set_xticks(iterations)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    figure.suptitle(f'City PPO evaluation progress through iteration {max(iterations)}')
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def save_per_city_plot(points: tuple[EvaluationPoint, ...], output_path: Path) -> None:
    city_names = tuple(city_name for city_name in CITY_LABELS if any(point.city_name == city_name for point in points))
    iterations = sorted({point.iteration for point in points})
    figure, axes = plt.subplots(len(city_names), 3, figsize=(15.5, 3.0 * len(city_names)), sharex=True)
    metric_settings = (
        ('throughput', 'Throughput', 'vehicles/hour'),
        ('completion', 'Completion', '% completed'),
        ('wait_density', 'Wait density (log)', 'seconds/meter'),
    )
    for row_index, city_name in enumerate(city_names):
        for column_index, (metric, title, y_label) in enumerate(metric_settings):
            axis = axes[row_index, column_index]
            draw_metric(
                axis,
                points,
                city_name,
                metric,
                f'{CITY_LABELS[city_name]} — {title}',
                y_label,
            )
            if row_index == len(city_names) - 1:
                axis.set_xlabel('PPO iteration')
                axis.set_xticks(iterations)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    figure.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    points = load_evaluation_points(arguments.evaluation_root)
    save_aggregate_plot(points, arguments.output_dir / 'city-ppo-progress-aggregate.png')
    save_per_city_plot(points, arguments.output_dir / 'city-ppo-progress-by-city.png')


if __name__ == '__main__':
    main()
