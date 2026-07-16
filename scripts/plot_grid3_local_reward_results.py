"""Generate figures for the 3x3 local-reward PPO validation run."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = 'grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01'
DEFAULT_EVALUATION_DIRECTORY = (
    PROJECT_ROOT
    / 'artifacts'
    / 'ppo_runs'
    / 'grid_3x3_local_reward_2hop_iter_0060_repeat_01'
    / 'checkpoints'
    / 'rl'
    / RUN_NAME
    / 'eval'
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / 'docs' / 'results' / 'assets'
EVALUATION_SEED_COUNT = 6
CONFIDENCE_MULTIPLIER = 1.96


class Policy(str, Enum):
    LEARNED = 'learned'
    LEARNED_GREEDY = 'learned-greedy'
    MAX_PRESSURE = 'max-pressure'
    QUEUE = 'queue'
    UNIFORM_RANDOM = 'uniform-random'
    FIXED_TIME = 'fixed-time'


class RowType(str, Enum):
    SEED = 'seed'
    MEAN = 'mean'
    STANDARD_DEVIATION = 'standard_deviation'


class Metric(str, Enum):
    THROUGHPUT = 'throughput_per_hour'
    COMPLETION = 'completion_rate'
    WAITING_TIME = 'average_waiting_time_s'
    WAIT_DENSITY = 'average_wait_density_s_per_m'
    SWITCH_FREQUENCY = 'switches_per_junction_per_minute'


@dataclass(frozen=True)
class Metrics:
    throughput_per_hour: float
    completion_rate: float
    average_waiting_time_s: float
    average_wait_density_s_per_m: float
    switches_per_junction_per_minute: float


@dataclass(frozen=True)
class EvaluationPoint:
    iteration: int
    policy: Policy
    mean: Metrics
    standard_deviation: Metrics


@dataclass(frozen=True)
class SummaryRow:
    policy: Policy
    row_type: RowType
    metrics: Metrics


@dataclass(frozen=True)
class MetricSpec:
    metric: Metric
    title: str
    ylabel: str
    percent: bool = False


POLICY_LABELS = {
    Policy.LEARNED: 'Learned (sampled)',
    Policy.LEARNED_GREEDY: 'Learned (greedy)',
    Policy.MAX_PRESSURE: 'Max pressure',
    Policy.QUEUE: 'Queue',
    Policy.UNIFORM_RANDOM: 'Uniform random',
    Policy.FIXED_TIME: 'Fixed time',
}
POLICY_COLORS = {
    Policy.LEARNED: '#2563EB',
    Policy.LEARNED_GREEDY: '#7C3AED',
    Policy.MAX_PRESSURE: '#EA580C',
    Policy.QUEUE: '#16A34A',
    Policy.UNIFORM_RANDOM: '#DC2626',
    Policy.FIXED_TIME: '#64748B',
}
LEARNING_METRICS = (
    MetricSpec(Metric.THROUGHPUT, 'Throughput', 'vehicles/hour'),
    MetricSpec(Metric.COMPLETION, 'Completion rate', 'completed/departed', percent=True),
    MetricSpec(Metric.WAITING_TIME, 'Completed-trip waiting time', 'seconds'),
    MetricSpec(Metric.SWITCH_FREQUENCY, 'Phase-switch frequency', 'switches/junction/minute'),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def _summary_row(row: dict[str, str]) -> SummaryRow:
    return SummaryRow(
        policy=Policy(row['policy']),
        row_type=RowType(row['row_type']),
        metrics=Metrics(
            throughput_per_hour=float(row['throughput_per_hour']),
            completion_rate=float(row['completion_rate']),
            average_waiting_time_s=float(row['average_waiting_time_s']),
            average_wait_density_s_per_m=float(row['average_wait_density_s_per_m']),
            switches_per_junction_per_minute=float(row['phase_switch_frequency_per_junction_per_minute']),
        ),
    )


def _policy_row(rows: tuple[SummaryRow, ...], policy: Policy, row_type: RowType) -> SummaryRow:
    matches = tuple(row for row in rows if row.policy is policy and row.row_type is row_type)
    if len(matches) != 1:
        raise ValueError(f'Expected one {row_type.value} row for {policy.value}, found {len(matches)}.')
    return matches[0]


def load_evaluation_points(evaluation_directory: Path) -> tuple[EvaluationPoint, ...]:
    points: list[EvaluationPoint] = []
    iteration_directories = sorted(evaluation_directory.glob('iter_*'))
    if not iteration_directories:
        raise ValueError(f'No iteration directories found in {evaluation_directory}.')

    for iteration_directory in iteration_directories:
        try:
            iteration = int(iteration_directory.name.removeprefix('iter_'))
        except ValueError as error:
            raise ValueError(f'Invalid evaluation directory name: {iteration_directory.name}.') from error
        with (iteration_directory / 'summary.csv').open(newline='', encoding='utf-8') as handle:
            rows = tuple(_summary_row(row) for row in csv.DictReader(handle))
        for policy in Policy:
            mean_row = _policy_row(rows, policy, RowType.MEAN)
            standard_deviation_row = _policy_row(rows, policy, RowType.STANDARD_DEVIATION)
            points.append(
                EvaluationPoint(
                    iteration=iteration,
                    policy=policy,
                    mean=mean_row.metrics,
                    standard_deviation=standard_deviation_row.metrics,
                )
            )
    return tuple(points)


def configure_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 120,
            'savefig.dpi': 220,
            'font.size': 10,
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.frameon': False,
        }
    )


def metric_value(metrics: Metrics, metric: MetricSpec) -> float:
    match metric.metric:
        case Metric.THROUGHPUT:
            return metrics.throughput_per_hour
        case Metric.COMPLETION:
            return metrics.completion_rate
        case Metric.WAITING_TIME:
            return metrics.average_waiting_time_s
        case Metric.WAIT_DENSITY:
            return metrics.average_wait_density_s_per_m
        case Metric.SWITCH_FREQUENCY:
            return metrics.switches_per_junction_per_minute


def confidence_interval(standard_deviation: float) -> float:
    return CONFIDENCE_MULTIPLIER * standard_deviation / math.sqrt(EVALUATION_SEED_COUNT)


def policy_points(points: tuple[EvaluationPoint, ...], policy: Policy) -> tuple[EvaluationPoint, ...]:
    return tuple(point for point in points if point.policy is policy)


def save_figure(figure: Figure, output_path: Path) -> None:
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def _plot_learning_series(
    axis: Axes,
    points: tuple[EvaluationPoint, ...],
    policy: Policy,
    metric: MetricSpec,
    *,
    linestyle: str = '-',
    marker: str | None = 'o',
    alpha: float = 1.0,
) -> None:
    series = policy_points(points, policy)
    iterations = tuple(point.iteration for point in series)
    means = tuple(metric_value(point.mean, metric) for point in series)
    confidence_intervals = tuple(
        confidence_interval(metric_value(point.standard_deviation, metric)) for point in series
    )
    axis.plot(
        iterations,
        means,
        color=POLICY_COLORS[policy],
        label=POLICY_LABELS[policy],
        linewidth=2.0,
        linestyle=linestyle,
        marker=marker,
        markersize=3.8,
        alpha=alpha,
    )
    if policy in (Policy.LEARNED, Policy.LEARNED_GREEDY):
        lower = tuple(mean - interval for mean, interval in zip(means, confidence_intervals, strict=True))
        upper = tuple(mean + interval for mean, interval in zip(means, confidence_intervals, strict=True))
        axis.fill_between(iterations, lower, upper, color=POLICY_COLORS[policy], alpha=0.12)


def plot_learning_curves(points: tuple[EvaluationPoint, ...], output_directory: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharex=True)
    for axis, metric in zip(axes.flat, LEARNING_METRICS, strict=True):
        _plot_learning_series(axis, points, Policy.LEARNED, metric)
        _plot_learning_series(axis, points, Policy.LEARNED_GREEDY, metric)
        _plot_learning_series(
            axis,
            points,
            Policy.MAX_PRESSURE,
            metric,
            linestyle='--',
            marker=None,
            alpha=0.9,
        )
        axis.set_title(metric.title)
        axis.set_ylabel(metric.ylabel)
        if metric.percent:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for axis in axes[-1]:
        axis.set_xlabel('PPO iteration')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle('3×3 local-reward PPO learning trajectory', fontweight='bold', y=1.04)
    figure.tight_layout()
    save_figure(figure, output_directory / 'grid3-local-reward-learning-curves.png')


def final_iteration_points(points: tuple[EvaluationPoint, ...]) -> tuple[EvaluationPoint, ...]:
    final_iteration = max(point.iteration for point in points)
    return tuple(point for point in points if point.iteration == final_iteration)


def _point_for_policy(points: tuple[EvaluationPoint, ...], policy: Policy) -> EvaluationPoint:
    matches = tuple(point for point in points if point.policy is policy)
    if len(matches) != 1:
        raise ValueError(f'Expected one final point for {policy.value}, found {len(matches)}.')
    return matches[0]


def plot_final_comparison(points: tuple[EvaluationPoint, ...], output_directory: Path) -> None:
    final_points = final_iteration_points(points)
    policies = (
        Policy.LEARNED,
        Policy.LEARNED_GREEDY,
        Policy.MAX_PRESSURE,
        Policy.FIXED_TIME,
        Policy.UNIFORM_RANDOM,
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for axis, metric in zip(axes.flat, LEARNING_METRICS, strict=True):
        selected_points = tuple(_point_for_policy(final_points, policy) for policy in policies)
        values = tuple(metric_value(point.mean, metric) for point in selected_points)
        errors = tuple(confidence_interval(metric_value(point.standard_deviation, metric)) for point in selected_points)
        labels = tuple(POLICY_LABELS[policy] for policy in policies)
        colors = tuple(POLICY_COLORS[policy] for policy in policies)
        bars = axis.bar(labels, values, yerr=errors, color=colors, alpha=0.88, capsize=3)
        if metric.percent:
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{value:.1%}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
        else:
            axis.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        axis.set_title(metric.title)
        axis.set_ylabel(metric.ylabel)
        axis.tick_params(axis='x', labelrotation=18)
        if metric.percent:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    figure.suptitle('3×3 policy comparison at iteration 60', fontweight='bold')
    figure.tight_layout()
    save_figure(figure, output_directory / 'grid3-local-reward-iteration-0060-comparison.png')


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    points = load_evaluation_points(arguments.evaluation_dir)
    plot_learning_curves(points, arguments.output_dir)
    plot_final_comparison(points, arguments.output_dir)
    print(f'Wrote 3x3 result figures to {arguments.output_dir}')


if __name__ == '__main__':
    main()
