"""Aggregate grid-study evaluations, paired intervals, and paper plots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict


class MetricName(str, Enum):
    THROUGHPUT = 'throughput_per_hour'
    COMPLETION = 'completion_rate'
    WAITING = 'average_waiting_time_s'
    SWITCHES = 'phase_switch_frequency_per_junction_per_minute'
    WAIT_DENSITY = 'average_wait_density_s_per_m'
    TELEPORTS = 'teleport_count'


@dataclass(frozen=True)
class EvaluationSeedRecord:
    training_design: str
    training_replica: str
    city_name: str
    policy: str
    seed: int
    demand_scale: float
    completed_vehicles: int
    departed_vehicles: int
    completion_rate: float
    teleport_count: int
    throughput_per_hour: float
    average_waiting_time_s: float
    average_wait_density_s_per_m: float
    phase_switch_frequency_per_junction_per_minute: float

    def metric(self, metric_name: MetricName) -> float:
        match metric_name:
            case MetricName.THROUGHPUT:
                return self.throughput_per_hour
            case MetricName.COMPLETION:
                return self.completion_rate
            case MetricName.WAITING:
                return self.average_waiting_time_s
            case MetricName.SWITCHES:
                return self.phase_switch_frequency_per_junction_per_minute
            case MetricName.WAIT_DENSITY:
                return self.average_wait_density_s_per_m
            case MetricName.TELEPORTS:
                return float(self.teleport_count)


class PairedConfidenceInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    training_design: str
    city_name: str
    demand_scale: float
    policy: str
    baseline_policy: str
    metric: str
    pair_count: int
    mean_difference: float
    confidence_interval_half_width: float


@dataclass(frozen=True)
class LabeledPath:
    label: str
    replica: str
    path: Path


def load_evaluation_records(
    summary_path: Path,
    training_design: str,
    training_replica: str | None = None,
) -> tuple[EvaluationSeedRecord, ...]:
    with summary_path.open(newline='', encoding='utf-8') as handle:
        rows = tuple(csv.DictReader(handle))
    replica = training_design if training_replica is None else training_replica
    return tuple(
        EvaluationSeedRecord(
            training_design=training_design,
            training_replica=replica,
            city_name=row['city_name'],
            policy=row['policy'],
            seed=int(row['seed']),
            demand_scale=float(row['demand_scale']),
            completed_vehicles=int(float(row['completed_vehicles'])),
            departed_vehicles=int(float(row['departed_vehicles'])),
            completion_rate=float(row['completion_rate']),
            teleport_count=int(float(row['teleport_count'])),
            throughput_per_hour=float(row['throughput_per_hour']),
            average_waiting_time_s=float(row['average_waiting_time_s']),
            average_wait_density_s_per_m=float(row['average_wait_density_s_per_m']),
            phase_switch_frequency_per_junction_per_minute=float(row['phase_switch_frequency_per_junction_per_minute']),
        )
        for row in rows
        if row['row_type'] == 'seed'
    )


def replicate_records_for_training_designs(
    records: Sequence[EvaluationSeedRecord],
    training_targets: Sequence[tuple[str, str]],
) -> tuple[EvaluationSeedRecord, ...]:
    return tuple(
        replace(
            record,
            training_design=training_design,
            training_replica=training_replica,
        )
        for training_design, training_replica in training_targets
        for record in records
    )


def paired_confidence_intervals(
    records: Sequence[EvaluationSeedRecord],
    policy: str,
    baseline_policy: str,
    metrics: tuple[MetricName, ...] = tuple(MetricName),
) -> tuple[PairedConfidenceInterval, ...]:
    grouping_keys = tuple(
        dict.fromkeys(
            (record.training_design, record.city_name, record.demand_scale)
            for record in records
            if record.policy == policy
        )
    )
    intervals: list[PairedConfidenceInterval] = []
    for training_design, city_name, demand_scale in grouping_keys:
        group = tuple(
            record
            for record in records
            if record.training_design == training_design
            and record.city_name == city_name
            and record.demand_scale == demand_scale
        )
        policy_by_pair = {(record.training_replica, record.seed): record for record in group if record.policy == policy}
        baseline_by_pair = {
            (record.training_replica, record.seed): record for record in group if record.policy == baseline_policy
        }
        paired_keys = tuple(sorted(policy_by_pair.keys() & baseline_by_pair.keys()))
        for metric_name in metrics:
            differences = tuple(
                policy_by_pair[pair_key].metric(metric_name) - baseline_by_pair[pair_key].metric(metric_name)
                for pair_key in paired_keys
            )
            if not differences:
                continue
            intervals.append(
                PairedConfidenceInterval(
                    training_design=training_design,
                    city_name=city_name,
                    demand_scale=demand_scale,
                    policy=policy,
                    baseline_policy=baseline_policy,
                    metric=metric_name.value,
                    pair_count=len(paired_keys),
                    mean_difference=statistics.fmean(differences),
                    confidence_interval_half_width=confidence_interval_half_width(differences),
                )
            )
    return tuple(intervals)


def confidence_interval_half_width(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return _student_t_critical_95(len(values) - 1) * standard_error


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    critical_values = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= 0:
        return 0.0
    if degrees_of_freedom <= len(critical_values):
        return critical_values[degrees_of_freedom - 1]
    return 1.96


def write_paired_intervals(
    path: Path,
    intervals: Sequence[PairedConfidenceInterval],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(PairedConfidenceInterval.model_fields.keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interval.model_dump() for interval in intervals)


def plot_training_matrix(
    records: Sequence[EvaluationSeedRecord],
    output_path: Path,
    policy: str,
    baseline_policy: str,
    demand_scale: float,
) -> None:
    filtered = tuple(
        record
        for record in records
        if record.demand_scale == demand_scale and record.policy in (policy, baseline_policy)
    )
    training_designs = tuple(dict.fromkeys(record.training_design for record in filtered))
    cities = tuple(dict.fromkeys(record.city_name for record in filtered))
    matrix = np.full((len(training_designs), len(cities)), np.nan)
    for training_index, training_design in enumerate(training_designs):
        for city_index, city_name in enumerate(cities):
            group = tuple(
                record
                for record in filtered
                if record.training_design == training_design and record.city_name == city_name
            )
            learned = tuple(record.throughput_per_hour for record in group if record.policy == policy)
            baseline = tuple(record.throughput_per_hour for record in group if record.policy == baseline_policy)
            if learned and baseline:
                matrix[training_index, city_index] = statistics.fmean(learned) - statistics.fmean(baseline)
    figure_width = max(8.0, 0.9 * len(cities))
    figure, axis = plt.subplots(figsize=(figure_width, max(3.5, 0.8 * len(training_designs))))
    maximum = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = axis.imshow(matrix, cmap='RdBu', vmin=-maximum, vmax=maximum, aspect='auto')
    axis.set_xticks(range(len(cities)), tuple(_short_scenario_name(city) for city in cities), rotation=45, ha='right')
    axis.set_yticks(range(len(training_designs)), training_designs)
    axis.set_title(f'{policy} throughput difference vs {baseline_policy} at demand {demand_scale:.1f}')
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f'{value:+.0f}', ha='center', va='center', fontsize=8)
    figure.colorbar(image, ax=axis, label='throughput difference (vehicles/hour)')
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_learning_curves(
    learning_runs: Sequence[LabeledPath],
    output_path: Path,
    policy: str,
    demand_scale: float,
    scenario_names: frozenset[str] | None,
) -> None:
    metric_specs: tuple[tuple[MetricName, str], ...] = (
        (MetricName.THROUGHPUT, 'throughput / hour'),
        (MetricName.COMPLETION, 'completion rate'),
        (MetricName.WAITING, 'completed-trip waiting time (s)'),
        (MetricName.SWITCHES, 'switches / junction / minute'),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for learning_run in learning_runs:
        points = _learning_points(
            evaluation_root=learning_run.path,
            training_design=learning_run.label,
            policy=policy,
            demand_scale=demand_scale,
            scenario_names=scenario_names,
        )
        scenario_names = tuple(dict.fromkeys(record.city_name for _iteration, record in points))
        for scenario_name in scenario_names:
            scenario_points = tuple(
                (iteration, record) for iteration, record in points if record.city_name == scenario_name
            )
            iterations = tuple(dict.fromkeys(iteration for iteration, _record in scenario_points))
            for axis, (metric_name, ylabel) in zip(axes.flat, metric_specs, strict=True):
                means = tuple(
                    statistics.fmean(
                        record.metric(metric_name)
                        for point_iteration, record in scenario_points
                        if point_iteration == iteration
                    )
                    for iteration in iterations
                )
                axis.plot(
                    iterations,
                    means,
                    marker='o',
                    markersize=2.5,
                    label=f'{learning_run.label}:{_short_scenario_name(scenario_name)}',
                )
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel('PPO iteration')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.suptitle(f'{policy} development evaluation at demand {demand_scale:.1f}')
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _learning_points(
    evaluation_root: Path,
    training_design: str,
    policy: str,
    demand_scale: float,
    scenario_names: frozenset[str] | None,
) -> tuple[tuple[int, EvaluationSeedRecord], ...]:
    points: list[tuple[int, EvaluationSeedRecord]] = []
    for summary_path in sorted(evaluation_root.glob('iter_*/summary.csv')):
        iteration = int(summary_path.parent.name.removeprefix('iter_'))
        records = load_evaluation_records(summary_path=summary_path, training_design=training_design)
        points.extend(
            (iteration, record)
            for record in records
            if record.policy == policy
            and record.demand_scale == demand_scale
            and (scenario_names is None or record.city_name in scenario_names)
        )
    return tuple(points)


def plot_signal_coverage(
    records: Sequence[EvaluationSeedRecord],
    output_path: Path,
    policies: tuple[str, ...],
    demand_scale: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metric_specs: tuple[tuple[MetricName, str], ...] = (
        (MetricName.THROUGHPUT, 'throughput / hour'),
        (MetricName.COMPLETION, 'completion rate'),
        (MetricName.WAITING, 'completed-trip waiting time (s)'),
        (MetricName.WAIT_DENSITY, 'wait density (s/m)'),
    )
    for policy in policies:
        policy_records = tuple(
            record for record in records if record.policy == policy and record.demand_scale == demand_scale
        )
        coverage_values = tuple(sorted(dict.fromkeys(_signal_coverage(record.city_name) for record in policy_records)))
        for axis, (metric_name, ylabel) in zip(axes.flat, metric_specs, strict=True):
            means = tuple(
                statistics.fmean(
                    record.metric(metric_name)
                    for record in policy_records
                    if _signal_coverage(record.city_name) == coverage
                )
                for coverage in coverage_values
            )
            axis.plot(coverage_values, means, marker='o', label=policy)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel('signal coverage')
    axes[0, 0].legend()
    figure.suptitle(f'Performance versus signal coverage at demand {demand_scale:.1f}')
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _signal_coverage(city_name: str) -> float:
    match = re.search(r'signals_(\d+)_of_(\d+)', city_name)
    if match is None:
        raise ValueError(f'Cannot parse signal coverage from scenario name: {city_name}')
    return int(match.group(1)) / int(match.group(2))


def _short_scenario_name(city_name: str) -> str:
    return city_name.removeprefix('matched_grid_').removeprefix('coverage_grid_')


def _parse_labeled_path(value: str) -> LabeledPath:
    raw_label, separator, raw_path = value.partition('=')
    if not separator or not raw_label or not raw_path:
        raise argparse.ArgumentTypeError('Expected LABEL=PATH.')
    label, replica_separator, replica = raw_label.partition('@')
    if not label or (replica_separator and not replica):
        raise argparse.ArgumentTypeError('Expected LABEL or LABEL@REPLICA before =PATH.')
    return LabeledPath(
        label=label,
        replica=replica if replica_separator else label,
        path=Path(raw_path),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-summary', action='append', type=_parse_labeled_path, default=[])
    parser.add_argument('--matrix-baseline-summary', type=Path, default=None)
    parser.add_argument('--learning-root', action='append', type=_parse_labeled_path, default=[])
    parser.add_argument('--learning-scenario', action='append', default=[])
    parser.add_argument('--coverage-summary', action='append', type=_parse_labeled_path, default=[])
    parser.add_argument('--coverage-baseline-summary', type=Path, default=None)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--policy', default='learned-greedy')
    parser.add_argument('--paired-policy', action='append', default=[])
    parser.add_argument('--baseline-policy', default='max-pressure')
    parser.add_argument('--demand-scale', type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    learned_matrix_records = tuple(
        record
        for labeled_path in arguments.matrix_summary
        for record in load_evaluation_records(
            summary_path=labeled_path.path,
            training_design=labeled_path.label,
            training_replica=labeled_path.replica,
        )
    )
    matrix_records = learned_matrix_records
    if arguments.matrix_baseline_summary is not None:
        baseline_records = load_evaluation_records(
            summary_path=arguments.matrix_baseline_summary,
            training_design='shared-baseline',
        )
        training_targets = tuple(
            dict.fromkeys((record.training_design, record.training_replica) for record in learned_matrix_records)
        )
        matrix_records += replicate_records_for_training_designs(
            records=baseline_records,
            training_targets=training_targets,
        )
    if matrix_records:
        paired_policies = tuple(arguments.paired_policy) or (arguments.policy,)
        intervals = tuple(
            interval
            for policy in paired_policies
            for interval in paired_confidence_intervals(
                records=matrix_records,
                policy=policy,
                baseline_policy=arguments.baseline_policy,
            )
        )
        write_paired_intervals(
            path=arguments.output_directory / 'paired_confidence_intervals.csv',
            intervals=intervals,
        )
        plot_training_matrix(
            records=matrix_records,
            output_path=arguments.output_directory / 'train_shape_by_evaluation_shape.png',
            policy=arguments.policy,
            baseline_policy=arguments.baseline_policy,
            demand_scale=arguments.demand_scale,
        )
    if arguments.learning_root:
        plot_learning_curves(
            learning_runs=arguments.learning_root,
            output_path=arguments.output_directory / 'learning_curves.png',
            policy=arguments.policy,
            demand_scale=arguments.demand_scale,
            scenario_names=(frozenset(arguments.learning_scenario) if arguments.learning_scenario else None),
        )
    if arguments.coverage_summary:
        coverage_records = tuple(
            record
            for labeled_path in arguments.coverage_summary
            for record in load_evaluation_records(
                summary_path=labeled_path.path,
                training_design='coverage',
                training_replica=labeled_path.replica,
            )
        )
        if arguments.coverage_baseline_summary is not None:
            coverage_replicas = tuple(dict.fromkeys(record.training_replica for record in coverage_records))
            coverage_records += replicate_records_for_training_designs(
                records=load_evaluation_records(
                    summary_path=arguments.coverage_baseline_summary,
                    training_design='shared-baseline',
                ),
                training_targets=tuple(('coverage', training_replica) for training_replica in coverage_replicas),
            )
        coverage_intervals = tuple(
            interval
            for policy in (tuple(arguments.paired_policy) or (arguments.policy,))
            for interval in paired_confidence_intervals(
                records=coverage_records,
                policy=policy,
                baseline_policy=arguments.baseline_policy,
            )
        )
        write_paired_intervals(
            path=arguments.output_directory / 'coverage_paired_confidence_intervals.csv',
            intervals=coverage_intervals,
        )
        plot_signal_coverage(
            records=coverage_records,
            output_path=arguments.output_directory / 'performance_vs_signal_coverage.png',
            policies=(arguments.policy, arguments.baseline_policy),
            demand_scale=arguments.demand_scale,
        )


if __name__ == '__main__':
    main()
