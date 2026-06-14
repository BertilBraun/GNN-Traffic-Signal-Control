"""Evaluation metric schema and serialization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class EvaluationMetrics:
    departed_vehicles: int
    completed_vehicles: int
    vehicles_remaining: int
    completion_rate: float
    teleport_count: int
    throughput_per_hour: float
    average_waiting_time_s: float
    average_travel_time_s: float
    average_time_loss_s: float
    average_queue_length_vehicles: float
    max_queue_length_vehicles: float
    average_wait_density_s_per_m: float
    phase_switch_frequency_per_junction_per_minute: float
    average_tls_passes_per_vehicle: float
    average_stops_before_tls_per_vehicle: float
    nonstop_tls_pass_rate: float
    average_best_nonstop_tls_streak: float
    per_junction_wait_density_s_per_m: dict[str, float]
    per_junction_max_queue_length_vehicles: dict[str, float]
    per_junction_phase_counts: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class EvaluationRecord:
    policy: str
    seed: int
    metrics: EvaluationMetrics


@dataclass(frozen=True)
class EvaluationAggregate:
    policy: str
    seeds: tuple[int, ...]
    mean: EvaluationMetrics
    standard_deviation: EvaluationMetrics


def parse_tripinfo_metrics(
    tripinfo_path: str | Path,
    episode_length_s: int,
) -> tuple[int, float, float, float, float]:
    """Parse SUMO tripinfo XML into completion, throughput, wait, and travel metrics."""
    path = Path(tripinfo_path)
    if not path.exists():
        return 0, 0.0, 0.0, 0.0, 0.0

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return 0, 0.0, 0.0, 0.0, 0.0

    waiting_times: list[float] = []
    travel_times: list[float] = []
    time_losses: list[float] = []
    for trip in root.findall('tripinfo'):
        waiting_times.append(float(trip.attrib.get('waitingTime', '0.0')))
        travel_times.append(float(trip.attrib.get('duration', '0.0')))
        time_losses.append(float(trip.attrib.get('timeLoss', '0.0')))

    completed = len(waiting_times)
    if completed == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    hours = float(episode_length_s) / 3600.0
    return (
        completed,
        completed / hours,
        sum(waiting_times) / completed,
        sum(travel_times) / completed,
        sum(time_losses) / completed,
    )


def aggregate_records(records: Sequence[EvaluationRecord]) -> list[EvaluationAggregate]:
    """Aggregate per-seed records into mean and standard deviation by policy."""
    policies = tuple(dict.fromkeys(record.policy for record in records))
    aggregates: list[EvaluationAggregate] = []
    for policy in policies:
        policy_records = tuple(record for record in records if record.policy == policy)
        aggregates.append(
            EvaluationAggregate(
                policy=policy,
                seeds=tuple(record.seed for record in policy_records),
                mean=_metrics_from_values(policy_records, statistics.fmean),
                standard_deviation=_metrics_from_values(policy_records, _population_standard_deviation),
            )
        )
    return aggregates


def write_records_csv(
    path: str | Path,
    records: Sequence[EvaluationRecord],
    aggregates: Sequence[EvaluationAggregate],
) -> None:
    """Write per-seed and aggregate metrics as a flat CSV table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for record in records:
            writer.writerow(_record_row(record))
        for aggregate in aggregates:
            writer.writerow(_aggregate_row(aggregate, 'mean', aggregate.mean))
            writer.writerow(_aggregate_row(aggregate, 'std', aggregate.standard_deviation))


def write_aggregate_json(
    path: str | Path,
    records: Sequence[EvaluationRecord],
    aggregates: Sequence[EvaluationAggregate],
) -> None:
    """Write evaluation results as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'records': [asdict(record) for record in records],
        'aggregates': [asdict(aggregate) for aggregate in aggregates],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _metrics_from_values(
    records: Sequence[EvaluationRecord],
    reducer: Callable[[Iterable[float]], float],
) -> EvaluationMetrics:
    return EvaluationMetrics(
        departed_vehicles=int(round(reducer(tuple(record.metrics.departed_vehicles for record in records)))),
        completed_vehicles=int(round(reducer(tuple(record.metrics.completed_vehicles for record in records)))),
        vehicles_remaining=int(round(reducer(tuple(record.metrics.vehicles_remaining for record in records)))),
        completion_rate=reducer(tuple(record.metrics.completion_rate for record in records)),
        teleport_count=int(round(reducer(tuple(record.metrics.teleport_count for record in records)))),
        throughput_per_hour=reducer(tuple(record.metrics.throughput_per_hour for record in records)),
        average_waiting_time_s=reducer(tuple(record.metrics.average_waiting_time_s for record in records)),
        average_travel_time_s=reducer(tuple(record.metrics.average_travel_time_s for record in records)),
        average_time_loss_s=reducer(tuple(record.metrics.average_time_loss_s for record in records)),
        average_queue_length_vehicles=reducer(
            tuple(record.metrics.average_queue_length_vehicles for record in records)
        ),
        max_queue_length_vehicles=reducer(tuple(record.metrics.max_queue_length_vehicles for record in records)),
        average_wait_density_s_per_m=reducer(tuple(record.metrics.average_wait_density_s_per_m for record in records)),
        phase_switch_frequency_per_junction_per_minute=reducer(
            tuple(record.metrics.phase_switch_frequency_per_junction_per_minute for record in records)
        ),
        average_tls_passes_per_vehicle=reducer(
            tuple(record.metrics.average_tls_passes_per_vehicle for record in records)
        ),
        average_stops_before_tls_per_vehicle=reducer(
            tuple(record.metrics.average_stops_before_tls_per_vehicle for record in records)
        ),
        nonstop_tls_pass_rate=reducer(tuple(record.metrics.nonstop_tls_pass_rate for record in records)),
        average_best_nonstop_tls_streak=reducer(
            tuple(record.metrics.average_best_nonstop_tls_streak for record in records)
        ),
        per_junction_wait_density_s_per_m=_aggregate_float_maps(
            records,
            reducer,
            lambda metrics: metrics.per_junction_wait_density_s_per_m,
        ),
        per_junction_max_queue_length_vehicles=_aggregate_float_maps(
            records,
            reducer,
            lambda metrics: metrics.per_junction_max_queue_length_vehicles,
        ),
        per_junction_phase_counts=_aggregate_phase_counts(records),
    )


def _population_standard_deviation(values: Iterable[float]) -> float:
    concrete_values = tuple(float(value) for value in values)
    if len(concrete_values) <= 1:
        return 0.0
    return statistics.pstdev(concrete_values)


def _aggregate_float_maps(
    records: Sequence[EvaluationRecord],
    reducer: Callable[[Iterable[float]], float],
    metric_map: Callable[[EvaluationMetrics], Mapping[str, float]],
) -> dict[str, float]:
    keys = tuple(dict.fromkeys(key for record in records for key in metric_map(record.metrics).keys()))
    return {key: reducer(tuple(float(metric_map(record.metrics).get(key, 0.0)) for record in records)) for key in keys}


def _aggregate_phase_counts(records: Sequence[EvaluationRecord]) -> dict[str, tuple[int, ...]]:
    keys = tuple(dict.fromkeys(key for record in records for key in record.metrics.per_junction_phase_counts.keys()))
    aggregate: dict[str, tuple[int, ...]] = {}
    for key in keys:
        width = max(
            (len(record.metrics.per_junction_phase_counts.get(key, ())) for record in records),
            default=0,
        )
        counts = []
        for index in range(width):
            counts.append(
                sum(
                    _phase_count_at(
                        record.metrics.per_junction_phase_counts.get(key, ()),
                        index,
                    )
                    for record in records
                )
            )
        aggregate[key] = tuple(counts)
    return aggregate


def _phase_count_at(counts: tuple[int, ...], index: int) -> int:
    if index >= len(counts):
        return 0
    return counts[index]


def _csv_fieldnames() -> list[str]:
    return [
        'policy',
        'seed',
        'row_type',
        'completed_vehicles',
        'departed_vehicles',
        'vehicles_remaining',
        'completion_rate',
        'teleport_count',
        'throughput_per_hour',
        'average_waiting_time_s',
        'average_travel_time_s',
        'average_time_loss_s',
        'average_queue_length_vehicles',
        'max_queue_length_vehicles',
        'average_wait_density_s_per_m',
        'phase_switch_frequency_per_junction_per_minute',
        'average_tls_passes_per_vehicle',
        'average_stops_before_tls_per_vehicle',
        'nonstop_tls_pass_rate',
        'average_best_nonstop_tls_streak',
    ]


def _record_row(record: EvaluationRecord) -> dict[str, str | int | float]:
    return {
        'policy': record.policy,
        'seed': record.seed,
        'row_type': 'seed',
        **_scalar_metrics(record.metrics),
    }


def _aggregate_row(
    aggregate: EvaluationAggregate,
    row_type: str,
    metrics: EvaluationMetrics,
) -> dict[str, str | int | float]:
    return {
        'policy': aggregate.policy,
        'seed': '',
        'row_type': row_type,
        **_scalar_metrics(metrics),
    }


def _scalar_metrics(metrics: EvaluationMetrics) -> dict[str, int | float]:
    return {
        'departed_vehicles': metrics.departed_vehicles,
        'completed_vehicles': metrics.completed_vehicles,
        'vehicles_remaining': metrics.vehicles_remaining,
        'completion_rate': metrics.completion_rate,
        'teleport_count': metrics.teleport_count,
        'throughput_per_hour': metrics.throughput_per_hour,
        'average_waiting_time_s': metrics.average_waiting_time_s,
        'average_travel_time_s': metrics.average_travel_time_s,
        'average_time_loss_s': metrics.average_time_loss_s,
        'average_queue_length_vehicles': metrics.average_queue_length_vehicles,
        'max_queue_length_vehicles': metrics.max_queue_length_vehicles,
        'average_wait_density_s_per_m': metrics.average_wait_density_s_per_m,
        'phase_switch_frequency_per_junction_per_minute': metrics.phase_switch_frequency_per_junction_per_minute,
        'average_tls_passes_per_vehicle': metrics.average_tls_passes_per_vehicle,
        'average_stops_before_tls_per_vehicle': metrics.average_stops_before_tls_per_vehicle,
        'nonstop_tls_pass_rate': metrics.nonstop_tls_pass_rate,
        'average_best_nonstop_tls_streak': metrics.average_best_nonstop_tls_streak,
    }
