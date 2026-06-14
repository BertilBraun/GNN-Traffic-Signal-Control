"""Console display helpers for movement-policy evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter

from src.movement.evaluation.metrics import EvaluationAggregate, EvaluationMetrics


class SummaryRow:
    def __init__(
        self,
        label: str,
        value: Callable[[EvaluationMetrics], float | int],
        precision: int,
        suffix: str,
    ) -> None:
        self.label = label
        self.value = value
        self.precision = precision
        self.suffix = suffix

    def format_value(self, value: float | int) -> str:
        if self.precision == 0:
            return f'{int(round(float(value)))}{self.suffix}'
        return f'{float(value):.{self.precision}f}{self.suffix}'


def current_timer_s() -> float:
    """Return a monotonic timestamp for evaluation timing."""
    return perf_counter()


def print_evaluation_start(
    policy: str,
    seed: int,
    run_index: int,
    total_runs: int,
) -> None:
    """Print one evaluation run start line."""
    print(f'[{run_index}/{total_runs}] policy={policy:<12} seed={seed:<5} running...')


def print_evaluation_result(
    policy: str,
    seed: int,
    metrics: EvaluationMetrics,
    run_index: int,
    total_runs: int,
    run_elapsed_s: float,
    batch_started_s: float,
) -> None:
    """Print one evaluation run result line with timing and key metrics."""
    elapsed_s = current_timer_s() - batch_started_s
    eta_s = _estimated_remaining_s(
        elapsed_s=elapsed_s,
        completed_runs=run_index,
        total_runs=total_runs,
    )
    print(
        f'[{run_index}/{total_runs}] '
        f'policy={policy:<12} seed={seed:<5} '
        f'done={_format_duration(run_elapsed_s):>7} '
        f'eta={_format_duration(eta_s):>7} | '
        f'completed={metrics.completed_vehicles:<5d} '
        f'finish={100.0 * metrics.completion_rate:>5.1f}% '
        f'active={metrics.vehicles_remaining:<4d} '
        f'tele={metrics.teleport_count:<3d} '
        f'throughput={metrics.throughput_per_hour:>7.1f}/h '
        f'wait={metrics.average_waiting_time_s:>6.2f}s '
        f'travel={metrics.average_travel_time_s:>6.2f}s '
        f'queue={metrics.average_queue_length_vehicles:>5.2f} '
        f'maxq={metrics.max_queue_length_vehicles:>4.1f} '
        f'wd={metrics.average_wait_density_s_per_m:>7.4f} '
        f'sw={metrics.phase_switch_frequency_per_junction_per_minute:>5.2f}/j/min'
    )


def print_aggregate_metric_table(
    title: str,
    aggregates: Sequence[EvaluationAggregate],
) -> None:
    """Print a metric-by-policy summary table."""
    if not aggregates:
        return
    print('')
    print(title)
    print(f'{"metric":<34}' + ''.join(f'{aggregate.policy:>16}' for aggregate in aggregates))
    print('-' * (34 + 16 * len(aggregates)))
    for row in _summary_rows():
        print(
            f'{row.label:<34}'
            + ''.join(f'{row.format_value(row.value(aggregate.mean)):>16}' for aggregate in aggregates)
        )
    print('')


def _summary_rows() -> tuple[SummaryRow, ...]:
    return (
        SummaryRow('completed vehicles', lambda metrics: metrics.completed_vehicles, 0, ''),
        SummaryRow('departed vehicles', lambda metrics: metrics.departed_vehicles, 0, ''),
        SummaryRow('vehicles remaining at end', lambda metrics: metrics.vehicles_remaining, 0, ''),
        SummaryRow('completion rate', lambda metrics: 100.0 * metrics.completion_rate, 1, '%'),
        SummaryRow('teleported vehicles', lambda metrics: metrics.teleport_count, 0, ''),
        SummaryRow('throughput vehicles/hour', lambda metrics: metrics.throughput_per_hour, 1, ''),
        SummaryRow('average waiting time', lambda metrics: metrics.average_waiting_time_s, 2, ' s'),
        SummaryRow('average travel time', lambda metrics: metrics.average_travel_time_s, 2, ' s'),
        SummaryRow('average time loss', lambda metrics: metrics.average_time_loss_s, 2, ' s'),
        SummaryRow('average queue length', lambda metrics: metrics.average_queue_length_vehicles, 2, ''),
        SummaryRow('max queue length', lambda metrics: metrics.max_queue_length_vehicles, 1, ''),
        SummaryRow('average wait density', lambda metrics: metrics.average_wait_density_s_per_m, 4, ' s/m'),
        SummaryRow(
            'switch frequency',
            lambda metrics: metrics.phase_switch_frequency_per_junction_per_minute,
            2,
            ' /j/min',
        ),
        SummaryRow('TLS passes per vehicle', lambda metrics: metrics.average_tls_passes_per_vehicle, 2, ''),
        SummaryRow('TLS stops per vehicle', lambda metrics: metrics.average_stops_before_tls_per_vehicle, 2, ''),
        SummaryRow('nonstop TLS pass rate', lambda metrics: 100.0 * metrics.nonstop_tls_pass_rate, 1, '%'),
        SummaryRow('best nonstop TLS streak', lambda metrics: metrics.average_best_nonstop_tls_streak, 2, ''),
    )


def _estimated_remaining_s(
    elapsed_s: float,
    completed_runs: int,
    total_runs: int,
) -> float:
    if completed_runs <= 0:
        return 0.0
    remaining_runs = max(0, total_runs - completed_runs)
    return (elapsed_s / completed_runs) * remaining_runs


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, seconds_remainder = divmod(rounded_seconds, 60)
    hours, minutes_remainder = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}:{minutes_remainder:02d}:{seconds_remainder:02d}'
    return f'{minutes_remainder:d}:{seconds_remainder:02d}'
