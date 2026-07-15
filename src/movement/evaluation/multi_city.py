"""Multi-city evaluation orchestration for movement policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from src.movement.evaluation.display import current_timer_s, print_evaluation_result
from src.movement.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationRecord,
    aggregate_records,
)
from src.movement.evaluation.runner import EvaluationPolicy, LearnedPolicyConfig, run_evaluation_episode
from src.movement.experiment_config import (
    CitySplit,
    ExperimentCityConfiguration,
    ExperimentConfiguration,
    resolve_experiment_path,
)
from src.movement.sumo_backend import SumoBackendKind

BASELINE_POLICY_IMPLEMENTATION_VERSION = 3


@dataclass(frozen=True)
class MultiCityEvaluationRunRequest:
    city_name: str
    city_split: CitySplit
    sumo_config_path: Path
    policy: EvaluationPolicy
    seed: int
    demand_scale: float
    steps: int
    decision_interval: int
    yellow_duration: int
    yellow_start_delay: int
    minimum_green_steps: int
    fixed_time_phase_duration: int
    queue_pressure_phase_duration: int
    minimum_initial_occupancy: float
    maximum_initial_occupancy: float
    time_to_teleport: int | None
    backend_kind: SumoBackendKind = SumoBackendKind.TRACI


@dataclass(frozen=True)
class MultiCityEvaluationRecord:
    city_name: str
    city_split: CitySplit
    policy: str
    seed: int
    demand_scale: float
    metrics: EvaluationMetrics


@dataclass(frozen=True)
class MultiCityEvaluationRunResult:
    run_index: int
    total_runs: int
    record: MultiCityEvaluationRecord
    elapsed_s: float


@dataclass(frozen=True)
class MultiCityEvaluationAggregate:
    city_name: str
    city_split: CitySplit
    policy: str
    demand_scale: float
    seeds: tuple[int, ...]
    mean: EvaluationMetrics
    standard_deviation: EvaluationMetrics


@dataclass(frozen=True)
class MultiCityEvaluationResult:
    records: tuple[MultiCityEvaluationRecord, ...]
    aggregates: tuple[MultiCityEvaluationAggregate, ...]


EpisodeRunner = Callable[[MultiCityEvaluationRunRequest, LearnedPolicyConfig | None], EvaluationMetrics]


@dataclass(frozen=True)
class FileCachedEpisodeRunner:
    cache_dir: Path
    episode_runner: EpisodeRunner

    def __call__(
        self,
        request: MultiCityEvaluationRunRequest,
        learned_policy_config: LearnedPolicyConfig | None,
    ) -> EvaluationMetrics:
        if request.policy == EvaluationPolicy.LEARNED:
            return self.episode_runner(request, learned_policy_config)
        key = MultiCityEvaluationCacheKey.from_request(request)
        path = self.cache_dir / f'{key.sha256()}.json'
        if path.exists():
            return MultiCityEvaluationCacheEntry.model_validate_json(
                path.read_text(encoding='utf-8')
            ).metrics_dataclass()
        metrics = self.episode_runner(request, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            MultiCityEvaluationCacheEntry(
                key=key,
                metrics=EvaluationMetricsOutput.model_validate(metrics),
            ).model_dump_json(indent=2),
            encoding='utf-8',
        )
        return metrics


class MultiCityEvaluationCacheKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    city_split: str
    sumo_config_path: str
    policy: str
    seed: int
    demand_scale: float
    steps: int
    decision_interval: int
    yellow_duration: int
    yellow_start_delay: int
    minimum_green_steps: int
    fixed_time_phase_duration: int
    queue_pressure_phase_duration: int
    minimum_initial_occupancy: float
    maximum_initial_occupancy: float
    time_to_teleport: int | None
    backend_kind: str
    baseline_policy_implementation_version: int

    @classmethod
    def from_request(cls, request: MultiCityEvaluationRunRequest) -> 'MultiCityEvaluationCacheKey':
        return cls(
            city_name=request.city_name,
            city_split=request.city_split.value,
            sumo_config_path=str(request.sumo_config_path.resolve()),
            policy=request.policy.value,
            seed=request.seed,
            demand_scale=request.demand_scale,
            steps=request.steps,
            decision_interval=request.decision_interval,
            yellow_duration=request.yellow_duration,
            yellow_start_delay=request.yellow_start_delay,
            minimum_green_steps=request.minimum_green_steps,
            fixed_time_phase_duration=request.fixed_time_phase_duration,
            queue_pressure_phase_duration=request.queue_pressure_phase_duration,
            minimum_initial_occupancy=request.minimum_initial_occupancy,
            maximum_initial_occupancy=request.maximum_initial_occupancy,
            time_to_teleport=request.time_to_teleport,
            backend_kind=request.backend_kind.value,
            baseline_policy_implementation_version=BASELINE_POLICY_IMPLEMENTATION_VERSION,
        )

    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
        return sha256(payload.encode('utf-8')).hexdigest()


class EvaluationMetricsOutput(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

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


class MultiCityEvaluationCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: MultiCityEvaluationCacheKey
    metrics: EvaluationMetricsOutput

    def metrics_dataclass(self) -> EvaluationMetrics:
        return EvaluationMetrics(**self.metrics.model_dump())


class MultiCityEvaluationRecordOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    city_split: str
    policy: str
    seed: int
    demand_scale: float
    metrics: EvaluationMetricsOutput


class MultiCityEvaluationAggregateOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    city_split: str
    policy: str
    demand_scale: float
    seeds: tuple[int, ...]
    mean: EvaluationMetricsOutput
    standard_deviation: EvaluationMetricsOutput


class MultiCityEvaluationJsonOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    records: tuple[MultiCityEvaluationRecordOutput, ...]
    aggregates: tuple[MultiCityEvaluationAggregateOutput, ...]


def run_multi_city_evaluation(
    configuration: ExperimentConfiguration,
    project_root: Path,
    policies: tuple[EvaluationPolicy, ...],
    seeds: tuple[int, ...],
    steps: int,
    demand_scales: tuple[float, ...],
    learned_policy_config: LearnedPolicyConfig | None,
    episode_runner: EpisodeRunner,
    backend_kind: SumoBackendKind = SumoBackendKind.TRACI,
    worker_count: int = 1,
) -> MultiCityEvaluationResult:
    if not policies:
        raise ValueError('at least one evaluation policy is required')
    if not seeds:
        raise ValueError('at least one evaluation seed is required')
    if not demand_scales:
        raise ValueError('at least one evaluation demand scale is required')
    if EvaluationPolicy.LEARNED in policies and learned_policy_config is None:
        raise ValueError('learned_policy_config is required when evaluating the learned policy')

    requests = _build_run_requests(
        configuration=configuration,
        project_root=project_root,
        policies=policies,
        seeds=seeds,
        steps=steps,
        demand_scales=demand_scales,
        backend_kind=backend_kind,
    )
    if worker_count <= 0:
        raise ValueError('worker_count must be positive')
    if worker_count > 1:
        return run_parallel_multi_city_evaluation(
            requests=requests,
            learned_policy_config=learned_policy_config,
            episode_runner=episode_runner,
            worker_count=worker_count,
        )
    records: list[MultiCityEvaluationRecord] = []
    total_runs = len(requests)
    batch_started_s = current_timer_s()
    for run_index, request in enumerate(requests, start=1):
        run_started_s = current_timer_s()
        print_multi_city_evaluation_start(
            request=request,
            run_index=run_index,
            total_runs=total_runs,
        )
        metrics = episode_runner(request, learned_policy_config if request.policy == EvaluationPolicy.LEARNED else None)
        records.append(
            MultiCityEvaluationRecord(
                city_name=request.city_name,
                city_split=request.city_split,
                policy=request.policy.value,
                seed=request.seed,
                demand_scale=request.demand_scale,
                metrics=metrics,
            )
        )
        print_evaluation_result(
            policy=request.policy.value,
            seed=request.seed,
            metrics=metrics,
            run_index=run_index,
            total_runs=total_runs,
            run_elapsed_s=current_timer_s() - run_started_s,
            batch_started_s=batch_started_s,
        )

    concrete_records = tuple(records)
    return MultiCityEvaluationResult(
        records=concrete_records,
        aggregates=aggregate_multi_city_records(concrete_records),
    )


def run_parallel_multi_city_evaluation(
    requests: tuple[MultiCityEvaluationRunRequest, ...],
    learned_policy_config: LearnedPolicyConfig | None,
    episode_runner: EpisodeRunner,
    worker_count: int,
) -> MultiCityEvaluationResult:
    records_by_run_index: dict[int, MultiCityEvaluationRecord] = {}
    total_runs = len(requests)
    batch_started_s = current_timer_s()
    multiprocessing_context = get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(worker_count, total_runs), mp_context=multiprocessing_context) as pool:
        futures = {
            pool.submit(
                run_multi_city_evaluation_request,
                run_index,
                total_runs,
                request,
                learned_policy_config if request.policy == EvaluationPolicy.LEARNED else None,
                episode_runner,
            ): request
            for run_index, request in enumerate(requests, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            records_by_run_index[result.run_index] = result.record
            print_evaluation_result(
                policy=result.record.policy,
                seed=result.record.seed,
                metrics=result.record.metrics,
                run_index=result.run_index,
                total_runs=result.total_runs,
                run_elapsed_s=result.elapsed_s,
                batch_started_s=batch_started_s,
            )
    records = tuple(records_by_run_index[run_index] for run_index in range(1, total_runs + 1))
    return MultiCityEvaluationResult(
        records=records,
        aggregates=aggregate_multi_city_records(records),
    )


def run_multi_city_evaluation_request(
    run_index: int,
    total_runs: int,
    request: MultiCityEvaluationRunRequest,
    learned_policy_config: LearnedPolicyConfig | None,
    episode_runner: EpisodeRunner,
) -> MultiCityEvaluationRunResult:
    run_started_s = current_timer_s()
    metrics = episode_runner(request, learned_policy_config)
    return MultiCityEvaluationRunResult(
        run_index=run_index,
        total_runs=total_runs,
        record=MultiCityEvaluationRecord(
            city_name=request.city_name,
            city_split=request.city_split,
            policy=request.policy.value,
            seed=request.seed,
            demand_scale=request.demand_scale,
            metrics=metrics,
        ),
        elapsed_s=current_timer_s() - run_started_s,
    )


def default_episode_runner(
    request: MultiCityEvaluationRunRequest,
    learned_policy_config: LearnedPolicyConfig | None,
) -> EvaluationMetrics:
    return run_evaluation_episode(
        cfg_path=request.sumo_config_path,
        policy=request.policy,
        seed=request.seed,
        steps=request.steps,
        decision_interval=request.decision_interval,
        yellow_duration=request.yellow_duration,
        yellow_start_delay=request.yellow_start_delay,
        min_green_steps=request.minimum_green_steps,
        learned_policy_config=learned_policy_config,
        demand_scale=request.demand_scale,
        initial_occupancy_min=request.minimum_initial_occupancy,
        initial_occupancy_max=request.maximum_initial_occupancy,
        time_to_teleport=request.time_to_teleport,
        fixed_time_phase_duration=request.fixed_time_phase_duration,
        queue_pressure_phase_duration=request.queue_pressure_phase_duration,
        backend_kind=request.backend_kind,
    )


def aggregate_multi_city_records(
    records: Sequence[MultiCityEvaluationRecord],
) -> tuple[MultiCityEvaluationAggregate, ...]:
    grouping_keys: list[tuple[str, CitySplit, str, float]] = []
    for record in records:
        grouping_key = (record.city_name, record.city_split, record.policy, record.demand_scale)
        if grouping_key not in grouping_keys:
            grouping_keys.append(grouping_key)
    aggregates: list[MultiCityEvaluationAggregate] = []
    for city_name, city_split, policy, demand_scale in grouping_keys:
        group_records = tuple(
            record
            for record in records
            if record.city_name == city_name
            and record.city_split == city_split
            and record.policy == policy
            and record.demand_scale == demand_scale
        )
        single_city_records = tuple(
            EvaluationRecord(policy=record.policy, seed=record.seed, metrics=record.metrics) for record in group_records
        )
        single_city_aggregate = aggregate_records(single_city_records)[0]
        aggregates.append(
            MultiCityEvaluationAggregate(
                city_name=city_name,
                city_split=city_split,
                policy=policy,
                demand_scale=demand_scale,
                seeds=single_city_aggregate.seeds,
                mean=single_city_aggregate.mean,
                standard_deviation=single_city_aggregate.standard_deviation,
            )
        )
    return tuple(aggregates)


def write_multi_city_json(
    path: Path,
    result: MultiCityEvaluationResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = MultiCityEvaluationJsonOutput(
        records=tuple(_record_output(record) for record in result.records),
        aggregates=tuple(_aggregate_output(aggregate) for aggregate in result.aggregates),
    )
    path.write_text(payload.model_dump_json(indent=2), encoding='utf-8')


def write_multi_city_csv(
    path: Path,
    result: MultiCityEvaluationResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(_csv_header())
        for record in result.records:
            writer.writerow(_record_csv_row(record))
        for aggregate in result.aggregates:
            writer.writerow(_aggregate_csv_row(aggregate, 'mean', aggregate.mean))
            writer.writerow(_aggregate_csv_row(aggregate, 'standard_deviation', aggregate.standard_deviation))


def write_multi_city_tensorboard(
    log_dir: Path,
    aggregates: Sequence[MultiCityEvaluationAggregate],
) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    for aggregate_index, aggregate in enumerate(aggregates):
        demand_tag = _demand_scale_tag(aggregate.demand_scale)
        tag_prefix = f'eval/{aggregate.city_split.value}/{aggregate.city_name}/{aggregate.policy}/{demand_tag}'
        writer.add_scalar(f'{tag_prefix}/throughput_per_hour', aggregate.mean.throughput_per_hour, aggregate_index)
        writer.add_scalar(f'{tag_prefix}/completion_rate', aggregate.mean.completion_rate, aggregate_index)
        writer.add_scalar(
            f'{tag_prefix}/average_wait_density_s_per_m',
            aggregate.mean.average_wait_density_s_per_m,
            aggregate_index,
        )
        writer.add_scalar(f'{tag_prefix}/teleport_count', aggregate.mean.teleport_count, aggregate_index)
    writer.close()


def print_multi_city_summary(
    aggregates: Sequence[MultiCityEvaluationAggregate],
) -> None:
    if not aggregates:
        return
    print('')
    print('Multi-city evaluation summary')
    print(
        f'{"city":<26} {"split":<8} {"policy":<12} {"demand":>8} '
        f'{"finish":>8} {"teleports":>9} {"throughput":>11} {"wait_density":>13}'
    )
    print('-' * 103)
    for aggregate in aggregates:
        metrics = aggregate.mean
        print(
            f'{aggregate.city_name:<26} '
            f'{aggregate.city_split.value:<8} '
            f'{aggregate.policy:<12} '
            f'{aggregate.demand_scale:>8.2f} '
            f'{100.0 * metrics.completion_rate:>7.1f}% '
            f'{metrics.teleport_count:>9d} '
            f'{metrics.throughput_per_hour:>11.1f} '
            f'{metrics.average_wait_density_s_per_m:>13.4f}'
        )
    print('')


def print_multi_city_evaluation_start(
    request: MultiCityEvaluationRunRequest,
    run_index: int,
    total_runs: int,
) -> None:
    print(
        f'[{run_index}/{total_runs}] '
        f'city={request.city_name:<24} '
        f'split={request.city_split.value:<8} '
        f'policy={request.policy.value:<12} '
        f'seed={request.seed:<5} '
        f'demand={request.demand_scale:<4.2f} running...'
    )


def _build_run_requests(
    configuration: ExperimentConfiguration,
    project_root: Path,
    policies: tuple[EvaluationPolicy, ...],
    seeds: tuple[int, ...],
    steps: int,
    demand_scales: tuple[float, ...],
    backend_kind: SumoBackendKind,
) -> tuple[MultiCityEvaluationRunRequest, ...]:
    requests: list[MultiCityEvaluationRunRequest] = []
    for city in configuration.cities:
        for policy in policies:
            for seed in seeds:
                for demand_scale in demand_scales:
                    requests.append(
                        _run_request(
                            city=city,
                            configuration=configuration,
                            project_root=project_root,
                            policy=policy,
                            seed=seed,
                            steps=steps,
                            demand_scale=demand_scale,
                            backend_kind=backend_kind,
                        )
                    )
    return tuple(requests)


def _run_request(
    city: ExperimentCityConfiguration,
    configuration: ExperimentConfiguration,
    project_root: Path,
    policy: EvaluationPolicy,
    seed: int,
    steps: int,
    demand_scale: float,
    backend_kind: SumoBackendKind,
) -> MultiCityEvaluationRunRequest:
    return MultiCityEvaluationRunRequest(
        city_name=city.name,
        city_split=city.split,
        sumo_config_path=resolve_experiment_path(path=city.sumo_config, project_root=project_root),
        policy=policy,
        seed=seed,
        demand_scale=demand_scale,
        steps=steps,
        decision_interval=configuration.simulation.decision_interval,
        yellow_duration=configuration.simulation.yellow_duration,
        yellow_start_delay=configuration.simulation.yellow_start_delay,
        minimum_green_steps=configuration.simulation.minimum_green_steps,
        fixed_time_phase_duration=configuration.evaluation.fixed_time_phase_duration,
        queue_pressure_phase_duration=configuration.evaluation.queue_pressure_phase_duration,
        minimum_initial_occupancy=configuration.simulation.minimum_initial_occupancy,
        maximum_initial_occupancy=configuration.simulation.maximum_initial_occupancy,
        time_to_teleport=configuration.simulation.time_to_teleport,
        backend_kind=backend_kind,
    )


def _record_output(record: MultiCityEvaluationRecord) -> MultiCityEvaluationRecordOutput:
    return MultiCityEvaluationRecordOutput(
        city_name=record.city_name,
        city_split=record.city_split.value,
        policy=record.policy,
        seed=record.seed,
        demand_scale=record.demand_scale,
        metrics=EvaluationMetricsOutput.model_validate(record.metrics),
    )


def _aggregate_output(aggregate: MultiCityEvaluationAggregate) -> MultiCityEvaluationAggregateOutput:
    return MultiCityEvaluationAggregateOutput(
        city_name=aggregate.city_name,
        city_split=aggregate.city_split.value,
        policy=aggregate.policy,
        demand_scale=aggregate.demand_scale,
        seeds=aggregate.seeds,
        mean=EvaluationMetricsOutput.model_validate(aggregate.mean),
        standard_deviation=EvaluationMetricsOutput.model_validate(aggregate.standard_deviation),
    )


def _csv_header() -> tuple[str, ...]:
    return (
        'city_name',
        'city_split',
        'policy',
        'seed',
        'demand_scale',
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
    )


def _record_csv_row(record: MultiCityEvaluationRecord) -> tuple[str | int | float, ...]:
    return (
        record.city_name,
        record.city_split.value,
        record.policy,
        record.seed,
        record.demand_scale,
        'seed',
        *_metric_csv_values(record.metrics),
    )


def _aggregate_csv_row(
    aggregate: MultiCityEvaluationAggregate,
    row_type: str,
    metrics: EvaluationMetrics,
) -> tuple[str | int | float, ...]:
    return (
        aggregate.city_name,
        aggregate.city_split.value,
        aggregate.policy,
        '',
        aggregate.demand_scale,
        row_type,
        *_metric_csv_values(metrics),
    )


def _metric_csv_values(metrics: EvaluationMetrics) -> tuple[int | float, ...]:
    return (
        metrics.completed_vehicles,
        metrics.departed_vehicles,
        metrics.vehicles_remaining,
        metrics.completion_rate,
        metrics.teleport_count,
        metrics.throughput_per_hour,
        metrics.average_waiting_time_s,
        metrics.average_travel_time_s,
        metrics.average_time_loss_s,
        metrics.average_queue_length_vehicles,
        metrics.max_queue_length_vehicles,
        metrics.average_wait_density_s_per_m,
        metrics.phase_switch_frequency_per_junction_per_minute,
        metrics.average_tls_passes_per_vehicle,
        metrics.average_stops_before_tls_per_vehicle,
        metrics.nonstop_tls_pass_rate,
        metrics.average_best_nonstop_tls_streak,
    )


def _demand_scale_tag(demand_scale: float) -> str:
    return f'demand_{demand_scale:.3f}'.replace('.', '_')
