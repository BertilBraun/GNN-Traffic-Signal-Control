"""Balanced multi-city imitation-learning sample collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
import tempfile

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.collect_il_data import collect_samples
from src.movement.dataset import MovementDatasetSample, load_jsonl_samples, save_jsonl_samples
from src.movement.experiment_config import (
    CitySplit,
    ExperimentCityConfiguration,
    ExperimentConfiguration,
    resolve_experiment_path,
)
from src.movement.initial_traffic import sample_target_occupancy


class MultiCityCollectionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_root: Path
    output_dir: Path
    samples_per_city: int
    samples_per_simulation: int
    collection_seed: int
    sample_stride: int
    workers: int

    @field_validator('samples_per_city')
    @classmethod
    def validate_samples_per_city(cls, samples_per_city: int) -> int:
        if samples_per_city <= 0:
            raise ValueError('samples_per_city must be positive')
        return samples_per_city

    @field_validator('samples_per_simulation')
    @classmethod
    def validate_samples_per_simulation(cls, samples_per_simulation: int) -> int:
        if samples_per_simulation <= 0:
            raise ValueError('samples_per_simulation must be positive')
        return samples_per_simulation

    @field_validator('sample_stride')
    @classmethod
    def validate_sample_stride(cls, sample_stride: int) -> int:
        if sample_stride <= 0:
            raise ValueError('sample_stride must be positive')
        return sample_stride

    @field_validator('workers')
    @classmethod
    def validate_workers(cls, workers: int) -> int:
        if workers <= 0:
            raise ValueError('workers must be positive')
        return workers


@dataclass(frozen=True)
class MultiCityCollectionJob:
    job_index: int
    city_name: str
    city_split: CitySplit
    sumo_config_path: Path
    retained_sample_target: int
    decision_interval: int
    seed: int
    demand_scale: float
    initial_occupancy: float
    sample_stride: int
    time_to_teleport: int | None
    raw_output_path: Path


@dataclass(frozen=True)
class MultiCityCollectionJobResult:
    job_index: int
    city_name: str
    city_split: CitySplit
    seed: int
    demand_scale: float
    initial_occupancy: float
    retained_sample_count: int
    raw_sample_count: int
    raw_output_path: Path


@dataclass(frozen=True)
class CityCollectionSummary:
    city_name: str
    city_split: CitySplit
    sample_count: int
    output_path: Path


@dataclass(frozen=True)
class MultiCityCollectionResult:
    combined_output_path: Path
    city_summaries: tuple[CityCollectionSummary, ...]
    job_results: tuple[MultiCityCollectionJobResult, ...]


class CollectedSampleMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    city_name: str
    city_split: str
    demand_scale: float
    initial_occupancy: float
    collection_seed: int
    sample_stride: int = Field(gt=0)
    source_decision_index: int = Field(ge=0)


CollectionJobRunner = Callable[[MultiCityCollectionJob], MultiCityCollectionJobResult]


def collect_multi_city_samples(
    configuration: ExperimentConfiguration,
    settings: MultiCityCollectionSettings,
    job_runner: CollectionJobRunner,
) -> MultiCityCollectionResult:
    jobs = build_balanced_collection_jobs(configuration=configuration, settings=settings)
    return _collect_jobs_to_outputs(settings=settings, jobs=jobs, job_runner=job_runner)


def build_balanced_collection_jobs(
    configuration: ExperimentConfiguration,
    settings: MultiCityCollectionSettings,
) -> tuple[MultiCityCollectionJob, ...]:
    train_cities = configuration.train_cities
    if not train_cities:
        raise ValueError('at least one training city is required for IL collection')

    city_targets = tuple(
        _city_job_targets(settings.samples_per_city, settings.samples_per_simulation) for _ in train_cities
    )
    maximum_jobs_per_city = max(len(targets) for targets in city_targets)
    jobs: list[MultiCityCollectionJob] = []
    job_index = 0
    for simulation_index in range(maximum_jobs_per_city):
        for city, targets in zip(train_cities, city_targets):
            if simulation_index >= len(targets):
                continue
            seed = settings.collection_seed + job_index
            jobs.append(
                _collection_job(
                    city=city,
                    configuration=configuration,
                    settings=settings,
                    retained_sample_target=targets[simulation_index],
                    seed=seed,
                    job_index=job_index,
                )
            )
            job_index += 1
    return tuple(jobs)


def run_collection_job(job: MultiCityCollectionJob) -> MultiCityCollectionJobResult:
    try:
        raw_sample_target = job.retained_sample_target * job.sample_stride
        raw_sample_count = collect_samples(
            cfg_path=job.sumo_config_path,
            output_path=job.raw_output_path,
            steps=raw_sample_target * job.decision_interval,
            decision_interval=job.decision_interval,
            seed=job.seed,
            gui=False,
            demand_scale=job.demand_scale,
            initial_occupancy=job.initial_occupancy,
            time_to_teleport=job.time_to_teleport,
        )
        retained_count = len(_retained_samples(samples=load_jsonl_samples(job.raw_output_path), job=job))
        return MultiCityCollectionJobResult(
            job_index=job.job_index,
            city_name=job.city_name,
            city_split=job.city_split,
            seed=job.seed,
            demand_scale=job.demand_scale,
            initial_occupancy=job.initial_occupancy,
            retained_sample_count=retained_count,
            raw_sample_count=raw_sample_count,
            raw_output_path=job.raw_output_path,
        )
    except Exception as error:
        raise RuntimeError(
            f'IL collection failed for city={job.city_name} seed={job.seed} demand_scale={job.demand_scale:.3f}'
        ) from error


def write_multi_city_collection_outputs(
    output_dir: Path,
    job_results: Sequence[MultiCityCollectionJobResult],
    sample_stride: int,
) -> MultiCityCollectionResult:
    if sample_stride <= 0:
        raise ValueError('sample_stride must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    city_names = _ordered_city_names(job_results)
    city_summaries: list[CityCollectionSummary] = []
    combined_samples: list[MovementDatasetSample] = []
    for city_name in city_names:
        city_results = tuple(result for result in job_results if result.city_name == city_name)
        city_samples: list[MovementDatasetSample] = []
        for result in city_results:
            raw_samples = load_jsonl_samples(result.raw_output_path)
            city_samples.extend(
                _retained_samples_from_result(samples=raw_samples, result=result, sample_stride=sample_stride)
            )
        city_output_path = output_dir / f'{city_name}.jsonl'
        save_jsonl_samples(city_output_path, city_samples)
        combined_samples.extend(city_samples)
        city_summaries.append(
            CityCollectionSummary(
                city_name=city_name,
                city_split=city_results[0].city_split,
                sample_count=len(city_samples),
                output_path=city_output_path,
            )
        )

    combined_output_path = output_dir / 'combined.jsonl'
    save_jsonl_samples(combined_output_path, combined_samples)
    return MultiCityCollectionResult(
        combined_output_path=combined_output_path,
        city_summaries=tuple(city_summaries),
        job_results=tuple(job_results),
    )


def collect_multi_city_samples_to_directory(
    configuration: ExperimentConfiguration,
    settings: MultiCityCollectionSettings,
) -> MultiCityCollectionResult:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='movement_multi_city_il_') as temporary_directory:
        temporary_root = Path(temporary_directory)
        jobs = tuple(
            replace(job, raw_output_path=temporary_root / job.raw_output_path.name)
            for job in build_balanced_collection_jobs(configuration=configuration, settings=settings)
        )
        return _collect_jobs_to_outputs(settings=settings, jobs=jobs, job_runner=run_collection_job)


def _collect_jobs_to_outputs(
    settings: MultiCityCollectionSettings,
    jobs: tuple[MultiCityCollectionJob, ...],
    job_runner: CollectionJobRunner,
) -> MultiCityCollectionResult:
    if settings.workers == 1:
        job_results = tuple(job_runner(job) for job in jobs)
    else:
        job_results = _run_jobs_in_parallel(jobs=jobs, workers=settings.workers, job_runner=job_runner)
    return write_multi_city_collection_outputs(
        output_dir=settings.output_dir,
        job_results=job_results,
        sample_stride=settings.sample_stride,
    )


def _run_jobs_in_parallel(
    jobs: tuple[MultiCityCollectionJob, ...],
    workers: int,
    job_runner: CollectionJobRunner,
) -> tuple[MultiCityCollectionJobResult, ...]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = tuple(pool.submit(job_runner, job) for job in jobs)
        results = tuple(future.result() for future in as_completed(futures))
    return tuple(sorted(results, key=lambda result: result.job_index))


def _collection_job(
    city: ExperimentCityConfiguration,
    configuration: ExperimentConfiguration,
    settings: MultiCityCollectionSettings,
    retained_sample_target: int,
    seed: int,
    job_index: int,
) -> MultiCityCollectionJob:
    return MultiCityCollectionJob(
        job_index=job_index,
        city_name=city.name,
        city_split=city.split,
        sumo_config_path=resolve_experiment_path(path=city.sumo_config, project_root=settings.project_root),
        retained_sample_target=retained_sample_target,
        decision_interval=configuration.simulation.decision_interval,
        seed=seed,
        demand_scale=_sample_demand_scale(
            minimum_scale=configuration.demand.minimum_train_scale,
            maximum_scale=configuration.demand.maximum_train_scale,
            seed=seed,
        ),
        initial_occupancy=sample_target_occupancy(
            minimum_occupancy=configuration.simulation.minimum_initial_occupancy,
            maximum_occupancy=configuration.simulation.maximum_initial_occupancy,
            seed=seed,
        ),
        sample_stride=settings.sample_stride,
        time_to_teleport=configuration.simulation.time_to_teleport,
        raw_output_path=settings.output_dir / '_jobs' / f'{job_index:05d}_{city.name}_seed_{seed}.jsonl',
    )


def _city_job_targets(samples_per_city: int, samples_per_simulation: int) -> tuple[int, ...]:
    remaining_samples = samples_per_city
    targets: list[int] = []
    while remaining_samples > 0:
        target = min(samples_per_simulation, remaining_samples)
        targets.append(target)
        remaining_samples -= target
    return tuple(targets)


def _retained_samples(
    samples: Sequence[MovementDatasetSample],
    job: MultiCityCollectionJob,
) -> tuple[MovementDatasetSample, ...]:
    return _retained_samples_with_metadata(
        samples=samples,
        city_name=job.city_name,
        city_split=job.city_split,
        demand_scale=job.demand_scale,
        initial_occupancy=job.initial_occupancy,
        collection_seed=job.seed,
        sample_stride=job.sample_stride,
        retained_sample_count=job.retained_sample_target,
    )


def _retained_samples_from_result(
    samples: Sequence[MovementDatasetSample],
    result: MultiCityCollectionJobResult,
    sample_stride: int,
) -> tuple[MovementDatasetSample, ...]:
    return _retained_samples_with_metadata(
        samples=samples,
        city_name=result.city_name,
        city_split=result.city_split,
        demand_scale=result.demand_scale,
        initial_occupancy=result.initial_occupancy,
        collection_seed=result.seed,
        sample_stride=sample_stride,
        retained_sample_count=result.retained_sample_count,
    )


def _retained_samples_with_metadata(
    samples: Sequence[MovementDatasetSample],
    city_name: str,
    city_split: CitySplit,
    demand_scale: float,
    initial_occupancy: float,
    collection_seed: int,
    sample_stride: int,
    retained_sample_count: int,
) -> tuple[MovementDatasetSample, ...]:
    retained_samples = tuple(samples[index] for index in range(0, len(samples), sample_stride))
    return tuple(
        _with_collection_metadata(
            sample=sample,
            city_name=city_name,
            city_split=city_split,
            demand_scale=demand_scale,
            initial_occupancy=initial_occupancy,
            collection_seed=collection_seed,
            sample_stride=sample_stride,
            source_decision_index=source_decision_index,
        )
        for source_decision_index, sample in zip(range(0, len(samples), sample_stride), retained_samples)
    )[:retained_sample_count]


def _with_collection_metadata(
    sample: MovementDatasetSample,
    city_name: str,
    city_split: CitySplit,
    demand_scale: float,
    initial_occupancy: float,
    collection_seed: int,
    sample_stride: int,
    source_decision_index: int,
) -> MovementDatasetSample:
    metadata = CollectedSampleMetadata(
        city_name=city_name,
        city_split=city_split.value,
        demand_scale=demand_scale,
        initial_occupancy=initial_occupancy,
        collection_seed=collection_seed,
        sample_stride=sample_stride,
        source_decision_index=source_decision_index,
    )
    return replace(
        sample,
        metadata={
            **sample.metadata,
            **metadata.model_dump(mode='json'),
        },
    )


def _ordered_city_names(job_results: Sequence[MultiCityCollectionJobResult]) -> tuple[str, ...]:
    city_names: list[str] = []
    for result in job_results:
        if result.city_name not in city_names:
            city_names.append(result.city_name)
    return tuple(city_names)


def _sample_demand_scale(minimum_scale: float, maximum_scale: float, seed: int) -> float:
    if minimum_scale == maximum_scale:
        return minimum_scale
    return Random(seed).uniform(minimum_scale, maximum_scale)
