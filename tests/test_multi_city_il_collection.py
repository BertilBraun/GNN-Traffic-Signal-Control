from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import (  # noqa: E402
    MovementDatasetSample,
    MovementEdgeIndices,
    StoredPhaseIncidence,
    load_jsonl_samples,
    save_jsonl_samples,
)
from src.movement.experiment_config import CitySplit, load_experiment_configuration  # noqa: E402
from src.movement.training.il.multi_city_collection import (  # noqa: E402
    MultiCityCollectionJob,
    MultiCityCollectionJobResult,
    MultiCityCollectionSettings,
    build_balanced_collection_jobs,
    collect_multi_city_samples,
    run_collection_job,
)
from src.movement.training.il import multi_city_collection  # noqa: E402


def test_balanced_collection_jobs_round_robin_train_cities_only(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )

    jobs = build_balanced_collection_jobs(
        configuration=configuration,
        settings=_settings(output_dir=tmp_path, samples_per_city=5, samples_per_simulation=2, sample_stride=3),
    )

    assert len(jobs) == 12
    assert tuple(job.city_name for job in jobs[:4]) == (
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
    )
    assert 'freiburg_altstadt' not in {job.city_name for job in jobs}
    assert all(job.city_split == CitySplit.TRAIN for job in jobs)
    assert [job.retained_sample_target for job in jobs[0::4]] == [2, 2, 1]
    assert tuple(job.job_index for job in jobs) == tuple(range(12))
    assert all(job.sample_stride == 3 for job in jobs)
    assert all(0.8 <= job.demand_scale <= 1.2 for job in jobs)


def test_multi_city_collection_writes_balanced_outputs_with_stride_metadata(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )

    result = collect_multi_city_samples(
        configuration=configuration,
        settings=_settings(output_dir=tmp_path, samples_per_city=3, samples_per_simulation=2, sample_stride=3),
        job_runner=_fake_collection_job_runner,
    )

    assert result.combined_output_path == tmp_path / 'combined.jsonl'
    assert tuple(summary.sample_count for summary in result.city_summaries) == (3, 3, 3, 3)
    assert tuple(summary.city_name for summary in result.city_summaries) == (
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
    )

    combined_samples = load_jsonl_samples(result.combined_output_path)
    assert len(combined_samples) == 12
    assert {sample.metadata['city_split'] for sample in combined_samples} == {'train'}
    assert {sample.metadata['sample_stride'] for sample in combined_samples} == {3}
    assert 'freiburg_altstadt' not in {sample.metadata['city_name'] for sample in combined_samples}

    first_city_samples = load_jsonl_samples(tmp_path / 'karlsruhe_oststadt.jsonl')
    assert [sample.metadata['source_decision_index'] for sample in first_city_samples] == [0, 3, 0]
    assert all(sample.metadata['city_name'] == 'karlsruhe_oststadt' for sample in first_city_samples)
    assert all('simulation_time_s' in sample.metadata for sample in first_city_samples)


@pytest.mark.parametrize(
    ('samples_per_city', 'samples_per_simulation', 'sample_stride', 'workers', 'expected_message'),
    (
        (0, 1, 1, 1, 'samples_per_city must be positive'),
        (1, 0, 1, 1, 'samples_per_simulation must be positive'),
        (1, 1, 0, 1, 'sample_stride must be positive'),
        (1, 1, 1, 0, 'workers must be positive'),
    ),
)
def test_collection_settings_validate_positive_values(
    tmp_path: Path,
    samples_per_city: int,
    samples_per_simulation: int,
    sample_stride: int,
    workers: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        MultiCityCollectionSettings(
            project_root=ROOT,
            output_dir=tmp_path,
            samples_per_city=samples_per_city,
            samples_per_simulation=samples_per_simulation,
            collection_seed=42,
            sample_stride=sample_stride,
            workers=workers,
        )


def test_collection_job_failures_include_city_seed_and_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )
    job = build_balanced_collection_jobs(
        configuration=configuration,
        settings=_settings(output_dir=tmp_path, samples_per_city=1, samples_per_simulation=1, sample_stride=3),
    )[0]

    def failing_collect_samples(
        cfg_path: Path,
        output_path: Path,
        steps: int,
        decision_interval: int,
        seed: int,
        gui: bool,
        demand_scale: float,
        initial_occupancy: float,
        time_to_teleport: int | None,
    ) -> int:
        raise RuntimeError('sumo failed')

    monkeypatch.setattr(multi_city_collection, 'collect_samples', failing_collect_samples)

    with pytest.raises(RuntimeError, match=f'city={job.city_name} seed={job.seed} demand_scale='):
        run_collection_job(job)


def _settings(
    output_dir: Path,
    samples_per_city: int,
    samples_per_simulation: int,
    sample_stride: int,
) -> MultiCityCollectionSettings:
    return MultiCityCollectionSettings(
        project_root=ROOT,
        output_dir=output_dir,
        samples_per_city=samples_per_city,
        samples_per_simulation=samples_per_simulation,
        collection_seed=42,
        sample_stride=sample_stride,
        workers=1,
    )


def _fake_collection_job_runner(job: MultiCityCollectionJob) -> MultiCityCollectionJobResult:
    raw_sample_count = job.retained_sample_target * job.sample_stride
    save_jsonl_samples(
        job.raw_output_path,
        (_sample(job=job, raw_sample_index=raw_sample_index) for raw_sample_index in range(raw_sample_count)),
    )
    return MultiCityCollectionJobResult(
        job_index=job.job_index,
        city_name=job.city_name,
        city_split=job.city_split,
        seed=job.seed,
        demand_scale=job.demand_scale,
        initial_occupancy=job.initial_occupancy,
        retained_sample_count=job.retained_sample_target,
        raw_sample_count=raw_sample_count,
        raw_output_path=job.raw_output_path,
    )


def _sample(job: MultiCityCollectionJob, raw_sample_index: int) -> MovementDatasetSample:
    return MovementDatasetSample(
        x_lane=((1.0, 2.0),),
        x_movement=((3.0, 4.0),),
        edge_indices=MovementEdgeIndices(
            input_lane_to_movement=((0, 0),),
            output_lane_to_movement=((0, 0),),
            movement_to_input_lane=((0, 0),),
            movement_to_output_lane=((0, 0),),
        ),
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0,),
                movement_ids=(0,),
                rows=((1,),),
            )
        },
        teacher_movement_scores=(1.0,),
        teacher_selected_phase_by_tls={'J0': 0},
        metadata={
            'simulation_time_s': raw_sample_index * job.decision_interval,
            'raw_sample_index': raw_sample_index,
        },
    )
