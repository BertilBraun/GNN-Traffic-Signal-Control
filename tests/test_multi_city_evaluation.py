from pathlib import Path
import csv
import json
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import eval_multi_city  # noqa: E402
from src.movement.evaluation import EvaluationPolicy  # noqa: E402
from src.movement.evaluation.metrics import EvaluationMetrics  # noqa: E402
from src.movement.evaluation.multi_city import (  # noqa: E402
    MultiCityEvaluationRunRequest,
    aggregate_multi_city_records,
    run_multi_city_evaluation,
    write_multi_city_csv,
    write_multi_city_json,
)
from src.movement.evaluation.runner import LearnedPolicyConfig  # noqa: E402
from src.movement.experiment_config import CitySplit, load_experiment_configuration  # noqa: E402


def test_multi_city_evaluation_runs_all_cities_and_splits() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )
    requests: list[MultiCityEvaluationRunRequest] = []

    def fake_episode_runner(
        request: MultiCityEvaluationRunRequest,
        learned_policy_config: LearnedPolicyConfig | None,
    ) -> EvaluationMetrics:
        requests.append(request)
        assert learned_policy_config is None
        return _metrics(completed_vehicles=10, departed_vehicles=12)

    result = run_multi_city_evaluation(
        configuration=configuration,
        project_root=ROOT,
        policies=(EvaluationPolicy.MAX_PRESSURE, EvaluationPolicy.QUEUE),
        seeds=(100,),
        steps=300,
        demand_scales=(0.8,),
        learned_policy_config=None,
        episode_runner=fake_episode_runner,
    )

    assert len(result.records) == 10
    assert len(requests) == 10
    assert {record.city_name for record in result.records} == {
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'stuttgart_mitte',
        'heidelberg_bergheim',
        'freiburg_altstadt',
    }
    freiburg_records = tuple(record for record in result.records if record.city_name == 'freiburg_altstadt')
    assert freiburg_records
    assert all(record.city_split == CitySplit.HELD_OUT for record in freiburg_records)
    assert all(
        record.city_split == CitySplit.TRAIN for record in result.records if record.city_name != 'freiburg_altstadt'
    )
    assert all(record.demand_scale == 0.8 for record in result.records)
    assert all(request.steps == 300 for request in requests)


def test_multi_city_learned_policy_requires_checkpoint() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )

    with pytest.raises(ValueError, match='learned_policy_config is required'):
        run_multi_city_evaluation(
            configuration=configuration,
            project_root=ROOT,
            policies=(EvaluationPolicy.LEARNED,),
            seeds=(100,),
            steps=300,
            demand_scales=(1.0,),
            learned_policy_config=None,
            episode_runner=_unused_episode_runner,
        )


def test_multi_city_writers_include_city_split_policy_seed_and_demand(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )
    result = run_multi_city_evaluation(
        configuration=configuration,
        project_root=ROOT,
        policies=(EvaluationPolicy.MAX_PRESSURE,),
        seeds=(100, 101),
        steps=300,
        demand_scales=(0.8,),
        learned_policy_config=None,
        episode_runner=_constant_episode_runner,
    )

    csv_path = tmp_path / 'summary.csv'
    json_path = tmp_path / 'summary.json'
    write_multi_city_csv(path=csv_path, result=result)
    write_multi_city_json(path=json_path, result=result)

    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = tuple(csv.DictReader(handle))
    seed_rows = tuple(row for row in rows if row['row_type'] == 'seed')
    assert seed_rows[0]['city_name'] == 'karlsruhe_oststadt'
    assert seed_rows[0]['city_split'] == 'train'
    assert seed_rows[0]['policy'] == 'max-pressure'
    assert seed_rows[0]['seed'] == '100'
    assert seed_rows[0]['demand_scale'] == '0.8'
    assert any(row['city_name'] == 'freiburg_altstadt' and row['city_split'] == 'held_out' for row in seed_rows)

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert payload['records'][0]['city_name'] == 'karlsruhe_oststadt'
    assert payload['records'][0]['city_split'] == 'train'
    assert payload['records'][0]['demand_scale'] == 0.8
    assert payload['aggregates'][0]['policy'] == 'max-pressure'


def test_multi_city_aggregation_groups_by_city_policy_and_demand() -> None:
    configuration = load_experiment_configuration(
        configuration_path=ROOT / 'configs' / 'training' / 'city_first_pass.yaml',
        project_root=ROOT,
    )
    result = run_multi_city_evaluation(
        configuration=configuration,
        project_root=ROOT,
        policies=(EvaluationPolicy.MAX_PRESSURE,),
        seeds=(100, 101),
        steps=300,
        demand_scales=(0.8, 1.0),
        learned_policy_config=None,
        episode_runner=_constant_episode_runner,
    )

    aggregates = aggregate_multi_city_records(result.records)

    assert len(aggregates) == 10
    assert {aggregate.demand_scale for aggregate in aggregates} == {0.8, 1.0}
    assert all(aggregate.policy == EvaluationPolicy.MAX_PRESSURE.value for aggregate in aggregates)
    assert all(aggregate.seeds == (100, 101) for aggregate in aggregates)


def test_eval_multi_city_learned_checkpoint_rules() -> None:
    assert (
        eval_multi_city._learned_policy_config(
            policies=(EvaluationPolicy.MAX_PRESSURE,),
            checkpoint_path=None,
            device='cpu',
        )
        is None
    )

    with pytest.raises(SystemExit, match='--checkpoint is required'):
        eval_multi_city._learned_policy_config(
            policies=(EvaluationPolicy.LEARNED,),
            checkpoint_path=None,
            device='cpu',
        )


def _constant_episode_runner(
    request: MultiCityEvaluationRunRequest,
    learned_policy_config: LearnedPolicyConfig | None,
) -> EvaluationMetrics:
    return _metrics(completed_vehicles=request.seed, departed_vehicles=request.seed + 10)


def _unused_episode_runner(
    request: MultiCityEvaluationRunRequest,
    learned_policy_config: LearnedPolicyConfig | None,
) -> EvaluationMetrics:
    raise AssertionError('episode runner should not be called')


def _metrics(completed_vehicles: int, departed_vehicles: int) -> EvaluationMetrics:
    return EvaluationMetrics(
        departed_vehicles=departed_vehicles,
        completed_vehicles=completed_vehicles,
        vehicles_remaining=departed_vehicles - completed_vehicles,
        completion_rate=completed_vehicles / departed_vehicles,
        teleport_count=0,
        throughput_per_hour=float(completed_vehicles),
        average_waiting_time_s=1.0,
        average_travel_time_s=2.0,
        average_time_loss_s=3.0,
        average_queue_length_vehicles=4.0,
        max_queue_length_vehicles=5.0,
        average_wait_density_s_per_m=6.0,
        phase_switch_frequency_per_junction_per_minute=7.0,
        average_tls_passes_per_vehicle=8.0,
        average_stops_before_tls_per_vehicle=9.0,
        nonstop_tls_pass_rate=0.5,
        average_best_nonstop_tls_streak=10.0,
        per_junction_wait_density_s_per_m={},
        per_junction_max_queue_length_vehicles={},
        per_junction_phase_counts={},
    )
