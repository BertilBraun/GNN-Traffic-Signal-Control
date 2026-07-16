from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_grid_generalization_results import (
    EvaluationSeedRecord,
    MetricName,
    _short_scenario_name,
    confidence_interval_half_width,
    coverage_paired_confidence_intervals,
    paired_confidence_intervals,
    replicate_records_for_training_designs,
)


def test_paired_confidence_intervals_pair_policy_and_baseline_by_seed() -> None:
    records = tuple(
        _record(policy=policy, seed=seed, throughput=throughput)
        for policy, seed, throughput in (
            ('learned-greedy', 1, 110.0),
            ('learned-greedy', 2, 130.0),
            ('max-pressure', 1, 100.0),
            ('max-pressure', 2, 100.0),
        )
    )

    intervals = paired_confidence_intervals(
        records=records,
        policy='learned-greedy',
        baseline_policy='max-pressure',
        metrics=(MetricName.THROUGHPUT,),
    )

    assert len(intervals) == 1
    assert intervals[0].pair_count == 2
    assert intervals[0].mean_difference == 20.0
    assert intervals[0].confidence_interval_half_width == pytest.approx(127.06)


def test_single_pair_has_zero_interval_width() -> None:
    assert confidence_interval_half_width((3.0,)) == 0.0


@pytest.mark.parametrize(
    ('scenario_name', 'expected'),
    (
        (
            'coverage_generalization_grid_4x4_validation_signals_06_of_12_mask_01',
            '4x4 validation 06/12 m01',
        ),
        ('matched_grid_5x3', '5x3'),
        ('coverage_grid_6x6_eligible_signals_16_of_32', '6x6_eligible_signals_16_of_32'),
    ),
)
def test_short_scenario_name_compacts_coverage_masks(scenario_name: str, expected: str) -> None:
    assert _short_scenario_name(scenario_name) == expected


def test_paired_intervals_keep_equal_evaluation_seeds_from_training_replicas() -> None:
    records = (
        _record('learned', 1, 110.0, training_replica='5101'),
        _record('max-pressure', 1, 100.0, training_replica='5101'),
        _record('learned', 1, 130.0, training_replica='5102'),
        _record('max-pressure', 1, 100.0, training_replica='5102'),
    )

    intervals = paired_confidence_intervals(
        records=records,
        policy='learned',
        baseline_policy='max-pressure',
        metrics=(MetricName.THROUGHPUT,),
    )

    assert intervals[0].pair_count == 2
    assert intervals[0].mean_difference == 20.0


def test_shared_baseline_records_are_replicated_for_each_training_design() -> None:
    baseline_records = (_record(policy='max-pressure', seed=1, throughput=100.0),)

    replicated = replicate_records_for_training_designs(
        records=baseline_records,
        training_targets=(('3x3', '5101'), ('mixed', '5102')),
    )

    assert tuple(record.training_design for record in replicated) == ('3x3', 'mixed')
    assert tuple(record.training_replica for record in replicated) == ('5101', '5102')
    assert all(record.policy == 'max-pressure' for record in replicated)


def test_coverage_intervals_pool_masks_at_the_same_coverage() -> None:
    records = tuple(
        _record(
            policy=policy,
            seed=seed,
            throughput=throughput,
            city_name=city_name,
        )
        for city_name, policy, seed, throughput in (
            ('coverage_grid_signals_06_of_12_mask_01', 'learned', 1, 110.0),
            ('coverage_grid_signals_06_of_12_mask_01', 'max-pressure', 1, 100.0),
            ('coverage_grid_signals_06_of_12_mask_02', 'learned', 1, 130.0),
            ('coverage_grid_signals_06_of_12_mask_02', 'max-pressure', 1, 100.0),
        )
    )

    intervals = coverage_paired_confidence_intervals(
        records=records,
        policy='learned',
        baseline_policy='max-pressure',
        metrics=(MetricName.THROUGHPUT,),
    )

    assert len(intervals) == 1
    assert intervals[0].city_name == 'signal_coverage_050_percent'
    assert intervals[0].pair_count == 2
    assert intervals[0].mean_difference == 20.0


def _record(
    policy: str,
    seed: int,
    throughput: float,
    training_replica: str = '5101',
    city_name: str = 'matched_grid_6x6_square_validation',
) -> EvaluationSeedRecord:
    return EvaluationSeedRecord(
        training_design='mixed',
        training_replica=training_replica,
        city_name=city_name,
        policy=policy,
        seed=seed,
        demand_scale=0.7,
        completed_vehicles=10,
        departed_vehicles=12,
        completion_rate=10 / 12,
        teleport_count=0,
        throughput_per_hour=throughput,
        average_waiting_time_s=1.0,
        average_wait_density_s_per_m=0.01,
        phase_switch_frequency_per_junction_per_minute=2.0,
    )
