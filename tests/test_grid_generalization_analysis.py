from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_grid_generalization_results import (
    EvaluationSeedRecord,
    MetricName,
    confidence_interval_half_width,
    paired_confidence_intervals,
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


def _record(
    policy: str,
    seed: int,
    throughput: float,
) -> EvaluationSeedRecord:
    return EvaluationSeedRecord(
        training_design='mixed',
        city_name='matched_grid_6x6_square_validation',
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
