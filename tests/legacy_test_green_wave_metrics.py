from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.eval_episode import GreenWaveTracker


def test_green_wave_tracker_counts_nonstop_tls_passes() -> None:
    tracker = GreenWaveTracker(approach_distance_m=150.0, stop_speed_mps=0.1)

    tracker.update(
        vehicle_ids=["veh0"],
        next_tls_by_vehicle={"veh0": [("J0", 0, 100.0, "G")]},
        speed_by_vehicle={"veh0": 8.0},
        arrived_ids=[],
    )
    tracker.update(
        vehicle_ids=["veh0"],
        next_tls_by_vehicle={"veh0": [("J1", 0, 120.0, "G")]},
        speed_by_vehicle={"veh0": 8.0},
        arrived_ids=[],
    )
    tracker.update(
        vehicle_ids=[],
        next_tls_by_vehicle={},
        speed_by_vehicle={},
        arrived_ids=["veh0"],
    )

    metrics = tracker.metrics()

    assert metrics["avg_tls_passes_per_vehicle"] == 2.0
    assert metrics["avg_stops_before_tls_per_vehicle"] == 0.0
    assert metrics["nonstop_tls_pass_rate"] == 1.0
    assert metrics["avg_best_nonstop_tls_streak"] == 2.0


def test_green_wave_tracker_counts_stopped_tls_passes() -> None:
    tracker = GreenWaveTracker(approach_distance_m=150.0, stop_speed_mps=0.1)

    tracker.update(
        vehicle_ids=["veh0"],
        next_tls_by_vehicle={"veh0": [("J0", 0, 80.0, "G")]},
        speed_by_vehicle={"veh0": 0.0},
        arrived_ids=[],
    )
    tracker.update(
        vehicle_ids=[],
        next_tls_by_vehicle={},
        speed_by_vehicle={},
        arrived_ids=["veh0"],
    )

    metrics = tracker.metrics()

    assert metrics["avg_tls_passes_per_vehicle"] == 1.0
    assert metrics["avg_stops_before_tls_per_vehicle"] == 1.0
    assert metrics["nonstop_tls_pass_rate"] == 0.0
    assert metrics["avg_best_nonstop_tls_streak"] == 0.0
