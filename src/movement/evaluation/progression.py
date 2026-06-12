"""Vehicle progression metrics around signalized junctions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GreenWaveTracker:
    approach_distance_m: float
    stop_speed_mps: float
    seen_vehicle_ids: set[str] = field(default_factory=set)
    completed_vehicle_ids: set[str] = field(default_factory=set)
    current_tls_by_vehicle: dict[str, str] = field(default_factory=dict)
    stopped_on_approach_by_vehicle: dict[str, bool] = field(default_factory=dict)
    tls_passes_by_vehicle: dict[str, int] = field(default_factory=dict)
    nonstop_tls_passes_by_vehicle: dict[str, int] = field(default_factory=dict)
    current_nonstop_streak_by_vehicle: dict[str, int] = field(default_factory=dict)
    best_nonstop_streak_by_vehicle: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        vehicle_ids: tuple[str, ...],
        next_tls_by_vehicle: dict[str, tuple[object, ...]],
        speed_by_vehicle: dict[str, float],
        arrived_vehicle_ids: tuple[str, ...],
    ) -> None:
        for vehicle_id in vehicle_ids:
            self.seen_vehicle_ids.add(vehicle_id)
            tls_id, distance_m = _first_tls(next_tls_by_vehicle.get(vehicle_id, ()))
            previous_tls_id = self.current_tls_by_vehicle.get(vehicle_id)
            if previous_tls_id is not None and tls_id != previous_tls_id:
                self._record_pass(vehicle_id)
                self._clear_current(vehicle_id)
            if tls_id is not None and vehicle_id not in self.current_tls_by_vehicle:
                self.current_tls_by_vehicle[vehicle_id] = tls_id
                self.stopped_on_approach_by_vehicle[vehicle_id] = False
            if tls_id is not None and distance_m is not None:
                speed_mps = speed_by_vehicle.get(vehicle_id, float('inf'))
                if distance_m <= self.approach_distance_m and speed_mps <= self.stop_speed_mps:
                    self.stopped_on_approach_by_vehicle[vehicle_id] = True

        for vehicle_id in arrived_vehicle_ids:
            self.completed_vehicle_ids.add(vehicle_id)
            if vehicle_id in self.current_tls_by_vehicle:
                self._record_pass(vehicle_id)
                self._clear_current(vehicle_id)

    def metric_values(self) -> tuple[float, float, float, float]:
        vehicle_ids = self.completed_vehicle_ids or self.seen_vehicle_ids
        if not vehicle_ids:
            return 0.0, 0.0, 0.0, 0.0
        total_passes = sum(self.tls_passes_by_vehicle.get(vehicle_id, 0) for vehicle_id in vehicle_ids)
        total_nonstop = sum(self.nonstop_tls_passes_by_vehicle.get(vehicle_id, 0) for vehicle_id in vehicle_ids)
        total_stops = total_passes - total_nonstop
        total_best_streak = sum(self.best_nonstop_streak_by_vehicle.get(vehicle_id, 0) for vehicle_id in vehicle_ids)
        vehicle_count = len(vehicle_ids)
        nonstop_rate = total_nonstop / total_passes if total_passes > 0 else 0.0
        return (
            total_passes / vehicle_count,
            total_stops / vehicle_count,
            nonstop_rate,
            total_best_streak / vehicle_count,
        )

    def _record_pass(self, vehicle_id: str) -> None:
        stopped = self.stopped_on_approach_by_vehicle.get(vehicle_id, False)
        self.tls_passes_by_vehicle[vehicle_id] = self.tls_passes_by_vehicle.get(vehicle_id, 0) + 1
        if stopped:
            self.current_nonstop_streak_by_vehicle[vehicle_id] = 0
            return
        self.nonstop_tls_passes_by_vehicle[vehicle_id] = self.nonstop_tls_passes_by_vehicle.get(vehicle_id, 0) + 1
        streak = self.current_nonstop_streak_by_vehicle.get(vehicle_id, 0) + 1
        self.current_nonstop_streak_by_vehicle[vehicle_id] = streak
        self.best_nonstop_streak_by_vehicle[vehicle_id] = max(
            self.best_nonstop_streak_by_vehicle.get(vehicle_id, 0),
            streak,
        )

    def _clear_current(self, vehicle_id: str) -> None:
        self.current_tls_by_vehicle.pop(vehicle_id, None)
        self.stopped_on_approach_by_vehicle.pop(vehicle_id, None)


def _first_tls(next_tls: tuple[object, ...]) -> tuple[str | None, float | None]:
    if not next_tls:
        return None, None
    first_tls = next_tls[0]
    try:
        return str(first_tls[0]), float(first_tls[2])
    except (IndexError, TypeError, ValueError):
        return None, None
