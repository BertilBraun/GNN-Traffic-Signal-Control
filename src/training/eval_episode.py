"""Evaluation episode runner for IL and RL training stages.

Runs one full simulation episode under a given policy, collects traffic
metrics, and returns them in an EvalMetrics dataclass.  SUMO's
``--tripinfo-output`` flag is used to obtain per-vehicle waiting time and
travel time; all other metrics are accumulated in-sim via TraCI.

Exported symbols
----------------
EvalMetrics         — all metrics from one evaluation episode
run_eval_episode    — runs one episode, closes SUMO, returns EvalMetrics
make_model_policy   — wraps GATPolicy → policy_fn(obs) → actions
make_expert_policy  — wraps GreedyExpert → policy_fn(obs) → actions

Lifecycle
---------
run_eval_episode starts a fresh SUMO process (via env.reset()), runs to
completion, then calls env.close() so SUMO flushes the tripinfo file before
parsing.  After it returns env._started is False; the caller's next
env.reset() starts a clean SUMO process.

    expert.reset()
    metrics = run_eval_episode(
        env, make_expert_policy(expert), junction_ids, junction_infos,
        on_step=lambda a: [expert.notify_applied(j, p) for j, p in a.items()],
    )
    # env is now closed; next env.reset() starts fresh SUMO
"""
from __future__ import annotations

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.sumo_env import TrafficEnv
    from src.environment.expert import GreedyExpert
    from src.model.gat_policy import GATPolicy
    from src.utils.graph_builder import GraphBuilder

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Point it to your SUMO installation directory."
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

from src.environment.phase_schema import phase_indices

# ---------------------------------------------------------------------------
# Green-wave tracker
# ---------------------------------------------------------------------------

class GreenWaveTracker:
    """Track whether vehicles pass signalized junctions without stopping.

    A pass is counted when a vehicle's next TLS changes from one junction to
    another, or when it arrives while still tracking an upcoming TLS.  A pass
    is "nonstop" if the vehicle did not fall below ``stop_speed_mps`` within
    ``approach_distance_m`` of that TLS while it was being tracked.
    """

    def __init__(self, approach_distance_m: float = 150.0, stop_speed_mps: float = 0.1) -> None:
        self.approach_distance_m = approach_distance_m
        self.stop_speed_mps = stop_speed_mps
        self._seen: set[str] = set()
        self._completed: set[str] = set()
        self._current_tls: dict[str, str] = {}
        self._stopped_on_approach: dict[str, bool] = {}
        self._passes: dict[str, int] = {}
        self._nonstop_passes: dict[str, int] = {}
        self._current_streak: dict[str, int] = {}
        self._best_streak: dict[str, int] = {}

    def update(
        self,
        vehicle_ids: list[str],
        next_tls_by_vehicle: dict[str, list],
        speed_by_vehicle: dict[str, float],
        arrived_ids: list[str],
    ) -> None:
        for vid in vehicle_ids:
            self._seen.add(vid)
            next_tls = next_tls_by_vehicle.get(vid, [])
            tls_id, distance = self._first_tls(next_tls)
            prev_tls = self._current_tls.get(vid)

            if prev_tls is not None and tls_id != prev_tls:
                self._record_pass(vid)
                self._clear_current(vid)

            if tls_id is not None and self._current_tls.get(vid) is None:
                self._current_tls[vid] = tls_id
                self._stopped_on_approach[vid] = False

            if tls_id is not None and distance is not None:
                speed = speed_by_vehicle.get(vid, float("inf"))
                if distance <= self.approach_distance_m and speed <= self.stop_speed_mps:
                    self._stopped_on_approach[vid] = True

        for vid in arrived_ids:
            self._completed.add(vid)
            if vid in self._current_tls:
                self._record_pass(vid)
                self._clear_current(vid)

    def metrics(self) -> dict[str, float]:
        vehicles = self._completed if self._completed else self._seen
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return {
                "avg_tls_passes_per_vehicle": 0.0,
                "avg_stops_before_tls_per_vehicle": 0.0,
                "nonstop_tls_pass_rate": 0.0,
                "avg_best_nonstop_tls_streak": 0.0,
            }

        total_passes = sum(self._passes.get(vid, 0) for vid in vehicles)
        total_nonstop = sum(self._nonstop_passes.get(vid, 0) for vid in vehicles)
        total_stops = total_passes - total_nonstop
        total_best_streak = sum(self._best_streak.get(vid, 0) for vid in vehicles)

        return {
            "avg_tls_passes_per_vehicle": total_passes / n_vehicles,
            "avg_stops_before_tls_per_vehicle": total_stops / n_vehicles,
            "nonstop_tls_pass_rate": total_nonstop / total_passes if total_passes else 0.0,
            "avg_best_nonstop_tls_streak": total_best_streak / n_vehicles,
        }

    @staticmethod
    def _first_tls(next_tls: list) -> tuple[str | None, float | None]:
        if not next_tls:
            return None, None
        first = next_tls[0]
        try:
            return str(first[0]), float(first[2])
        except (IndexError, TypeError, ValueError):
            return None, None

    def _record_pass(self, vid: str) -> None:
        stopped = self._stopped_on_approach.get(vid, False)
        self._passes[vid] = self._passes.get(vid, 0) + 1
        if stopped:
            self._current_streak[vid] = 0
            return

        self._nonstop_passes[vid] = self._nonstop_passes.get(vid, 0) + 1
        streak = self._current_streak.get(vid, 0) + 1
        self._current_streak[vid] = streak
        self._best_streak[vid] = max(self._best_streak.get(vid, 0), streak)

    def _clear_current(self, vid: str) -> None:
        self._current_tls.pop(vid, None)
        self._stopped_on_approach.pop(vid, None)

# ---------------------------------------------------------------------------
# EvalMetrics
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """All traffic metrics collected from one evaluation episode.

    Tripinfo-derived (per-vehicle, only completed trips)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    avg_waiting_time     : float   — mean accumulated wait per vehicle (s)
    avg_travel_time      : float   — mean door-to-door trip duration (s)
    throughput_per_hour  : float   — completed trips / simulated hour

    In-sim (all vehicles / all lanes)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    max_queue_length     : float   — peak halting-vehicle count on any single
                                     approach lane during the episode
    phase_switch_freq    : float   — mean phase switches per junction per minute
    avg_wait_density     : float   — mean global wait density (s/m) — same unit
                                     as the RL reward signal, for reference

    Per-junction breakdowns (keyed by junction_id)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    per_junction_wait_density  : dict[str, float]      — avg wait density (s/m)
    per_junction_max_queue     : dict[str, int]        — peak halting vehicles
                                                          on any approach lane
    per_junction_phase_counts  : dict[str, list[int]]  — phase-selection counts
                                                          [ph0, ..., ph7]
    """
    # Tripinfo
    avg_waiting_time:    float
    avg_travel_time:     float
    throughput_per_hour: float
    # In-sim
    max_queue_length:    float
    phase_switch_freq:   float
    avg_wait_density:    float
    # Per-junction
    per_junction_wait_density:  dict[str, float]      = field(default_factory=dict)
    per_junction_max_queue:     dict[str, int]        = field(default_factory=dict)
    per_junction_phase_counts:  dict[str, list[int]]  = field(default_factory=dict)
    # Green-wave / progression
    avg_tls_passes_per_vehicle:       float = 0.0
    avg_stops_before_tls_per_vehicle: float = 0.0
    nonstop_tls_pass_rate:             float = 0.0
    avg_best_nonstop_tls_streak:       float = 0.0


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------

def make_model_policy(
    model: GATPolicy,
    builder: GraphBuilder,
    device: str | torch.device,
) -> Callable[[dict[str, np.ndarray]], dict[str, int]]:
    """Return a policy_fn that runs the GATPolicy on the current observation.

    The returned callable is stateless — no on_step callback needed.
    """
    dev = torch.device(device)
    junction_ids = builder.junction_ids

    def policy_fn(obs: dict[str, np.ndarray]) -> dict[str, int]:
        graph  = builder.build(obs, update_normalizer=False).to(dev)
        phases = model.predict(graph).cpu().tolist()
        return {jid: phases[i] for i, jid in enumerate(junction_ids)}

    return policy_fn


def make_expert_policy(
    expert: GreedyExpert,
) -> Callable[[dict[str, np.ndarray]], dict[str, int]]:
    """Return a policy_fn that calls GreedyExpert.act().

    The expert reads TraCI state directly; obs is ignored.
    Remember to call expert.reset() before the episode and pass an on_step
    callback to run_eval_episode that calls expert.notify_applied().
    """
    def policy_fn(obs: dict[str, np.ndarray]) -> dict[str, int]:
        return expert.act()

    return policy_fn


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_eval_episode(
    env: TrafficEnv,
    policy_fn: Callable[[dict[str, np.ndarray]], dict[str, int]],
    on_step: Optional[Callable[[dict[str, int]], None]] = None,
) -> EvalMetrics:
    """Run one evaluation episode and return collected metrics.

    Parameters
    ----------
    env :
        TrafficEnv instance.  Its SUMO process is started and stopped inside
        this function (env.reset() then env.close()).
    policy_fn :
        Callable mapping obs dict → actions dict.  Produced by
        make_model_policy or make_expert_policy.
    on_step :
        Optional callback invoked with the actions dict after every env.step().
        Use this to call expert.notify_applied() for stateful expert policies.

    Returns
    -------
    EvalMetrics with all traffic statistics for the episode.
    """
    junction_ids   = env.junction_ids
    junction_infos = env.junction_infos
    n_junctions    = len(junction_ids)

    # ------------------------------------------------------------------
    # Temporary tripinfo file — SUMO writes it on close().
    # ------------------------------------------------------------------
    fd, tripinfo_path = tempfile.mkstemp(suffix=".xml", prefix="tripinfo_eval_")
    os.close(fd)   # SUMO will open/overwrite it; we just needed a safe path

    env.tripinfo_output = tripinfo_path
    obs  = env.reset()
    done = False

    # ------------------------------------------------------------------
    # In-sim accumulators
    # ------------------------------------------------------------------
    total_wait_density = 0.0
    total_switches     = 0
    max_queue_ever     = 0
    n_steps            = 0

    # Per-junction
    junc_wait_acc   = {jid: 0.0 for jid in junction_ids}
    junc_max_queue  = {jid: 0   for jid in junction_ids}
    junc_phase_cnt = {jid: [0 for _ in phase_indices()] for jid in junction_ids}
    green_wave = GreenWaveTracker()

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    while not done:
        actions = policy_fn(obs)

        # Record phase choices before the step.
        for jid, phase in actions.items():
            junc_phase_cnt[jid][phase] += 1

        obs, _rewards, done, info = env.step(actions)

        if on_step is not None:
            on_step(actions)

        _update_green_wave_tracker(green_wave)

        n_steps        += 1
        total_switches += sum(1 for sw in info["switches"].values() if sw)
        total_wait_density += info["global_wait"]

        # Per-junction wait density.
        for jid, w in info["local_waits"].items():
            junc_wait_acc[jid] += w

        # Per-lane peak queue (raw vehicle count, not density).
        for ji in junction_infos.values():
            jid = ji.junction_id
            for lane_id, _ in ji.all_lane_det:
                q = traci.lane.getLastStepHaltingNumber(lane_id)
                if q > max_queue_ever:
                    max_queue_ever = q
                if q > junc_max_queue[jid]:
                    junc_max_queue[jid] = q

    # ------------------------------------------------------------------
    # Close SUMO → flushes tripinfo file.
    # ------------------------------------------------------------------
    env.close()
    env.tripinfo_output = None   # clear so next reset() runs without flag

    # ------------------------------------------------------------------
    # Parse tripinfo.
    # ------------------------------------------------------------------
    avg_wait, avg_travel, throughput = _parse_tripinfo(
        tripinfo_path, env.episode_length
    )
    try:
        os.unlink(tripinfo_path)
    except OSError:
        pass

    # ------------------------------------------------------------------
    # Aggregate in-sim metrics.
    # ------------------------------------------------------------------
    avg_wait_density = total_wait_density / max(1, n_steps)
    switch_freq      = total_switches / max(1, n_junctions) / (env.episode_length / 60.0)

    per_junction_wait_density = {
        jid: junc_wait_acc[jid] / max(1, n_steps)
        for jid in junction_ids
    }
    green_wave_metrics = green_wave.metrics()

    return EvalMetrics(
        avg_waiting_time           = avg_wait,
        avg_travel_time            = avg_travel,
        throughput_per_hour        = throughput,
        max_queue_length           = float(max_queue_ever),
        phase_switch_freq          = switch_freq,
        avg_wait_density           = avg_wait_density,
        per_junction_wait_density  = per_junction_wait_density,
        per_junction_max_queue     = junc_max_queue,
        per_junction_phase_counts  = junc_phase_cnt,
        avg_tls_passes_per_vehicle       = green_wave_metrics["avg_tls_passes_per_vehicle"],
        avg_stops_before_tls_per_vehicle = green_wave_metrics["avg_stops_before_tls_per_vehicle"],
        nonstop_tls_pass_rate            = green_wave_metrics["nonstop_tls_pass_rate"],
        avg_best_nonstop_tls_streak      = green_wave_metrics["avg_best_nonstop_tls_streak"],
    )


def _update_green_wave_tracker(tracker: GreenWaveTracker) -> None:
    vehicle_ids = list(traci.vehicle.getIDList())
    next_tls_by_vehicle = {}
    speed_by_vehicle = {}
    for vid in vehicle_ids:
        try:
            next_tls_by_vehicle[vid] = list(traci.vehicle.getNextTLS(vid))
            speed_by_vehicle[vid] = float(traci.vehicle.getSpeed(vid))
        except traci.exceptions.TraCIException:
            continue
    tracker.update(
        vehicle_ids=vehicle_ids,
        next_tls_by_vehicle=next_tls_by_vehicle,
        speed_by_vehicle=speed_by_vehicle,
        arrived_ids=list(traci.simulation.getArrivedIDList()),
    )


# ---------------------------------------------------------------------------
# Multi-seed averaging
# ---------------------------------------------------------------------------

def average_eval_metrics(metrics_list: list[EvalMetrics]) -> EvalMetrics:
    """Return element-wise mean of a list of EvalMetrics.

    Scalar fields are averaged directly.  Per-junction phase counts are summed
    (preserving the relative distribution shape for histograms).
    """
    n = len(metrics_list)
    if n == 1:
        return metrics_list[0]

    def mean(attr: str) -> float:
        return sum(getattr(m, attr) for m in metrics_list) / n

    jids = list(metrics_list[0].per_junction_wait_density.keys())
    return EvalMetrics(
        avg_waiting_time    = mean('avg_waiting_time'),
        avg_travel_time     = mean('avg_travel_time'),
        throughput_per_hour = mean('throughput_per_hour'),
        max_queue_length    = mean('max_queue_length'),
        phase_switch_freq   = mean('phase_switch_freq'),
        avg_wait_density    = mean('avg_wait_density'),
        avg_tls_passes_per_vehicle=mean('avg_tls_passes_per_vehicle'),
        avg_stops_before_tls_per_vehicle=mean('avg_stops_before_tls_per_vehicle'),
        nonstop_tls_pass_rate=mean('nonstop_tls_pass_rate'),
        avg_best_nonstop_tls_streak=mean('avg_best_nonstop_tls_streak'),
        per_junction_wait_density={
            jid: sum(m.per_junction_wait_density[jid] for m in metrics_list) / n
            for jid in jids
        },
        per_junction_max_queue={
            jid: int(round(sum(m.per_junction_max_queue[jid] for m in metrics_list) / n))
            for jid in jids
        },
        per_junction_phase_counts={
            jid: [
                sum(m.per_junction_phase_counts[jid][ph] for m in metrics_list)
                for ph in phase_indices()
            ]
            for jid in jids
        },
    )


# ---------------------------------------------------------------------------
# Tripinfo parser
# ---------------------------------------------------------------------------

def _parse_tripinfo(
    path: str,
    episode_length: float,
) -> tuple[float, float, float]:
    """Parse SUMO tripinfo XML into per-vehicle traffic metrics.

    Parameters
    ----------
    path :
        Path to the ``--tripinfo-output`` XML file.
    episode_length :
        Simulation seconds in the episode — used to compute throughput_per_hour.

    Returns
    -------
    (avg_waiting_time, avg_travel_time, throughput_per_hour)
    All zero if the file is absent or contains no completed trips.
    """
    if not os.path.exists(path):
        return 0.0, 0.0, 0.0

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return 0.0, 0.0, 0.0

    wait_sum   = 0.0
    travel_sum = 0.0
    n_trips    = 0

    for trip in root.findall("tripinfo"):
        try:
            wait_sum   += float(trip.get("waitingTime", 0.0))
            travel_sum += float(trip.get("duration",    0.0))
            n_trips    += 1
        except (TypeError, ValueError):
            continue

    if n_trips == 0:
        return 0.0, 0.0, 0.0

    hours = episode_length / 3600.0
    return (
        wait_sum   / n_trips,
        travel_sum / n_trips,
        n_trips    / hours,
    )
