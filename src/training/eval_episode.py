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
                                                          [ph0, ph1, ph2, ph3]
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
    junc_phase_cnt  = {jid: [0, 0, 0, 0] for jid in junction_ids}

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
