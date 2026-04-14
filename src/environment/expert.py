"""Greedy expert controller for imitation learning (PLAN §8).

The expert scores each of the 4 canonical phases by attributing every
waiting vehicle's accumulated wait to the phase that serves its intended
turn (from-edge → to-edge lookup via vehicle route).  This avoids the
shared-lane over-counting problem and produces clean (state, phase) labels
in the GNN's output space.

Usage
-----
    expert = GreedyExpert(env.junction_infos)
    actions = expert.act()                      # dict[jid → int 0–3]
    obs, rewards, done, info = env.step(actions)
    for jid in env.junction_ids:
        expert.notify_applied(jid, actions[jid])
"""
from __future__ import annotations

import os
import sys

from pathlib import Path

# Make sure traci is importable.
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Point it to your SUMO installation directory."
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

from .junction_info import JunctionInfo

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

MIN_HOLD_INTERVALS = 3   # commit to each phase for at least 3 intervals (45 s)
MAX_HOLD_INTERVALS = 9   # starvation cap: force switch after 9 intervals (135 s)


# ---------------------------------------------------------------------------
# GreedyExpert
# ---------------------------------------------------------------------------

class GreedyExpert:
    """Per-junction greedy expert that scores phases by per-vehicle wait attribution.

    Algorithm (PLAN §8):
      1. Score each phase by the sum of accumulated waiting times of vehicles
         whose intended movement is served by that phase (vehicle route lookup).
      2. Commit to the selected phase for MIN_HOLD_INTERVALS (3 × 15 s = 45 s).
      3. Before MIN_HOLD, allow an early switch only if every protected-green
         lane in the current phase has zero halting vehicles (queue cleared).
      4. After MIN_HOLD, re-score.  Switch if a different phase scores higher.
         Re-commit for another MIN_HOLD if the current phase is still best.
      5. Hard starvation cap: after MAX_HOLD_INTERVALS, force-switch to the
         next-best phase even if the current phase is still scoring highest.
    """

    def __init__(self, junction_infos: dict[str, JunctionInfo]) -> None:
        self._junctions:      dict[str, JunctionInfo] = junction_infos
        self._intervals_held: dict[str, int]           = {jid: 0 for jid in junction_infos}
        self._current_phase:  dict[str, int]           = {jid: 0 for jid in junction_infos}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self) -> dict[str, int]:
        """Return target phase for every junction based on current traci state."""
        return {jid: self._decide(jid, ji) for jid, ji in self._junctions.items()}

    def notify_applied(self, jid: str, applied_phase: int) -> None:
        """Update internal state after env.step() applies the action for *jid*.

        Must be called once per step for each junction so hold-interval
        counting stays in sync with the environment.
        """
        if applied_phase != self._current_phase[jid]:
            self._current_phase[jid]  = applied_phase
            self._intervals_held[jid] = 0
        else:
            self._intervals_held[jid] += 1

    def reset(self) -> None:
        """Reset internal counters (call when env.reset() is called)."""
        for jid in self._junctions:
            self._intervals_held[jid] = 0
            self._current_phase[jid]  = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_phases(self, ji: JunctionInfo) -> list[float]:
        """Score phases by attributing each vehicle's accumulated wait to the
        phase that serves its intended movement (from-edge → to-edge lookup).
        """
        scores = [0.0] * 4
        for lane_id, _ in ji.all_lane_det:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                if wait <= 0.0:
                    continue
                route: list = list(traci.vehicle.getRoute(vid))
                ridx: int   = traci.vehicle.getRouteIndex(vid)
                if ridx + 1 >= len(route):
                    continue
                phase = ji.conn_to_phase.get((route[ridx], route[ridx + 1]))
                if phase is not None:
                    scores[phase] += wait
        return scores

    def _served_lanes_clear(self, ji: JunctionInfo, phase: int) -> bool:
        """True when every protected-green lane for *phase* has no halting vehicles."""
        return all(
            traci.lane.getLastStepHaltingNumber(lid) == 0
            for lid in ji.phase_served_lanes[phase]
        )

    def _decide(self, jid: str, ji: JunctionInfo) -> int:
        current = self._current_phase[jid]
        held    = self._intervals_held[jid]

        # ---- Before minimum hold ----------------------------------------
        # Only leave early if the current phase's queue is completely clear.
        if held < MIN_HOLD_INTERVALS:
            if self._served_lanes_clear(ji, current):
                scores = self._score_phases(ji)
                ranked = sorted(range(4), key=lambda p: scores[p], reverse=True)
                best_other = next((p for p in ranked if p != current), current)
                if scores[best_other] > 0:
                    return best_other   # early switch to something with demand
            return current              # stay committed

        # ---- Starvation cap ----------------------------------------------
        if held >= MAX_HOLD_INTERVALS:
            scores = self._score_phases(ji)
            ranked = sorted(range(4), key=lambda p: scores[p], reverse=True)
            return next((p for p in ranked if p != current), current)

        # ---- Normal re-evaluation at / after MIN_HOLD --------------------
        scores = self._score_phases(ji)
        ranked = sorted(range(4), key=lambda p: scores[p], reverse=True)
        return ranked[0]   # may equal current → re-commit for another MIN_HOLD
