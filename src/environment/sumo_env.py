"""TrafficEnv — SUMO-backed multi-junction traffic control environment.

Interface
---------
env = TrafficEnv(cfg_path, gui=False)
obs          = env.reset()                    # dict[jid -> np.ndarray(41,)]
obs, rew, done, info = env.step(actions)      # actions: dict[jid -> phase 0-3]
obs          = env.observe()                  # can be called between steps
env.close()

Observation vector per junction (41 dims, §3 of PLAN):
  [0..35]  12 movements × 3 features (queue_density, approach_density, wait_density)
           ordered: slot0-left, slot0-through, slot0-right,
                    slot1-left, slot1-through, slot1-right,
                    slot2-left, slot2-through, slot2-right,
                    slot3-left, slot3-through, slot3-right
           Slot 3 is zero-padded for 3-way junctions.
  [36..39] current phase one-hot (4 dims)
  [40]     elapsed time in phase, normalised to [0, 1]

Reward per junction per step (§8):
  r_i = -α·Δlocal_wait_i - β·Δglobal_wait - γ·switch_i
  α=1.0, β=0.1, γ=0.5

Step timing (§6):
  Decision interval = 15 s.
  Switching junctions: 3 s yellow → 2 s all-red → 10 s new green.
  Holding junctions:   15 s current green.
  Minimum green (10 s) is satisfied by construction.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

# Make sure SUMO tools are on the path.
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Point it to your SUMO installation directory."
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib
import traci

from .junction_info import JunctionInfo, build_junction_info
from src.utils.demand_generator import DemandRandomizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YELLOW_DUR        = 3     # seconds
ALLRED_DUR        = 2     # seconds
MIN_GREEN_DUR     = 10    # seconds (guaranteed by the fixed 15-s interval)
DECISION_INTERVAL = 15    # seconds between GNN decisions

ALPHA = 1.0   # local wait weight
BETA  = 0.1   # global wait weight
GAMMA = 0.5   # switch penalty

MAX_PHASE_HOLD_S  = 45.0  # 3 intervals; used to normalise elapsed time

# Movement directions in canonical order (matches feature vector layout)
MOVEMENT_ORDER = ('l', 's', 'r')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_net_path(cfg_path: str) -> str:
    """Extract net-file path from a .sumocfg file."""
    cfg = Path(cfg_path)
    tree = ET.parse(cfg)
    net_file = tree.find(".//net-file")
    if net_file is None:
        raise ValueError(f"No <net-file> element found in {cfg_path}")
    rel = net_file.get("value")
    return str(cfg.parent / rel)


# ---------------------------------------------------------------------------
# TrafficEnv
# ---------------------------------------------------------------------------

class TrafficEnv:
    """Multi-junction traffic signal environment backed by SUMO/TraCI.

    Parameters
    ----------
    cfg_path :
        Path to the SUMO configuration file (.sumocfg).  All other files
        (net, routes, additional/detectors) are resolved relative to it.
    gui :
        If True, launches sumo-gui instead of headless sumo.
    episode_length :
        Simulation seconds per episode.
    flow_range :
        ``(min, max)`` total vehicles-per-hour across the network.  A fresh
        value is drawn uniformly at each ``reset()``.
    demand_seed :
        Seed for the demand RNG.  ``None`` (default) means a different
        traffic pattern every episode.  Pass an integer for reproducible runs.
    """

    def __init__(
        self,
        cfg_path: str,
        gui: bool = False,
        episode_length: int = 3600,
        gui_settings_file: Optional[str] = None,
        tripinfo_output: Optional[str] = None,
        flow_range: tuple[int, int] = (300, 1000),
        demand_seed: Optional[int] = None,
    ) -> None:
        self.cfg_path          = cfg_path
        self.gui               = gui
        self.episode_length    = episode_length
        self.gui_settings_file = gui_settings_file
        # Set to a file path before env.reset() to make SUMO write per-vehicle
        # trip statistics (waiting time, travel time) to that file.
        # eval_episode.run_eval_episode() manages this automatically.
        self.tripinfo_output   = tripinfo_output

        # Parse network statically (no simulation needed).
        net_path = _parse_net_path(cfg_path)
        self._net = sumolib.net.readNet(net_path, withConnections=True)

        # Build per-junction canonical info (static, computed once).
        self._junctions: dict[str, JunctionInfo] = {}
        for node in sorted(self._net.getNodes(), key=lambda n: n.getID()):
            if node.getType() != "traffic_light":
                continue
            ji = build_junction_info(node)
            if ji is not None:
                self._junctions[node.getID()] = ji

        if not self._junctions:
            raise RuntimeError("No supported TL junctions found in network.")

        # Demand randomization — always active; seed=None means a different
        # demand pattern each episode, seed=<int> gives reproducible demand.
        self._rng              = np.random.default_rng(demand_seed)
        self._temp_routes: Optional[str] = None
        self._demand_randomizer = DemandRandomizer(self._net, flow_range)

        # Runtime state (populated by reset()).
        self._started          = False
        self._current_phases:  dict[str, int]   = {}
        self._phase_start_t:   dict[str, float] = {}   # sim-time when phase last changed
        self._prev_local_wait: dict[str, float] = {}
        self._prev_global_wait: float           = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        """Start (or restart) a simulation episode.

        Returns
        -------
        Initial observations (dict[junction_id → np.ndarray(41,)]).
        """
        if self._started:
            traci.close()
            self._started = False

        # Rotate demand: delete the previous temp file and generate a new one.
        if self._temp_routes is not None:
            try:
                os.unlink(self._temp_routes)
            except OSError:
                pass
            self._temp_routes = None

        self._temp_routes = self._demand_randomizer.generate(
            self.episode_length, self._rng
        )

        self._start_sumo()

        # Initialise all junctions to phase 0.
        for jid, ji in self._junctions.items():
            self._current_phases[jid] = 0
            self._phase_start_t[jid]  = 0.0
            traci.trafficlight.setRedYellowGreenState(jid, ji.phase_states[0])

        # Advance one step so sensors have data.
        traci.simulationStep()

        self._prev_local_wait  = self._compute_local_waits()
        self._prev_global_wait = self._compute_global_wait()

        return self.observe()

    def observe(self) -> dict[str, np.ndarray]:
        """Return the current observation for every managed junction.

        Returns
        -------
        dict mapping junction_id → np.ndarray of shape (41,).
        """
        now = traci.simulation.getTime()
        obs = {}
        for jid, ji in self._junctions.items():
            obs[jid] = self._observe_junction(ji, now)
        return obs

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        """Execute one decision interval (15 simulation seconds).

        Parameters
        ----------
        actions :
            Target phase (0–3) for each junction.  Missing junctions hold
            their current phase.

        Returns
        -------
        observations : dict[jid → np.ndarray(41,)]
        rewards      : dict[jid → float]
        done         : bool — True when episode_length is reached
        info         : dict with diagnostic values
        """
        # Fill missing junctions with hold.
        full_actions: dict[str, int] = {
            jid: actions.get(jid, self._current_phases[jid])
            for jid in self._junctions
        }

        # Determine which junctions will switch.
        switches: dict[str, bool] = {
            jid: (full_actions[jid] != self._current_phases[jid])
            for jid in self._junctions
        }

        # Snapshot wait densities before the interval.
        pre_local  = self._compute_local_waits()
        pre_global = self._compute_global_wait()

        # ---- Run 15 simulation seconds -----------------------------------
        # t = 1..15 within the interval (1-based so we step first, then set).
        for t in range(1, DECISION_INTERVAL + 1):
            # Set each junction's signal state for this second.
            for jid, ji in self._junctions.items():
                current = self._current_phases[jid]
                target  = full_actions[jid]
                if switches[jid]:
                    if t <= YELLOW_DUR:
                        state = ji.yellow_states[current]
                    elif t <= YELLOW_DUR + ALLRED_DUR:
                        state = ji.all_red
                    else:
                        state = ji.phase_states[target]
                else:
                    state = ji.phase_states[current]
                traci.trafficlight.setRedYellowGreenState(jid, state)

            traci.simulationStep()

        now = traci.simulation.getTime()

        # Update phase tracking.
        for jid, target in full_actions.items():
            if switches[jid]:
                self._current_phases[jid] = target
                self._phase_start_t[jid]  = now

        # ---- Rewards -----------------------------------------------------
        post_local  = self._compute_local_waits()
        post_global = self._compute_global_wait()
        delta_global = post_global - pre_global

        rewards: dict[str, float] = {}
        for jid in self._junctions:
            delta_local    = post_local[jid] - pre_local[jid]
            switch_penalty = 1.0 if switches[jid] else 0.0
            rewards[jid] = (
                -ALPHA * delta_local
                - BETA  * delta_global
                - GAMMA * switch_penalty
            )

        self._prev_local_wait  = post_local
        self._prev_global_wait = post_global

        done = now >= self.episode_length
        obs  = self.observe()
        info = {
            "sim_time":     now,
            "global_wait":  post_global,
            "local_waits":  post_local,
            "switches":     switches,
        }
        return obs, rewards, done, info

    def close(self) -> None:
        """Shut down the SUMO process."""
        if self._started:
            traci.close()
            self._started = False
        if self._temp_routes is not None:
            try:
                os.unlink(self._temp_routes)
            except OSError:
                pass
            self._temp_routes = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def junction_ids(self) -> list[str]:
        return list(self._junctions.keys())

    @property
    def junction_infos(self) -> dict[str, JunctionInfo]:
        return self._junctions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_sumo(self) -> None:
        binary = sumolib.checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary,
            "-c", self.cfg_path,
            "--no-step-log", "true",
            "--no-warnings",  "true",
        ]
        if self.gui and self.gui_settings_file:
            cmd += ["--gui-settings-file", self.gui_settings_file]
        if self.tripinfo_output:
            cmd += ["--tripinfo-output", self.tripinfo_output]
        if self._temp_routes is not None:
            cmd += ["--route-files", self._temp_routes]
        traci.start(cmd)
        self._started = True

    # -- Observation -------------------------------------------------------

    def _observe_junction(
        self,
        ji: JunctionInfo,
        now: float,
    ) -> np.ndarray:
        """Build the 41-dim feature vector for one junction."""
        feat = np.zeros(41, dtype=np.float32)
        idx  = 0

        for slot_idx in range(4):
            slot = ji.slots[slot_idx]
            if slot is None:
                idx += 9   # 3 movements × 3 features — zero-padded
                continue

            for direction in MOVEMENT_ORDER:
                mv = slot.movements.get(direction)
                if mv is None or not mv.lane_ids:
                    idx += 3   # zero-pad absent movement
                    continue

                total_len = mv.total_det_length

                # queue_density: halting vehicles / total det length
                halting = sum(
                    traci.lane.getLastStepHaltingNumber(lid)
                    for lid in mv.lane_ids
                )
                # approach_density: all vehicles in lane / det length
                # (uses lane counts as proxy — det is within lane bounds)
                approaching = sum(
                    traci.lane.getLastStepVehicleNumber(lid)
                    for lid in mv.lane_ids
                )
                # wait_density: accumulated waiting time / det length
                waiting = sum(
                    traci.lane.getWaitingTime(lid)
                    for lid in mv.lane_ids
                )

                feat[idx]   = halting    / total_len
                feat[idx+1] = approaching / total_len
                feat[idx+2] = waiting    / total_len
                idx += 3

        # Junction-level features: current phase one-hot [36..39]
        phase = self._current_phases.get(ji.junction_id, 0)
        feat[36 + phase] = 1.0

        # Elapsed time in phase, normalised to [0, 1] [40]
        start_t  = self._phase_start_t.get(ji.junction_id, 0.0)
        elapsed  = max(0.0, now - start_t)
        feat[40] = min(1.0, elapsed / MAX_PHASE_HOLD_S)

        return feat

    # -- Reward components -------------------------------------------------

    def _compute_local_waits(self) -> dict[str, float]:
        """Wait density per junction: sum(lane_wait) / sum(det_len)."""
        result = {}
        for jid, ji in self._junctions.items():
            total_wait = sum(
                traci.lane.getWaitingTime(lid)
                for lid, _ in ji.all_lane_det
            )
            total_len  = sum(dl for _, dl in ji.all_lane_det) or 1.0
            result[jid] = total_wait / total_len
        return result

    def _compute_global_wait(self) -> float:
        """Network-wide wait density: sum_all_lanes(wait) / sum_all_lanes(det_len)."""
        total_wait = 0.0
        total_len  = 0.0
        for ji in self._junctions.values():
            for lid, dl in ji.all_lane_det:
                total_wait += traci.lane.getWaitingTime(lid)
                total_len  += dl
        return total_wait / (total_len or 1.0)
