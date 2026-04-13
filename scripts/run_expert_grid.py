"""Expert controller demo on the irregular grid network.

Runs the canonical greedy expert (PLAN §8) through TrafficEnv and prints
live per-junction rewards every decision step.  Pass --gui to open SUMO-GUI.

With --gui, a RewardOverlay draws a colour-coded floating label above every
junction in SUMO-GUI after each decision step:

    green  = positive reward (queues clearing)
    red    = negative reward (queues growing)
    amber  = near-zero reward

The label format is:  <junction_id> [>] <reward:+.2f>
The ">" marker appears when the junction switched phase this step.

Usage
-----
    python scripts/run_expert_grid.py          # headless
    python scripts/run_expert_grid.py --gui    # visual
    python scripts/run_expert_grid.py --gui --episode-length 1200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import traci

from src.environment import TrafficEnv
from src.environment.junction_info import JunctionInfo

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

GRID_CFG        = str(ROOT / "configs" / "grid_4x4" / "grid.sumocfg")

# Vertical offset (metres) to place the label above the junction centre.
_POI_Y_OFFSET   = 30.0
# SUMO layer — above roads (layer 0) and vehicles (layer ~10).
_POI_LAYER      = 200.0

# ---------------------------------------------------------------------------
# Reward overlay (SUMO-GUI only)
# ---------------------------------------------------------------------------

# Colours as (R, G, B, A) 0-255.
_COL_POS   = (30,  210,  60, 230)   # green  — positive reward
_COL_NEG   = (220,  40,  40, 230)   # red    — negative reward
_COL_ZERO  = (210, 170,   0, 230)   # amber  — near-zero reward
_THRESHOLD = 0.3                     # |reward| below this → amber


class RewardOverlay:
    """Manages floating POI labels in SUMO-GUI showing per-junction rewards.

    Each label is a SUMO POI whose *ID* is the text string displayed in the
    GUI (SUMO shows POI IDs as labels when 'Show POI name' is enabled via the
    view-settings file).  Because the reward value changes every step we
    cannot keep a stable ID, so we track the previous ID per junction and
    remove it before adding the new one.

    Parameters
    ----------
    junction_infos :
        The same dict the environment uses.  We need junction coordinates.
    net :
        sumolib network object for junction coordinates.
    """

    def __init__(self, junction_infos: dict[str, JunctionInfo], net) -> None:
        self._net      = net
        self._junctions = junction_infos
        # Tracks the POI ID currently displayed for each junction.
        self._active_ids: dict[str, str] = {}

    def update(
        self,
        rewards:  dict[str, float],
        switches: dict[str, bool],
        phase:    dict[str, int],
    ) -> None:
        """Refresh all junction labels.  Call once per decision step."""
        for jid in self._junctions:
            r   = rewards.get(jid, 0.0)
            sw  = switches.get(jid, False)
            ph  = phase.get(jid, 0)

            # Remove previous POI for this junction.
            prev_id = self._active_ids.get(jid)
            if prev_id is not None:
                try:
                    traci.poi.remove(prev_id)
                except traci.exceptions.TraCIException:
                    pass

            # Build label: "J5 P2> +1.24"  or  "J5 P2  -0.31"
            sw_marker = ">" if sw else " "
            new_id = f"{jid} P{ph}{sw_marker} {r:+.2f}"

            # Pick colour.
            if r > _THRESHOLD:
                color = _COL_POS
            elif r < -_THRESHOLD:
                color = _COL_NEG
            else:
                color = _COL_ZERO

            # Place POI slightly above the junction centre.
            node = self._net.getNode(jid)
            x, y = node.getCoord()
            traci.poi.add(
                new_id,
                x, y + _POI_Y_OFFSET,
                color,
                layer=_POI_LAYER,
                imgFile="",
                width=3.0,
                height=3.0,
                angle=0.0,
            )
            self._active_ids[jid] = new_id

    def clear(self) -> None:
        """Remove all overlay POIs (call on episode end / close)."""
        for poi_id in self._active_ids.values():
            try:
                traci.poi.remove(poi_id)
            except traci.exceptions.TraCIException:
                pass
        self._active_ids.clear()


# ---------------------------------------------------------------------------
# Expert controller
# ---------------------------------------------------------------------------

MIN_HOLD_INTERVALS = 3   # commit to each phase for at least 3 intervals (45 s)
MAX_HOLD_INTERVALS = 9   # hard starvation cap: force switch after 9 intervals (135 s)


class GreedyExpert:
    """Per-junction greedy expert that scores phases by per-vehicle wait attribution.

    Implements PLAN §8 expert logic:
      1. Score each phase by the sum of accumulated waiting times of vehicles
         whose intended turn is served by that phase (vehicle-route lookup).
         This avoids the shared-lane over-counting problem: a lane serving both
         through and left no longer inflates the through phase's score with
         stuck left-turning vehicles.
      2. Commit to the selected phase for MIN_HOLD_INTERVALS (3 × 15 s = 45 s).
      3. Before MIN_HOLD, allow an early switch ONLY if every protected-green
         lane in the current phase has zero halting vehicles (queue cleared).
      4. After MIN_HOLD, re-score.  Switch if a different phase scores higher.
         Re-commit for another MIN_HOLD if the current phase is still best.
      5. Hard starvation cap: after MAX_HOLD_INTERVALS, force-switch to the
         next-best phase even if the current phase is still scoring highest.
    """

    def __init__(self, junction_infos: dict[str, JunctionInfo]) -> None:
        self._junctions      = junction_infos
        self._intervals_held: dict[str, int] = {jid: 0 for jid in junction_infos}
        self._current_phase:  dict[str, int] = {jid: 0 for jid in junction_infos}

    def act(self) -> dict[str, int]:
        """Return target phase for every junction based on current traci state."""
        return {jid: self._decide(jid, ji) for jid, ji in self._junctions.items()}

    def notify_applied(self, jid: str, applied_phase: int) -> None:
        """Update internal state after env.step() applies the action for *jid*."""
        if applied_phase != self._current_phase[jid]:
            self._current_phase[jid]  = applied_phase
            self._intervals_held[jid] = 0
        else:
            self._intervals_held[jid] += 1

    # -- internal ----------------------------------------------------------

    def _score_phases(self, ji: JunctionInfo) -> list[float]:
        """Score phases by attributing each vehicle's accumulated wait to the
        phase that serves its intended movement (from-edge → to-edge lookup).

        This avoids the shared-lane bias where a lane serving both through and
        left-turn traffic would inflate the through phase's score while
        left-turning vehicles are stuck.
        """
        scores = [0.0] * 4
        for lane_id, _ in ji.all_lane_det:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                if wait <= 0.0:
                    continue
                route: list = list(traci.vehicle.getRoute(vid))  # type: ignore[arg-type]
                ridx: int   = traci.vehicle.getRouteIndex(vid)   # type: ignore[assignment]
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
        best   = ranked[0]
        return best   # may equal current (re-commit for another MIN_HOLD)


# ---------------------------------------------------------------------------
# Reward display
# ---------------------------------------------------------------------------

def _format_rewards(rewards: dict[str, float], switches: dict[str, bool]) -> str:
    parts = []
    for jid in sorted(rewards):
        sw  = "→" if switches.get(jid, False) else " "
        r   = rewards[jid]
        bar = "▓" * min(20, max(0, int((r + 5) * 2)))   # crude ASCII bar
        parts.append(f"  {jid:6s}{sw} r={r:+6.2f}  {bar}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(cfg: str, gui: bool, episode_length: int) -> None:
    env     = TrafficEnv(cfg, gui=gui, episode_length=episode_length)
    expert  = GreedyExpert(env.junction_infos)
    overlay = RewardOverlay(env.junction_infos, env._net) if gui else None

    print(f"\n{'='*60}")
    print(f"  Expert controller — {len(env.junction_ids)} junctions")
    print(f"  Episode length: {episode_length} s")
    print(f"  GUI: {gui}")
    if gui:
        print(f"\n  Reward labels are drawn as POIs above each junction.")
        print(f"  If labels are not visible, enable them once in SUMO-GUI:")
        print(f"    View > Visualization Settings > POIs > tick 'Show name'")
    print(f"{'='*60}\n")

    env.reset()

    step      = 0
    done      = False
    ep_reward: dict[str, float] = {jid: 0.0 for jid in env.junction_ids}

    while not done:
        step += 1
        actions = expert.act()
        obs, rewards, done, info = env.step(actions)

        # Update expert phase tracking.
        for jid in env.junction_ids:
            expert.notify_applied(jid, actions[jid])

        # Accumulate episode reward.
        for jid, r in rewards.items():
            ep_reward[jid] = ep_reward.get(jid, 0.0) + r

        # Update in-sim reward labels.
        if overlay is not None:
            overlay.update(rewards, info["switches"], env._current_phases)

        # Print live step summary.
        t = info["sim_time"]
        gw = info["global_wait"]
        print(f"[t={t:5.0f}s  step={step:3d}]  global_wait_density={gw:.4f}")
        print(_format_rewards(rewards, info["switches"]))
        print()

    # Episode summary.
    if overlay is not None:
        overlay.clear()
    print(f"\n{'='*60}  EPISODE DONE")
    print(f"  Cumulative rewards per junction:")
    for jid in sorted(ep_reward):
        print(f"    {jid:6s}  {ep_reward[jid]:+.1f}")
    total = sum(ep_reward.values())
    mean  = total / max(1, len(ep_reward))
    print(f"  Total: {total:+.1f}   Mean per junction: {mean:+.2f}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert controller demo on grid_4x4")
    parser.add_argument("--gui",            action="store_true",
                        help="Launch SUMO-GUI")
    parser.add_argument("--episode-length", type=int, default=3600,
                        help="Simulation seconds per episode (default: 3600)")
    parser.add_argument("--cfg",            default=GRID_CFG,
                        help="Path to .sumocfg (default: configs/grid_4x4/grid.sumocfg)")
    args = parser.parse_args()

    main(cfg=args.cfg, gui=args.gui, episode_length=args.episode_length)
