"""Run the canonical expert controller on the single 4-way junction.

Expert logic per PLAN §8:
  - Commit to the selected phase for MIN_INTERVALS (3 x 15s = 45s).
  - Early exit: if the served lanes' queue drops to zero AND another phase has
    waiting traffic, switch immediately (don't wait out the 3 intervals).
  - After MIN_INTERVALS: re-score. If a different phase wins, switch. If the
    same phase wins, re-commit for another 3 intervals.
  - Starvation cap: after MAX_INTERVALS consecutive intervals in one phase,
    force switch to next-best regardless of scores (per PLAN §8 item 5).
"""
import os
import sys
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    sys.exit("SUMO_HOME not set")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
import sumolib

HERE = Path(__file__).parent.parent / "configs" / "single_junction"
CFG = str(HERE / "single.sumocfg")

GREEN_STATES = [
    "GGGrrrrrGGGrrrrr",  # phase 0: NS through+right
    "rrrGgrrrrrrGgrrr",  # phase 1: NS left + EW right (permissive)
    "rrrrGGGrrrrrGGGr",  # phase 2: EW through+right
    "grrrrrrGgrrrrrrG",  # phase 3: EW left + NS right (permissive)
]

def yellow_for(green: str) -> str:
    return "".join("y" if c in "Gg" else "r" for c in green)

YELLOW_STATES = [yellow_for(g) for g in GREEN_STATES]
ALL_RED = "r" * 16

DECISION_INTERVAL = 15   # seconds between checks
YELLOW_DUR        = 3
ALLRED_DUR        = 2
MIN_INTERVALS     = 3    # commit for at least 3 intervals (45s)
MAX_INTERVALS     = 3    # starvation cap: force re-eval after 3 intervals

# Each turn movement (from_edge, to_edge) -> primary phase: the first phase
# that gives it a protected green ('G').  Built from link layout inspected
# via sumolib (see configs/single_junction/single.net.xml).
LINK_TO_MOVEMENT = {
    0:  ("N_C", "C_W"),  1:  ("N_C", "C_S"),  2:  ("N_C", "C_S"),  3:  ("N_C", "C_E"),
    4:  ("E_C", "C_N"),  5:  ("E_C", "C_W"),  6:  ("E_C", "C_W"),  7:  ("E_C", "C_S"),
    8:  ("S_C", "C_E"),  9:  ("S_C", "C_N"), 10:  ("S_C", "C_N"), 11:  ("S_C", "C_W"),
    12: ("W_C", "C_S"), 13: ("W_C", "C_E"), 14: ("W_C", "C_E"), 15: ("W_C", "C_N"),
}
INCOMING_LANES = [
    "N_C_0", "N_C_1", "E_C_0", "E_C_1",
    "S_C_0", "S_C_1", "W_C_0", "W_C_1",
]

MOVEMENT_TO_PHASE: dict[tuple[str, str], int] = {}
for _phase_idx, _state in enumerate(GREEN_STATES):
    for _link_idx, _ch in enumerate(_state):
        if _ch == "G":
            _mv = LINK_TO_MOVEMENT[_link_idx]
            MOVEMENT_TO_PHASE.setdefault(_mv, _phase_idx)

# Lanes whose vehicles are cleared by each phase (protected G only).
_lane_of = {
    0: "N_C_0", 1: "N_C_0", 2: "N_C_1", 3: "N_C_1",
    4: "E_C_0", 5: "E_C_0", 6: "E_C_1", 7: "E_C_1",
    8: "S_C_0", 9: "S_C_0", 10: "S_C_1", 11: "S_C_1",
    12: "W_C_0", 13: "W_C_0", 14: "W_C_1", 15: "W_C_1",
}
PHASE_SERVED_LANES: list[set[str]] = []
for _state in GREEN_STATES:
    PHASE_SERVED_LANES.append({_lane_of[i] for i, c in enumerate(_state) if c == "G"})


def score_phases() -> list[float]:
    """Attribute each waiting vehicle's accumulated wait to its primary phase."""
    scores = [0.0] * 4
    for lane in INCOMING_LANES:
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            wait = traci.vehicle.getAccumulatedWaitingTime(vid)
            if wait <= 0:
                continue
            route = traci.vehicle.getRoute(vid)
            ridx = traci.vehicle.getRouteIndex(vid)
            if ridx + 1 >= len(route):
                continue
            phase = MOVEMENT_TO_PHASE.get((route[ridx], route[ridx + 1]))
            if phase is not None:
                scores[phase] += wait
    return scores


def served_queue_empty(phase: int) -> bool:
    """True when every lane the current phase serves has zero halting vehicles."""
    return all(
        traci.lane.getLastStepHaltingNumber(ln) == 0
        for ln in PHASE_SERVED_LANES[phase]
    )


def do_switch(from_phase: int, to_phase: int) -> None:
    """Execute yellow -> all-red -> new green transition."""
    traci.trafficlight.setRedYellowGreenState("C", YELLOW_STATES[from_phase])
    for _ in range(YELLOW_DUR):
        traci.simulationStep()
    traci.trafficlight.setRedYellowGreenState("C", ALL_RED)
    for _ in range(ALLRED_DUR):
        traci.simulationStep()
    traci.trafficlight.setRedYellowGreenState("C", GREEN_STATES[to_phase])


def main():
    gui = "--gui" in sys.argv
    sumo_bin = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    traci.start([
        sumo_bin, "-c", CFG, "--no-step-log", "true",
        "--duration-log.statistics", "true",
        "--tripinfo-output", str(HERE / "tripinfo_expert.xml"),
    ])

    current_phase  = 0
    intervals_held = 0
    traci.trafficlight.setRedYellowGreenState("C", GREEN_STATES[current_phase])
    next_decision_t = DECISION_INTERVAL

    end_t = 3600
    while traci.simulation.getTime() < end_t:
        traci.simulationStep()
        t = traci.simulation.getTime()

        if t < next_decision_t:
            continue

        intervals_held += 1
        scores = score_phases()

        # --- Early exit (before MIN_INTERVALS) ----------------------------
        # Switch only if our lanes are fully clear AND another phase has wait.
        if intervals_held < MIN_INTERVALS:
            other_has_wait = any(scores[p] > 0 for p in range(4) if p != current_phase)
            if served_queue_empty(current_phase) and other_has_wait:
                best_other = max(
                    (p for p in range(4) if p != current_phase),
                    key=lambda p: scores[p],
                )
                do_switch(current_phase, best_other)
                current_phase  = best_other
                intervals_held = 0
            next_decision_t = traci.simulation.getTime() + (DECISION_INTERVAL - YELLOW_DUR - ALLRED_DUR)
            continue

        # --- Committed MIN_INTERVALS: re-score and decide -----------------
        ranked = sorted(range(4), key=lambda p: scores[p], reverse=True)

        if ranked[0] == current_phase and intervals_held < MAX_INTERVALS:
            # Best phase is still current — re-commit for another round.
            intervals_held = 0
            next_decision_t = t + DECISION_INTERVAL
            continue

        # Switch to best (or force-switch on starvation cap).
        target = ranked[0] if ranked[0] != current_phase else ranked[1]
        do_switch(current_phase, target)
        current_phase  = target
        intervals_held = 0
        next_decision_t = traci.simulation.getTime() + (DECISION_INTERVAL - YELLOW_DUR - ALLRED_DUR)

    traci.close()


if __name__ == "__main__":
    main()
