"""Run the movement-aware max-pressure controller in SUMO or SUMO-GUI."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Point it to your SUMO installation directory."
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib  # noqa: E402
import traci  # noqa: E402

from src.movement.sumo_adapter import (  # noqa: E402
    extract_programs_from_trafficlight_api,
    select_max_pressure_states,
)
from src.movement.transition import SignalTransitionController  # noqa: E402

DEFAULT_CFG = ROOT / "configs" / "grid_4x4" / "grid.sumocfg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the movement-aware max-pressure phase selector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run sumo-gui instead of headless sumo")
    parser.add_argument("--steps", type=int, default=1800, help="Maximum simulation seconds to run")
    parser.add_argument("--decision-interval", type=int, default=15, help="Seconds between phase decisions")
    parser.add_argument("--yellow-duration", type=int, default=3, help="Yellow seconds inserted before a new green")
    parser.add_argument("--seed", type=int, default=42, help="SUMO random seed")
    parser.add_argument("--verbose", action="store_true", help="Print selected phases each decision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary = sumolib.checkBinary("sumo-gui" if args.gui else "sumo")
    command = [
        binary,
        "-c",
        args.cfg,
        "--seed",
        str(args.seed),
        "--no-step-log",
        "true",
    ]

    traci.start(command)
    try:
        programs = extract_programs_from_trafficlight_api(traci.trafficlight)
        if not programs:
            raise RuntimeError("No traffic lights with selectable movement phases were found.")

        print(f"Loaded {len(programs)} movement-aware traffic-light programs.")
        controller = SignalTransitionController(yellow_duration=args.yellow_duration)
        controller.set_targets(select_max_pressure_states(programs, traci.lane))
        for step in range(args.steps):
            if step % args.decision_interval == 0:
                target_states = select_max_pressure_states(programs, traci.lane)
                controller.set_targets(target_states)
                if args.verbose:
                    print(f"t={step:5d}s targets={target_states}")

            current_states = controller.current_states()
            for tls_id, state in current_states.items():
                traci.trafficlight.setRedYellowGreenState(tls_id, state)
            if args.verbose and any("y" in state or "Y" in state for state in current_states.values()):
                print(f"t={step:5d}s transition={current_states}")

            traci.simulationStep()
            controller.advance()
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
    finally:
        traci.close()


if __name__ == "__main__":
    main()
