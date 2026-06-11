"""Run movement-aware heuristic controllers in SUMO or SUMO-GUI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.phase_selection import select_highest_scoring_phase
from src.movement.runtime import LaneAPI, MovementControlRuntime
from src.movement.schema import TrafficLightProgram

DEFAULT_CFG = ROOT / 'configs' / 'grid_4x4_dedicated' / 'grid.sumocfg'
PHASE_SCORE_AGGREGATION = 'incoming_lane'


def compute_movement_scores(
    program: TrafficLightProgram,
    lane_api: LaneAPI,
    method: str,
) -> dict[int, float]:
    """Compute runner-owned movement scores for a single traffic light."""
    if method not in {'max-pressure', 'queue'}:
        raise ValueError(f'Unsupported control method: {method}')

    scores: dict[int, float] = {}
    for movement in program.movements:
        incoming_queue = lane_api.getLastStepHaltingNumber(movement.incoming_lane_id)
        if method == 'queue':
            scores[movement.movement_index] = float(incoming_queue)
            continue

        outgoing_queue = lane_api.getLastStepHaltingNumber(movement.outgoing_lane_id)
        scores[movement.movement_index] = float(incoming_queue - outgoing_queue)
    return scores


def select_control_states(
    programs: Mapping[str, TrafficLightProgram],
    lane_api: LaneAPI,
    method: str,
    phase_score_aggregation: str = PHASE_SCORE_AGGREGATION,
) -> dict[str, str]:
    """Return held green states selected by the runner's scoring method."""
    states: dict[str, str] = {}
    for tls_id, program in programs.items():
        movement_scores = compute_movement_scores(program, lane_api, method)
        selection = select_highest_scoring_phase(
            program,
            movement_scores,
            phase_score_aggregation=phase_score_aggregation,
        )
        states[tls_id] = program.selectable_phases[selection.local_phase_index].state
    return states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize movement-aware phase-selection heuristics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', default=str(DEFAULT_CFG), help='Path to SUMO .sumocfg file')
    parser.add_argument('--gui', action='store_true', help='Run sumo-gui instead of headless sumo')
    parser.add_argument('--steps', type=int, default=1800, help='Maximum simulation seconds to run')
    parser.add_argument(
        '--decision-interval', type=int, default=15, help='Seconds between phase decisions'
    )
    parser.add_argument(
        '--yellow-duration', type=int, default=3, help='Yellow seconds inserted before a new green'
    )
    parser.add_argument(
        '--min-green-steps',
        type=int,
        default=2,
        help='Minimum accepted decision intervals before a traffic light may switch again',
    )
    parser.add_argument(
        '--method',
        choices=('max-pressure', 'queue'),
        default='max-pressure',
        help='Control heuristic used to score selectable phases',
    )
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument(
        '--verbose', action='store_true', help='Print selected phases each decision'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = MovementControlRuntime(
        cfg_path=args.cfg,
        gui=args.gui,
        seed=args.seed,
        yellow_duration=args.yellow_duration,
        min_green_steps=args.min_green_steps,
    )
    if False:
        for decision in runtime.decision_loop(args.steps, args.decision_interval):
            desired_states = select_control_states(
                decision.programs,
                decision.lane_api,
                method=args.method,
            )
            accepted_states = decision.request_targets(desired_states)
            if args.verbose:
                print(f'method={args.method} desired={desired_states} accepted={accepted_states}')

    try:
        runtime.start()
        print(f'Loaded {len(runtime.programs)} movement-aware traffic-light programs.')
        for step in range(args.steps):
            if step % args.decision_interval == 0:
                desired_states = select_control_states(
                    runtime.programs,
                    runtime.lane_api,
                    method=args.method,
                )
                accepted_states = runtime.request_targets(desired_states)
                if args.verbose:
                    print(
                        f't={step:5d}s method={args.method} '
                        f'desired={desired_states} accepted={accepted_states}'
                    )

            current_states = runtime.step()
            if args.verbose and any(
                'y' in state or 'Y' in state for state in current_states.values()
            ):
                print(f't={step:5d}s transition={current_states}')

            if not runtime.is_running():
                break
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
