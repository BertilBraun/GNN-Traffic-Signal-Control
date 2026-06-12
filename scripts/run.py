"""Run movement-aware heuristic controllers in SUMO or SUMO-GUI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path

import torch
import traci
from traci._vehicle import VehicleDomain

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import (  # noqa: E402
    lane_inputs_from_net,
    resolve_sumocfg_net_path,
    vehicle_snapshots_from_api,
)
from src.movement.dataset import build_dataset_sample  # noqa: E402
from src.movement.features import (  # noqa: E402
    LaneGroupGeometry,
    MovementControlState,
    build_feature_frame,
)
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_schema import MovementGraph  # noqa: E402
from src.movement.models.bipartite_gnn import MovementScorer  # noqa: E402
from src.movement.normalization import RunningNormalizer  # noqa: E402
from src.movement.phase_selection import select_highest_scoring_phase
from src.movement.policies import MovementScoringMethod, compute_movement_scores
from src.movement.runtime import LaneQueueApi, MovementControlRuntime
from src.movement.schema import TrafficLightProgram
from src.movement.training.il import (  # noqa: E402
    edge_tensors_from_sample,
    load_movement_checkpoint,
    normalizer_from_state,
    tensors_from_sample,
)

DEFAULT_CFG = ROOT / 'configs' / 'grid_4x4_dedicated' / 'grid.sumocfg'


def select_control_states(
    programs: Mapping[str, TrafficLightProgram],
    lane_api: LaneQueueApi,
    method: MovementScoringMethod,
) -> dict[str, str]:
    """Return held green states selected by the runner's scoring method."""
    states: dict[str, str] = {}
    for tls_id, program in programs.items():
        movement_scores = compute_movement_scores(program, lane_api, method)
        selection = select_highest_scoring_phase(program, movement_scores)
        states[tls_id] = program.selectable_phases[selection.local_phase_index].state
    return states


def select_graph_score_control_states(
    programs: Mapping[str, TrafficLightProgram],
    graph: MovementGraph,
    graph_movement_scores: tuple[float, ...],
) -> dict[str, str]:
    """Select phase states from graph-level movement scores."""
    states: dict[str, str] = {}
    for tls_id, program in programs.items():
        incidence = graph.phase_incidences[program.traffic_light_id]
        movement_ids = tuple(int(value) for value in incidence.movement_ids)
        best_local_idx = 0
        best_score = _phase_incidence_score(
            incidence.rows[0],
            movement_ids,
            graph_movement_scores,
        )
        for local_idx, row in enumerate(incidence.rows[1:], start=1):
            score = _phase_incidence_score(row, movement_ids, graph_movement_scores)
            if score > best_score:
                best_local_idx = local_idx
                best_score = score
        states[tls_id] = program.selectable_phases[best_local_idx].state
    return states


def select_learned_control_states(
    programs: Mapping[str, TrafficLightProgram],
    lane_api: LaneQueueApi,
    graph: MovementGraph,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
    lane_geometries: dict[str, LaneGroupGeometry],
    vehicle_api: VehicleDomain,
    model: MovementScorer,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: str,
) -> dict[str, str]:
    """Score graph movements with a learned checkpoint."""
    feature_frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        lane_api=lane_api,
        control_state=MovementControlState(),
        vehicles=vehicle_snapshots_from_api(vehicle_api),
    )
    sample = build_dataset_sample(
        graph=graph,
        feature_frame=feature_frame,
        programs=programs,
        teacher_controlled_scores={tls_id: {} for tls_id in programs},
        metadata={},
    )
    x_lane, x_movement, _target = tensors_from_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        scores = tuple(
            float(value)
            for value in model(
                x_lane,
                x_movement,
                edge_tensors_from_sample(sample, device=device),
            ).cpu()
        )
    return select_graph_score_control_states(
        programs=programs,
        graph=graph,
        graph_movement_scores=scores,
    )


def _phase_incidence_score(
    row: tuple[int, ...],
    movement_ids: tuple[int, ...],
    graph_movement_scores: tuple[float, ...],
) -> float:
    return sum(graph_movement_scores[movement_id] for enabled, movement_id in zip(row, movement_ids) if enabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize movement-aware phase-selection heuristics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--cfg',
        dest='sumo_config_path',
        default=str(DEFAULT_CFG),
        help='Path to SUMO .sumocfg file',
    )
    parser.add_argument('--gui', action='store_true', help='Run sumo-gui instead of headless sumo')
    parser.add_argument('--steps', type=int, default=1800, help='Maximum simulation seconds to run')
    parser.add_argument('--decision-interval', type=int, default=15, help='Seconds between phase decisions')
    parser.add_argument('--yellow-duration', type=int, default=3, help='Yellow seconds inserted before a new green')
    parser.add_argument(
        '--min-green-steps',
        type=int,
        default=2,
        help='Minimum accepted decision intervals before a traffic light may switch again',
    )
    parser.add_argument(
        '--method',
        choices=tuple(method.value for method in MovementScoringMethod) + ('learned',),
        default=MovementScoringMethod.MAX_PRESSURE.value,
        help='Control method used to score selectable phases',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='movement policy checkpoint, required for --method learned',
    )
    parser.add_argument('--device', default='cpu', help='Torch device for learned policy')
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument('--verbose', action='store_true', help='Print selected phases each decision')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learned_policy = args.method == 'learned'
    scoring_method = None if learned_policy else MovementScoringMethod(args.method)
    if learned_policy and args.checkpoint is None:
        raise SystemExit('--checkpoint is required when --method learned')
    runtime = MovementControlRuntime(
        cfg_path=args.sumo_config_path,
        gui=args.gui,
        seed=args.seed,
        yellow_duration=args.yellow_duration,
        min_green_steps=args.min_green_steps,
    )

    try:
        runtime.start()
        print(f'Loaded {len(runtime.programs)} movement-aware traffic-light programs.')
        learned_context = None
        if learned_policy:
            model, metadata = load_movement_checkpoint(args.checkpoint, device=args.device)
            graph = build_movement_graph(runtime.programs)
            lane_ids_by_edge, lane_geometries = lane_inputs_from_net(resolve_sumocfg_net_path(args.sumo_config_path))
            learned_context = (
                model,
                graph,
                lane_ids_by_edge,
                lane_geometries,
                normalizer_from_state(metadata.lane_normalizer),
                normalizer_from_state(metadata.movement_normalizer),
            )
        for step in range(args.steps):
            if step % args.decision_interval == 0:
                if learned_policy:
                    (
                        model,
                        graph,
                        lane_ids_by_edge,
                        lane_geometries,
                        lane_normalizer,
                        movement_normalizer,
                    ) = learned_context
                    desired_states = select_learned_control_states(
                        programs=runtime.programs,
                        lane_api=runtime.lane_api,
                        graph=graph,
                        lane_ids_by_edge=lane_ids_by_edge,
                        lane_geometries=lane_geometries,
                        vehicle_api=traci.vehicle,
                        model=model,
                        lane_normalizer=lane_normalizer,
                        movement_normalizer=movement_normalizer,
                        device=args.device,
                    )
                else:
                    desired_states = select_control_states(
                        runtime.programs,
                        runtime.lane_api,
                        method=scoring_method,
                    )
                accepted_states = runtime.request_targets(desired_states)
                if args.verbose:
                    print(f't={step:5d}s method={args.method} desired={desired_states} accepted={accepted_states}')

            current_states = runtime.step()
            if args.verbose and any('y' in state or 'Y' in state for state in current_states.values()):
                print(f't={step:5d}s transition={current_states}')

            if not runtime.is_running():
                break
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
