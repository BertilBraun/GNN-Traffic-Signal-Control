"""Policy and sample helpers for movement PPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from scripts.collect_il_data import lane_inputs_from_net, resolve_sumocfg_net_path
from src.movement.dataset import MovementDatasetSample, build_dataset_sample
from src.movement.features import (
    LaneGroupFlowTracker,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
)
from src.movement.graph import build_movement_graph
from src.movement.models.bipartite_gnn import MovementActorCritic
from src.movement.normalization import RunningNormalizer
from src.movement.runtime import MovementControlRuntime
from src.movement.schema import TrafficLightProgram
from src.movement.training.il.tensors import edge_tensors_from_sample, tensors_from_sample
from src.movement.training.ppo.types import PolicyContext, RolloutContext


def rollout_context(
    cfg_path: Path,
    programs: Mapping[str, TrafficLightProgram],
) -> RolloutContext:
    net_path = resolve_sumocfg_net_path(cfg_path)
    graph = build_movement_graph(programs, net_path=net_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    traffic_light_ids = tuple(sorted((str(traffic_light_id) for traffic_light_id in programs), key=str))
    movement_ids_by_traffic_light = tuple(
        tuple(int(value) for value in graph.phase_incidences[programs[traffic_light_id].traffic_light_id].movement_ids)
        for traffic_light_id in traffic_light_ids
    )
    incoming_lanes_by_traffic_light = {
        traffic_light_id: tuple(
            dict.fromkeys(str(movement.incoming_lane_id) for movement in programs[traffic_light_id].movements)
        )
        for traffic_light_id in traffic_light_ids
    }
    incoming_lane_length_by_traffic_light = {
        traffic_light_id: sum(float(lane_geometries[edge_id_from_lane_id(lane_id)].length_m) for lane_id in lane_ids)
        for traffic_light_id, lane_ids in incoming_lanes_by_traffic_light.items()
    }
    speed_limit_by_lane = {
        lane_id: float(lane_geometries[edge_id_from_lane_id(lane_id)].speed_limit_mps)
        for lane_ids in incoming_lanes_by_traffic_light.values()
        for lane_id in lane_ids
    }
    all_incoming_lane_ids = tuple(
        dict.fromkeys(
            lane_id
            for traffic_light_id in traffic_light_ids
            for lane_id in incoming_lanes_by_traffic_light[traffic_light_id]
        )
    )
    all_incoming_lane_length_m = sum(
        float(lane_geometries[edge_id_from_lane_id(lane_id)].length_m) for lane_id in all_incoming_lane_ids
    )
    return RolloutContext(
        graph=graph,
        traffic_light_ids=traffic_light_ids,
        movement_ids_by_traffic_light=movement_ids_by_traffic_light,
        lane_ids_by_edge=lane_ids_by_edge,
        lane_geometries=lane_geometries,
        incoming_lanes_by_traffic_light=incoming_lanes_by_traffic_light,
        incoming_lane_length_by_traffic_light=incoming_lane_length_by_traffic_light,
        speed_limit_by_lane=speed_limit_by_lane,
        all_incoming_lane_ids=all_incoming_lane_ids,
        all_incoming_lane_length_m=all_incoming_lane_length_m,
    )


def policy_context_from_sample(sample: MovementDatasetSample) -> PolicyContext:
    traffic_light_ids = tuple(sorted(sample.phase_incidences.keys(), key=str))
    movement_ids_by_traffic_light = tuple(
        sample.phase_incidences[traffic_light_id].movement_ids for traffic_light_id in traffic_light_ids
    )
    return PolicyContext(
        traffic_light_ids=traffic_light_ids,
        movement_ids_by_traffic_light=movement_ids_by_traffic_light,
    )


def current_sample(
    context: RolloutContext,
    programs: Mapping[str, TrafficLightProgram],
    vehicle_snapshot_collector: VehicleSnapshotCollector,
    lane_flow_tracker: LaneGroupFlowTracker,
    control_state: MovementControlState,
) -> MovementDatasetSample:
    vehicles = vehicle_snapshot_collector.capture()
    feature_frame = build_feature_frame(
        graph=context.graph,
        lane_ids_by_edge=context.lane_ids_by_edge,
        lane_geometries=context.lane_geometries,
        control_state=control_state,
        vehicles=vehicles,
        lane_flow_rates=lane_flow_tracker.observe(vehicles),
    )
    return build_dataset_sample(
        graph=context.graph,
        feature_frame=feature_frame,
        programs=programs,
        teacher_controlled_scores={traffic_light_id: {} for traffic_light_id in programs},
        metadata={},
    )


def forward_policy(
    model: MovementActorCritic,
    sample: MovementDatasetSample,
    context: RolloutContext | PolicyContext,
    lane_normalizer: RunningNormalizer,
    movement_normalizer: RunningNormalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    x_lane, x_movement, _target = tensors_from_sample(
        sample=sample,
        lane_normalizer=lane_normalizer,
        movement_normalizer=movement_normalizer,
        device=device,
    )
    movement_scores, values = model.forward_actor_critic(
        x_lane=x_lane,
        x_movement=x_movement,
        edge_index_dict=edge_tensors_from_sample(sample, device=device),
        movement_ids_by_traffic_light=context.movement_ids_by_traffic_light,
    )
    return (
        movement_scores,
        values,
        phase_logits(
            sample=sample,
            traffic_light_ids=context.traffic_light_ids,
            movement_scores=movement_scores,
        ),
    )


def phase_logits(
    sample: MovementDatasetSample,
    traffic_light_ids: Sequence[str],
    movement_scores: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    logits = []
    for traffic_light_id in traffic_light_ids:
        incidence = sample.phase_incidences[traffic_light_id]
        phase_scores = []
        for row in incidence.rows:
            enabled_scores = [
                movement_scores[movement_id]
                for enabled, movement_id in zip(row, incidence.movement_ids)
                if enabled == 1
            ]
            phase_scores.append(torch.stack(tuple(enabled_scores)).sum())
        logits.append(torch.stack(tuple(phase_scores)))
    return tuple(logits)


def states_from_actions(
    programs: Mapping[str, TrafficLightProgram],
    traffic_light_ids: Sequence[str],
    actions: Sequence[int],
) -> dict[str, str]:
    return {
        traffic_light_id: str(programs[traffic_light_id].selectable_phases[action].state)
        for traffic_light_id, action in zip(traffic_light_ids, actions)
    }


def actions_from_states(
    programs: Mapping[str, TrafficLightProgram],
    traffic_light_ids: Sequence[str],
    states: Mapping[str, str],
) -> tuple[int, ...]:
    return tuple(
        next(
            local_phase_index
            for local_phase_index, phase in enumerate(programs[traffic_light_id].selectable_phases)
            if str(phase.state) == states[traffic_light_id]
        )
        for traffic_light_id in traffic_light_ids
    )


def allowed_action_masks(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    programs: Mapping[str, TrafficLightProgram],
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            str(phase.state) in set(runtime.allowed_target_states(traffic_light_id))
            for phase in programs[traffic_light_id].selectable_phases
        )
        for traffic_light_id in context.traffic_light_ids
    )


def masked_phase_logits(
    phase_logits: Sequence[torch.Tensor],
    action_masks: Sequence[Sequence[bool]],
) -> tuple[torch.Tensor, ...]:
    if len(phase_logits) != len(action_masks):
        raise ValueError('Phase-logit count does not match action-mask count.')
    masked_logits = []
    for logits, mask in zip(phase_logits, action_masks):
        if len(mask) != len(logits):
            raise ValueError('Action-mask width does not match phase-logit width.')
        if not any(mask):
            raise ValueError('Each traffic light must permit at least one action.')
        mask_tensor = torch.tensor(tuple(mask), dtype=torch.bool, device=logits.device)
        masked_logits.append(logits.masked_fill(~mask_tensor, float('-inf')))
    return tuple(masked_logits)


def edge_id_from_lane_id(lane_id: str) -> str:
    edge_id, separator, lane_index = lane_id.rpartition('_')
    if separator and lane_index.isdigit() and edge_id:
        return edge_id
    return lane_id
