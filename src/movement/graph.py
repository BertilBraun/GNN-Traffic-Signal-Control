"""Build static LaneGroup/Movement graphs from movement-aware programs."""
from __future__ import annotations

from collections.abc import Mapping

from .graph_schema import (
    GraphMovementId,
    LaneGroupId,
    LaneGroupNode,
    MovementGraph,
    MovementNode,
    PhaseIncidence,
    TypedMovementEdges,
)
from .schema import EdgeId, LaneId, MovementIndex, TrafficLightId, TrafficLightProgram


def build_movement_graph(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> MovementGraph:
    """Build a deterministic static graph from extracted traffic-light programs."""
    lane_group_edges = _collect_lane_group_edges(programs)
    lane_groups = tuple(
        LaneGroupNode(lane_group_id=LaneGroupId(index), edge_id=edge_id)
        for index, edge_id in enumerate(lane_group_edges)
    )
    lane_group_id_by_edge = {
        lane_group.edge_id: lane_group.lane_group_id
        for lane_group in lane_groups
    }

    movement_keys = _collect_movement_keys(programs)
    movement_id_by_key = {
        key: GraphMovementId(index)
        for index, key in enumerate(movement_keys)
    }
    controlled_indices_by_key = _collect_controlled_indices_by_key(programs)
    movements = tuple(
        MovementNode(
            movement_id=movement_id_by_key[key],
            traffic_light_id=key[0],
            input_lane_group_id=lane_group_id_by_edge[key[1]],
            output_lane_group_id=lane_group_id_by_edge[key[2]],
            controlled_movement_indices=controlled_indices_by_key[key],
        )
        for key in movement_keys
    )

    return MovementGraph(
        lane_groups=lane_groups,
        movements=movements,
        edges=_build_edges(movements),
        phase_incidences=_build_phase_incidences(
            programs=programs,
            movement_id_by_key=movement_id_by_key,
        ),
        lane_group_id_by_edge=lane_group_id_by_edge,
        movement_id_by_key=movement_id_by_key,
    )


def edge_id_from_lane_id(lane_id: LaneId | str) -> EdgeId:
    """Return the SUMO edge id for a lane id such as `edge_with_underscores_0`."""
    text = str(lane_id)
    edge_text, separator, lane_index = text.rpartition("_")
    if separator and lane_index.isdigit() and edge_text:
        return EdgeId(edge_text)
    return EdgeId(text)


def _collect_lane_group_edges(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> tuple[EdgeId, ...]:
    edge_ids: set[EdgeId] = set()
    for program in programs.values():
        for movement in program.movements:
            edge_ids.add(edge_id_from_lane_id(movement.incoming_lane_id))
            edge_ids.add(edge_id_from_lane_id(movement.outgoing_lane_id))
    return tuple(sorted(edge_ids, key=str))


def _collect_movement_keys(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> tuple[tuple[TrafficLightId, EdgeId, EdgeId], ...]:
    keys: list[tuple[TrafficLightId, EdgeId, EdgeId]] = []
    seen: set[tuple[TrafficLightId, EdgeId, EdgeId]] = set()
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
            key = _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def _collect_controlled_indices_by_key(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
) -> dict[tuple[TrafficLightId, EdgeId, EdgeId], tuple[MovementIndex, ...]]:
    indices_by_key: dict[tuple[TrafficLightId, EdgeId, EdgeId], list[MovementIndex]] = {}
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
            key = _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
            indices_by_key.setdefault(key, []).append(movement.movement_index)
    return {
        key: tuple(indices)
        for key, indices in indices_by_key.items()
    }


def _movement_key(
    traffic_light_id: TrafficLightId,
    incoming_lane_id: LaneId,
    outgoing_lane_id: LaneId,
) -> tuple[TrafficLightId, EdgeId, EdgeId]:
    return (
        traffic_light_id,
        edge_id_from_lane_id(incoming_lane_id),
        edge_id_from_lane_id(outgoing_lane_id),
    )


def _build_edges(movements: tuple[MovementNode, ...]) -> TypedMovementEdges:
    return TypedMovementEdges(
        input_lane_to_movement=tuple(
            (movement.input_lane_group_id, movement.movement_id)
            for movement in movements
        ),
        output_lane_to_movement=tuple(
            (movement.output_lane_group_id, movement.movement_id)
            for movement in movements
        ),
        movement_to_input_lane=tuple(
            (movement.movement_id, movement.input_lane_group_id)
            for movement in movements
        ),
        movement_to_output_lane=tuple(
            (movement.movement_id, movement.output_lane_group_id)
            for movement in movements
        ),
    )


def _build_phase_incidences(
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> dict[TrafficLightId, PhaseIncidence]:
    incidences: dict[TrafficLightId, PhaseIncidence] = {}
    for program in sorted(programs.values(), key=lambda item: str(item.traffic_light_id)):
        local_movement_ids = _local_movement_ids(program, movement_id_by_key)
        rows = []
        controlled_to_graph_id = _controlled_to_graph_id(program, movement_id_by_key)
        for phase in program.selectable_phases:
            enabled = {
                controlled_to_graph_id[movement_index]
                for movement_index in phase.enabled_movement_indices
            }
            rows.append(
                tuple(
                    1 if movement_id in enabled else 0
                    for movement_id in local_movement_ids
                )
            )
        incidences[program.traffic_light_id] = PhaseIncidence(
            traffic_light_id=program.traffic_light_id,
            sumo_phase_indices=tuple(
                int(phase.sumo_phase_index)
                for phase in program.selectable_phases
            ),
            movement_ids=local_movement_ids,
            rows=tuple(rows),
        )
    return incidences


def _local_movement_ids(
    program: TrafficLightProgram,
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> tuple[GraphMovementId, ...]:
    movement_ids: list[GraphMovementId] = []
    seen: set[GraphMovementId] = set()
    for movement in sorted(program.movements, key=lambda item: int(item.movement_index)):
        key = _movement_key(
            program.traffic_light_id,
            movement.incoming_lane_id,
            movement.outgoing_lane_id,
        )
        movement_id = movement_id_by_key[key]
        if movement_id in seen:
            continue
        seen.add(movement_id)
        movement_ids.append(movement_id)
    return tuple(movement_ids)


def _controlled_to_graph_id(
    program: TrafficLightProgram,
    movement_id_by_key: Mapping[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId],
) -> dict[MovementIndex, GraphMovementId]:
    return {
        movement.movement_index: movement_id_by_key[
            _movement_key(
                program.traffic_light_id,
                movement.incoming_lane_id,
                movement.outgoing_lane_id,
            )
        ]
        for movement in program.movements
    }
