"""Static graph schema for movement-score learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from .schema import EdgeId, MovementIndex, TrafficLightId

LaneGroupId = NewType('LaneGroupId', int)
GraphMovementId = NewType('GraphMovementId', int)


@dataclass(frozen=True)
class LaneGroupNode:
    """One directed road segment or lane group."""

    lane_group_id: LaneGroupId
    edge_ids: tuple[EdgeId, ...]

    @property
    def edge_id(self) -> EdgeId:
        """Return the downstream-most edge for compatibility with local features."""
        return self.edge_ids[-1]


@dataclass(frozen=True)
class MovementNode:
    """One graph-level movement through a signalized junction."""

    movement_id: GraphMovementId
    traffic_light_id: TrafficLightId
    input_lane_group_id: LaneGroupId
    output_lane_group_id: LaneGroupId
    controlled_movement_indices: tuple[MovementIndex, ...]


@dataclass(frozen=True)
class LaneLaneConnectorEdge:
    """One legal directed pass-through connector across a non-controllable junction."""

    source_lane_group_id: LaneGroupId
    target_lane_group_id: LaneGroupId
    source_edge_id: EdgeId
    target_edge_id: EdgeId
    via_junction_id: str
    distance_m: float
    freeflow_time_s: float
    lane_count: float
    connector_type: str


@dataclass(frozen=True)
class LaneMovementEdgeMetadata:
    """Deterministic metadata for a LaneGroup/Movement message edge."""

    lane_group_id: LaneGroupId
    movement_id: GraphMovementId
    distance_m: float
    freeflow_time_s: float
    lane_count: float
    connector_type: str


@dataclass(frozen=True)
class TypedMovementEdges:
    """Typed bipartite edges between lane groups and graph movements."""

    input_lane_to_movement: tuple[tuple[LaneGroupId, GraphMovementId], ...]
    output_lane_to_movement: tuple[tuple[LaneGroupId, GraphMovementId], ...]
    movement_to_input_lane: tuple[tuple[GraphMovementId, LaneGroupId], ...]
    movement_to_output_lane: tuple[tuple[GraphMovementId, LaneGroupId], ...]


@dataclass(frozen=True)
class PhaseIncidence:
    """Local selectable phase to graph movement incidence for one traffic light."""

    traffic_light_id: TrafficLightId
    sumo_phase_indices: tuple[int, ...]
    movement_ids: tuple[GraphMovementId, ...]
    rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MovementGraph:
    """Static graph and lookup metadata used by learning and replay."""

    lane_groups: tuple[LaneGroupNode, ...]
    movements: tuple[MovementNode, ...]
    edges: TypedMovementEdges
    lane_lane_connectors: tuple[LaneLaneConnectorEdge, ...]
    lane_movement_metadata: tuple[LaneMovementEdgeMetadata, ...]
    phase_incidences: dict[TrafficLightId, PhaseIncidence]
    lane_group_id_by_edge: dict[EdgeId, LaneGroupId]
    movement_id_by_key: dict[tuple[TrafficLightId, EdgeId, EdgeId], GraphMovementId]
    pass_through_traffic_light_ids: tuple[TrafficLightId, ...] = ()
