"""Dataset sample schema and serialization for movement-score imitation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .features import (
    LaneGroupFeatureRow,
    MovementFeatureFrame,
    MovementFeatureRow,
)
from .graph_schema import MovementGraph, PhaseIncidence
from .schema import MovementIndex, TrafficLightId, TrafficLightProgram

JsonScalar: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = list['JsonValue'] | tuple['JsonValue', ...]
JsonValue: TypeAlias = JsonScalar | JsonArray | dict[str, 'JsonValue']
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class MovementEdgeIndices:
    input_lane_to_movement: tuple[tuple[int, int], ...]
    output_lane_to_movement: tuple[tuple[int, int], ...]
    movement_to_input_lane: tuple[tuple[int, int], ...]
    movement_to_output_lane: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StoredPhaseIncidence:
    sumo_phase_indices: tuple[int, ...]
    movement_ids: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MovementDatasetSample:
    """One decision-time imitation sample."""

    x_lane: tuple[tuple[float, ...], ...]
    x_movement: tuple[tuple[float, ...], ...]
    edge_indices: MovementEdgeIndices
    phase_incidences: dict[str, StoredPhaseIncidence]
    teacher_movement_scores: tuple[float, ...]
    teacher_selected_phase_by_tls: dict[str, int]
    metadata: JsonObject


def build_dataset_sample(
    graph: MovementGraph,
    feature_frame: MovementFeatureFrame,
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    teacher_controlled_scores: Mapping[
        str | TrafficLightId,
        Mapping[MovementIndex, float],
    ],
    teacher_graph_scores: Sequence[float] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> MovementDatasetSample:
    """Build a serializable imitation sample aligned with graph movement IDs."""
    teacher_scores = (
        tuple(float(value) for value in teacher_graph_scores)
        if teacher_graph_scores is not None
        else _teacher_scores_by_graph_movement(
            graph=graph,
            teacher_controlled_scores=teacher_controlled_scores,
        )
    )
    if len(teacher_scores) != len(graph.movements):
        raise ValueError(f'Expected {len(graph.movements)} graph teacher scores, got {len(teacher_scores)}.')
    teacher_selected = _teacher_selected_phases(
        graph=graph,
        programs=programs,
        teacher_scores=teacher_scores,
    )
    return MovementDatasetSample(
        x_lane=tuple(_lane_row_vector(row) for row in feature_frame.lane_group_rows),
        x_movement=tuple(_movement_row_vector(row) for row in feature_frame.movement_rows),
        edge_indices=_edge_indices(graph),
        phase_incidences={
            str(tls_id): _stored_phase_incidence(incidence) for tls_id, incidence in graph.phase_incidences.items()
        },
        teacher_movement_scores=teacher_scores,
        teacher_selected_phase_by_tls=teacher_selected,
        metadata=dict(metadata or {}),
    )


def replay_teacher_selected_phases(sample: MovementDatasetSample) -> dict[str, int]:
    """Recompute selected local phase indices from stored teacher scores."""
    selected: dict[str, int] = {}
    for traffic_light_id, incidence in sample.phase_incidences.items():
        best_local_idx = 0
        best_score = _phase_score(incidence.rows[0], incidence.movement_ids, sample.teacher_movement_scores)
        for local_idx, row in enumerate(incidence.rows[1:], start=1):
            score = _phase_score(row, incidence.movement_ids, sample.teacher_movement_scores)
            if score > best_score:
                best_local_idx = local_idx
                best_score = score
        selected[traffic_light_id] = best_local_idx
    return selected


def save_jsonl_samples(
    path: str | Path,
    samples: Iterable[MovementDatasetSample],
) -> None:
    """Write samples as one JSON object per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), sort_keys=True))
            handle.write('\n')


def load_jsonl_samples(path: str | Path) -> list[MovementDatasetSample]:
    """Load samples written by `save_jsonl_samples`."""
    samples: list[MovementDatasetSample] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(_sample_from_dict(json.loads(line)))
    return samples


def _teacher_scores_by_graph_movement(
    graph: MovementGraph,
    teacher_controlled_scores: Mapping[
        str | TrafficLightId,
        Mapping[MovementIndex, float],
    ],
) -> tuple[float, ...]:
    scores: list[float] = [0.0 for _ in graph.movements]
    teacher_by_tls = {
        str(tls_id): scores_by_movement for tls_id, scores_by_movement in teacher_controlled_scores.items()
    }
    for movement in graph.movements:
        local_scores = teacher_by_tls.get(str(movement.traffic_light_id), {})
        scores[int(movement.movement_id)] = sum(
            float(local_scores.get(controlled_idx, 0.0)) for controlled_idx in movement.controlled_movement_indices
        )
    return tuple(scores)


def _teacher_selected_phases(
    graph: MovementGraph,
    programs: Mapping[str | TrafficLightId, TrafficLightProgram],
    teacher_scores: Sequence[float],
) -> dict[str, int]:
    selected: dict[str, int] = {}
    program_ids = {str(program.traffic_light_id) for program in programs.values()}
    for tls_id, incidence in graph.phase_incidences.items():
        if str(tls_id) not in program_ids:
            continue
        selected[str(tls_id)] = _select_from_incidence(incidence, teacher_scores)
    return selected


def _select_from_incidence(
    incidence: PhaseIncidence,
    teacher_scores: Sequence[float],
) -> int:
    movement_ids = tuple(int(value) for value in incidence.movement_ids)
    best_local_idx = 0
    best_score = _phase_score(incidence.rows[0], movement_ids, teacher_scores)
    for local_idx, row in enumerate(incidence.rows[1:], start=1):
        score = _phase_score(row, movement_ids, teacher_scores)
        if score > best_score:
            best_local_idx = local_idx
            best_score = score
    return best_local_idx


def _edge_indices(graph: MovementGraph) -> MovementEdgeIndices:
    return MovementEdgeIndices(
        input_lane_to_movement=_int_edges(graph.edges.input_lane_to_movement),
        output_lane_to_movement=_int_edges(graph.edges.output_lane_to_movement),
        movement_to_input_lane=_int_edges(graph.edges.movement_to_input_lane),
        movement_to_output_lane=_int_edges(graph.edges.movement_to_output_lane),
    )


def _int_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(src), int(dst)) for src, dst in edges)


def _stored_phase_incidence(incidence: PhaseIncidence) -> StoredPhaseIncidence:
    return StoredPhaseIncidence(
        sumo_phase_indices=tuple(int(value) for value in incidence.sumo_phase_indices),
        movement_ids=tuple(int(value) for value in incidence.movement_ids),
        rows=tuple(tuple(int(value) for value in row) for row in incidence.rows),
    )


def _lane_row_vector(row: LaneGroupFeatureRow) -> tuple[float, ...]:
    static = row.static
    dynamic = row.dynamic
    return (
        static.length_m,
        static.detector_length_m,
        static.num_lanes,
        static.speed_limit_mps,
        static.freeflow_travel_time_s,
        static.estimated_storage_capacity,
        static.is_short_link,
        dynamic.vehicle_count_detector,
        dynamic.moving_count_detector,
        dynamic.queue_length_m_detector,
        dynamic.occupancy_detector,
        dynamic.mean_speed_detector,
        dynamic.density_detector,
        dynamic.available_storage_detector_ratio,
        dynamic.arrival_rate_15s,
        dynamic.departure_rate_15s,
        dynamic.arrival_rate_60s,
        dynamic.departure_rate_60s,
        dynamic.detector_saturation,
        dynamic.vehicle_count_norm_detector,
        dynamic.moving_count_norm_detector,
        dynamic.queue_length_norm_detector,
        dynamic.approaching_queue_tail_count,
        dynamic.fast_approaching_queue_tail_count,
        dynamic.min_eta_to_queue_tail_s,
        dynamic.mean_eta_to_queue_tail_s,
        dynamic.predicted_arrivals_to_queue_tail_5s,
        dynamic.predicted_arrivals_to_queue_tail_10s,
        dynamic.predicted_arrivals_to_queue_tail_15s,
    )


def _movement_row_vector(row: MovementFeatureRow) -> tuple[float, ...]:
    static = row.static
    dynamic = row.dynamic
    return (
        static.num_underlying_controlled_links,
        dynamic.oracle_movement_demand,
        dynamic.oracle_movement_demand_norm,
        dynamic.was_green_last_decision,
    )


def _phase_score(
    row: Sequence[int],
    movement_ids: Sequence[int],
    teacher_scores: Sequence[float],
) -> float:
    return sum(float(teacher_scores[movement_id]) for enabled, movement_id in zip(row, movement_ids) if enabled)


def _sample_from_dict(data: Mapping[str, JsonValue]) -> MovementDatasetSample:
    phase_incidences = _require_json_object(data['phase_incidences'])
    teacher_selected_phase_by_tls = _require_json_object(data['teacher_selected_phase_by_tls'])
    return MovementDatasetSample(
        x_lane=_tuple2_float(data['x_lane']),
        x_movement=_tuple2_float(data['x_movement']),
        edge_indices=_edge_indices_from_json(data['edge_indices']),
        phase_incidences={
            traffic_light_id: _phase_incidence_from_json(incidence)
            for traffic_light_id, incidence in phase_incidences.items()
        },
        teacher_movement_scores=tuple(
            _json_float(value) for value in _require_sequence(data['teacher_movement_scores'])
        ),
        teacher_selected_phase_by_tls={
            str(traffic_light_id): _json_int(local_phase_index)
            for traffic_light_id, local_phase_index in teacher_selected_phase_by_tls.items()
        },
        metadata=_require_json_object(data['metadata']),
    )


def _edge_indices_from_json(value: JsonValue) -> MovementEdgeIndices:
    edge_indices = _require_json_object(value)
    return MovementEdgeIndices(
        input_lane_to_movement=_tuple2_int_pairs(edge_indices['input_lane_to_movement']),
        output_lane_to_movement=_tuple2_int_pairs(edge_indices['output_lane_to_movement']),
        movement_to_input_lane=_tuple2_int_pairs(edge_indices['movement_to_input_lane']),
        movement_to_output_lane=_tuple2_int_pairs(edge_indices['movement_to_output_lane']),
    )


def _phase_incidence_from_json(value: JsonValue) -> StoredPhaseIncidence:
    incidence = _require_json_object(value)
    return StoredPhaseIncidence(
        sumo_phase_indices=tuple(_json_int(item) for item in _require_sequence(incidence['sumo_phase_indices'])),
        movement_ids=tuple(_json_int(item) for item in _require_sequence(incidence['movement_ids'])),
        rows=_tuple2_int(incidence['rows']),
    )


def _tuple2_float(rows: JsonValue) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(_json_float(value) for value in _require_sequence(row)) for row in _require_sequence(rows))


def _tuple2_int(rows: JsonValue) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(_json_int(value) for value in _require_sequence(row)) for row in _require_sequence(rows))


def _tuple2_int_pairs(rows: JsonValue) -> tuple[tuple[int, int], ...]:
    pairs = []
    for row in _require_sequence(rows):
        values = tuple(_json_int(value) for value in _require_sequence(row))
        if len(values) != 2:
            raise ValueError('Expected edge indices with two columns.')
        pairs.append((values[0], values[1]))
    return tuple(pairs)


def _json_float(value: JsonValue) -> float:
    match value:
        case str() | int() | float() | bool():
            return float(value)
        case _:
            raise ValueError('Expected a JSON scalar convertible to float.')


def _json_int(value: JsonValue) -> int:
    match value:
        case str() | int() | float() | bool():
            return int(value)
        case _:
            raise ValueError('Expected a JSON scalar convertible to int.')


def _require_sequence(value: JsonValue) -> tuple[JsonValue, ...]:
    match value:
        case tuple():
            return value
        case list():
            return tuple(value)
        case _:
            raise ValueError('Expected a JSON array.')


def _require_json_object(value: JsonValue) -> JsonObject:
    match value:
        case dict():
            return value
        case _:
            raise ValueError('Expected a JSON object.')
