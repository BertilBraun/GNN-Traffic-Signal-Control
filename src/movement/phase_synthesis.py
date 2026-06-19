"""Traffic-light phase synthesis from SUMO movement conflicts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .schema import (
    ControlledMovement,
    LaneId,
    MovementIndex,
    PhaseState,
    SelectablePhase,
    SignalIndex,
    SumoPhaseIndex,
    TrafficLightProgram,
    TrafficLightId,
)


@dataclass(frozen=True)
class TrafficLightLinkSpec:
    traffic_light_link_index: int
    incoming_lane_id: LaneId | None = None
    outgoing_lane_id: LaneId | None = None
    outgoing_edge_id: str | None = None
    request_index: int | None = None


@dataclass(frozen=True)
class _AtomicLinkGroup:
    links: tuple[TrafficLightLinkSpec, ...]
    enabled_indices: frozenset[int]


MAX_SYNTHESIZED_PHASES = 128


def synthesize_traffic_light_program(
    traffic_light_id: TrafficLightId,
    links: list[TrafficLightLinkSpec],
    are_foes: Callable[[int, int], bool],
) -> TrafficLightProgram:
    number_of_links = max(link.traffic_light_link_index for link in links) + 1
    phase_states = build_conflict_phase_states(
        links=links,
        number_of_links=number_of_links,
        are_foes=are_foes,
    )
    movements = tuple(
        ControlledMovement(
            movement_index=MovementIndex(index),
            signal_index=SignalIndex(link.traffic_light_link_index),
            incoming_lane_id=_required_lane_id(link.incoming_lane_id, 'incoming', link),
            outgoing_lane_id=_required_lane_id(link.outgoing_lane_id, 'outgoing', link),
        )
        for index, link in enumerate(sorted(links, key=lambda item: item.traffic_light_link_index))
    )
    selectable_phases = tuple(
        SelectablePhase(
            sumo_phase_index=SumoPhaseIndex(phase_index),
            state=phase_state,
            enabled_movement_indices=tuple(
                movement.movement_index for movement in movements if phase_state[movement.signal_index] == 'G'
            ),
        )
        for phase_index, phase_state in enumerate(phase_states)
    )
    return TrafficLightProgram(
        traffic_light_id=traffic_light_id,
        movements=movements,
        selectable_phases=selectable_phases,
    )


def _required_lane_id(
    lane_id: LaneId | None,
    lane_role: str,
    link: TrafficLightLinkSpec,
) -> LaneId:
    if lane_id is None:
        raise ValueError(f'Missing {lane_role} lane id for controlled link {link.traffic_light_link_index}.')
    return lane_id


def build_conflict_phase_states(
    links: list[TrafficLightLinkSpec],
    number_of_links: int,
    are_foes: Callable[[int, int], bool],
) -> tuple[PhaseState, ...]:
    indexed_links = sorted(links, key=lambda link: link.traffic_light_link_index)
    atomic_groups = _atomic_link_groups(indexed_links, are_foes)
    maximal_sets = _maximal_compatible_group_sets(atomic_groups, are_foes)
    states: list[PhaseState] = []
    for phase_set in sorted(maximal_sets, key=lambda item: (-len(item), sorted(item))):
        chars = ['r'] * number_of_links
        for traffic_light_link_index in phase_set:
            chars[traffic_light_link_index] = 'G'
        state = PhaseState(''.join(chars))
        if state not in states:
            states.append(state)
    return tuple(states)


def _atomic_link_groups(
    indexed_links: list[TrafficLightLinkSpec],
    are_foes: Callable[[int, int], bool],
) -> tuple[_AtomicLinkGroup, ...]:
    grouped_links: dict[LaneId | int, list[TrafficLightLinkSpec]] = {}
    for link in indexed_links:
        key: LaneId | int = (
            link.incoming_lane_id if link.incoming_lane_id is not None else link.traffic_light_link_index
        )
        grouped_links.setdefault(key, []).append(link)
    groups = tuple(
        _AtomicLinkGroup(
            links=tuple(links),
            enabled_indices=frozenset(link.traffic_light_link_index for link in links),
        )
        for _key, links in sorted(
            grouped_links.items(),
            key=lambda item: min(link.traffic_light_link_index for link in item[1]),
        )
        if not _has_sumo_or_outgoing_edge_conflict(links, are_foes)
    )
    return groups


def _maximal_compatible_group_sets(
    atomic_groups: tuple[_AtomicLinkGroup, ...],
    are_foes: Callable[[int, int], bool],
) -> tuple[frozenset[int], ...]:
    neighbors = _compatible_group_neighbors(atomic_groups, are_foes)
    maximal_group_sets: list[frozenset[int]] = []
    _bron_kerbosch_maximal_cliques(
        chosen=frozenset(),
        candidates=frozenset(range(len(atomic_groups))),
        excluded=frozenset(),
        neighbors=neighbors,
        maximal_group_sets=maximal_group_sets,
    )
    if len(maximal_group_sets) > MAX_SYNTHESIZED_PHASES:
        raise ValueError(
            f'Traffic-light program synthesis produced more than {MAX_SYNTHESIZED_PHASES} '
            'maximal compatible phases. Inspect or simplify this junction before training.'
        )
    return tuple(
        frozenset(link_index for group_index in group_set for link_index in atomic_groups[group_index].enabled_indices)
        for group_set in maximal_group_sets
        if group_set
    )


def _compatible_group_neighbors(
    atomic_groups: tuple[_AtomicLinkGroup, ...],
    are_foes: Callable[[int, int], bool],
) -> dict[int, frozenset[int]]:
    neighbors: dict[int, set[int]] = {index: set() for index in range(len(atomic_groups))}
    for first_index, first in enumerate(atomic_groups):
        for second_index, second in enumerate(atomic_groups[first_index + 1 :], start=first_index + 1):
            if _groups_are_compatible(first, second, are_foes):
                neighbors[first_index].add(second_index)
                neighbors[second_index].add(first_index)
    return {index: frozenset(values) for index, values in neighbors.items()}


def _groups_are_compatible(
    first: _AtomicLinkGroup,
    second: _AtomicLinkGroup,
    are_foes: Callable[[int, int], bool],
) -> bool:
    return not _has_sumo_or_outgoing_edge_conflict([*first.links, *second.links], are_foes)


def _bron_kerbosch_maximal_cliques(
    chosen: frozenset[int],
    candidates: frozenset[int],
    excluded: frozenset[int],
    neighbors: dict[int, frozenset[int]],
    maximal_group_sets: list[frozenset[int]],
) -> None:
    if not candidates and not excluded:
        maximal_group_sets.append(chosen)
        if len(maximal_group_sets) > MAX_SYNTHESIZED_PHASES:
            raise ValueError(
                f'Traffic-light program synthesis produced more than {MAX_SYNTHESIZED_PHASES} '
                'maximal compatible phases. Inspect or simplify this junction before training.'
            )
        return
    pivot = _max_degree_pivot(candidates | excluded, neighbors)
    pivot_neighbors = neighbors[pivot] if pivot is not None else frozenset()
    for group_index in sorted(candidates - pivot_neighbors):
        _bron_kerbosch_maximal_cliques(
            chosen=chosen | {group_index},
            candidates=candidates & neighbors[group_index],
            excluded=excluded & neighbors[group_index],
            neighbors=neighbors,
            maximal_group_sets=maximal_group_sets,
        )
        candidates = candidates - {group_index}
        excluded = excluded | {group_index}


def _max_degree_pivot(
    group_indices: frozenset[int],
    neighbors: dict[int, frozenset[int]],
) -> int | None:
    if not group_indices:
        return None
    return max(group_indices, key=lambda group_index: len(neighbors[group_index] & group_indices))


def _has_sumo_or_outgoing_edge_conflict(
    links: list[TrafficLightLinkSpec],
    are_foes: Callable[[int, int], bool],
) -> bool:
    for index, first in enumerate(links):
        for second in links[index + 1 :]:
            if _sumo_requests_are_foes(first, second, are_foes):
                return True
            if _same_outgoing_edge_conflict(first, second):
                return True
    return False


def _same_outgoing_edge_conflict(
    first: TrafficLightLinkSpec,
    second: TrafficLightLinkSpec,
) -> bool:
    if first.outgoing_edge_id is None or first.outgoing_edge_id != second.outgoing_edge_id:
        return False
    return first.incoming_lane_id is None or first.incoming_lane_id != second.incoming_lane_id


def _sumo_requests_are_foes(
    first: TrafficLightLinkSpec,
    second: TrafficLightLinkSpec,
    are_foes: Callable[[int, int], bool],
) -> bool:
    first_request = first.request_index if first.request_index is not None else first.traffic_light_link_index
    second_request = second.request_index if second.request_index is not None else second.traffic_light_link_index
    return bool(are_foes(first_request, second_request) or are_foes(second_request, first_request))
