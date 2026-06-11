"""Traffic-light phase synthesis from SUMO movement conflicts."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

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


class PhaseGenerationMode(str, Enum):
    CONFLICT_EDGE = "conflict-edge"
    PROTECTED = "protected"


@dataclass(frozen=True)
class TrafficLightLinkSpec:
    traffic_light_link_index: int
    approach: str
    direction: str
    incoming_lane_id: LaneId | None = None
    outgoing_lane_id: LaneId | None = None
    outgoing_edge_id: str | None = None
    request_index: int | None = None

    @property
    def axis(self) -> str:
        return "vertical" if self.approach in {"north", "south"} else "horizontal"


def synthesize_traffic_light_program(
    traffic_light_id: TrafficLightId,
    links: list[TrafficLightLinkSpec],
    mode: PhaseGenerationMode,
    are_foes: Callable[[int, int], bool],
) -> TrafficLightProgram:
    number_of_links = max(link.traffic_light_link_index for link in links) + 1
    phase_states = synthesize_phase_states(
        links=links,
        number_of_links=number_of_links,
        mode=mode,
        are_foes=are_foes,
    )
    movements = tuple(
        ControlledMovement(
            movement_index=MovementIndex(index),
            signal_index=SignalIndex(link.traffic_light_link_index),
            incoming_lane_id=_required_lane_id(link.incoming_lane_id, "incoming", link),
            outgoing_lane_id=_required_lane_id(link.outgoing_lane_id, "outgoing", link),
        )
        for index, link in enumerate(sorted(links, key=lambda item: item.traffic_light_link_index))
    )
    selectable_phases = tuple(
        SelectablePhase(
            sumo_phase_index=SumoPhaseIndex(phase_index),
            state=phase_state,
            enabled_movement_indices=tuple(
                movement.movement_index
                for movement in movements
                if phase_state[movement.signal_index] == "G"
            ),
        )
        for phase_index, phase_state in enumerate(phase_states)
    )
    return TrafficLightProgram(
        traffic_light_id=traffic_light_id,
        movements=movements,
        selectable_phases=selectable_phases,
    )


def synthesize_phase_states(
    links: list[TrafficLightLinkSpec],
    number_of_links: int,
    mode: PhaseGenerationMode,
    are_foes: Callable[[int, int], bool],
) -> tuple[PhaseState, ...]:
    match mode:
        case PhaseGenerationMode.CONFLICT_EDGE:
            return build_conflict_phase_states(
                links=links,
                number_of_links=number_of_links,
                are_foes=are_foes,
            )
        case PhaseGenerationMode.PROTECTED:
            return build_protected_phase_states(
                links=links,
                number_of_links=number_of_links,
            )


def build_protected_phase_states(
    links: list[TrafficLightLinkSpec],
    number_of_links: int,
) -> tuple[PhaseState, ...]:
    phase_rules: list[list[tuple[set[str], set[str]]]] = [
        [({"north", "south"}, {"r", "s"})],
        [({"north", "south"}, {"l"}), ({"east", "west"}, {"r"})],
        [({"east", "west"}, {"r", "s"})],
        [({"east", "west"}, {"l"}), ({"north", "south"}, {"r"})],
        [({"north"}, {"r", "s", "l"})],
        [({"south"}, {"r", "s", "l"})],
        [({"east"}, {"r", "s", "l"})],
        [({"west"}, {"r", "s", "l"})],
    ]
    states: list[PhaseState] = []
    for rule in phase_rules:
        chars = ["r"] * number_of_links
        phase_links = [
            link for link in links
            if any(
                link.approach in approaches and link.direction.lower() in directions
                for approaches, directions in rule
            )
        ]
        if (
            _has_movement_conflict(phase_links)
            or _partially_serves_shared_incoming_lane(phase_links, links)
        ):
            continue
        for link in phase_links:
            chars[link.traffic_light_link_index] = "G"
        state = PhaseState("".join(chars))
        if "G" in state and state not in states:
            states.append(state)
    return tuple(states)


def _required_lane_id(
    lane_id: LaneId | None,
    lane_role: str,
    link: TrafficLightLinkSpec,
) -> LaneId:
    if lane_id is None:
        raise ValueError(
            f"Missing {lane_role} lane id for controlled link "
            f"{link.traffic_light_link_index}."
        )
    return lane_id


def build_conflict_phase_states(
    links: list[TrafficLightLinkSpec],
    number_of_links: int,
    are_foes: Callable[[int, int], bool],
) -> tuple[PhaseState, ...]:
    indexed_links = sorted(links, key=lambda link: link.traffic_light_link_index)
    valid_sets: list[frozenset[int]] = []
    for size in range(1, len(indexed_links) + 1):
        for phase_links in itertools.combinations(indexed_links, size):
            candidate_links = list(phase_links)
            if (
                _has_sumo_or_outgoing_edge_conflict(candidate_links, are_foes)
                or _partially_serves_shared_incoming_lane(candidate_links, indexed_links)
            ):
                continue
            valid_sets.append(frozenset(link.traffic_light_link_index for link in phase_links))

    maximal_sets = [
        candidate for candidate in valid_sets
        if not any(candidate < other for other in valid_sets)
    ]
    states: list[PhaseState] = []
    for phase_set in sorted(maximal_sets, key=lambda item: (-len(item), sorted(item))):
        chars = ["r"] * number_of_links
        for traffic_light_link_index in phase_set:
            chars[traffic_light_link_index] = "G"
        state = PhaseState("".join(chars))
        if state not in states:
            states.append(state)
    return tuple(states)


def _has_sumo_or_outgoing_edge_conflict(
    links: list[TrafficLightLinkSpec],
    are_foes: Callable[[int, int], bool],
) -> bool:
    for index, first in enumerate(links):
        for second in links[index + 1:]:
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


def _partially_serves_shared_incoming_lane(
    phase_links: list[TrafficLightLinkSpec],
    all_links: list[TrafficLightLinkSpec],
) -> bool:
    enabled_indices = {
        link.traffic_light_link_index
        for link in phase_links
    }
    lane_groups: dict[LaneId, set[int]] = {}
    for link in all_links:
        if link.incoming_lane_id is None:
            continue
        lane_groups.setdefault(link.incoming_lane_id, set()).add(link.traffic_light_link_index)

    for shared_indices in lane_groups.values():
        if len(shared_indices) < 2:
            continue
        enabled_shared_indices = enabled_indices & shared_indices
        if enabled_shared_indices and enabled_shared_indices != shared_indices:
            return True
    return False


def _sumo_requests_are_foes(
    first: TrafficLightLinkSpec,
    second: TrafficLightLinkSpec,
    are_foes: Callable[[int, int], bool],
) -> bool:
    first_request = (
        first.request_index
        if first.request_index is not None
        else first.traffic_light_link_index
    )
    second_request = (
        second.request_index
        if second.request_index is not None
        else second.traffic_light_link_index
    )
    return bool(
        are_foes(first_request, second_request)
        or are_foes(second_request, first_request)
    )


def _has_movement_conflict(links: list[TrafficLightLinkSpec]) -> bool:
    for index, first in enumerate(links):
        for second in links[index + 1:]:
            if _movements_conflict(first, second):
                return True
    return False


def _movements_conflict(
    first: TrafficLightLinkSpec,
    second: TrafficLightLinkSpec,
) -> bool:
    if first.approach == second.approach:
        return False

    first_direction = first.direction.lower()
    second_direction = second.direction.lower()
    if _opposite_approaches(first.approach, second.approach):
        return not (
            first_direction in {"r", "s"} and second_direction in {"r", "s"}
            or first_direction == "l" and second_direction == "l"
        )

    return not (
        first_direction == "l" and second_direction == "r"
        or first_direction == "r" and second_direction == "l"
    )


def _opposite_approaches(first: str, second: str) -> bool:
    return {first, second} in (
        {"north", "south"},
        {"east", "west"},
    )
