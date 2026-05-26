"""Extract movement-aware phase programs from SUMO-compatible data."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from .schema import ControlledMovement, SelectablePhase, TrafficLightProgram

GREEN_CHARS = frozenset({"G", "g"})
TRANSITION_CHARS = frozenset({"y", "Y"})


def is_selectable_green_state(state: str) -> bool:
    """Return True when a phase state is a direct selectable green phase."""
    if not state:
        return False
    if any(char in TRANSITION_CHARS for char in state):
        return False
    return any(char in GREEN_CHARS for char in state)


ControlledLink = tuple[str, str, str | None]
ControlledLinkInput = Sequence[ControlledLink | Sequence[ControlledLink]]


def _iter_signal_links(
    controlled_links: ControlledLinkInput,
) -> Iterable[tuple[int, ControlledLink]]:
    for signal_idx, entry in enumerate(controlled_links):
        if _is_controlled_link(entry):
            yield signal_idx, entry
            continue
        for link in entry:
            yield signal_idx, link


def _is_controlled_link(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    return isinstance(value[0], str) and isinstance(value[1], str)


def extract_traffic_light_program(
    tls_id: str,
    phase_states: Sequence[str],
    controlled_links: ControlledLinkInput,
) -> TrafficLightProgram:
    """Build a movement-aware traffic-light program from SUMO phase data.

    `controlled_links` must be ordered by SUMO signal index. Each item is
    `(incoming_lane_id, outgoing_lane_id, via_lane_id)`.
    """
    signal_links = list(_iter_signal_links(controlled_links))
    movements = tuple(
        ControlledMovement(
            movement_index=movement_idx,
            signal_index=signal_idx,
            incoming_lane_id=incoming_lane,
            outgoing_lane_id=outgoing_lane,
            via_lane_id=via_lane,
        )
        for movement_idx, (signal_idx, (incoming_lane, outgoing_lane, via_lane))
        in enumerate(signal_links)
    )

    selectable_phases: list[SelectablePhase] = []
    n_signals = len(controlled_links)
    for phase_idx, state in enumerate(phase_states):
        if len(state) != n_signals:
            raise ValueError(
                f"Phase {phase_idx} state length {len(state)} does not match "
                f"{n_signals} controlled links for traffic light {tls_id}."
            )
        if not is_selectable_green_state(state):
            continue

        green_signal_indices = {
            signal_idx
            for signal_idx, char in enumerate(state)
            if char in GREEN_CHARS
        }
        enabled = tuple(
            movement.movement_index
            for movement in movements
            if movement.signal_index in green_signal_indices
        )
        if enabled:
            selectable_phases.append(
                SelectablePhase(
                    sumo_phase_index=phase_idx,
                    state=state,
                    enabled_movement_indices=enabled,
                )
            )

    return TrafficLightProgram(
        tls_id=tls_id,
        movements=movements,
        selectable_phases=tuple(selectable_phases),
    )
