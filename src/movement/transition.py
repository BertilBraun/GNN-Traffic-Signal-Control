"""Signal-state transition helpers for movement-based controllers."""
from __future__ import annotations

from dataclasses import dataclass, field


def yellow_from_green_state(state: str) -> str:
    """Convert active green links in `state` to yellow, preserving red links."""
    return "".join("y" if char in {"G", "g"} else char for char in state)


def transition_yellow_state(current_state: str, target_state: str) -> str:
    """Yellow only green links that will not remain green in `target_state`."""
    if len(current_state) != len(target_state):
        raise ValueError("Current and target signal states must have the same length.")

    chars: list[str] = []
    for current_char, target_char in zip(current_state, target_state):
        current_green = current_char in {"G", "g"}
        target_green = target_char in {"G", "g"}
        if current_green and not target_green:
            chars.append("y")
        else:
            chars.append(current_char)
    return "".join(chars)


@dataclass
class SignalTransitionController:
    """Hold selected greens and insert yellow when targets change."""

    yellow_duration: int = 3
    _current_states: dict[str, str] = field(default_factory=dict)
    _pending_green_states: dict[str, str] = field(default_factory=dict)
    _yellow_remaining: dict[str, int] = field(default_factory=dict)

    def set_targets(self, target_states: dict[str, str]) -> None:
        """Update desired green states for each traffic light."""
        for tls_id, target_state in target_states.items():
            current_state = self._current_states.get(tls_id)
            if current_state is None or current_state == target_state:
                self._current_states[tls_id] = target_state
                self._pending_green_states.pop(tls_id, None)
                self._yellow_remaining[tls_id] = 0
                continue

            if self.yellow_duration <= 0:
                self._current_states[tls_id] = target_state
                self._pending_green_states.pop(tls_id, None)
                self._yellow_remaining[tls_id] = 0
                continue

            self._current_states[tls_id] = transition_yellow_state(current_state, target_state)
            self._pending_green_states[tls_id] = target_state
            self._yellow_remaining[tls_id] = self.yellow_duration

    def current_states(self) -> dict[str, str]:
        """Return states that should be applied this simulation step."""
        return dict(self._current_states)

    def advance(self) -> None:
        """Advance one simulation second after current states were applied."""
        for tls_id in list(self._yellow_remaining):
            remaining = self._yellow_remaining[tls_id]
            if remaining <= 0:
                continue

            remaining -= 1
            self._yellow_remaining[tls_id] = remaining
            if remaining == 0:
                pending = self._pending_green_states.pop(tls_id, None)
                if pending is not None:
                    self._current_states[tls_id] = pending
