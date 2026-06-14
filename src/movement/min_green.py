"""Minimum-green decision filtering for movement-based controllers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MinGreenController:
    """Hold accepted green targets for a minimum number of decisions."""

    min_green_steps: int = 2
    _current_targets: dict[str, str] = field(default_factory=dict)
    _held_steps: dict[str, int] = field(default_factory=dict)

    def can_switch(self, traffic_light_id: str) -> bool:
        """Return whether a different target may be accepted now."""
        current_target = self._current_targets.get(traffic_light_id)
        if current_target is None or self.min_green_steps <= 0:
            return True
        return self._held_steps.get(traffic_light_id, 0) >= self.min_green_steps

    def current_target(self, traffic_light_id: str) -> str | None:
        """Return the currently accepted green target."""
        return self._current_targets.get(traffic_light_id)

    def filter_targets(self, desired_targets: dict[str, str]) -> dict[str, str]:
        """Return targets after enforcing per-traffic-light minimum green time."""
        if self.min_green_steps <= 0:
            self._current_targets.update(desired_targets)
            for tls_id in desired_targets:
                self._held_steps[tls_id] = 0
            return dict(desired_targets)

        accepted: dict[str, str] = {}
        for tls_id, desired_target in desired_targets.items():
            current_target = self._current_targets.get(tls_id)
            if current_target is None:
                self._current_targets[tls_id] = desired_target
                self._held_steps[tls_id] = 1
                accepted[tls_id] = desired_target
                continue

            held_steps = self._held_steps.get(tls_id, 0)
            if desired_target == current_target:
                self._held_steps[tls_id] = held_steps + 1
                accepted[tls_id] = current_target
                continue

            if held_steps >= self.min_green_steps:
                self._current_targets[tls_id] = desired_target
                self._held_steps[tls_id] = 1
                accepted[tls_id] = desired_target
            else:
                self._held_steps[tls_id] = held_steps + 1
                accepted[tls_id] = current_target

        stale_tls_ids = set(self._current_targets) - set(desired_targets)
        for tls_id in stale_tls_ids:
            self._current_targets.pop(tls_id, None)
            self._held_steps.pop(tls_id, None)

        return accepted
