"""Centralized SUMO/TraCI runtime access for movement-based controllers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .min_green import MinGreenController
from .schema import TrafficLightProgram
from .sumo_backend import (
    LaneApi,
    SimulationApi,
    SumoBackend,
    SumoBackendKind,
    VehicleApi,
    check_sumo_binary,
    create_sumo_backend,
)
from .sumo_adapter import extract_programs_from_trafficlight_api
from .transition import SignalTransitionController


@dataclass
class MovementControlRuntime:
    """Own SUMO lifecycle, signal state application, and control guardrails."""

    cfg_path: str | Path
    gui: bool = False
    seed: int = 42
    yellow_duration: int = 3
    yellow_start_delay: int = 0
    min_green_steps: int = 2
    time_to_teleport: int | None = None
    additional_sumo_args: Sequence[str] = ()
    backend_kind: SumoBackendKind = SumoBackendKind.TRACI
    programs: dict[str, TrafficLightProgram] = field(default_factory=dict, init=False)
    _transition_controller: SignalTransitionController = field(init=False)
    _min_green_controller: MinGreenController = field(init=False)
    _backend: SumoBackend = field(init=False)

    def __post_init__(self) -> None:
        self._transition_controller = SignalTransitionController(
            yellow_duration=self.yellow_duration,
            yellow_start_delay=self.yellow_start_delay,
        )
        self._min_green_controller = MinGreenController(min_green_steps=self.min_green_steps)
        self._backend = create_sumo_backend(self.backend_kind)

    @property
    def lane_api(self) -> LaneApi:
        self._require_started()
        return self._backend.lane

    @property
    def vehicle_api(self) -> VehicleApi:
        self._require_started()
        return self._backend.vehicle

    @property
    def simulation_api(self) -> SimulationApi:
        self._require_started()
        return self._backend.simulation

    def start(self) -> None:
        """Start SUMO and extract movement-aware traffic-light programs."""
        binary = check_sumo_binary(self.gui)
        command = [
            binary,
            '-c',
            str(self.cfg_path),
            '--seed',
            str(self.seed),
            '--no-step-log',
            'true',
            '--no-warnings',
            'true',
        ]
        if self.time_to_teleport is not None:
            command.extend(('--time-to-teleport', str(self.time_to_teleport)))
        command.extend(self.additional_sumo_args)
        self._backend.start(command)
        self.programs = extract_programs_from_trafficlight_api(self._backend.trafficlight)
        if not self.programs:
            raise RuntimeError('No traffic lights with selectable movement phases were found.')

    def close(self) -> None:
        self._backend.close()

    def decision_loop(self, steps: int, decision_interval: int):
        try:
            self.start()
            print(f'Loaded {len(self.programs)} movement-aware traffic-light programs.')
            for step in range(steps):
                if step % decision_interval == 0:
                    yield self

                self.step()
                if not self.is_running():
                    break
        finally:
            self.close()

    def request_targets(self, desired_targets: dict[str, str]) -> dict[str, str]:
        """Apply min-green filtering and enqueue accepted targets for transition."""
        accepted_targets = self._min_green_controller.filter_targets(desired_targets)
        self._transition_controller.set_targets(accepted_targets)
        return accepted_targets

    def allowed_target_states(self, traffic_light_id: str) -> tuple[str, ...]:
        """Return target greens that min-green permits at the next decision."""
        self._require_started()
        program = self.programs[traffic_light_id]
        current_target = self._min_green_controller.current_target(traffic_light_id)
        if current_target is None or self._min_green_controller.can_switch(traffic_light_id):
            return tuple(str(phase.state) for phase in program.selectable_phases)
        return (current_target,)

    def current_target_state(self, traffic_light_id: str) -> str | None:
        """Return the most recently accepted green target."""
        self._require_started()
        return self._min_green_controller.current_target(traffic_light_id)

    def current_states(self) -> dict[str, str]:
        return self._transition_controller.current_states()

    def apply_current_states(self) -> dict[str, str]:
        self._require_started()
        current_states = self.current_states()
        for tls_id, state in current_states.items():
            self._backend.trafficlight.setRedYellowGreenState(tls_id, state)
        return current_states

    def step(self) -> dict[str, str]:
        """Apply current signal states, advance SUMO one step, then transition timers."""
        applied_states = self.apply_current_states()
        self._backend.simulation_step()
        self._transition_controller.advance()
        return applied_states

    def is_running(self) -> bool:
        self._require_started()
        return self._backend.simulation.getMinExpectedNumber() > 0

    def _require_started(self) -> None:
        if not self.programs:
            raise RuntimeError('MovementControlRuntime.start() must be called first.')
