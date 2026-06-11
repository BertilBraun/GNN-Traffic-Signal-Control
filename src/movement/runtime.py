"""Centralized SUMO/TraCI runtime access for movement-based controllers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .min_green import MinGreenController
from .schema import LaneId, TrafficLightProgram
from .sumo_adapter import extract_programs_from_trafficlight_api
from .transition import SignalTransitionController

if 'SUMO_HOME' not in os.environ:
    raise EnvironmentError(
        'SUMO_HOME environment variable is not set. Point it to your SUMO installation directory.'
    )
import traci
import sumolib


class LaneQueueApi(Protocol):
    def getLastStepHaltingNumber(self, lane_id: LaneId) -> int:
        ...


@dataclass
class MovementControlRuntime:
    """Own SUMO lifecycle, signal state application, and control guardrails."""

    cfg_path: str | Path
    gui: bool = False
    seed: int = 42
    yellow_duration: int = 3
    min_green_steps: int = 2
    programs: dict[str, TrafficLightProgram] = field(default_factory=dict, init=False)
    _transition_controller: SignalTransitionController = field(init=False)
    _min_green_controller: MinGreenController = field(init=False)

    def __post_init__(self) -> None:
        self._transition_controller = SignalTransitionController(
            yellow_duration=self.yellow_duration
        )
        self._min_green_controller = MinGreenController(min_green_steps=self.min_green_steps)

    @property
    def lane_api(self) -> LaneQueueApi:
        self._require_started()
        return traci.lane

    def start(self) -> None:
        """Start SUMO and extract movement-aware traffic-light programs."""
        binary = sumolib.checkBinary('sumo-gui' if self.gui else 'sumo')
        command = [
            binary,
            '-c',
            str(self.cfg_path),
            '--seed',
            str(self.seed),
            '--no-step-log',
            'true',
        ]
        traci.start(command)
        self.programs = extract_programs_from_trafficlight_api(traci.trafficlight)
        if not self.programs:
            raise RuntimeError('No traffic lights with selectable movement phases were found.')

    def close(self) -> None:
        if traci is None:
            return
        traci.close()

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

    def current_states(self) -> dict[str, str]:
        return self._transition_controller.current_states()

    def apply_current_states(self) -> dict[str, str]:
        self._require_started()
        current_states = self.current_states()
        for tls_id, state in current_states.items():
            traci.trafficlight.setRedYellowGreenState(tls_id, state)
        return current_states

    def step(self) -> dict[str, str]:
        """Apply current signal states, advance SUMO one step, then transition timers."""
        applied_states = self.apply_current_states()
        traci.simulationStep()
        self._transition_controller.advance()
        return applied_states

    def is_running(self) -> bool:
        self._require_started()
        return traci.simulation.getMinExpectedNumber() > 0

    def _require_started(self) -> None:
        if not self.programs:
            raise RuntimeError('MovementControlRuntime.start() must be called first.')
