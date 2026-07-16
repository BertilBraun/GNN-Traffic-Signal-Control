from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.training.ppo.reward import SpeedChangeTracker, advance_and_reward
from src.movement.training.ppo.types import PpoRewardMode, PpoSpeedChangeMode, RolloutContext
from src.movement.graph_schema import MovementGraph, TypedMovementEdges


class FakeRuntime:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> dict[str, str]:
        self.step_count += 1
        return {}

    def is_running(self) -> bool:
        return True


class FakeSimulation:
    def __init__(self) -> None:
        self.departed_number = 0
        self.arrived_number = 0

    def getStartingTeleportNumber(self) -> int:
        return 0

    def getDepartedNumber(self) -> int:
        return self.departed_number

    def getArrivedNumber(self) -> int:
        return self.arrived_number


class FakeLane:
    def __init__(self) -> None:
        self.vehicle_number_calls = 0
        self.mean_speed_calls = 0
        self.vehicle_ids_calls = 0
        self.mean_speeds_by_lane = {
            'lane_a': [5.0, 10.0],
            'lane_b': [10.0, 5.0],
        }
        self.mean_speed_index_by_lane = {'lane_a': 0, 'lane_b': 0}
        self.vehicle_ids_by_lane = {
            'lane_a': [('lane_a_vehicle_1', 'lane_a_vehicle_2'), ('lane_a_vehicle_2',)],
            'lane_b': [('lane_b_vehicle',), ('lane_b_vehicle',)],
        }
        self.vehicle_ids_index_by_lane = {'lane_a': 0, 'lane_b': 0}

    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        self.vehicle_number_calls += 1
        return {'lane_a': 2, 'lane_b': 1}[lane_id]

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        self.mean_speed_calls += 1
        index = self.mean_speed_index_by_lane[lane_id]
        self.mean_speed_index_by_lane[lane_id] += 1
        return self.mean_speeds_by_lane[lane_id][min(index, len(self.mean_speeds_by_lane[lane_id]) - 1)]

    def getLastStepVehicleIDs(self, lane_id: str) -> tuple[str, ...]:
        self.vehicle_ids_calls += 1
        index = self.vehicle_ids_index_by_lane[lane_id]
        self.vehicle_ids_index_by_lane[lane_id] += 1
        values = self.vehicle_ids_by_lane[lane_id]
        return values[min(index, len(values) - 1)]


class FakeVehicle:
    def __init__(self) -> None:
        self.speed_calls = 0

    def getSpeed(self, vehicle_id: str) -> float:
        self.speed_calls += 1
        return 5.0


class FakeTraci:
    def __init__(self) -> None:
        self.simulation = FakeSimulation()
        self.lane = FakeLane()
        self.vehicle = FakeVehicle()


def test_advance_and_reward_reuses_lane_delay_snapshot() -> None:
    fake_traci = FakeTraci()

    advance_and_reward(
        runtime=FakeRuntime(),
        lane_api=fake_traci.lane,
        simulation_api=fake_traci.simulation,
        context=_context(),
        decision_interval=3,
        global_reward_weight=0.1,
        flow_reward_weight=0.1,
        reward_mode=PpoRewardMode.DELAY_DENSITY,
        throughput_reward_weight=0.0,
        progress_reward_weight=0.0,
        discharge_reward_weight=0.0,
        gridlock_penalty_weight=0.0,
        speed_change_weight=0.0,
        speed_change_mode=PpoSpeedChangeMode.ABSOLUTE,
        switch_penalty_weight=0.0,
        phase_switches=(False, False),
        reward_sample_interval=3,
        reward_clip=1.0,
        teleport_penalty=0.0,
        speed_change_tracker=SpeedChangeTracker(),
    )

    assert fake_traci.lane.vehicle_number_calls == 2
    assert fake_traci.lane.mean_speed_calls == 2
    assert fake_traci.lane.vehicle_ids_calls == 0
    assert fake_traci.vehicle.speed_calls == 0


def test_advance_and_reward_uses_lane_speed_changes_without_vehicle_queries() -> None:
    fake_traci = FakeTraci()

    result = advance_and_reward(
        runtime=FakeRuntime(),
        lane_api=fake_traci.lane,
        simulation_api=fake_traci.simulation,
        context=_context(),
        decision_interval=2,
        global_reward_weight=0.1,
        flow_reward_weight=0.1,
        reward_mode=PpoRewardMode.DELAY_DENSITY,
        throughput_reward_weight=0.0,
        progress_reward_weight=0.0,
        discharge_reward_weight=0.0,
        gridlock_penalty_weight=0.0,
        speed_change_weight=0.02,
        speed_change_mode=PpoSpeedChangeMode.ABSOLUTE,
        switch_penalty_weight=0.0,
        phase_switches=(False, False),
        reward_sample_interval=1,
        reward_clip=1.0,
        teleport_penalty=0.0,
        speed_change_tracker=SpeedChangeTracker(),
    )

    assert fake_traci.lane.vehicle_ids_calls == 0
    assert fake_traci.vehicle.speed_calls == 0
    assert result.speed_change_densities == pytest.approx((0.005, 0.0025))


def test_advance_and_reward_braking_mode_does_not_penalize_acceleration() -> None:
    fake_traci = FakeTraci()

    result = advance_and_reward(
        runtime=FakeRuntime(),
        lane_api=fake_traci.lane,
        simulation_api=fake_traci.simulation,
        context=_context(),
        decision_interval=2,
        global_reward_weight=0.0,
        flow_reward_weight=0.0,
        reward_mode=PpoRewardMode.THROUGHPUT,
        throughput_reward_weight=0.0,
        progress_reward_weight=0.0,
        discharge_reward_weight=0.0,
        gridlock_penalty_weight=0.0,
        speed_change_weight=1.0,
        speed_change_mode=PpoSpeedChangeMode.BRAKING,
        switch_penalty_weight=0.0,
        phase_switches=(False, False),
        reward_sample_interval=1,
        reward_clip=1.0,
        teleport_penalty=0.0,
        speed_change_tracker=SpeedChangeTracker(),
    )

    assert result.speed_change_densities == pytest.approx((0.0, 0.0025))


def test_advance_and_reward_adds_local_discharge_density() -> None:
    fake_traci = FakeTraci()

    result = advance_and_reward(
        runtime=FakeRuntime(),
        lane_api=fake_traci.lane,
        simulation_api=fake_traci.simulation,
        context=_context(),
        decision_interval=1,
        global_reward_weight=0.0,
        flow_reward_weight=0.0,
        reward_mode=PpoRewardMode.THROUGHPUT,
        throughput_reward_weight=0.0,
        progress_reward_weight=0.0,
        discharge_reward_weight=1.0,
        gridlock_penalty_weight=0.0,
        speed_change_weight=0.0,
        speed_change_mode=PpoSpeedChangeMode.BRAKING,
        switch_penalty_weight=0.0,
        phase_switches=(False, False),
        reward_sample_interval=1,
        reward_clip=1.0,
        teleport_penalty=0.0,
        speed_change_tracker=SpeedChangeTracker(),
    )

    assert result.discharge_densities == pytest.approx((0.01, 0.0))
    assert result.raw_rewards == pytest.approx((0.01, 0.0))


def test_advance_and_reward_flow_bonus_uses_arrived_vehicles() -> None:
    fake_traci = FakeTraci()
    fake_traci.simulation.departed_number = 20
    fake_traci.simulation.arrived_number = 4

    result = advance_and_reward(
        runtime=FakeRuntime(),
        lane_api=fake_traci.lane,
        simulation_api=fake_traci.simulation,
        context=_context(),
        decision_interval=1,
        global_reward_weight=0.0,
        flow_reward_weight=1.0,
        reward_mode=PpoRewardMode.DELAY_DENSITY,
        throughput_reward_weight=0.0,
        progress_reward_weight=0.0,
        discharge_reward_weight=0.0,
        gridlock_penalty_weight=0.0,
        speed_change_weight=0.0,
        speed_change_mode=PpoSpeedChangeMode.ABSOLUTE,
        switch_penalty_weight=0.0,
        phase_switches=(False, False),
        reward_sample_interval=1,
        reward_clip=1.0,
        teleport_penalty=0.0,
        speed_change_tracker=SpeedChangeTracker(),
    )

    assert result.raw_rewards == pytest.approx((1.99, 2.0))


def _context() -> RolloutContext:
    return RolloutContext(
        graph=MovementGraph(
            lane_groups=(),
            movements=(),
            edges=TypedMovementEdges(
                input_lane_to_movement=(),
                output_lane_to_movement=(),
                movement_to_input_lane=(),
                movement_to_output_lane=(),
            ),
            lane_lane_connectors=(),
            lane_movement_metadata=(),
            phase_incidences={},
            lane_group_id_by_edge={},
            movement_id_by_key={},
        ),
        traffic_light_ids=('tls_a', 'tls_b'),
        movement_ids_by_traffic_light=(),
        lane_ids_by_edge={},
        lane_geometries={},
        incoming_lanes_by_traffic_light={
            'tls_a': ('lane_a',),
            'tls_b': ('lane_b',),
        },
        incoming_lane_length_by_traffic_light={
            'tls_a': 100.0,
            'tls_b': 100.0,
        },
        speed_limit_by_lane={
            'lane_a': 10.0,
            'lane_b': 10.0,
        },
        all_incoming_lane_ids=('lane_a', 'lane_b'),
        all_incoming_lane_length_m=200.0,
    )
