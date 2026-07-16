"""Reward calculation for movement PPO rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from src.movement.runtime import MovementControlRuntime
from src.movement.sumo_backend import LaneApi, SimulationApi
from src.movement.training.ppo.types import (
    IntervalRewardResult,
    PpoRewardMode,
    PpoRewardObjective,
    RolloutContext,
)


def objective_rewards(
    rewards: Sequence[float],
    objective: PpoRewardObjective,
) -> tuple[float, ...]:
    return tuple(objective.multiplier * reward for reward in rewards)


def advance_and_reward(
    runtime: MovementControlRuntime,
    lane_api: LaneApi,
    simulation_api: SimulationApi,
    context: RolloutContext,
    decision_interval: int,
    global_reward_weight: float,
    flow_reward_weight: float,
    reward_mode: PpoRewardMode,
    throughput_reward_weight: float,
    progress_reward_weight: float,
    gridlock_penalty_weight: float,
    speed_change_weight: float,
    switch_penalty_weight: float,
    phase_switches: Sequence[bool],
    reward_sample_interval: int,
    reward_clip: float,
    teleport_penalty: float,
    speed_change_tracker: 'SpeedChangeTracker',
) -> IntervalRewardResult:
    if teleport_penalty < 0.0:
        raise ValueError('teleport_penalty must not be negative.')
    if flow_reward_weight < 0.0:
        raise ValueError('flow_reward_weight must not be negative.')
    if throughput_reward_weight < 0.0:
        raise ValueError('throughput_reward_weight must not be negative.')
    if progress_reward_weight < 0.0:
        raise ValueError('progress_reward_weight must not be negative.')
    if gridlock_penalty_weight < 0.0:
        raise ValueError('gridlock_penalty_weight must not be negative.')
    if speed_change_weight < 0.0:
        raise ValueError('speed_change_weight must not be negative.')
    if switch_penalty_weight < 0.0:
        raise ValueError('switch_penalty_weight must not be negative.')
    if len(phase_switches) != len(context.traffic_light_ids):
        raise ValueError('phase_switches must contain one value per traffic light.')
    if reward_sample_interval <= 0:
        raise ValueError('reward_sample_interval must be positive.')
    if reward_sample_interval > decision_interval:
        raise ValueError('reward_sample_interval must not exceed decision_interval.')
    local_delay_sums = {traffic_light_id: 0.0 for traffic_light_id in context.traffic_light_ids}
    progress_sums = {traffic_light_id: 0.0 for traffic_light_id in context.traffic_light_ids}
    speed_change_sums = {traffic_light_id: 0.0 for traffic_light_id in context.traffic_light_ids}
    global_delay_sum = 0.0
    arrived_vehicle_count = 0
    teleport_count = 0
    simulated_steps = 0
    reward_sumo_step_seconds = 0.0
    reward_lane_query_seconds = 0.0
    reward_vehicle_query_seconds = 0.0
    aggregation_started = perf_counter()
    previous_reward_sample_step = 0
    for _step in range(decision_interval):
        step_started = perf_counter()
        runtime.step()
        reward_sumo_step_seconds += perf_counter() - step_started
        simulated_steps += 1
        arrived_vehicle_count += int(simulation_api.getArrivedNumber())
        teleport_count += int(simulation_api.getStartingTeleportNumber())
        running = runtime.is_running()
        should_sample_reward = (
            simulated_steps % reward_sample_interval == 0 or simulated_steps == decision_interval or not running
        )
        if not should_sample_reward:
            continue
        represented_steps = simulated_steps - previous_reward_sample_step
        previous_reward_sample_step = simulated_steps
        lane_query_started = perf_counter()
        delay_snapshot = lane_delay_snapshot(
            lane_api=lane_api,
            lane_ids=context.all_incoming_lane_ids,
            speed_limit_by_lane=context.speed_limit_by_lane,
        )
        reward_lane_query_seconds += perf_counter() - lane_query_started
        speed_change_by_lane = speed_change_tracker.observe_lane_snapshot(
            snapshot=delay_snapshot,
            speed_limit_by_lane=context.speed_limit_by_lane,
        )
        for traffic_light_id in context.traffic_light_ids:
            local_delay_sums[traffic_light_id] += (
                delay_snapshot.density(
                    lane_ids=context.incoming_lanes_by_traffic_light[traffic_light_id],
                    total_lane_length_m=context.incoming_lane_length_by_traffic_light[traffic_light_id],
                )
                * represented_steps
            )
            progress_sums[traffic_light_id] += (
                delay_snapshot.progress_density(
                    lane_ids=context.incoming_lanes_by_traffic_light[traffic_light_id],
                    total_lane_length_m=context.incoming_lane_length_by_traffic_light[traffic_light_id],
                    speed_limit_by_lane=context.speed_limit_by_lane,
                )
                * represented_steps
            )
            speed_change_sums[traffic_light_id] += speed_change_density(
                lane_ids=context.incoming_lanes_by_traffic_light[traffic_light_id],
                speed_change_by_lane=speed_change_by_lane,
                total_lane_length_m=context.incoming_lane_length_by_traffic_light[traffic_light_id],
            )
        global_delay_sum += (
            delay_snapshot.density(
                lane_ids=context.all_incoming_lane_ids,
                total_lane_length_m=context.all_incoming_lane_length_m,
            )
            * represented_steps
        )
        if not running:
            break
    reward_aggregation_seconds = perf_counter() - aggregation_started - reward_sumo_step_seconds
    reward_aggregation_seconds -= reward_lane_query_seconds + reward_vehicle_query_seconds
    local_delay_densities = tuple(
        local_delay_sums[traffic_light_id] / max(1, simulated_steps) for traffic_light_id in context.traffic_light_ids
    )
    speed_change_densities = tuple(
        speed_change_sums[traffic_light_id] / max(1, simulated_steps) for traffic_light_id in context.traffic_light_ids
    )
    progress_densities = tuple(
        progress_sums[traffic_light_id] / max(1, simulated_steps) for traffic_light_id in context.traffic_light_ids
    )
    global_delay_density = global_delay_sum / max(1, simulated_steps)
    flow_rate_per_signal = arrived_vehicle_count / max(1, simulated_steps) / max(1, len(context.traffic_light_ids))
    raw_rewards = tuple(
        interval_reward(
            reward_mode=reward_mode,
            local_delay_density=local_delay_density,
            global_delay_density=global_delay_density,
            flow_rate_per_signal=flow_rate_per_signal,
            progress_density=progress_density,
            speed_change_density=speed_change_density_value,
            throughput_reward_weight=throughput_reward_weight,
            progress_reward_weight=progress_reward_weight,
            gridlock_penalty_weight=gridlock_penalty_weight,
            speed_change_weight=speed_change_weight,
            switch_penalty_weight=switch_penalty_weight,
            phase_switched=phase_switched,
            global_reward_weight=global_reward_weight,
            flow_reward_weight=flow_reward_weight,
            teleport_penalty=teleport_penalty,
            teleport_count=teleport_count,
        )
        for local_delay_density, progress_density, speed_change_density_value, phase_switched in zip(
            local_delay_densities,
            progress_densities,
            speed_change_densities,
            phase_switches,
        )
    )
    rewards = tuple(clip_reward(reward, reward_clip=reward_clip) for reward in raw_rewards)
    return IntervalRewardResult(
        rewards=rewards,
        raw_rewards=raw_rewards,
        local_delay_densities=local_delay_densities,
        global_delay_density=global_delay_density,
        flow_rate_per_signal=flow_rate_per_signal,
        progress_densities=progress_densities,
        speed_change_densities=speed_change_densities,
        phase_switches=tuple(phase_switches),
        teleport_count=teleport_count,
        simulated_steps=simulated_steps,
        reward_sumo_step_seconds=reward_sumo_step_seconds,
        reward_lane_query_seconds=reward_lane_query_seconds,
        reward_vehicle_query_seconds=reward_vehicle_query_seconds,
        reward_aggregation_seconds=max(0.0, reward_aggregation_seconds),
    )


def interval_reward(
    reward_mode: PpoRewardMode,
    local_delay_density: float,
    global_delay_density: float,
    flow_rate_per_signal: float,
    progress_density: float,
    speed_change_density: float,
    throughput_reward_weight: float,
    progress_reward_weight: float,
    gridlock_penalty_weight: float,
    global_reward_weight: float,
    flow_reward_weight: float,
    speed_change_weight: float,
    switch_penalty_weight: float,
    phase_switched: bool,
    teleport_penalty: float,
    teleport_count: int,
) -> float:
    match reward_mode:
        case PpoRewardMode.DELAY_DENSITY:
            return delay_density_reward(
                local_delay_density=local_delay_density,
                global_delay_density=global_delay_density,
                flow_rate_per_signal=flow_rate_per_signal,
                speed_change_density=speed_change_density,
                global_reward_weight=global_reward_weight,
                flow_reward_weight=flow_reward_weight,
                speed_change_weight=speed_change_weight,
                switch_penalty_weight=switch_penalty_weight,
                phase_switched=phase_switched,
                teleport_penalty=teleport_penalty,
                teleport_count=teleport_count,
            )
        case PpoRewardMode.THROUGHPUT:
            return throughput_reward(
                local_delay_density=local_delay_density,
                global_delay_density=global_delay_density,
                flow_rate_per_signal=flow_rate_per_signal,
                progress_density=progress_density,
                speed_change_density=speed_change_density,
                throughput_reward_weight=throughput_reward_weight,
                progress_reward_weight=progress_reward_weight,
                gridlock_penalty_weight=gridlock_penalty_weight,
                global_reward_weight=global_reward_weight,
                speed_change_weight=speed_change_weight,
                switch_penalty_weight=switch_penalty_weight,
                phase_switched=phase_switched,
                teleport_penalty=teleport_penalty,
                teleport_count=teleport_count,
            )


def delay_density_reward(
    local_delay_density: float,
    global_delay_density: float,
    flow_rate_per_signal: float,
    speed_change_density: float,
    global_reward_weight: float,
    flow_reward_weight: float,
    speed_change_weight: float,
    switch_penalty_weight: float,
    phase_switched: bool,
    teleport_penalty: float,
    teleport_count: int,
) -> float:
    return (
        -local_delay_density
        - global_reward_weight * global_delay_density
        + flow_reward_weight * flow_rate_per_signal
        - speed_change_weight * speed_change_density
        - switch_penalty_weight * int(phase_switched)
        - teleport_penalty * teleport_count
    )


def throughput_reward(
    local_delay_density: float,
    global_delay_density: float,
    flow_rate_per_signal: float,
    progress_density: float,
    speed_change_density: float,
    throughput_reward_weight: float,
    progress_reward_weight: float,
    gridlock_penalty_weight: float,
    global_reward_weight: float,
    speed_change_weight: float,
    switch_penalty_weight: float,
    phase_switched: bool,
    teleport_penalty: float,
    teleport_count: int,
) -> float:
    gridlock_density = local_delay_density + global_reward_weight * global_delay_density
    return (
        throughput_reward_weight * flow_rate_per_signal
        + progress_reward_weight * progress_density
        - gridlock_penalty_weight * gridlock_density
        - speed_change_weight * speed_change_density
        - switch_penalty_weight * int(phase_switched)
        - teleport_penalty * teleport_count
    )


def speed_deficit_density(
    lane_api: LaneApi,
    lane_ids: Sequence[str],
    speed_limit_by_lane: Mapping[str, float],
    total_lane_length_m: float,
) -> float:
    if total_lane_length_m <= 0.0:
        return 0.0
    delayed_vehicle_equivalents = 0.0
    for lane_id in lane_ids:
        vehicle_count = int(lane_api.getLastStepVehicleNumber(lane_id))
        if vehicle_count <= 0:
            continue
        speed_limit = speed_limit_by_lane[lane_id]
        if speed_limit <= 0.0:
            continue
        mean_speed = max(0.0, float(lane_api.getLastStepMeanSpeed(lane_id)))
        speed_deficit = max(0.0, 1.0 - min(mean_speed / speed_limit, 1.0))
        delayed_vehicle_equivalents += vehicle_count * speed_deficit
    return delayed_vehicle_equivalents / total_lane_length_m


@dataclass(frozen=True)
class LaneDelaySnapshot:
    delayed_vehicle_equivalents_by_lane: dict[str, float]
    vehicle_count_by_lane: dict[str, int]
    mean_speed_by_lane: dict[str, float]

    def density(
        self,
        lane_ids: Sequence[str],
        total_lane_length_m: float,
    ) -> float:
        if total_lane_length_m <= 0.0:
            return 0.0
        return sum(self.delayed_vehicle_equivalents_by_lane.get(lane_id, 0.0) for lane_id in lane_ids) / (
            total_lane_length_m
        )

    def progress_density(
        self,
        lane_ids: Sequence[str],
        total_lane_length_m: float,
        speed_limit_by_lane: Mapping[str, float],
    ) -> float:
        if total_lane_length_m <= 0.0:
            return 0.0
        progress = 0.0
        for lane_id in lane_ids:
            speed_limit = speed_limit_by_lane[lane_id]
            if speed_limit <= 0.0:
                continue
            speed_fraction = min(max(self.mean_speed_by_lane.get(lane_id, 0.0) / speed_limit, 0.0), 1.0)
            progress += self.vehicle_count_by_lane.get(lane_id, 0) * speed_fraction
        return progress / total_lane_length_m


def lane_delay_snapshot(
    lane_api: LaneApi,
    lane_ids: Sequence[str],
    speed_limit_by_lane: Mapping[str, float],
) -> LaneDelaySnapshot:
    delayed_vehicle_equivalents_by_lane: dict[str, float] = {}
    vehicle_count_by_lane: dict[str, int] = {}
    mean_speed_by_lane: dict[str, float] = {}
    for lane_id in lane_ids:
        vehicle_count = int(lane_api.getLastStepVehicleNumber(lane_id))
        vehicle_count_by_lane[lane_id] = vehicle_count
        if vehicle_count <= 0:
            delayed_vehicle_equivalents_by_lane[lane_id] = 0.0
            mean_speed_by_lane[lane_id] = 0.0
            continue
        speed_limit = speed_limit_by_lane[lane_id]
        if speed_limit <= 0.0:
            delayed_vehicle_equivalents_by_lane[lane_id] = 0.0
            mean_speed_by_lane[lane_id] = 0.0
            continue
        mean_speed = max(0.0, float(lane_api.getLastStepMeanSpeed(lane_id)))
        mean_speed_by_lane[lane_id] = mean_speed
        speed_deficit = max(0.0, 1.0 - min(mean_speed / speed_limit, 1.0))
        delayed_vehicle_equivalents_by_lane[lane_id] = vehicle_count * speed_deficit
    return LaneDelaySnapshot(
        delayed_vehicle_equivalents_by_lane=delayed_vehicle_equivalents_by_lane,
        vehicle_count_by_lane=vehicle_count_by_lane,
        mean_speed_by_lane=mean_speed_by_lane,
    )


@dataclass
class SpeedChangeTracker:
    previous_mean_speed_by_lane: dict[str, float] = field(default_factory=dict)

    def observe_lane_snapshot(
        self,
        snapshot: LaneDelaySnapshot,
        speed_limit_by_lane: Mapping[str, float],
    ) -> dict[str, float]:
        speed_change_by_lane: dict[str, float] = {}
        for lane_id, mean_speed in snapshot.mean_speed_by_lane.items():
            speed_limit = speed_limit_by_lane[lane_id]
            previous_mean_speed = self.previous_mean_speed_by_lane.get(lane_id)
            if speed_limit <= 0.0 or previous_mean_speed is None:
                speed_change_by_lane[lane_id] = 0.0
                continue
            vehicle_count = snapshot.vehicle_count_by_lane[lane_id]
            speed_change_by_lane[lane_id] = vehicle_count * abs(mean_speed - previous_mean_speed) / speed_limit
        self.previous_mean_speed_by_lane = dict(snapshot.mean_speed_by_lane)
        return speed_change_by_lane


def speed_change_density(
    lane_ids: Sequence[str],
    speed_change_by_lane: Mapping[str, float],
    total_lane_length_m: float,
) -> float:
    if total_lane_length_m <= 0.0:
        return 0.0
    return sum(speed_change_by_lane.get(lane_id, 0.0) for lane_id in lane_ids) / total_lane_length_m


def clip_reward(reward: float, reward_clip: float) -> float:
    if reward_clip <= 0.0:
        raise ValueError('reward_clip must be positive.')
    return max(-reward_clip, min(reward_clip, reward))
