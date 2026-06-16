"""Reward calculation for movement PPO rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import traci

from src.movement.runtime import MovementControlRuntime
from src.movement.training.ppo.types import IntervalRewardResult, RolloutContext


def advance_and_reward(
    runtime: MovementControlRuntime,
    context: RolloutContext,
    decision_interval: int,
    global_reward_weight: float,
    reward_clip: float,
    teleport_penalty: float,
) -> IntervalRewardResult:
    if teleport_penalty < 0.0:
        raise ValueError('teleport_penalty must not be negative.')
    local_delay_sums = {traffic_light_id: 0.0 for traffic_light_id in context.traffic_light_ids}
    global_delay_sum = 0.0
    teleport_count = 0
    simulated_steps = 0
    for _step in range(decision_interval):
        runtime.step()
        simulated_steps += 1
        teleport_count += int(traci.simulation.getStartingTeleportNumber())
        for traffic_light_id in context.traffic_light_ids:
            local_delay_sums[traffic_light_id] += speed_deficit_density(
                lane_ids=context.incoming_lanes_by_traffic_light[traffic_light_id],
                speed_limit_by_lane=context.speed_limit_by_lane,
                total_lane_length_m=context.incoming_lane_length_by_traffic_light[traffic_light_id],
            )
        global_delay_sum += speed_deficit_density(
            lane_ids=context.all_incoming_lane_ids,
            speed_limit_by_lane=context.speed_limit_by_lane,
            total_lane_length_m=context.all_incoming_lane_length_m,
        )
        if not runtime.is_running():
            break
    local_delay_densities = tuple(
        local_delay_sums[traffic_light_id] / max(1, simulated_steps) for traffic_light_id in context.traffic_light_ids
    )
    global_delay_density = global_delay_sum / max(1, simulated_steps)
    raw_rewards = tuple(
        delay_density_reward(
            local_delay_density=local_delay_density,
            global_delay_density=global_delay_density,
            global_reward_weight=global_reward_weight,
            teleport_penalty=teleport_penalty,
            teleport_count=teleport_count,
        )
        for local_delay_density in local_delay_densities
    )
    rewards = tuple(clip_reward(reward, reward_clip=reward_clip) for reward in raw_rewards)
    return IntervalRewardResult(
        rewards=rewards,
        raw_rewards=raw_rewards,
        local_delay_densities=local_delay_densities,
        global_delay_density=global_delay_density,
        teleport_count=teleport_count,
    )


def delay_density_reward(
    local_delay_density: float,
    global_delay_density: float,
    global_reward_weight: float,
    teleport_penalty: float,
    teleport_count: int,
) -> float:
    return -local_delay_density - global_reward_weight * global_delay_density - teleport_penalty * teleport_count


def speed_deficit_density(
    lane_ids: Sequence[str],
    speed_limit_by_lane: Mapping[str, float],
    total_lane_length_m: float,
) -> float:
    if total_lane_length_m <= 0.0:
        return 0.0
    delayed_vehicle_equivalents = 0.0
    for lane_id in lane_ids:
        vehicle_count = int(traci.lane.getLastStepVehicleNumber(lane_id))
        if vehicle_count <= 0:
            continue
        speed_limit = speed_limit_by_lane[lane_id]
        if speed_limit <= 0.0:
            continue
        mean_speed = max(0.0, float(traci.lane.getLastStepMeanSpeed(lane_id)))
        speed_deficit = max(0.0, 1.0 - min(mean_speed / speed_limit, 1.0))
        delayed_vehicle_equivalents += vehicle_count * speed_deficit
    return delayed_vehicle_equivalents / total_lane_length_m


def clip_reward(reward: float, reward_clip: float) -> float:
    if reward_clip <= 0.0:
        raise ValueError('reward_clip must be positive.')
    return max(-reward_clip, min(reward_clip, reward))
