from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import MovementDatasetSample, MovementEdgeIndices, StoredPhaseIncidence
from src.movement.evaluation import EvaluationMetrics, EvaluationPolicy
from src.movement.experiment_config import CitySplit
from src.movement.training.ppo import validate_config
from src.movement.training.ppo.evaluation import checkpoint_selection_score
from src.movement.training.ppo.reward import (
    SpeedChangeTracker,
    clip_reward,
    delay_density_reward,
    speed_change_density,
    speed_deficit_density,
)
from src.movement.training.ppo.rollout import (
    effective_rollout_cities,
    rollout_schedule,
    rollout_seed,
    sample_demand_scale,
)
from src.movement.training.ppo.types import MovementPpoConfig, RolloutCity
from src.movement.training.ppo.stats import standard_deviation, training_diagnostics
from src.movement.training.ppo.update import gradient_norm
from src.movement.training.rollout import MovementRolloutBuffer
from src.movement.training.rollout.types import MovementTransition


def _sample() -> MovementDatasetSample:
    return MovementDatasetSample(
        x_lane=((1.0,),),
        x_movement=((0.0, 1.0, 1.0, 0.0, 0.0),),
        edge_indices=MovementEdgeIndices(
            input_lane_to_movement=(),
            output_lane_to_movement=(),
            movement_to_input_lane=(),
            movement_to_output_lane=(),
        ),
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0,),
                movement_ids=(0,),
                rows=((1,),),
            )
        },
        teacher_movement_scores=(0.0,),
        teacher_selected_phase_by_tls={},
        metadata={},
    )


def test_rollout_buffer_computes_discounted_returns_for_value_warmup() -> None:
    buffer = MovementRolloutBuffer(
        traffic_light_count=1,
        gamma=0.5,
        lam=0.95,
    )
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(1.0,),
            values=(0.0,),
            done=False,
        )
    )
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(2.0,),
            values=(0.0,),
            done=True,
        )
    )

    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(0.0,),
    )

    assert buffer.returns is not None
    assert tuple(float(value) for value in buffer.returns[:, 0]) == (2.0, 2.0)


def test_reward_clipping_limits_gridlock_outliers() -> None:
    assert clip_reward(4.0, reward_clip=1.0) == 1.0
    assert clip_reward(-4.0, reward_clip=1.0) == -1.0
    assert clip_reward(0.25, reward_clip=1.0) == 0.25


def test_rollout_seed_can_be_fixed_for_overfit_experiments() -> None:
    assert (
        rollout_seed(
            training_seed=42,
            iteration=3,
            rollout_index=0,
            rollouts_per_update=1,
            fixed_rollout_seed=None,
        )
        == 45
    )


def test_rollout_schedule_uses_configured_train_cities_evenly() -> None:
    config = _ppo_config(
        rollouts_per_update=4,
        rollout_cities=(
            RolloutCity(
                city_name='karlsruhe_oststadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('karlsruhe.sumocfg'),
                rollout_workers=2,
            ),
            RolloutCity(
                city_name='mannheim_innenstadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('mannheim.sumocfg'),
                rollout_workers=2,
            ),
        ),
    )

    schedule = rollout_schedule(config=config, iteration=1)

    assert tuple(rollout.rollout_city.city_name for rollout in schedule) == (
        'karlsruhe_oststadt',
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'mannheim_innenstadt',
    )
    assert tuple(rollout.rollout_seed for rollout in schedule) == (46, 47, 48, 49)


def test_rollout_schedule_rejects_worker_count_mismatch() -> None:
    config = _ppo_config(
        rollouts_per_update=3,
        rollout_cities=(
            RolloutCity(
                city_name='karlsruhe_oststadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('karlsruhe.sumocfg'),
                rollout_workers=2,
            ),
        ),
    )

    with pytest.raises(ValueError, match='rollout city worker counts must equal rollouts_per_update'):
        rollout_schedule(config=config, iteration=1)


def test_validate_config_rejects_held_out_rollout_city() -> None:
    config = _ppo_config(
        rollouts_per_update=1,
        rollout_cities=(
            RolloutCity(
                city_name='freiburg_altstadt',
                city_split=CitySplit.HELD_OUT,
                sumo_config_path=Path('freiburg.sumocfg'),
                rollout_workers=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match='rollout_cities must not include held-out cities'):
        validate_config(config)


def test_effective_rollout_cities_keeps_single_city_default() -> None:
    config = _ppo_config(rollouts_per_update=3, rollout_cities=())

    rollout_cities = effective_rollout_cities(config)

    assert len(rollout_cities) == 1
    assert rollout_cities[0].sumo_config_path == config.cfg_path
    assert rollout_cities[0].rollout_workers == 3


def test_rollout_seed_uses_iteration_and_rollout_index() -> None:
    assert (
        rollout_seed(
            training_seed=42,
            iteration=3,
            rollout_index=2,
            rollouts_per_update=4,
            fixed_rollout_seed=None,
        )
        == 56
    )
    assert (
        rollout_seed(
            training_seed=42,
            iteration=3,
            rollout_index=2,
            rollouts_per_update=4,
            fixed_rollout_seed=100,
        )
        == 100
    )


def test_demand_scale_sampling_is_seeded_and_bounded() -> None:
    first = sample_demand_scale(demand_scale_min=0.4, demand_scale_max=0.85, seed=42)
    second = sample_demand_scale(demand_scale_min=0.4, demand_scale_max=0.85, seed=42)
    other = sample_demand_scale(demand_scale_min=0.4, demand_scale_max=0.85, seed=43)

    assert first == second
    assert first != other
    assert 0.4 <= first <= 0.85
    assert sample_demand_scale(demand_scale_min=0.65, demand_scale_max=0.65, seed=42) == 0.65


def test_delay_density_reward_penalizes_local_and_global_delay() -> None:
    assert delay_density_reward(
        local_delay_density=0.2,
        global_delay_density=0.1,
        speed_change_density=0.3,
        global_reward_weight=0.1,
        speed_change_weight=0.02,
        teleport_penalty=0.5,
        teleport_count=2,
    ) == pytest.approx(-1.216)


def test_speed_deficit_density_counts_slow_moving_vehicles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle_counts = {'stopped': 2, 'slow': 2, 'free': 2}
    mean_speeds = {'stopped': 0.0, 'slow': 5.0, 'free': 10.0}
    monkeypatch.setattr(
        'src.movement.training.ppo.reward.traci.lane.getLastStepVehicleNumber',
        lambda lane_id: vehicle_counts[lane_id],
    )
    monkeypatch.setattr(
        'src.movement.training.ppo.reward.traci.lane.getLastStepMeanSpeed',
        lambda lane_id: mean_speeds[lane_id],
    )

    density = speed_deficit_density(
        lane_ids=('stopped', 'slow', 'free'),
        speed_limit_by_lane={'stopped': 10.0, 'slow': 10.0, 'free': 10.0},
        total_lane_length_m=300.0,
    )

    assert density == pytest.approx(3.0 / 300.0)


def test_speed_change_density_tracks_vehicle_speed_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane_vehicle_ids = {'lane': ['veh0', 'veh1']}
    speeds = {'veh0': 5.0, 'veh1': 10.0}
    monkeypatch.setattr(
        'src.movement.training.ppo.reward.traci.lane.getLastStepVehicleIDs',
        lambda lane_id: lane_vehicle_ids[lane_id],
    )
    monkeypatch.setattr(
        'src.movement.training.ppo.reward.traci.vehicle.getSpeed',
        lambda vehicle_id: speeds[vehicle_id],
    )

    tracker = SpeedChangeTracker()
    assert tracker.observe(lane_ids=('lane',), speed_limit_by_lane={'lane': 10.0}) == {'lane': 0.0}
    speeds['veh0'] = 8.0
    speeds['veh1'] = 4.0

    speed_change_by_lane = tracker.observe(lane_ids=('lane',), speed_limit_by_lane={'lane': 10.0})

    assert speed_change_by_lane == pytest.approx({'lane': 0.9})
    assert speed_change_density(
        lane_ids=('lane',),
        speed_change_by_lane=speed_change_by_lane,
        total_lane_length_m=100.0,
    ) == pytest.approx(0.009)


def test_checkpoint_selection_score_penalizes_incomplete_and_teleported_vehicles() -> None:
    metrics = EvaluationMetrics(
        departed_vehicles=100,
        completed_vehicles=80,
        vehicles_remaining=20,
        completion_rate=0.8,
        teleport_count=1,
        throughput_per_hour=480.0,
        average_waiting_time_s=30.0,
        average_travel_time_s=100.0,
        average_time_loss_s=40.0,
        average_queue_length_vehicles=1.0,
        max_queue_length_vehicles=5.0,
        average_wait_density_s_per_m=0.1,
        phase_switch_frequency_per_junction_per_minute=1.0,
        average_tls_passes_per_vehicle=2.0,
        average_stops_before_tls_per_vehicle=1.0,
        nonstop_tls_pass_rate=0.5,
        average_best_nonstop_tls_streak=1.0,
        per_junction_wait_density_s_per_m={},
        per_junction_max_queue_length_vehicles={},
        per_junction_phase_counts={},
    )

    score = checkpoint_selection_score(metrics=metrics, evaluation_steps=600)

    assert score == pytest.approx(56.0)


def test_diagnostics_report_reward_and_return_scale() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=0.95)
    for reward, value, done in ((1.0, 0.25, False), (2.0, 0.5, True)):
        buffer.add(
            MovementTransition(
                sample=_sample(),
                actions=(0,),
                old_log_probs=(0.0,),
                action_masks=((True,),),
                rewards=(reward,),
                values=(value,),
                done=done,
            )
        )
    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(0.0,),
    )

    diagnostics = training_diagnostics(buffer)

    assert diagnostics.mean_return == 2.0
    assert diagnostics.return_standard_deviation == 0.0
    assert diagnostics.mean_value == 0.375
    assert diagnostics.value_standard_deviation > 0.0
    assert diagnostics.advantage_standard_deviation > 0.0
    assert standard_deviation((1.0, 2.0, 3.0)) == 1.0


def test_gradient_norm_uses_all_available_parameter_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor((0.0,)))
    second = torch.nn.Parameter(torch.tensor((0.0,)))
    first.grad = torch.tensor((3.0,))
    second.grad = torch.tensor((4.0,))

    assert gradient_norm((first, second)) == 5.0


def test_rollout_excludes_forced_actions_from_policy_advantages() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.99, lam=0.95)
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(1.0,),
            values=(0.0,),
            done=True,
        )
    )
    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=False,
        bootstrap_values=(0.0,),
    )

    batch = next(buffer.iterate_minibatches(transitions_per_batch=1, device='cpu'))

    assert batch.policy_mask.tolist() == [[False]]
    assert batch.advantages.tolist() == [[0.0]]


def test_rollout_bootstraps_truncated_returns_from_next_state_value() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    for reward, value in ((1.0, 0.25), (2.0, 0.5)):
        buffer.add(
            MovementTransition(
                sample=_sample(),
                actions=(0,),
                old_log_probs=(0.0,),
                action_masks=((True,),),
                rewards=(reward,),
                values=(value,),
                done=False,
            )
        )

    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(4.0,),
    )

    assert buffer.returns is not None
    assert buffer.advantages is not None
    assert tuple(float(value) for value in buffer.returns[:, 0]) == (3.0, 4.0)
    assert tuple(float(value) for value in buffer.advantages[:, 0]) == (2.75, 3.5)


def test_rollout_ignores_bootstrap_after_true_terminal_state() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(2.0,),
            values=(0.5,),
            done=True,
        )
    )

    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(4.0,),
    )

    assert buffer.returns is not None
    assert buffer.advantages is not None
    assert float(buffer.returns[0, 0]) == 2.0
    assert float(buffer.advantages[0, 0]) == 1.5


def test_rollout_buffer_concatenates_precomputed_rollouts_without_recomputing_bootstrap() -> None:
    first = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    first.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(1.0,),
            values=(0.0,),
            done=False,
        )
    )
    first.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(4.0,),
    )
    second = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    second.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(2.0,),
            values=(0.0,),
            done=False,
        )
    )
    second.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(10.0,),
    )

    combined = MovementRolloutBuffer.concatenate_computed((first, second))

    assert len(combined) == 2
    assert combined.returns is not None
    assert combined.advantages is not None
    assert tuple(float(value) for value in combined.returns[:, 0]) == (3.0, 7.0)
    assert tuple(float(value) for value in combined.advantages[:, 0]) == (3.0, 7.0)


def _ppo_config(
    rollouts_per_update: int,
    rollout_cities: tuple[RolloutCity, ...],
) -> MovementPpoConfig:
    return MovementPpoConfig(
        cfg_path=Path('grid.sumocfg'),
        il_checkpoint_path=Path('movement_policy.pt'),
        iterations=2,
        steps_per_rollout=3,
        rollouts_per_update=rollouts_per_update,
        num_workers=1,
        decision_interval=10,
        learning_rate=2e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.1,
        update_epochs=4,
        value_warmup_iterations=1,
        warmup_epochs=8,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        transitions_per_batch=32,
        yellow_duration=3,
        min_green_steps=2,
        demand_scale_min=0.8,
        demand_scale_max=1.2,
        global_reward_weight=0.1,
        speed_change_weight=0.02,
        reward_clip=1.0,
        teleport_penalty=0.0,
        max_teleports_per_rollout=999,
        time_to_teleport=-1,
        target_kl=0.03,
        gui=False,
        initial_occupancy_min=0.05,
        initial_occupancy_max=0.08,
        eval_every=10,
        eval_steps=600,
        eval_seeds=(42,),
        eval_policies=(EvaluationPolicy.MAX_PRESSURE,),
        eval_demand_scale=1.0,
        eval_demand_scales=(1.0,),
        save_every=10,
        print_every=1,
        checkpoint_dir=Path('checkpoints/rl/unit'),
        log_dir=Path('runs/rl/unit'),
        device='cpu',
        seed=42,
        fixed_rollout_seed=None,
        resume_checkpoint_path=None,
        rollout_cities=rollout_cities,
        experiment_configuration=None,
        project_root=ROOT,
    )
