from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import MovementDatasetSample, MovementEdgeIndices, StoredPhaseIncidence
from src.movement.evaluation import EvaluationMetrics
from src.movement.training.ppo.evaluation import checkpoint_selection_score
from src.movement.training.ppo.reward import clip_reward, delay_density_reward, speed_deficit_density
from src.movement.training.ppo.rollout import rollout_seed
from src.movement.training.ppo.stats import standard_deviation, training_diagnostics
from src.movement.training.ppo.update import gradient_norm
from src.movement.training.rollout import MovementRolloutBuffer, MovementTransition


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


def test_delay_density_reward_penalizes_local_and_global_delay() -> None:
    assert delay_density_reward(
        local_delay_density=0.2,
        global_delay_density=0.1,
        global_reward_weight=0.1,
        teleport_penalty=0.5,
        teleport_count=2,
    ) == pytest.approx(-1.21)


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
