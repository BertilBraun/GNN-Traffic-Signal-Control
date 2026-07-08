from dataclasses import replace
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import MovementDatasetSample, MovementEdgeIndices, StoredPhaseIncidence
from src.movement.evaluation import EvaluationMetrics, EvaluationPolicy
from src.movement.evaluation import LearnedPolicyConfig
from src.movement.evaluation.multi_city import (
    FileCachedEpisodeRunner,
    MultiCityEvaluationAggregate,
    MultiCityEvaluationRunRequest,
)
from src.movement.experiment_config import CitySplit
from src.movement.sumo_backend import SumoBackendKind
from src.movement.training.il.types import MovementILTrainingConfig
from src.movement.training.normalizer_state import NormalizerState
from src.movement.training.ppo import configure_sumo_backend_environment, validate_config
from src.movement.training.ppo.batch import ppo_batch_data_loader
from src.movement.training.ppo.evaluation import checkpoint_selection_score, held_out_learned_checkpoint_score
from src.movement.training.ppo.reward import (
    LaneDelaySnapshot,
    SpeedChangeTracker,
    clip_reward,
    delay_density_reward,
    speed_change_density,
    speed_deficit_density,
)
from src.movement.training.ppo.rollout import (
    city_rollout_job_allocations,
    effective_rollout_cities,
    load_serialized_rollout,
    rollout_schedule,
    rollout_seed,
    sample_demand_scale,
    save_serialized_rollout,
)
from src.movement.training.ppo.run_metadata import build_run_metadata, write_run_metadata
from src.movement.training.ppo.state import cuda_random_states_for_restore, validate_resume_experiment_configuration
from src.movement.training.ppo.types import (
    CollectedRollout,
    MovementPpoCheckpoint,
    MovementPpoConfig,
    RolloutCity,
    RolloutStats,
)
from src.movement.training.ppo.stats import standard_deviation, training_diagnostics
from src.movement.training.ppo.update import gradient_norm
from src.movement.training.movement_batch import (
    MovementPhaseTensor,
    MovementTensorSample,
    movement_tensor_sample_from_dataset_sample,
    phase_tensors_from_sample,
)
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
        teacher_selected_phase_by_tls={'J0': 0},
        metadata={},
    )


def _tensor_sample() -> MovementTensorSample:
    sample = movement_tensor_sample_from_dataset_sample(
        sample=_sample(),
        city_name='unit',
        device=torch.device('cpu'),
    )
    return replace(
        sample,
        phase_tensors=phase_tensors_from_sample(sample=sample, device=torch.device('cpu')),
    )


def _two_traffic_light_tensor_sample() -> MovementTensorSample:
    sample = MovementTensorSample(
        x_lane=torch.tensor(((1.0,),), dtype=torch.float32),
        x_movement=torch.tensor(
            (
                (0.0, 1.0, 1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
        ),
        target=torch.tensor((0.0, 0.0), dtype=torch.float32),
        edge_index_dict={
            'input_lane_to_movement': torch.empty((2, 0), dtype=torch.long),
            'output_lane_to_movement': torch.empty((2, 0), dtype=torch.long),
            'movement_to_input_lane': torch.empty((2, 0), dtype=torch.long),
            'movement_to_output_lane': torch.empty((2, 0), dtype=torch.long),
            'lane_to_lane': torch.empty((2, 0), dtype=torch.long),
            'lane_to_lane_weight': torch.empty((0,), dtype=torch.float32),
        },
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0,),
                movement_ids=(0,),
                rows=((1,),),
            ),
            'J1': StoredPhaseIncidence(
                sumo_phase_indices=(0, 1),
                movement_ids=(1,),
                rows=((1,), (0,)),
            ),
        },
        teacher_selected_phase_by_tls={'J0': 0, 'J1': 0},
        city_name='unit',
    )
    return replace(
        sample,
        phase_tensors=phase_tensors_from_sample(sample=sample, device=torch.device('cpu')),
    )


def test_rollout_buffer_computes_discounted_returns_for_value_warmup() -> None:
    buffer = MovementRolloutBuffer(
        traffic_light_count=1,
        gamma=0.5,
        lam=0.95,
    )
    buffer.add(
        MovementTransition(
            tensor_sample=_tensor_sample(),
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
            tensor_sample=_tensor_sample(),
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
    assert tuple(float(row[0]) for row in buffer.returns) == (2.0, 2.0)


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


def test_rollout_schedule_submits_equal_jobs_by_priority() -> None:
    config = _ppo_config(
        rollouts_per_update=4,
        rollout_cities=(
            RolloutCity(
                city_name='small_city',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('small.sumocfg'),
                rollout_workers=2,
                rollout_priority=1,
            ),
            RolloutCity(
                city_name='large_city',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('large.sumocfg'),
                rollout_workers=2,
                rollout_priority=5,
            ),
        ),
    )

    schedule = rollout_schedule(config=config, iteration=1)

    assert tuple(rollout.rollout_city.city_name for rollout in schedule) == (
        'large_city',
        'large_city',
        'small_city',
        'small_city',
    )
    assert tuple(rollout.rollout_index for rollout in schedule) == (0, 1, 2, 3)
    assert tuple(rollout.rollout_seed for rollout in schedule) == (46, 47, 48, 49)


def test_rollout_schedule_uses_city_jobs_as_weights_when_total_is_overridden() -> None:
    config = _ppo_config(
        rollouts_per_update=5,
        rollout_cities=(
            RolloutCity(
                city_name='mannheim_innenstadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('mannheim.sumocfg'),
                rollout_workers=1,
            ),
            RolloutCity(
                city_name='karlsruhe_oststadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('karlsruhe.sumocfg'),
                rollout_workers=2,
            ),
        ),
    )

    schedule = rollout_schedule(config=config, iteration=1)

    assert tuple(rollout.rollout_city.city_name for rollout in schedule) == (
        'karlsruhe_oststadt',
        'karlsruhe_oststadt',
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
        'mannheim_innenstadt',
    )
    assert tuple(rollout.rollout_index for rollout in schedule) == (0, 1, 2, 3, 4)
    assert tuple(rollout.rollout_seed for rollout in schedule) == (47, 48, 49, 50, 51)


def test_rollout_job_allocations_reject_zero_total_city_jobs() -> None:
    config = _ppo_config(
        rollouts_per_update=1,
        rollout_cities=(
            RolloutCity(
                city_name='karlsruhe_oststadt',
                city_split=CitySplit.TRAIN,
                sumo_config_path=Path('karlsruhe.sumocfg'),
                rollout_workers=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match='total rollout city jobs must be positive'):
        city_rollout_job_allocations(config)


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


def test_validate_config_rejects_reward_sampling_slower_than_decisions() -> None:
    config = replace(
        _ppo_config(rollouts_per_update=1, rollout_cities=()),
        decision_interval=5,
        reward_sample_interval=10,
    )

    with pytest.raises(ValueError, match='reward_sample_interval must not exceed decision_interval'):
        validate_config(config)


def test_validate_config_rejects_libsumo_gui() -> None:
    config = replace(
        _ppo_config(rollouts_per_update=1, rollout_cities=()),
        gui=True,
        sumo_backend=SumoBackendKind.LIBSUMO,
    )

    with pytest.raises(ValueError, match='SUMO-GUI rollout collection requires the traci backend'):
        validate_config(config)


def test_libsumo_environment_limits_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.delenv(variable, raising=False)
    config = replace(
        _ppo_config(rollouts_per_update=1, rollout_cities=()),
        sumo_backend=SumoBackendKind.LIBSUMO,
    )

    configure_sumo_backend_environment(config)

    assert os.environ['OMP_NUM_THREADS'] == '1'
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
    assert os.environ['MKL_NUM_THREADS'] == '1'


def test_resume_validation_rejects_experiment_hash_mismatch() -> None:
    config = replace(_ppo_config(rollouts_per_update=1, rollout_cities=()), experiment_configuration_sha256='current')
    checkpoint = _ppo_checkpoint(experiment_configuration_sha256='checkpoint')

    with pytest.raises(ValueError, match='resume experiment configuration hash mismatch'):
        validate_resume_experiment_configuration(config=config, checkpoint=checkpoint)


def test_cuda_random_states_for_restore_are_cpu_byte_tensors() -> None:
    states = cuda_random_states_for_restore(
        (
            torch.tensor((1, 2, 3), dtype=torch.int64),
            torch.tensor((4, 5), dtype=torch.float32),
        )
    )

    assert all(state.device.type == 'cpu' for state in states)
    assert all(state.dtype == torch.uint8 for state in states)
    assert tuple(int(value) for value in states[0]) == (1, 2, 3)


def test_held_out_learned_score_ignores_train_city_aggregates() -> None:
    train_metrics = _evaluation_metrics(completed_vehicles=10, departed_vehicles=10, average_time_loss_s=5.0)
    held_out_metrics = _evaluation_metrics(completed_vehicles=8, departed_vehicles=10, average_time_loss_s=40.0)

    score = held_out_learned_checkpoint_score(
        aggregates=(
            _evaluation_aggregate(
                city_name='karlsruhe_oststadt',
                city_split=CitySplit.TRAIN,
                policy=EvaluationPolicy.LEARNED.value,
                metrics=train_metrics,
            ),
            _evaluation_aggregate(
                city_name='freiburg_altstadt',
                city_split=CitySplit.HELD_OUT,
                policy=EvaluationPolicy.LEARNED.value,
                metrics=held_out_metrics,
            ),
        ),
        evaluation_steps=600,
    )

    assert score == pytest.approx(checkpoint_selection_score(metrics=held_out_metrics, evaluation_steps=600))


def test_file_cached_episode_runner_caches_baselines_only(tmp_path: Path) -> None:
    calls: list[EvaluationPolicy] = []

    def fake_episode_runner(
        request: MultiCityEvaluationRunRequest,
        learned_policy_config: LearnedPolicyConfig | None,
    ) -> EvaluationMetrics:
        calls.append(request.policy)
        return _evaluation_metrics(completed_vehicles=1, departed_vehicles=1, average_time_loss_s=1.0)

    cached_runner = FileCachedEpisodeRunner(
        cache_dir=tmp_path / 'cache',
        episode_runner=fake_episode_runner,
    )
    baseline_request = _multi_city_request(policy=EvaluationPolicy.MAX_PRESSURE)
    learned_request = _multi_city_request(policy=EvaluationPolicy.LEARNED)
    learned_policy_config = LearnedPolicyConfig(
        checkpoint_path=tmp_path / 'movement_policy.pt',
        device='cpu',
    )

    cached_runner(baseline_request, None)
    cached_runner(baseline_request, None)
    cached_runner(learned_request, learned_policy_config)
    cached_runner(learned_request, learned_policy_config)

    assert calls.count(EvaluationPolicy.MAX_PRESSURE) == 1
    assert calls.count(EvaluationPolicy.LEARNED) == 2


def test_serialized_rollout_round_trip_removes_handoff_file(tmp_path: Path) -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    buffer.add(
        MovementTransition(
            tensor_sample=_tensor_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(1.0,),
            values=(0.0,),
            done=True,
        )
    )
    buffer.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(0.0,),
    )
    rollout = CollectedRollout(
        buffer=buffer,
        stats=_rollout_stats(),
        seed=123,
        city_name='karlsruhe_oststadt',
        city_split=CitySplit.TRAIN,
    )

    serialized_rollout = save_serialized_rollout(
        rollout=rollout,
        handoff_directory=tmp_path,
    )
    loaded_rollout = load_serialized_rollout(serialized_rollout)

    assert not serialized_rollout.path.exists()
    assert loaded_rollout.seed == 123
    assert loaded_rollout.city_name == 'karlsruhe_oststadt'
    assert len(loaded_rollout.buffer) == 1
    assert loaded_rollout.buffer.returns is not None
    assert float(loaded_rollout.buffer.returns[0][0]) == 1.0


def test_run_metadata_writes_checkpoint_and_log_records(tmp_path: Path) -> None:
    config = replace(
        _ppo_config(rollouts_per_update=1, rollout_cities=()),
        checkpoint_dir=tmp_path / 'checkpoints',
        log_dir=tmp_path / 'runs',
        experiment_configuration_sha256='unit-sha',
    )
    metadata = build_run_metadata(config=config, completed_iteration_at_start=4)

    write_run_metadata(
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        metadata=metadata,
    )

    assert (config.checkpoint_dir / 'run_metadata.json').read_text(encoding='utf-8')
    assert (config.log_dir / 'run_metadata.json').read_text(encoding='utf-8')
    assert metadata.completed_iteration_at_start == 4
    assert metadata.experiment_configuration_sha256 == 'unit-sha'
    assert metadata.rollout_jobs_per_update == 1
    assert metadata.rollout_process_workers == 1


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


def test_speed_deficit_density_counts_slow_moving_vehicles() -> None:
    vehicle_counts = {'stopped': 2, 'slow': 2, 'free': 2}
    mean_speeds = {'stopped': 0.0, 'slow': 5.0, 'free': 10.0}
    lane_api = FakeLaneApi(vehicle_counts=vehicle_counts, mean_speeds=mean_speeds)

    density = speed_deficit_density(
        lane_api=lane_api,
        lane_ids=('stopped', 'slow', 'free'),
        speed_limit_by_lane={'stopped': 10.0, 'slow': 10.0, 'free': 10.0},
        total_lane_length_m=300.0,
    )

    assert density == pytest.approx(3.0 / 300.0)


def test_speed_change_density_tracks_lane_mean_speed_changes() -> None:
    tracker = SpeedChangeTracker()
    first_snapshot = LaneDelaySnapshot(
        delayed_vehicle_equivalents_by_lane={'lane': 0.0},
        vehicle_count_by_lane={'lane': 2},
        mean_speed_by_lane={'lane': 5.0},
    )
    second_snapshot = LaneDelaySnapshot(
        delayed_vehicle_equivalents_by_lane={'lane': 0.0},
        vehicle_count_by_lane={'lane': 2},
        mean_speed_by_lane={'lane': 9.5},
    )
    assert tracker.observe_lane_snapshot(snapshot=first_snapshot, speed_limit_by_lane={'lane': 10.0}) == {'lane': 0.0}

    speed_change_by_lane = tracker.observe_lane_snapshot(
        snapshot=second_snapshot,
        speed_limit_by_lane={'lane': 10.0},
    )

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
                tensor_sample=_tensor_sample(),
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
            tensor_sample=_tensor_sample(),
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

    assert buffer.advantages is not None
    assert buffer.returns is not None
    loader = ppo_batch_data_loader(
        transitions=buffer.transitions,
        advantages=buffer.advantages,
        returns=buffer.returns,
        transitions_per_batch=1,
        update_batch_workers=0,
    )
    batch = next(iter(loader))

    assert batch.policy_mask.tolist() == [False]
    assert batch.advantages.tolist() == [0.0]


def test_ppo_batch_collation_reuses_cached_phase_tensors() -> None:
    tensor_sample = replace(
        _tensor_sample(),
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0,),
                movement_ids=(0,),
                rows=((0,),),
            )
        },
        phase_tensors={
            'J0': MovementPhaseTensor(
                incidence_matrix=torch.tensor(((1.0,),), dtype=torch.float32),
                movement_ids=torch.tensor((0,), dtype=torch.long),
                target_phase=0,
            )
        },
    )
    transition = MovementTransition(
        tensor_sample=tensor_sample,
        actions=(0,),
        old_log_probs=(0.0,),
        action_masks=((True,),),
        rewards=(1.0,),
        values=(0.0,),
        done=True,
    )
    loader = ppo_batch_data_loader(
        transitions=(transition,),
        advantages=(torch.tensor((0.0,), dtype=torch.float32),),
        returns=(torch.tensor((1.0,), dtype=torch.float32),),
        transitions_per_batch=1,
        update_batch_workers=0,
    )

    batch = next(iter(loader))

    assert batch.phase_logit_groups[0].incidence_matrices.tolist() == [[[1.0]]]


def test_rollout_bootstraps_truncated_returns_from_next_state_value() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    for reward, value in ((1.0, 0.25), (2.0, 0.5)):
        buffer.add(
            MovementTransition(
                tensor_sample=_tensor_sample(),
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
    assert tuple(float(row[0]) for row in buffer.returns) == (3.0, 4.0)
    assert tuple(float(row[0]) for row in buffer.advantages) == (2.75, 3.5)


def test_rollout_ignores_bootstrap_after_true_terminal_state() -> None:
    buffer = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    buffer.add(
        MovementTransition(
            tensor_sample=_two_traffic_light_tensor_sample(),
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
    assert float(buffer.returns[0][0]) == 2.0
    assert float(buffer.advantages[0][0]) == 1.5


def test_rollout_buffer_concatenates_precomputed_rollouts_without_recomputing_bootstrap() -> None:
    first = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    first.add(
        MovementTransition(
            tensor_sample=_tensor_sample(),
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
            tensor_sample=_tensor_sample(),
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
    assert tuple(float(row[0]) for row in combined.returns) == (3.0, 7.0)
    assert tuple(float(row[0]) for row in combined.advantages) == (3.0, 7.0)


def test_rollout_buffer_concatenates_different_traffic_light_counts() -> None:
    first = MovementRolloutBuffer(traffic_light_count=1, gamma=0.5, lam=1.0)
    first.add(
        MovementTransition(
            tensor_sample=_tensor_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            action_masks=((True,),),
            rewards=(1.0,),
            values=(0.0,),
            done=True,
        )
    )
    first.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(0.0,),
    )
    second = MovementRolloutBuffer(traffic_light_count=2, gamma=0.5, lam=1.0)
    second.add(
        MovementTransition(
            tensor_sample=_two_traffic_light_tensor_sample(),
            actions=(0, 1),
            old_log_probs=(0.0, -0.5),
            action_masks=((True,), (True, True)),
            rewards=(2.0, 3.0),
            values=(0.0, 1.0),
            done=True,
        )
    )
    second.compute_returns_and_advantages(
        use_discounted_return_targets=True,
        bootstrap_values=(0.0, 0.0),
    )

    combined = MovementRolloutBuffer.concatenate_computed((first, second))
    assert combined.advantages is not None
    assert combined.returns is not None
    loader = ppo_batch_data_loader(
        transitions=combined.transitions,
        advantages=combined.advantages,
        returns=combined.returns,
        transitions_per_batch=2,
        update_batch_workers=0,
    )
    batch = next(iter(loader))

    assert len(combined) == 2
    assert batch.old_log_probs.shape == (3,)
    assert sorted(batch.returns.tolist()) == [1.0, 2.0, 3.0]
    assert batch.policy_mask.tolist().count(True) == 1


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
        update_batch_workers=0,
        yellow_duration=3,
        min_green_steps=2,
        demand_scale_min=0.8,
        demand_scale_max=1.2,
        global_reward_weight=0.1,
        speed_change_weight=0.02,
        reward_sample_interval=10,
        reward_clip=1.0,
        teleport_penalty=0.0,
        max_teleports_per_rollout=999,
        time_to_teleport=-1,
        target_kl=0.03,
        gui=False,
        sumo_backend=SumoBackendKind.TRACI,
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
        allow_resume_config_mismatch=False,
        rollout_cities=rollout_cities,
        experiment_configuration=None,
        experiment_configuration_path=None,
        experiment_configuration_text=None,
        experiment_configuration_sha256=None,
        project_root=ROOT,
    )


class FakeLaneApi:
    def __init__(self, vehicle_counts: dict[str, int], mean_speeds: dict[str, float]) -> None:
        self._vehicle_counts = vehicle_counts
        self._mean_speeds = mean_speeds

    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        return self._vehicle_counts[lane_id]

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        return self._mean_speeds[lane_id]


def _ppo_checkpoint(experiment_configuration_sha256: str | None) -> MovementPpoCheckpoint:
    normalizer = NormalizerState(
        count=1,
        mean=(0.0,),
        squared_differences=(0.0,),
        frozen=True,
        epsilon=1e-8,
    )
    return MovementPpoCheckpoint(
        model_state={},
        optimizer_state={},
        lane_feature_dim=1,
        movement_feature_dim=1,
        hidden_dim=1,
        num_hops=0,
        lane_normalizer=normalizer,
        movement_normalizer=normalizer,
        il_config=MovementILTrainingConfig(),
        iteration=1,
        best_checkpoint_score=1.0,
        experiment_configuration_sha256=experiment_configuration_sha256,
        experiment_configuration_text='name: checkpoint\n',
        torch_random_state=torch.get_rng_state(),
        cuda_random_states=(),
    )


def _evaluation_aggregate(
    city_name: str,
    city_split: CitySplit,
    policy: str,
    metrics: EvaluationMetrics,
) -> MultiCityEvaluationAggregate:
    return MultiCityEvaluationAggregate(
        city_name=city_name,
        city_split=city_split,
        policy=policy,
        demand_scale=1.0,
        seeds=(100,),
        mean=metrics,
        standard_deviation=_evaluation_metrics(
            completed_vehicles=0,
            departed_vehicles=0,
            average_time_loss_s=0.0,
        ),
    )


def _multi_city_request(policy: EvaluationPolicy) -> MultiCityEvaluationRunRequest:
    return MultiCityEvaluationRunRequest(
        city_name='karlsruhe_oststadt',
        city_split=CitySplit.TRAIN,
        sumo_config_path=Path('karlsruhe.sumocfg'),
        policy=policy,
        seed=100,
        demand_scale=1.0,
        steps=120,
        decision_interval=10,
        yellow_duration=3,
        minimum_green_steps=2,
        minimum_initial_occupancy=0.05,
        maximum_initial_occupancy=0.08,
        time_to_teleport=-1,
    )


def _evaluation_metrics(
    completed_vehicles: int,
    departed_vehicles: int,
    average_time_loss_s: float,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        departed_vehicles=departed_vehicles,
        completed_vehicles=completed_vehicles,
        vehicles_remaining=departed_vehicles - completed_vehicles,
        completion_rate=completed_vehicles / departed_vehicles if departed_vehicles > 0 else 0.0,
        teleport_count=0,
        throughput_per_hour=float(completed_vehicles),
        average_waiting_time_s=1.0,
        average_travel_time_s=2.0,
        average_time_loss_s=average_time_loss_s,
        average_queue_length_vehicles=4.0,
        max_queue_length_vehicles=5.0,
        average_wait_density_s_per_m=6.0,
        phase_switch_frequency_per_junction_per_minute=7.0,
        average_tls_passes_per_vehicle=8.0,
        average_stops_before_tls_per_vehicle=9.0,
        nonstop_tls_pass_rate=0.5,
        average_best_nonstop_tls_streak=10.0,
        per_junction_wait_density_s_per_m={},
        per_junction_max_queue_length_vehicles={},
        per_junction_phase_counts={},
    )


def _rollout_stats() -> RolloutStats:
    return RolloutStats(
        mean_reward=1.0,
        reward_standard_deviation=0.0,
        minimum_reward=1.0,
        maximum_reward=1.0,
        raw_reward_standard_deviation=0.0,
        minimum_raw_reward=1.0,
        maximum_raw_reward=1.0,
        reward_clip_fraction=0.0,
        mean_local_delay_density=0.0,
        mean_global_delay_density=0.0,
        mean_speed_change_density=0.0,
        normalized_entropy=1.0,
        mean_top_action_probability=1.0,
        policy_decision_fraction=1.0,
        teleport_count=0,
        mean_demand_scale=1.0,
        minimum_demand_scale=1.0,
        maximum_demand_scale=1.0,
        simulation_elapsed_s=1.0,
    )
