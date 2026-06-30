from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_il
from src.movement.evaluation import EvaluationMetrics, EvaluationPolicy
from src.movement.training.il.batching import CityBalancedBatchPlanner, RandomBatchPlanner
from src.movement.training.il.types import MovementILTrainingConfig


def test_train_il_cli_accepts_dataset_and_checkpoint_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_il.py',
            '--data',
            'data/il/samples.jsonl',
            '--epochs',
            '3',
            '--progress-every',
            '1',
            '--checkpoint-every-epochs',
            '2',
            '--progress-every-batches',
            '5',
            '--progress-every-seconds',
            '30',
            '--validation-every-epochs',
            '0',
            '--max-train-samples',
            '64',
            '--cache-workers',
            '4',
            '--train-workers',
            '8',
            '--prefetch-batches',
            '10',
            '--num-hops',
            '1',
            '--ckpt-dir',
            'checkpoints/il/unit',
        ],
    )

    args = train_il.parse_args()

    assert args.data == Path('data/il/samples.jsonl')
    assert args.epochs == 3
    assert args.progress_every == 1
    assert args.checkpoint_every_epochs == 2
    assert args.progress_every_batches == 5
    assert args.progress_every_seconds == 30
    assert args.validation_every_epochs == 0
    assert args.max_train_samples == 64
    assert args.cache_workers == 4
    assert args.train_workers == 8
    assert args.prefetch_batches == 10
    assert args.num_hops == 1
    assert args.ckpt_dir == Path('checkpoints/il/unit')
    assert args.experiment_config is None
    assert args.validation_fraction == 0.1


def test_train_il_cli_accepts_experiment_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_il.py',
            '--experiment-config',
            'configs/training/city_first_pass.yaml',
            '--data',
            'data/il/city_first_pass/combined.jsonl',
        ],
    )

    args = train_il.parse_args()

    assert args.experiment_config == Path('configs/training/city_first_pass.yaml')
    assert args.data == Path('data/il/city_first_pass/combined.jsonl')


def test_train_il_cli_accepts_collection_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_il.py',
            '--cfg',
            'configs/grid_3x3_dedicated/grid.sumocfg',
            '--samples',
            '120',
            '--samples-per-simulation',
            '30',
            '--collection-seed',
            '50',
            '--decision-interval',
            '15',
            '--demand-scale',
            '0.5',
            '--seed',
            '7',
            '--eval-cfg',
            'configs/grid_3x3_dedicated/grid.sumocfg',
            '--eval-demand-scale',
            '0.75',
            '--time-to-teleport',
            '-1',
        ],
    )

    args = train_il.parse_args()

    assert args.data is None
    assert args.sumo_config_path == Path('configs/grid_3x3_dedicated/grid.sumocfg')
    assert args.samples == 120
    assert args.samples_per_simulation == 30
    assert args.collection_seed == 50
    assert args.decision_interval == 15
    assert args.demand_scale == 0.5
    assert args.seed == 7
    assert args.eval_cfg == Path('configs/grid_3x3_dedicated/grid.sumocfg')
    assert args.eval_demand_scale == 0.75
    assert args.time_to_teleport == -1


def test_train_il_cli_uses_dataset_and_evaluation_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_il.py',
            '--cfg',
            'configs/grid_3x3_dedicated/grid.sumocfg',
        ],
    )

    args = train_il.parse_args()

    assert args.samples == 4800
    assert args.samples_per_simulation == 240
    assert args.collection_seed == 42
    assert args.decision_interval == 10
    assert args.samples_per_batch == 1024
    assert args.eval_every_epochs == 0
    assert args.eval_seeds == [100, 101]
    assert args.determinism_check_samples == 20
    assert args.time_to_teleport == -1


def test_training_evaluation_caches_deterministic_baseline(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[EvaluationPolicy, int]] = []

    def fake_run_evaluation_episode(**kwargs) -> EvaluationMetrics:
        calls.append((kwargs['policy'], kwargs['seed']))
        return _empty_metrics()

    monkeypatch.setattr(train_il, 'run_evaluation_episode', fake_run_evaluation_episode)
    observer = train_il.TrainingEvaluationObserver(
        cfg_path=Path('grid.sumocfg'),
        policies=(EvaluationPolicy.MAX_PRESSURE, EvaluationPolicy.LEARNED),
        seeds=(100, 101),
        steps=60,
        decision_interval=15,
        yellow_duration=3,
        min_green_steps=2,
        demand_scale=1.0,
        initial_occupancy_min=0.05,
        initial_occupancy_max=0.08,
        time_to_teleport=-1,
        output_dir=tmp_path / 'eval',
        log_dir=tmp_path / 'runs',
        device='cpu',
        every_epochs=10,
    )

    first_records = observer._run_epoch_evaluation(tmp_path / 'first.pt')
    observer.baseline_records = tuple(
        record for record in first_records if record.policy == EvaluationPolicy.MAX_PRESSURE.value
    )
    observer._run_epoch_evaluation(tmp_path / 'second.pt')

    assert calls.count((EvaluationPolicy.MAX_PRESSURE, 100)) == 1
    assert calls.count((EvaluationPolicy.MAX_PRESSURE, 101)) == 1
    assert calls.count((EvaluationPolicy.LEARNED, 100)) == 2
    assert calls.count((EvaluationPolicy.LEARNED, 101)) == 2


def test_train_il_batch_planner_uses_city_balance_only_for_experiment_config() -> None:
    config = MovementILTrainingConfig(
        epochs=1,
        lr=0.01,
        hidden_dim=8,
        checkpoint_dir=Path('checkpoints/il/unit'),
        seed=42,
        num_hops=0,
    )
    experiment_configuration = train_il._experiment_configuration(
        ROOT / 'configs' / 'training' / 'city_first_pass.yaml'
    )

    assert isinstance(
        train_il._batch_planner(config=config, experiment_configuration=experiment_configuration),
        CityBalancedBatchPlanner,
    )
    assert isinstance(
        train_il._batch_planner(config=config, experiment_configuration=None),
        RandomBatchPlanner,
    )


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        departed_vehicles=0,
        completed_vehicles=0,
        vehicles_remaining=0,
        completion_rate=0.0,
        teleport_count=0,
        throughput_per_hour=0.0,
        average_waiting_time_s=0.0,
        average_travel_time_s=0.0,
        average_time_loss_s=0.0,
        average_queue_length_vehicles=0.0,
        max_queue_length_vehicles=0.0,
        average_wait_density_s_per_m=0.0,
        phase_switch_frequency_per_junction_per_minute=0.0,
        average_tls_passes_per_vehicle=0.0,
        average_stops_before_tls_per_vehicle=0.0,
        nonstop_tls_pass_rate=0.0,
        average_best_nonstop_tls_streak=0.0,
        per_junction_wait_density_s_per_m={},
        per_junction_max_queue_length_vehicles={},
        per_junction_phase_counts={},
    )
