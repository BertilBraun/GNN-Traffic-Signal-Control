from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_il
from src.movement.evaluation import EvaluationMetrics, EvaluationPolicy


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
    assert args.num_hops == 1
    assert args.ckpt_dir == Path('checkpoints/il/unit')


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
    assert args.samples_per_batch == 32
    assert args.eval_every_epochs == 10
    assert args.eval_seeds == [100, 101]
    assert args.determinism_check_samples == 20


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
