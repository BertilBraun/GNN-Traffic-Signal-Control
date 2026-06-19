"""Train a movement-score imitation model from JSONL samples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'

from src.movement.training.il import train_movement_il_from_jsonl  # noqa: E402
from src.movement.training.il.checkpoint import save_movement_checkpoint  # noqa: E402
from src.movement.training.il.types import (  # noqa: E402
    MovementILLoss,
    MovementILTrainingConfig,
    MovementILTrainingSnapshot,
)
from src.movement.dataset import load_jsonl_samples, save_jsonl_samples  # noqa: E402
from src.movement.initial_traffic import sample_target_occupancy  # noqa: E402
from src.movement.evaluation import (  # noqa: E402
    EvaluationAggregate,
    EvaluationPolicy,
    EvaluationRecord,
    LearnedPolicyConfig,
    aggregate_records,
    current_timer_s,
    print_aggregate_metric_table,
    print_evaluation_result,
    print_evaluation_start,
    run_evaluation_episode,
    write_aggregate_json,
    write_records_csv,
)
from scripts.collect_il_data import collect_samples, verify_max_pressure_determinism  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train movement-score imitation from collected JSONL data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', type=Path, default=None, help='Input JSONL dataset')
    parser.add_argument(
        '--cfg',
        dest='sumo_config_path',
        type=Path,
        default=None,
        help='SUMO config to collect before training',
    )
    parser.add_argument('--samples', type=int, default=4800, help='Total decision samples collected with --cfg')
    parser.add_argument(
        '--samples-per-simulation',
        type=int,
        default=240,
        help='Decision samples collected from each simulation',
    )
    parser.add_argument('--collection-seed', type=int, default=42, help='First automatic collection seed')
    parser.add_argument('--decision-interval', type=int, default=10, help='Seconds between collected samples')
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=1.0,
        help='Multiplier applied to route-file flow demand during collection',
    )
    parser.add_argument('--initial-occupancy-min', type=float, default=0.04)
    parser.add_argument('--initial-occupancy-max', type=float, default=0.08)
    parser.add_argument('--epochs', type=int, default=400, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='MLP hidden dimension')
    parser.add_argument('--samples-per-batch', type=int, default=32, help='IL samples per optimizer update')
    parser.add_argument('--num-hops', type=int, default=1, help='LaneGroup/Movement macro-hops')
    parser.add_argument(
        '--loss',
        choices=('huber', 'mse'),
        default='huber',
        help='Movement-score regression loss',
    )
    parser.add_argument(
        '--phase-loss-coeff',
        type=float,
        default=1.0,
        help='Weight for teacher phase-ranking cross-entropy',
    )
    parser.add_argument('--seed', type=int, default=42, help='Torch random seed')
    parser.add_argument('--device', default='cpu', help='Torch device')
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10,
        help='Print loss every N epochs (0 disables progress output)',
    )
    parser.add_argument(
        '--ckpt-dir',
        type=Path,
        default=None,
        help='Checkpoint directory (default: checkpoints/il/<timestamp>)',
    )
    parser.add_argument(
        '--eval-cfg',
        type=Path,
        default=DEFAULT_CFG,
        help='SUMO config used for periodic learned-policy evaluation',
    )
    parser.add_argument(
        '--eval-every-epochs',
        type=int,
        default=10,
        help='Run evaluation every N epochs when --eval-cfg is set (0 disables)',
    )
    parser.add_argument('--eval-steps', type=int, default=600, help='Simulation seconds per evaluation episode')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[100, 101], help='Evaluation SUMO seeds')
    parser.add_argument(
        '--determinism-check-samples',
        type=int,
        default=20,
        help='Decision steps compared across two identical max-pressure simulations; 0 disables',
    )
    parser.add_argument('--log-dir', type=Path, default=None, help='TensorBoard directory')
    parser.add_argument(
        '--eval-policies',
        nargs='+',
        choices=tuple(policy.value for policy in EvaluationPolicy),
        default=[EvaluationPolicy.MAX_PRESSURE.value, EvaluationPolicy.LEARNED.value],
        help='Policies run during training-time evaluation',
    )
    parser.add_argument(
        '--eval-output-dir',
        type=Path,
        default=None,
        help='Directory for training-time eval reports (default: <ckpt-dir>/eval)',
    )
    parser.add_argument('--eval-yellow-duration', type=int, default=3, help='Evaluation yellow transition seconds')
    parser.add_argument('--eval-min-green-steps', type=int, default=2, help='Evaluation minimum green intervals')
    parser.add_argument(
        '--eval-demand-scale',
        type=float,
        default=1.0,
        help='Multiplier applied to route-file flow demand during periodic evaluation',
    )
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=-1,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    return parser.parse_args()


@dataclass
class TrainingEvaluationObserver:
    cfg_path: Path
    policies: tuple[EvaluationPolicy, ...]
    seeds: tuple[int, ...]
    steps: int
    decision_interval: int
    yellow_duration: int
    min_green_steps: int
    demand_scale: float
    initial_occupancy_min: float
    initial_occupancy_max: float
    time_to_teleport: int | None
    output_dir: Path
    log_dir: Path
    device: str
    every_epochs: int
    baseline_records: tuple[EvaluationRecord, ...] = field(default_factory=tuple, init=False)
    best_learned_wait_density: float = field(default=float('inf'), init=False)

    def on_epoch_completed(self, snapshot: MovementILTrainingSnapshot) -> None:
        if not self._should_evaluate(snapshot):
            return
        epoch_dir = self.output_dir / f'epoch_{snapshot.epoch:04d}'
        checkpoint_path = epoch_dir / 'movement_policy.pt'
        save_movement_checkpoint(
            checkpoint_path=checkpoint_path,
            model=snapshot.model,
            config=snapshot.config,
            lane_feature_dim=snapshot.lane_feature_dim,
            movement_feature_dim=snapshot.movement_feature_dim,
            lane_normalizer=snapshot.lane_normalizer,
            movement_normalizer=snapshot.movement_normalizer,
            loss=snapshot.loss,
        )
        records = self._run_epoch_evaluation(checkpoint_path)
        self.baseline_records = tuple(record for record in records if record.policy != EvaluationPolicy.LEARNED.value)
        aggregates = aggregate_records(records)
        self._write_tensorboard(snapshot.epoch, aggregates)
        write_aggregate_json(epoch_dir / 'summary.json', records, aggregates)
        write_records_csv(epoch_dir / 'summary.csv', records, aggregates)
        print_aggregate_metric_table(f'Evaluation summary at epoch {snapshot.epoch}', aggregates)
        learned_aggregate = next(
            (aggregate for aggregate in aggregates if aggregate.policy == EvaluationPolicy.LEARNED.value),
            None,
        )
        if (
            learned_aggregate is not None
            and learned_aggregate.mean.average_wait_density_s_per_m < self.best_learned_wait_density
        ):
            self.best_learned_wait_density = learned_aggregate.mean.average_wait_density_s_per_m
            save_movement_checkpoint(
                checkpoint_path=self.output_dir.parent / 'movement_policy_eval_best.pt',
                model=snapshot.model,
                config=snapshot.config,
                lane_feature_dim=snapshot.lane_feature_dim,
                movement_feature_dim=snapshot.movement_feature_dim,
                lane_normalizer=snapshot.lane_normalizer,
                movement_normalizer=snapshot.movement_normalizer,
                loss=snapshot.loss,
            )
            print(f'  new best learned wait density={self.best_learned_wait_density:.4f} s/m at epoch {snapshot.epoch}')

    def _should_evaluate(self, snapshot: MovementILTrainingSnapshot) -> bool:
        return snapshot.epoch % self.every_epochs == 0 or snapshot.epoch == snapshot.epochs

    def _run_epoch_evaluation(self, checkpoint_path: Path) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = list(self.baseline_records)
        cached_baseline_keys = {(record.policy, record.seed) for record in self.baseline_records}
        learned_policy_config = LearnedPolicyConfig(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        pending_runs = tuple(
            (policy, seed)
            for policy in self.policies
            for seed in self.seeds
            if policy == EvaluationPolicy.LEARNED or (policy.value, seed) not in cached_baseline_keys
        )
        total_runs = len(pending_runs)
        run_index = 0
        batch_started_s = current_timer_s()
        for policy, seed in pending_runs:
            run_index += 1
            run_started_s = current_timer_s()
            print_evaluation_start(
                policy=policy.value,
                seed=seed,
                run_index=run_index,
                total_runs=total_runs,
            )
            metrics = run_evaluation_episode(
                cfg_path=self.cfg_path,
                policy=policy,
                seed=seed,
                steps=self.steps,
                decision_interval=self.decision_interval,
                yellow_duration=self.yellow_duration,
                min_green_steps=self.min_green_steps,
                learned_policy_config=learned_policy_config,
                demand_scale=self.demand_scale,
                initial_occupancy_min=self.initial_occupancy_min,
                initial_occupancy_max=self.initial_occupancy_max,
                time_to_teleport=self.time_to_teleport,
            )
            records.append(
                EvaluationRecord(
                    policy=policy.value,
                    seed=seed,
                    metrics=metrics,
                )
            )
            print_evaluation_result(
                policy=policy.value,
                seed=seed,
                metrics=metrics,
                run_index=run_index,
                total_runs=total_runs,
                run_elapsed_s=current_timer_s() - run_started_s,
                batch_started_s=batch_started_s,
            )
        return records

    def _write_tensorboard(
        self,
        epoch: int,
        aggregates: Sequence[EvaluationAggregate],
    ) -> None:
        writer = SummaryWriter(log_dir=str(self.log_dir))
        for aggregate in aggregates:
            writer.add_scalar(
                f'eval/{aggregate.policy}/throughput_per_hour',
                aggregate.mean.throughput_per_hour,
                epoch,
            )
            writer.add_scalar(
                f'eval/{aggregate.policy}/average_wait_density_s_per_m',
                aggregate.mean.average_wait_density_s_per_m,
                epoch,
            )
            writer.add_scalar(
                f'eval/{aggregate.policy}/completion_rate',
                aggregate.mean.completion_rate,
                epoch,
            )
        writer.close()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = args.ckpt_dir or ROOT / 'checkpoints' / 'il' / stamp
    log_dir = args.log_dir or ROOT / 'runs' / 'il' / stamp
    if args.data is None and args.sumo_config_path is None:
        raise SystemExit('Either --data or --cfg is required.')
    observer = _training_evaluation_observer(
        args=args,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    config = MovementILTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        loss=MovementILLoss(args.loss),
        device=args.device,
        progress_every=args.progress_every,
        num_hops=args.num_hops,
        phase_loss_coefficient=args.phase_loss_coeff,
        samples_per_batch=args.samples_per_batch,
        log_dir=log_dir,
    )
    if args.data is not None:
        result = train_movement_il_from_jsonl(
            dataset_path=args.data,
            config=config,
            observer=observer,
        )
    else:
        with tempfile.TemporaryDirectory(prefix='movement_il_') as temporary_directory:
            dataset_path = Path(temporary_directory) / 'samples.jsonl'
            if args.samples <= 0:
                raise SystemExit('--samples must be positive.')
            if args.samples_per_simulation <= 0:
                raise SystemExit('--samples-per-simulation must be positive.')
            if args.determinism_check_samples > 0:
                verify_max_pressure_determinism(
                    cfg_path=args.sumo_config_path,
                    decision_samples=args.determinism_check_samples,
                    decision_interval=args.decision_interval,
                    seed=args.collection_seed,
                    demand_scale=args.demand_scale,
                    initial_occupancy=sample_target_occupancy(
                        minimum_occupancy=args.initial_occupancy_min,
                        maximum_occupancy=args.initial_occupancy_max,
                        seed=args.collection_seed,
                    ),
                    time_to_teleport=args.time_to_teleport,
                )
            combined_samples = []
            simulation_index = 0
            while len(combined_samples) < args.samples:
                collection_seed = args.collection_seed + simulation_index
                remaining_samples = args.samples - len(combined_samples)
                simulation_samples = min(args.samples_per_simulation, remaining_samples)
                seed_dataset_path = Path(temporary_directory) / f'samples_seed_{collection_seed}.jsonl'
                collected_count = collect_samples(
                    cfg_path=args.sumo_config_path,
                    output_path=seed_dataset_path,
                    steps=simulation_samples * args.decision_interval,
                    decision_interval=args.decision_interval,
                    seed=collection_seed,
                    demand_scale=args.demand_scale,
                    initial_occupancy=sample_target_occupancy(
                        minimum_occupancy=args.initial_occupancy_min,
                        maximum_occupancy=args.initial_occupancy_max,
                        seed=collection_seed,
                    ),
                    time_to_teleport=args.time_to_teleport,
                )
                if collected_count == 0:
                    raise RuntimeError(f'Collection produced no samples for seed {collection_seed}.')
                combined_samples.extend(load_jsonl_samples(seed_dataset_path))
                print(f'Collected {collected_count} IL samples with seed={collection_seed}')
                simulation_index += 1
            save_jsonl_samples(dataset_path, combined_samples)
            save_jsonl_samples(checkpoint_dir / 'training_samples.jsonl', combined_samples)
            print(f'Combined {len(combined_samples)} IL samples from {simulation_index} simulations')
            result = train_movement_il_from_jsonl(
                dataset_path=dataset_path,
                config=config,
                observer=observer,
            )
    print(
        f'Training complete: epochs={result.epochs} '
        f'final_loss={result.final_loss:.6f} checkpoint={result.checkpoint_path}'
    )


def _training_evaluation_observer(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    log_dir: Path,
) -> TrainingEvaluationObserver | None:
    if args.eval_cfg is None or args.eval_every_epochs <= 0:
        return None
    return TrainingEvaluationObserver(
        cfg_path=args.eval_cfg,
        policies=tuple(EvaluationPolicy(policy) for policy in args.eval_policies),
        seeds=tuple(args.eval_seeds),
        steps=args.eval_steps,
        decision_interval=args.decision_interval,
        yellow_duration=args.eval_yellow_duration,
        min_green_steps=args.eval_min_green_steps,
        demand_scale=args.eval_demand_scale,
        initial_occupancy_min=args.initial_occupancy_min,
        initial_occupancy_max=args.initial_occupancy_max,
        time_to_teleport=args.time_to_teleport,
        output_dir=args.eval_output_dir or checkpoint_dir / 'eval',
        log_dir=log_dir,
        device=args.device,
        every_epochs=args.eval_every_epochs,
    )


if __name__ == '__main__':
    main()
