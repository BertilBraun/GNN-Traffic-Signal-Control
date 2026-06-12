"""Train a movement-score imitation model from JSONL samples."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.training.il import (  # noqa: E402
    MovementILLoss,
    MovementILTrainingConfig,
    MovementILTrainingSnapshot,
    save_movement_checkpoint,
    train_movement_il_from_jsonl,
)
from src.movement.evaluation import (  # noqa: E402
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
from scripts.collect_il_data import collect_samples  # noqa: E402


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
    parser.add_argument('--samples', type=int, default=120, help='Number of decision samples to collect with --cfg')
    parser.add_argument('--decision-interval', type=int, default=15, help='Seconds between collected samples')
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=1.0,
        help='Multiplier applied to route-file flow demand during collection',
    )
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='MLP hidden dimension')
    parser.add_argument('--num-hops', type=int, default=1, help='LaneGroup/Movement macro-hops')
    parser.add_argument(
        '--loss',
        choices=('huber', 'mse'),
        default='huber',
        help='Movement-score regression loss',
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
        default=None,
        help='SUMO config used for periodic learned-policy evaluation',
    )
    parser.add_argument(
        '--eval-every-epochs',
        type=int,
        default=0,
        help='Run evaluation every N epochs when --eval-cfg is set (0 disables)',
    )
    parser.add_argument('--eval-steps', type=int, default=600, help='Simulation seconds per evaluation episode')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[42], help='Evaluation SUMO seeds')
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
    return parser.parse_args()


@dataclass(frozen=True)
class TrainingEvaluationObserver:
    cfg_path: Path
    policies: tuple[EvaluationPolicy, ...]
    seeds: tuple[int, ...]
    steps: int
    decision_interval: int
    yellow_duration: int
    min_green_steps: int
    demand_scale: float
    output_dir: Path
    device: str
    every_epochs: int

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
        aggregates = aggregate_records(records)
        write_aggregate_json(epoch_dir / 'summary.json', records, aggregates)
        write_records_csv(epoch_dir / 'summary.csv', records, aggregates)
        print_aggregate_metric_table(f'Evaluation summary at epoch {snapshot.epoch}', aggregates)

    def _should_evaluate(self, snapshot: MovementILTrainingSnapshot) -> bool:
        return snapshot.epoch % self.every_epochs == 0 or snapshot.epoch == snapshot.epochs

    def _run_epoch_evaluation(self, checkpoint_path: Path) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        learned_policy_config = LearnedPolicyConfig(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        total_runs = len(self.policies) * len(self.seeds)
        run_index = 0
        batch_started_s = current_timer_s()
        for policy in self.policies:
            for seed in self.seeds:
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


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = args.ckpt_dir or ROOT / 'checkpoints' / 'il' / stamp
    if args.data is None and args.sumo_config_path is None:
        raise SystemExit('Either --data or --cfg is required.')
    observer = _training_evaluation_observer(
        args=args,
        checkpoint_dir=checkpoint_dir,
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
            collect_samples(
                cfg_path=args.sumo_config_path,
                output_path=dataset_path,
                steps=args.samples * args.decision_interval,
                decision_interval=args.decision_interval,
                seed=args.seed,
                demand_scale=args.demand_scale,
            )
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
        output_dir=args.eval_output_dir or checkpoint_dir / 'eval',
        device=args.device,
        every_epochs=args.eval_every_epochs,
    )


if __name__ == '__main__':
    main()
