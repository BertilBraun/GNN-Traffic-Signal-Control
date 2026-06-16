"""Evaluate movement-based traffic signal policies on identical SUMO seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
DEFAULT_OUTPUT_DIR = ROOT / 'reports' / 'movement_eval'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate movement policies on identical SUMO seeds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', type=Path, default=DEFAULT_CFG, help='SUMO .sumocfg path')
    parser.add_argument(
        '--policies',
        nargs='+',
        choices=tuple(policy.value for policy in EvaluationPolicy),
        default=[EvaluationPolicy.MAX_PRESSURE.value, EvaluationPolicy.QUEUE.value],
        help='Policies to evaluate',
    )
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint for the learned policy')
    parser.add_argument('--device', default='cpu', help='Torch device for learned policy inference')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42], help='SUMO random seeds')
    parser.add_argument('--steps', type=int, default=1800, help='Maximum simulation seconds per episode')
    parser.add_argument('--decision-interval', type=int, default=15, help='Seconds between policy decisions')
    parser.add_argument('--yellow-duration', type=int, default=3, help='Yellow transition duration in seconds')
    parser.add_argument(
        '--min-green-steps',
        type=int,
        default=2,
        help='Minimum accepted decision intervals before switching',
    )
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory for JSON and CSV outputs')
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=1.0,
        help='Multiplier applied to route-file flow demand at runtime',
    )
    parser.add_argument('--initial-occupancy-min', type=float, default=0.05)
    parser.add_argument('--initial-occupancy-max', type=float, default=0.08)
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=None,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = tuple(EvaluationPolicy(policy) for policy in args.policies)
    learned_policy_config = _learned_policy_config(policies, args.checkpoint, args.device)

    records: list[EvaluationRecord] = []
    total_runs = len(policies) * len(args.seeds)
    run_index = 0
    batch_started_s = current_timer_s()
    for policy in policies:
        for seed in args.seeds:
            run_index += 1
            run_started_s = current_timer_s()
            print_evaluation_start(
                policy=policy.value,
                seed=seed,
                run_index=run_index,
                total_runs=total_runs,
            )
            metrics = run_evaluation_episode(
                cfg_path=args.cfg,
                policy=policy,
                seed=seed,
                steps=args.steps,
                decision_interval=args.decision_interval,
                yellow_duration=args.yellow_duration,
                min_green_steps=args.min_green_steps,
                learned_policy_config=learned_policy_config,
                demand_scale=args.demand_scale,
                initial_occupancy_min=args.initial_occupancy_min,
                initial_occupancy_max=args.initial_occupancy_max,
                time_to_teleport=args.time_to_teleport,
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

    aggregates = aggregate_records(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / 'summary.json'
    csv_path = args.out_dir / 'summary.csv'
    write_aggregate_json(json_path, records, aggregates)
    write_records_csv(csv_path, records, aggregates)
    print_aggregate_metric_table('Evaluation summary (mean across seeds)', aggregates)
    print(f'Wrote {json_path}')
    print(f'Wrote {csv_path}')


def _learned_policy_config(
    policies: tuple[EvaluationPolicy, ...],
    checkpoint_path: Path | None,
    device: str,
) -> LearnedPolicyConfig | None:
    if EvaluationPolicy.LEARNED not in policies:
        return None
    if checkpoint_path is None:
        raise SystemExit('--checkpoint is required when evaluating the learned policy')
    return LearnedPolicyConfig(
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == '__main__':
    main()
