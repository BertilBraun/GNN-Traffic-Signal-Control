"""Evaluate movement policies across all cities in an experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.evaluation import EvaluationPolicy, LearnedPolicyConfig, is_learned_evaluation_policy  # noqa: E402
from src.movement.evaluation.multi_city import (  # noqa: E402
    FileCachedEpisodeRunner,
    default_episode_runner,
    print_multi_city_summary,
    run_multi_city_evaluation,
    write_multi_city_csv,
    write_multi_city_json,
    write_multi_city_tensorboard,
)
from src.movement.experiment_config import ExperimentEvaluationPolicy, load_experiment_configuration  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate movement policies across train and held-out experiment cities.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--experiment-config', type=Path, required=True, help='Experiment YAML path')
    parser.add_argument(
        '--policies',
        nargs='+',
        choices=tuple(policy.value for policy in EvaluationPolicy),
        default=None,
        help='Policies to evaluate; defaults to experiment configuration policies',
    )
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint for the learned policy')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for CSV and JSON outputs')
    parser.add_argument('--device', default='cpu', help='Torch device for learned policy inference')
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Override the configured number of parallel evaluation workers',
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Override experiment evaluation seeds')
    parser.add_argument('--steps', type=int, default=None, help='Override experiment evaluation steps')
    parser.add_argument(
        '--demand-scales',
        nargs='+',
        type=float,
        default=None,
        help='Override experiment evaluation demand scales',
    )
    parser.add_argument('--log-dir', type=Path, default=None, help='Optional TensorBoard output directory')
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=ROOT / '.cache' / 'evaluation',
        help='Directory for deterministic baseline evaluation cache files',
    )
    return parser.parse_args()


def main() -> None:
    parsed_arguments = parse_arguments()
    configuration = load_experiment_configuration(
        configuration_path=parsed_arguments.experiment_config,
        project_root=ROOT,
    )
    policies = _evaluation_policies(
        configuration_policy_values=configuration.evaluation.policies,
        parsed_arguments=parsed_arguments,
    )
    learned_policy_config = _learned_policy_config(
        policies=policies,
        checkpoint_path=parsed_arguments.checkpoint,
        device=parsed_arguments.device,
    )
    seeds = tuple(configuration.evaluation.seeds if parsed_arguments.seeds is None else parsed_arguments.seeds)
    steps = configuration.evaluation.steps if parsed_arguments.steps is None else parsed_arguments.steps
    demand_scales = tuple(
        configuration.demand.evaluation_scales
        if parsed_arguments.demand_scales is None
        else parsed_arguments.demand_scales
    )

    result = run_multi_city_evaluation(
        configuration=configuration,
        project_root=ROOT,
        policies=policies,
        seeds=seeds,
        steps=steps,
        demand_scales=demand_scales,
        learned_policy_config=learned_policy_config,
        episode_runner=FileCachedEpisodeRunner(
            cache_dir=parsed_arguments.cache_dir,
            episode_runner=default_episode_runner,
        ),
        worker_count=(
            configuration.proximal_policy_optimization.evaluation_workers
            if parsed_arguments.workers is None
            else parsed_arguments.workers
        ),
    )
    parsed_arguments.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = parsed_arguments.output_dir / 'summary.json'
    csv_path = parsed_arguments.output_dir / 'summary.csv'
    write_multi_city_json(path=json_path, result=result)
    write_multi_city_csv(path=csv_path, result=result)
    if parsed_arguments.log_dir is not None:
        write_multi_city_tensorboard(log_dir=parsed_arguments.log_dir, aggregates=result.aggregates)
    print_multi_city_summary(result.aggregates)
    print(f'Wrote {json_path}')
    print(f'Wrote {csv_path}')


def _evaluation_policies(
    configuration_policy_values: tuple[ExperimentEvaluationPolicy, ...],
    parsed_arguments: argparse.Namespace,
) -> tuple[EvaluationPolicy, ...]:
    if parsed_arguments.policies is None:
        return tuple(EvaluationPolicy(policy.value) for policy in configuration_policy_values)
    return tuple(EvaluationPolicy(policy) for policy in parsed_arguments.policies)


def _learned_policy_config(
    policies: tuple[EvaluationPolicy, ...],
    checkpoint_path: Path | None,
    device: str,
) -> LearnedPolicyConfig | None:
    if not any(is_learned_evaluation_policy(policy) for policy in policies):
        return None
    if checkpoint_path is None:
        raise SystemExit('--checkpoint is required when evaluating the learned policy')
    return LearnedPolicyConfig(
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == '__main__':
    main()
