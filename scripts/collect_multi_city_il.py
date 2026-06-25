"""Collect balanced multi-city movement imitation samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.experiment_config import load_experiment_configuration  # noqa: E402
from src.movement.training.il.multi_city_collection import (  # noqa: E402
    MultiCityCollectionSettings,
    collect_multi_city_samples_to_directory,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect balanced movement-score imitation samples from train cities.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--experiment-config', required=True, type=Path, help='Experiment YAML path')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory for JSONL datasets')
    parser.add_argument('--workers', type=int, default=None, help='Parallel SUMO collection workers')
    parser.add_argument('--samples-per-city', type=int, default=None, help='Retained samples per train city')
    parser.add_argument(
        '--samples-per-simulation',
        type=int,
        default=None,
        help='Retained samples collected from each SUMO simulation',
    )
    parser.add_argument('--collection-seed', type=int, default=42, help='First deterministic collection seed')
    parser.add_argument(
        '--sample-stride',
        type=int,
        default=3,
        help='Retain every Nth decision sample from each simulation',
    )
    return parser.parse_args()


def main() -> None:
    parsed_arguments = parse_arguments()
    configuration = load_experiment_configuration(
        configuration_path=parsed_arguments.experiment_config,
        project_root=ROOT,
    )
    result = collect_multi_city_samples_to_directory(
        configuration=configuration,
        settings=MultiCityCollectionSettings(
            project_root=ROOT,
            output_dir=parsed_arguments.output_dir,
            samples_per_city=(
                configuration.imitation_learning.samples_per_city
                if parsed_arguments.samples_per_city is None
                else parsed_arguments.samples_per_city
            ),
            samples_per_simulation=(
                configuration.imitation_learning.samples_per_simulation
                if parsed_arguments.samples_per_simulation is None
                else parsed_arguments.samples_per_simulation
            ),
            collection_seed=parsed_arguments.collection_seed,
            sample_stride=parsed_arguments.sample_stride,
            workers=(
                configuration.imitation_learning.collection_workers
                if parsed_arguments.workers is None
                else parsed_arguments.workers
            ),
        ),
    )
    for summary in result.city_summaries:
        print(
            f'city={summary.city_name} split={summary.city_split.value} '
            f'samples={summary.sample_count} output={summary.output_path}'
        )
    print(f'combined={result.combined_output_path}')


if __name__ == '__main__':
    main()
