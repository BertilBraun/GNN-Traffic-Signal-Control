"""Generate matched square, rectangular, and signal-coverage grid suites."""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_grid_network import generate_grid
from src.movement.grid_study import (
    COVERAGE_GENERALIZATION_4X4_SCENARIOS,
    GRID_COVERAGE_SCENARIOS,
    MATCHED_GRID_SCENARIOS,
)


class GridSuite(str, Enum):
    MATCHED = 'matched'
    COVERAGE = 'coverage'
    COVERAGE_GENERALIZATION_4X4 = 'coverage-generalization-4x4'
    ALL = 'all'


def generate_matched_suite(output_root: Path, skip_existing: bool) -> None:
    for scenario in MATCHED_GRID_SCENARIOS:
        output_dir = output_root / scenario.name
        if _should_skip(output_dir=output_dir, skip_existing=skip_existing):
            continue
        generate_grid(
            rows=scenario.rows,
            cols=scenario.cols,
            output_dir=output_dir,
        )


def generate_coverage_suite(output_root: Path, skip_existing: bool) -> None:
    for scenario in GRID_COVERAGE_SCENARIOS:
        output_dir = output_root / scenario.name
        if _should_skip(output_dir=output_dir, skip_existing=skip_existing):
            continue
        generate_grid(
            rows=scenario.rows,
            cols=scenario.cols,
            output_dir=output_dir,
            unsignalized_node_ids=scenario.unsignalized_node_ids,
        )


def generate_coverage_generalization_4x4_suite(output_root: Path, skip_existing: bool) -> None:
    for scenario in COVERAGE_GENERALIZATION_4X4_SCENARIOS:
        output_dir = output_root / scenario.name
        if _should_skip(output_dir=output_dir, skip_existing=skip_existing):
            continue
        generate_grid(
            rows=scenario.rows,
            cols=scenario.cols,
            output_dir=output_dir,
            unsignalized_node_ids=scenario.unsignalized_node_ids,
        )


def _should_skip(output_dir: Path, skip_existing: bool) -> bool:
    if not output_dir.exists():
        return False
    if skip_existing:
        print(f'Skipping existing grid directory without modification: {output_dir}')
        return True
    raise FileExistsError(f'Refusing to overwrite existing grid directory: {output_dir}')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ROOT / 'configs' / 'grid_generalization',
        help='Parent directory for newly generated scenarios',
    )
    parser.add_argument(
        '--suite',
        choices=tuple(suite.value for suite in GridSuite),
        default=GridSuite.ALL.value,
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Leave existing scenario directories untouched instead of failing',
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    suite = GridSuite(arguments.suite)
    if suite in (GridSuite.MATCHED, GridSuite.ALL):
        generate_matched_suite(
            output_root=arguments.output_root,
            skip_existing=arguments.skip_existing,
        )
    if suite in (GridSuite.COVERAGE, GridSuite.ALL):
        generate_coverage_suite(
            output_root=arguments.output_root,
            skip_existing=arguments.skip_existing,
        )
    if suite is GridSuite.COVERAGE_GENERALIZATION_4X4:
        generate_coverage_generalization_4x4_suite(
            output_root=arguments.output_root,
            skip_existing=arguments.skip_existing,
        )


if __name__ == '__main__':
    main()
