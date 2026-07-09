"""Regenerate SUMO route files from existing city build recipes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

import sumolib  # noqa: E402

from scripts import build_network  # noqa: E402
from src.movement.city_build.recipe import city_build_paths, load_build_recipe  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Regenerate city route XML files from existing net XML files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'build_files',
        nargs='+',
        type=Path,
        help='City build YAML files to regenerate',
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    for build_file in arguments.build_files:
        regenerate_routes(build_file.resolve())


def regenerate_routes(build_file: Path) -> None:
    recipe = load_build_recipe(build_file)
    paths = city_build_paths(recipe_path=build_file, recipe=recipe)
    network = sumolib.net.readNet(str(paths.net_path), withConnections=True)
    route_count = build_network._write_routes(
        net=network,
        rou_path=paths.route_path,
        route_count=recipe.demand.route_count,
        demand_vehicles_per_hour=recipe.demand.demand_vehicles_per_hour,
    )
    print(f'{recipe.name}: regenerated {route_count} fastest-path routes at {paths.route_path}')


if __name__ == '__main__':
    main()
