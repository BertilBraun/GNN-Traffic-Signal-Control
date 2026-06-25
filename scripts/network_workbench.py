"""Orchestrate repeatable city SUMO network builds from saved inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_network  # noqa: E402
from src.movement.city_build.recipe import (  # noqa: E402
    CityBuildPaths,
    CityBuildRecipe,
    DemandRecipe,
    NetconvertRecipe,
    OsmCachePolicy,
    SourceRecipe,
    VerificationRecipe,
    city_build_paths,
    load_build_recipe,
    save_build_recipe,
)


class WorkbenchCommand(str, Enum):
    FETCH = 'fetch'
    BUILD_INITIAL = 'build-initial'
    PRUNE = 'prune'
    REBUILD = 'rebuild'
    INSPECT = 'inspect'
    VISUALIZE = 'visualize'
    RUN_GUI = 'run-gui'
    EVALUATE = 'evaluate'
    ALL = 'all'


@dataclass(frozen=True)
class WorkbenchContext:
    recipe: CityBuildRecipe
    paths: CityBuildPaths


@dataclass(frozen=True)
class CompletedWorkbenchCommand:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the city network build workbench.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--build-file',
        '--recipe',
        dest='recipe',
        type=Path,
        default=None,
        metavar='BUILD_FILE',
        help='Saved city build YAML file',
    )
    parser.add_argument('--name', default=None, help='City build name, used under configs/<name>')
    parser.add_argument(
        '--bbox',
        default=None,
        metavar='S,W,N,E',
        help='Bounding box to download from Overpass when creating a new build file',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='City output directory when creating a new build file',
    )
    parser.add_argument('--join-dist', type=float, default=40.0, help='SUMO junction join distance in metres')
    parser.add_argument('--route-count', type=int, default=300, help='Maximum city O-D routes')
    parser.add_argument(
        '--demand-vehicles-per-hour',
        type=float,
        default=900.0,
        help='Base route-file demand in vehicles per hour',
    )
    parser.add_argument('--demand-scale', type=float, default=1.0, help='Runtime demand scale for checks')
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open generated/served visualizations for subcommands that support it',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Prune UI host')
    parser.add_argument('--port', type=int, default=8765, help='Prune UI port')
    parser.add_argument(
        'command',
        choices=tuple(command.value for command in WorkbenchCommand),
        help='Workbench subcommand',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = _context_from_args(args)
    command = WorkbenchCommand(args.command)
    _run_workbench_command(
        context=context,
        command=command,
        open_browser=args.open,
        host=args.host,
        port=args.port,
    )


def _context_from_args(args: argparse.Namespace) -> WorkbenchContext:
    if args.recipe is not None:
        recipe_path = args.recipe.resolve()
        recipe = load_build_recipe(recipe_path)
        return WorkbenchContext(
            recipe=recipe,
            paths=city_build_paths(recipe_path=recipe_path, recipe=recipe),
        )
    if args.name is None:
        raise SystemExit('Provide either --build-file, or --name.')
    city_directory = args.out_dir.resolve() if args.out_dir is not None else ROOT / 'configs' / args.name
    recipe_path = city_directory / f'{args.name}.build.yaml'
    if recipe_path.exists():
        recipe = load_build_recipe(recipe_path)
        print(f'Using existing build file: {recipe_path}')
        return WorkbenchContext(
            recipe=recipe,
            paths=city_build_paths(recipe_path=recipe_path, recipe=recipe),
        )
    if args.bbox is None:
        raise SystemExit(f'Missing build file {recipe_path}. Provide --bbox to create it.')
    recipe = CityBuildRecipe(
        name=args.name,
        source=SourceRecipe(bbox=args.bbox, cache_policy=OsmCachePolicy.REUSE),
        netconvert=NetconvertRecipe(join_dist=args.join_dist, promote_all_junctions_to_tl=False),
        demand=DemandRecipe(route_count=args.route_count, demand_vehicles_per_hour=args.demand_vehicles_per_hour),
        verification=VerificationRecipe(demand_scale=args.demand_scale),
    )
    save_build_recipe(recipe_path=recipe_path, recipe=recipe)
    print(f'Wrote build file: {recipe_path}')
    return WorkbenchContext(
        recipe=recipe,
        paths=city_build_paths(recipe_path=recipe_path, recipe=recipe),
    )


def _run_workbench_command(
    context: WorkbenchContext,
    command: WorkbenchCommand,
    open_browser: bool,
    host: str,
    port: int,
) -> None:
    match command:
        case WorkbenchCommand.FETCH:
            _fetch(context)
            _print_next_step('Run build-initial to create the first editable network.')
        case WorkbenchCommand.BUILD_INITIAL:
            _build(context=context, include_prune=False)
            _print_next_step('Run prune --open to edit topology with auto-save and auto-rebuild.')
        case WorkbenchCommand.PRUNE:
            _prune(
                context=context,
                open_browser=open_browser,
                host=host,
                port=port,
            )
        case WorkbenchCommand.REBUILD:
            _build(context=context, include_prune=True)
            _print_next_step('Run inspect, visualize, or all to verify the rebuilt network.')
        case WorkbenchCommand.INSPECT:
            _inspect(context)
            _print_next_step(f'Review {context.paths.inspection_report_path}.')
        case WorkbenchCommand.VISUALIZE:
            _visualize(context=context, open_browser=open_browser)
            _print_next_step(f'Review HTML reports in {context.paths.reports_directory}.')
        case WorkbenchCommand.RUN_GUI:
            _run_gui(context)
        case WorkbenchCommand.EVALUATE:
            _evaluate(context)
        case WorkbenchCommand.ALL:
            _run_interactive_pipeline(context=context, host=host, port=port)


def _fetch(context: WorkbenchContext) -> None:
    context.paths.city_directory.mkdir(parents=True, exist_ok=True)
    if context.recipe.source.bbox is None:
        print(f'Using explicit OSM source: {context.recipe.source.osm}')
        return
    source = build_network._fetch_osm_for_bbox(
        bbox=context.recipe.source.bbox,
        output_path=context.paths.osm_path,
        cache_directory=build_network.DEFAULT_OSM_CACHE_DIR,
        refresh_osm=context.recipe.source.cache_policy == OsmCachePolicy.REFRESH,
    )
    print(f'OSM source: {source.kind.value}')
    if source.cache_path is not None:
        print(f'Cache path: {source.cache_path}')


def _build(context: WorkbenchContext, include_prune: bool) -> CompletedWorkbenchCommand:
    command = _build_network_command(context=context, include_prune=include_prune)
    return _run_checked_command(command=command, report_path=None)


def _run_interactive_pipeline(
    context: WorkbenchContext,
    host: str,
    port: int,
) -> None:
    print('[workbench] Building network before prune review.')
    _build(context=context, include_prune=context.paths.prune_path.exists())
    print('[workbench] Opening prune editor. Click Finish Pruning to continue.')
    _prune(context=context, open_browser=True, host=host, port=port)
    print('[workbench] Rebuilding final network from saved prune JSON.')
    _build(context=context, include_prune=context.paths.prune_path.exists())
    if context.recipe.verification.inspect:
        _inspect(context)
    if context.recipe.verification.movement_graph_html or context.recipe.verification.detection_html:
        _visualize(context=context, open_browser=True)
    _write_build_summary(context)
    if context.recipe.verification.gui:
        _run_gui(context)
    else:
        _print_next_step('Set verification.gui: true, or run run-gui manually, for SUMO-GUI demand calibration.')


def _build_network_command(context: WorkbenchContext, include_prune: bool) -> tuple[str, ...]:
    source_arguments = _build_source_arguments(context)
    command = (
        sys.executable,
        str(ROOT / 'scripts' / 'build_network.py'),
        *source_arguments,
        '--out-dir',
        str(context.paths.city_directory),
        '--name',
        context.recipe.name,
        '--join-dist',
        str(context.recipe.netconvert.join_dist),
        '--route-count',
        str(context.recipe.demand.route_count),
        '--demand-vehicles-per-hour',
        str(context.recipe.demand.demand_vehicles_per_hour),
    )
    if context.recipe.netconvert.promote_all_junctions_to_tl:
        command = (*command, '--promote-all-junctions-to-tl')
    if include_prune and context.paths.prune_path.exists():
        command = (*command, '--prune', str(context.paths.prune_path))
    return command


def _build_source_arguments(context: WorkbenchContext) -> tuple[str, ...]:
    if context.recipe.source.bbox is not None:
        arguments = ('--bbox', context.recipe.source.bbox)
        if context.recipe.source.cache_policy == OsmCachePolicy.REFRESH:
            arguments = (*arguments, '--refresh-osm')
        return arguments
    if context.recipe.source.osm is None:
        raise ValueError('Recipe source does not include bbox or osm.')
    return ('--osm', str(_resolve_recipe_relative_path(context.paths.recipe_path, context.recipe.source.osm)))


def _resolve_recipe_relative_path(recipe_path: Path, configured_path: Path) -> Path:
    if configured_path.is_absolute():
        return configured_path
    return recipe_path.parent / configured_path


def _prune(
    context: WorkbenchContext,
    open_browser: bool,
    host: str,
    port: int,
) -> CompletedWorkbenchCommand:
    if not context.paths.net_path.exists():
        raise SystemExit(f'Missing network file: {context.paths.net_path}. Run build-initial first.')
    command = _prune_command(
        context=context,
        open_browser=open_browser,
        host=host,
        port=port,
    )
    return _run_passthrough_command(command=command)


def _prune_command(
    context: WorkbenchContext,
    open_browser: bool,
    host: str,
    port: int,
) -> tuple[str, ...]:
    command = (
        sys.executable,
        str(ROOT / 'scripts' / 'visualize_network_prune.py'),
        '--net',
        str(context.paths.net_path),
        '--prune',
        str(context.paths.prune_path),
        '--save-prune',
        str(context.paths.prune_path),
        '--serve',
        '--host',
        host,
        '--port',
        str(port),
    )
    if open_browser:
        command = (*command, '--open')
    return command


def _inspect(context: WorkbenchContext) -> CompletedWorkbenchCommand:
    context.paths.reports_directory.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        str(ROOT / 'scripts' / 'inspect_movement_city.py'),
        '--cfg',
        str(context.paths.sumo_config_path),
        '--time-to-teleport',
        str(context.recipe.verification.time_to_teleport),
    )
    return _run_checked_command(command=command, report_path=context.paths.inspection_report_path)


def _visualize(context: WorkbenchContext, open_browser: bool) -> None:
    context.paths.reports_directory.mkdir(parents=True, exist_ok=True)
    if context.recipe.verification.movement_graph_html:
        command = (
            sys.executable,
            str(ROOT / 'scripts' / 'visualize_movement_graph.py'),
            '--cfg',
            str(context.paths.sumo_config_path),
            '--out',
            str(context.paths.movement_graph_path),
        )
        if open_browser:
            command = (*command, '--open')
        _run_checked_command(command=command, report_path=None)
    if context.recipe.verification.detection_html:
        command = (
            sys.executable,
            str(ROOT / 'scripts' / 'visualize_movement_detection.py'),
            '--cfg',
            str(context.paths.sumo_config_path),
            '--out',
            str(context.paths.movement_detection_path),
            '--steps',
            str(context.recipe.verification.detection_steps),
            '--sample-every',
            str(context.recipe.verification.detection_sample_every),
            '--demand-scale',
            str(context.recipe.verification.demand_scale),
            '--time-to-teleport',
            str(context.recipe.verification.time_to_teleport),
        )
        if open_browser:
            command = (*command, '--open')
        _run_checked_command(command=command, report_path=None)


def _run_gui(context: WorkbenchContext) -> CompletedWorkbenchCommand:
    command = (
        sys.executable,
        str(ROOT / 'scripts' / 'run.py'),
        '--cfg',
        str(context.paths.sumo_config_path),
        '--method',
        'max-pressure',
        '--gui',
        '--steps',
        str(context.recipe.verification.gui_steps),
        '--demand-scale',
        str(context.recipe.verification.demand_scale),
        '--time-to-teleport',
        str(context.recipe.verification.time_to_teleport),
    )
    return _run_checked_command(command=command, report_path=None)


def _evaluate(context: WorkbenchContext) -> CompletedWorkbenchCommand:
    command = (
        sys.executable,
        str(ROOT / 'scripts' / 'eval_policy.py'),
        '--cfg',
        str(context.paths.sumo_config_path),
        '--out-dir',
        str(context.paths.evaluation_directory),
        '--policies',
        *context.recipe.evaluation.policies,
        '--seeds',
        *(str(seed) for seed in context.recipe.evaluation.seeds),
        '--steps',
        str(context.recipe.evaluation.steps),
        '--demand-scale',
        str(context.recipe.verification.demand_scale),
        '--time-to-teleport',
        str(context.recipe.verification.time_to_teleport),
    )
    return _run_checked_command(command=command, report_path=None)


def _run_checked_command(command: tuple[str, ...], report_path: Path | None) -> CompletedWorkbenchCommand:
    print(_display_command(command))
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    result = CompletedWorkbenchCommand(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if report_path is not None:
        report_path.write_text(result.stdout + result.stderr, encoding='utf-8')
        print(f'Wrote {report_path}')
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def _run_passthrough_command(command: tuple[str, ...]) -> CompletedWorkbenchCommand:
    print(_display_command(command))
    completed = subprocess.run(command, cwd=ROOT)
    result = CompletedWorkbenchCommand(
        command=command,
        returncode=completed.returncode,
        stdout='',
        stderr='',
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def _display_command(command: tuple[str, ...]) -> str:
    return ' '.join(f'"{part}"' if ' ' in part else part for part in command)


def _write_build_summary(context: WorkbenchContext) -> None:
    context.paths.reports_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        'name': context.recipe.name,
        'build_file': str(context.paths.recipe_path),
        'osm': str(context.paths.osm_path),
        'net': str(context.paths.net_path),
        'prune': str(context.paths.prune_path) if context.paths.prune_path.exists() else None,
        'sumocfg': str(context.paths.sumo_config_path),
        'reports': {
            'inspection': str(context.paths.inspection_report_path),
            'movement_graph': str(context.paths.movement_graph_path),
            'movement_detection': str(context.paths.movement_detection_path),
        },
        'demand': {
            'route_count': context.recipe.demand.route_count,
            'demand_vehicles_per_hour': context.recipe.demand.demand_vehicles_per_hour,
            'demand_scale': context.recipe.verification.demand_scale,
        },
    }
    context.paths.build_summary_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {context.paths.build_summary_path}')


def _print_next_step(message: str) -> None:
    print(f'Next: {message}')


if __name__ == '__main__':
    main()
