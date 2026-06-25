from pathlib import Path
import sys
from argparse import Namespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import network_workbench  # noqa: E402
from src.movement.city_build.recipe import city_build_paths, load_build_recipe  # noqa: E402


def test_load_build_recipe_from_yaml() -> None:
    recipe_path = ROOT / 'configs' / 'karlsruhe_oststadt' / 'karlsruhe_oststadt.build.yaml'

    recipe = load_build_recipe(recipe_path)
    paths = city_build_paths(recipe_path=recipe_path, recipe=recipe)

    assert recipe.name == 'karlsruhe_oststadt'
    assert recipe.source.bbox == '49.0000,8.4050,49.0230,8.4520'
    assert recipe.netconvert.join_dist == 35
    assert recipe.verification.demand_scale == 4.5
    assert paths.prune_path.name == 'karlsruhe_oststadt.prune.json'
    assert paths.movement_graph_path.name == 'movement_graph.html'


def test_build_initial_command_omits_prune() -> None:
    recipe_path = ROOT / 'configs' / 'karlsruhe_oststadt' / 'karlsruhe_oststadt.build.yaml'
    recipe = load_build_recipe(recipe_path)
    context = network_workbench.WorkbenchContext(
        recipe=recipe,
        paths=city_build_paths(recipe_path=recipe_path, recipe=recipe),
    )

    command = network_workbench._build_network_command(context=context, include_prune=False)

    assert '--bbox' in command
    assert '--prune' not in command
    assert '--route-count' in command
    assert '300' in command


def test_rebuild_command_includes_existing_prune() -> None:
    recipe_path = ROOT / 'configs' / 'karlsruhe_oststadt' / 'karlsruhe_oststadt.build.yaml'
    recipe = load_build_recipe(recipe_path)
    paths = city_build_paths(recipe_path=recipe_path, recipe=recipe)
    context = network_workbench.WorkbenchContext(recipe=recipe, paths=paths)

    command = network_workbench._build_network_command(context=context, include_prune=True)

    assert '--prune' in command
    assert str(paths.prune_path) in command


def test_prune_command_defaults_to_served_auto_rebuild_editor() -> None:
    recipe_path = ROOT / 'configs' / 'karlsruhe_oststadt' / 'karlsruhe_oststadt.build.yaml'
    recipe = load_build_recipe(recipe_path)
    paths = city_build_paths(recipe_path=recipe_path, recipe=recipe)
    context = network_workbench.WorkbenchContext(recipe=recipe, paths=paths)

    command = network_workbench._prune_command(
        context=context,
        open_browser=True,
        host='127.0.0.1',
        port=8765,
    )

    assert '--serve' in command
    assert '--open' in command
    assert '--save-prune' in command
    assert str(paths.prune_path) in command
    assert '--out' not in command


def test_name_and_bbox_create_saved_build_file(tmp_path: Path) -> None:
    args = Namespace(
        recipe=None,
        name='test_city',
        bbox='1,2,3,4',
        out_dir=tmp_path / 'test_city',
        join_dist=25.0,
        route_count=12,
        demand_vehicles_per_hour=450.0,
        demand_scale=2.0,
    )

    context = network_workbench._context_from_args(args)

    assert context.recipe.name == 'test_city'
    assert context.recipe.source.bbox == '1,2,3,4'
    assert context.recipe.netconvert.join_dist == 25.0
    assert context.recipe.demand.route_count == 12
    assert context.recipe.verification.demand_scale == 2.0
    assert context.paths.recipe_path == tmp_path / 'test_city' / 'test_city.build.yaml'
    assert context.paths.recipe_path.exists()


def test_name_reuses_existing_build_file_without_overwriting(tmp_path: Path) -> None:
    build_file = tmp_path / 'test_city' / 'test_city.build.yaml'
    build_file.parent.mkdir()
    build_file.write_text(
        """
name: test_city
source:
  bbox: "1,2,3,4"
  cache_policy: reuse
netconvert:
  join_dist: 35
demand:
  route_count: 300
  demand_vehicles_per_hour: 900
verification:
  demand_scale: 4.5
""",
        encoding='utf-8',
    )
    original_content = build_file.read_text(encoding='utf-8')
    args = Namespace(
        recipe=None,
        name='test_city',
        bbox='9,9,9,9',
        out_dir=tmp_path / 'test_city',
        join_dist=40.0,
        route_count=12,
        demand_vehicles_per_hour=450.0,
        demand_scale=1.0,
    )

    context = network_workbench._context_from_args(args)

    assert context.recipe.source.bbox == '1,2,3,4'
    assert context.recipe.netconvert.join_dist == 35
    assert context.recipe.verification.demand_scale == 4.5
    assert build_file.read_text(encoding='utf-8') == original_content


def test_all_runs_interactive_prune_checkpoint(monkeypatch) -> None:
    recipe_path = ROOT / 'configs' / 'karlsruhe_oststadt' / 'karlsruhe_oststadt.build.yaml'
    recipe = load_build_recipe(recipe_path)
    paths = city_build_paths(recipe_path=recipe_path, recipe=recipe)
    context = network_workbench.WorkbenchContext(recipe=recipe, paths=paths)
    calls = []

    def fake_build(context: network_workbench.WorkbenchContext, include_prune: bool) -> None:
        calls.append(('build', include_prune))

    def fake_prune(
        context: network_workbench.WorkbenchContext,
        open_browser: bool,
        host: str,
        port: int,
    ) -> None:
        calls.append(('prune', open_browser, host, port))

    def fake_inspect(context: network_workbench.WorkbenchContext) -> None:
        calls.append(('inspect',))

    def fake_visualize(context: network_workbench.WorkbenchContext, open_browser: bool) -> None:
        calls.append(('visualize', open_browser))

    def fake_summary(context: network_workbench.WorkbenchContext) -> None:
        calls.append(('summary',))

    def fake_calibrate_demand(context: network_workbench.WorkbenchContext) -> None:
        calls.append(('calibrate-demand',))

    monkeypatch.setattr(network_workbench, '_build', fake_build)
    monkeypatch.setattr(network_workbench, '_prune', fake_prune)
    monkeypatch.setattr(network_workbench, '_inspect', fake_inspect)
    monkeypatch.setattr(network_workbench, '_visualize', fake_visualize)
    monkeypatch.setattr(network_workbench, '_write_build_summary', fake_summary)
    monkeypatch.setattr(network_workbench, '_calibrate_demand', fake_calibrate_demand)

    network_workbench._run_workbench_command(
        context=context,
        command=network_workbench.WorkbenchCommand.ALL,
        open_browser=False,
        host='127.0.0.1',
        port=8765,
    )

    assert calls == [
        ('build', True),
        ('prune', True, '127.0.0.1', 8765),
        ('build', True),
        ('inspect',),
        ('visualize', True),
        ('summary',),
        ('calibrate-demand',),
    ]


def test_runtime_demand_scale_action_updates_build_file_without_rebuild(tmp_path: Path, monkeypatch) -> None:
    context = _write_test_build_file(tmp_path)
    calls = []

    def fake_build(context: network_workbench.WorkbenchContext, include_prune: bool) -> None:
        calls.append(('build', include_prune))

    monkeypatch.setattr(network_workbench, '_build', fake_build)

    updated_context = network_workbench._apply_demand_action(
        context=context,
        action=network_workbench.DemandCalibrationAction(
            base_demand_vehicles_per_hour=None,
            base_demand_multiplier=None,
            demand_scale=2.5,
            demand_scale_multiplier=None,
            rebuild=False,
        ),
    )

    loaded = load_build_recipe(updated_context.paths.recipe_path)
    assert updated_context.recipe.verification.demand_scale == 2.5
    assert loaded.verification.demand_scale == 2.5
    assert loaded.demand.demand_vehicles_per_hour == 900.0
    assert calls == []


def test_base_demand_multiplier_action_updates_build_file_and_rebuilds(tmp_path: Path, monkeypatch) -> None:
    context = _write_test_build_file(tmp_path)
    calls = []

    def fake_build(context: network_workbench.WorkbenchContext, include_prune: bool) -> None:
        calls.append(('build', include_prune, context.recipe.demand.demand_vehicles_per_hour))

    monkeypatch.setattr(network_workbench, '_build', fake_build)

    updated_context = network_workbench._apply_demand_action(
        context=context,
        action=network_workbench.DemandCalibrationAction(
            base_demand_vehicles_per_hour=None,
            base_demand_multiplier=2.0,
            demand_scale=None,
            demand_scale_multiplier=None,
            rebuild=True,
        ),
    )

    loaded = load_build_recipe(updated_context.paths.recipe_path)
    assert updated_context.recipe.demand.demand_vehicles_per_hour == 1800.0
    assert loaded.demand.demand_vehicles_per_hour == 1800.0
    assert loaded.verification.demand_scale == 4.5
    assert calls == [('build', False, 1800.0)]


def test_parse_base_scale_demand_action() -> None:
    action = network_workbench._parse_demand_action('base-scale 2')

    assert action == network_workbench.DemandCalibrationAction(
        base_demand_vehicles_per_hour=None,
        base_demand_multiplier=2.0,
        demand_scale=None,
        demand_scale_multiplier=None,
        rebuild=True,
    )


def _write_test_build_file(directory: Path) -> network_workbench.WorkbenchContext:
    args = Namespace(
        recipe=None,
        name='test_city',
        bbox='1,2,3,4',
        out_dir=directory / 'test_city',
        join_dist=35.0,
        route_count=300,
        demand_vehicles_per_hour=900.0,
        demand_scale=4.5,
    )
    return network_workbench._context_from_args(args)
