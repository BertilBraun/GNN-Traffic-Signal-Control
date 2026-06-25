from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import sumolib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import build_network  # noqa: E402
from scripts import visualize_network_prune  # noqa: E402
from scripts.inspect_movement_city import (  # noqa: E402
    _connectivity_report,
    _suspicious_lane_groups,
    _suspicious_movements,
)
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402


GRID_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
GRID_NET = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.net.xml'


def test_city_tll_builder_writes_movement_safe_programs(tmp_path: Path) -> None:
    network = sumolib.net.readNet(str(GRID_NET), withConnections=True, withFoes=True)
    tll_path = tmp_path / 'city.tll.xml'

    written = build_network._build_tll(network, GRID_NET, tll_path)
    content = tll_path.read_text(encoding='utf-8')

    assert written == 5
    assert 'programID="movement_safe"' in content
    assert 'programID="canonical"' not in content


def test_city_route_builder_writes_destination_routes(tmp_path: Path) -> None:
    network = sumolib.net.readNet(str(GRID_NET), withConnections=True)
    route_path = tmp_path / 'city.rou.xml'

    route_count = build_network._write_routes(
        net=network,
        rou_path=route_path,
        route_count=12,
        demand_vehicles_per_hour=600.0,
    )
    content = route_path.read_text(encoding='utf-8')

    assert route_count > 0
    assert '<route id="city_route_' in content
    assert '<flow id="city_flow_' in content
    assert 'probability="' in content
    assert 'vehsPerHour="' not in content


def test_city_routes_include_internal_origins_and_destinations() -> None:
    network = sumolib.net.readNet(str(GRID_NET), withConnections=True)

    routes = build_network._city_routes(network, route_count=30)
    source_edges = {route[0] for route in routes}
    destination_edges = {route[-1] for route in routes}

    assert any(not edge_id.split('_to_', 1)[0].startswith('S_') for edge_id in source_edges)
    assert any(not edge_id.split('_to_', 1)[1].startswith('S_') for edge_id in destination_edges)


def test_city_route_search_uses_passenger_vehicle_class() -> None:
    class FakeEdge:
        def __init__(self, edge_id: str) -> None:
            self._edge_id = edge_id

        def getID(self) -> str:
            return self._edge_id

    class FakeNet:
        def __init__(self) -> None:
            self.requested_vclass = None

        def getOptimalPath(self, source, sink, fastest=False, vClass=None):
            self.requested_vclass = vClass
            return (source, sink), 2.0

    network = FakeNet()
    route = build_network._shortest_route(network, FakeEdge('source'), FakeEdge('sink'))

    assert route == ('source', 'sink')
    assert network.requested_vclass == 'passenger'


def test_city_additional_file_matches_empty_grid_additional(tmp_path: Path) -> None:
    add_path = tmp_path / 'city.add.xml'

    build_network._write_additional(add_path)
    content = add_path.read_text(encoding='utf-8')

    assert '<additional/>' in content
    assert 'laneAreaDetector' not in content


def test_osm_cache_key_ignores_bbox_whitespace() -> None:
    bbox = '49.0000,8.4050,49.0230,8.4520'
    query = build_network._overpass_query(bbox)

    clean_key = build_network._osm_cache_key(bbox=bbox, query=query)
    spaced_key = build_network._osm_cache_key(
        bbox='49.0000, 8.4050, 49.0230, 8.4520',
        query=query,
    )

    assert clean_key == spaced_key


def test_fetch_osm_for_bbox_reuses_existing_cache(tmp_path: Path) -> None:
    bbox = '49.0000,8.4050,49.0230,8.4520'
    query = build_network._overpass_query(bbox)
    cache_path = tmp_path / 'cache' / f'{build_network._osm_cache_key(bbox=bbox, query=query)}.osm'
    output_path = tmp_path / 'city.osm'
    cache_path.parent.mkdir()
    cache_path.write_text('<osm version="0.6"></osm>', encoding='utf-8')

    source = build_network._fetch_osm_for_bbox(
        bbox=bbox,
        output_path=output_path,
        cache_directory=cache_path.parent,
        refresh_osm=False,
    )

    assert source.kind == build_network.OsmSourceKind.CACHE
    assert source.cache_path == cache_path
    assert output_path.read_text(encoding='utf-8') == '<osm version="0.6"></osm>'


def test_prune_recipe_deletes_junction_and_incident_edges(tmp_path: Path) -> None:
    node_path, edge_path = _write_plain_network(tmp_path)
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=('n1',),
        delete_edges=(),
        keep_junctions=(),
        notes=(build_network.PruneNote(target_id='n1', text='remove side area'),),
    )

    report = build_network._apply_prune_recipe_to_plain(
        nod_file=node_path,
        edg_file=edge_path,
        prune_recipe=prune_recipe,
    )

    node_ids = _plain_ids(node_path, 'node')
    edge_ids = _plain_ids(edge_path, 'edge')
    assert report.deleted_junction_count == 1
    assert report.deleted_edge_count == 2
    assert report.missing_junctions == ()
    assert report.missing_edges == ()
    assert node_ids == {'n0', 'n2'}
    assert edge_ids == {'e02'}


def test_prune_recipe_deletes_explicit_edge_only(tmp_path: Path) -> None:
    node_path, edge_path = _write_plain_network(tmp_path)
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=(),
        delete_edges=('e02',),
        keep_junctions=('n0',),
        notes=(),
    )

    report = build_network._apply_prune_recipe_to_plain(
        nod_file=node_path,
        edg_file=edge_path,
        prune_recipe=prune_recipe,
    )

    assert report.deleted_junction_count == 0
    assert report.deleted_edge_count == 1
    assert report.missing_junctions == ()
    assert report.missing_edges == ()
    assert _plain_ids(node_path, 'node') == {'n0', 'n1', 'n2'}
    assert _plain_ids(edge_path, 'edge') == {'e01', 'e12'}


def test_prune_recipe_deletes_bidirectional_road_segment(tmp_path: Path) -> None:
    node_path, edge_path = _write_plain_bidirectional_network(tmp_path)
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=(),
        delete_edges=('e01',),
        keep_junctions=(),
        notes=(),
    )

    report = build_network._apply_prune_recipe_to_plain(
        nod_file=node_path,
        edg_file=edge_path,
        prune_recipe=prune_recipe,
    )

    assert report.deleted_junction_count == 1
    assert report.deleted_edge_count == 2
    assert report.missing_junctions == ()
    assert report.missing_edges == ()
    assert _plain_ids(node_path, 'node') == {'n1', 'n2'}
    assert _plain_ids(edge_path, 'edge') == {'e12', 'e21'}


def test_prune_recipe_rejects_kept_junction_deletion(tmp_path: Path) -> None:
    node_path, edge_path = _write_plain_network(tmp_path)
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=('n1',),
        delete_edges=(),
        keep_junctions=('n1',),
        notes=(),
    )

    try:
        build_network._apply_prune_recipe_to_plain(
            nod_file=node_path,
            edg_file=edge_path,
            prune_recipe=prune_recipe,
        )
    except ValueError as exc:
        assert 'deletes kept junctions' in str(exc)
    else:
        raise AssertionError('Expected kept junction deletion to fail')


def test_prune_recipe_skips_missing_plain_ids(tmp_path: Path) -> None:
    node_path, edge_path = _write_plain_network(tmp_path)
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=('n1', 'missing_node'),
        delete_edges=('missing_edge',),
        keep_junctions=(),
        notes=(),
    )

    report = build_network._apply_prune_recipe_to_plain(
        nod_file=node_path,
        edg_file=edge_path,
        prune_recipe=prune_recipe,
    )

    assert report.deleted_junction_count == 1
    assert report.deleted_edge_count == 2
    assert report.missing_junctions == ('missing_node',)
    assert report.missing_edges == ('missing_edge',)


def test_load_prune_recipe_from_json(tmp_path: Path) -> None:
    recipe_path = tmp_path / 'city.prune.json'
    recipe_path.write_text(
        """
{
  "delete_junctions": ["n1"],
  "delete_edges": ["e02"],
  "keep_junctions": ["n0"],
  "notes": [{"target_id": "n1", "text": "not needed"}]
}
""",
        encoding='utf-8',
    )

    prune_recipe = build_network._load_prune_recipe(recipe_path)

    assert prune_recipe.delete_junctions == ('n1',)
    assert prune_recipe.delete_edges == ('e02',)
    assert prune_recipe.keep_junctions == ('n0',)
    assert prune_recipe.notes[0].target_id == 'n1'


def test_prune_visualization_uses_normal_edges_only() -> None:
    prune_recipe = build_network.PruneRecipe(
        delete_junctions=('N1_1',),
        delete_edges=('N0_0_to_N0_1',),
        keep_junctions=('N0_0',),
        notes=(),
    )

    visualization = visualize_network_prune.build_prune_visualization(
        net_path=GRID_NET,
        prune_recipe=prune_recipe,
    )

    assert visualization.existing_prune_recipe == prune_recipe
    assert visualization.edges
    assert all(not edge.edge_id.startswith(':') for edge in visualization.edges)
    assert {edge.edge_id for edge in visualization.edges} >= {'N0_0_to_N0_1'}
    assert {junction.junction_id for junction in visualization.junctions} >= {'N0_0', 'N1_1'}


def test_generate_prune_editor_writes_html(tmp_path: Path) -> None:
    output_path = tmp_path / 'prune.html'

    written_path = visualize_network_prune.generate_prune_editor(
        net_path=GRID_NET,
        output_path=output_path,
        prune_path=None,
    )
    content = written_path.read_text(encoding='utf-8')

    assert written_path == output_path
    assert 'Network Prune' in content
    assert 'N0_0_to_N0_1' in content
    assert 'delete_junctions' in content
    assert 'id="undo-last"' in content
    assert 'id="finish-pruning"' in content
    assert 'id="mode-keep"' not in content


def test_generate_prune_editor_preloads_existing_save_recipe(tmp_path: Path) -> None:
    output_path = tmp_path / 'prune.html'
    save_prune_path = tmp_path / 'grid.prune.json'
    save_prune_path.write_text(
        """
{
  "delete_junctions": ["N0_0"],
  "delete_edges": ["N0_0_to_N0_1"],
  "keep_junctions": [],
  "notes": []
}
""",
        encoding='utf-8',
    )

    visualize_network_prune.generate_prune_editor(
        net_path=GRID_NET,
        output_path=output_path,
        prune_path=None,
        save_prune_path=save_prune_path,
    )
    content = output_path.read_text(encoding='utf-8')

    assert '"delete_junctions":["N0_0"]' in content
    assert '"delete_edges":["N0_0_to_N0_1"]' in content


def test_prune_editor_config_reports_missing_osm_for_grid_fixture() -> None:
    save_path = visualize_network_prune._default_prune_path(GRID_NET)
    editor_config = visualize_network_prune.build_editor_config(
        net_path=GRID_NET,
        save_prune_path=save_path,
        save_url='/prune',
        rebuild_url='/rebuild',
    )

    assert save_path.name == 'grid.prune.json'
    assert editor_config.save_url == '/prune'
    assert editor_config.rebuild_url == '/rebuild'
    assert 'Cannot infer rebuild command' in editor_config.rebuild_command


def test_prune_editor_config_builds_rebuild_command_with_sibling_osm(tmp_path: Path) -> None:
    net_path = tmp_path / 'city.net.xml'
    prune_path = tmp_path / 'city.prune.json'
    (tmp_path / 'city.osm').write_text('<osm version="0.6"></osm>', encoding='utf-8')
    net_path.write_text('<net></net>', encoding='utf-8')

    editor_config = visualize_network_prune.build_editor_config(
        net_path=net_path,
        save_prune_path=prune_path,
        save_url='/prune',
        rebuild_url='/rebuild',
    )

    assert editor_config.rebuild_url == '/rebuild'
    assert '--osm' in editor_config.rebuild_command
    assert '--prune' in editor_config.rebuild_command


def test_inspection_suspicion_checks_accept_current_grid_graph() -> None:
    runtime = MovementControlRuntime(cfg_path=GRID_CFG, gui=False, seed=42)
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=GRID_NET)
        network = sumolib.net.readNet(str(GRID_NET), withConnections=True)
        connectivity = _connectivity_report(graph)
        lane_group_warnings = _suspicious_lane_groups(graph=graph, network=network)
        movement_warnings = _suspicious_movements(graph)
    finally:
        runtime.close()

    assert connectivity.component_count == 1
    assert connectivity.unused_lane_groups == ()
    assert lane_group_warnings == ()
    assert movement_warnings == ()


def _write_plain_network(directory: Path) -> tuple[Path, Path]:
    node_path = directory / 'plain.nod.xml'
    edge_path = directory / 'plain.edg.xml'
    node_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="n0" x="0" y="0" type="priority"/>
    <node id="n1" x="1" y="0" type="traffic_light"/>
    <node id="n2" x="2" y="0" type="priority"/>
</nodes>
""",
        encoding='utf-8',
    )
    edge_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="e01" from="n0" to="n1"/>
    <edge id="e12" from="n1" to="n2"/>
    <edge id="e02" from="n0" to="n2"/>
</edges>
""",
        encoding='utf-8',
    )
    return node_path, edge_path


def _write_plain_bidirectional_network(directory: Path) -> tuple[Path, Path]:
    node_path = directory / 'plain.nod.xml'
    edge_path = directory / 'plain.edg.xml'
    node_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="n0" x="0" y="0" type="priority"/>
    <node id="n1" x="1" y="0" type="traffic_light"/>
    <node id="n2" x="2" y="0" type="priority"/>
</nodes>
""",
        encoding='utf-8',
    )
    edge_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="e01" from="n0" to="n1"/>
    <edge id="e10" from="n1" to="n0"/>
    <edge id="e12" from="n1" to="n2"/>
    <edge id="e21" from="n2" to="n1"/>
</edges>
""",
        encoding='utf-8',
    )
    return node_path, edge_path


def _plain_ids(path: Path, tag: str) -> set[str]:
    root = ET.parse(path).getroot()
    return {str(element.get('id')) for element in root.findall(tag)}
