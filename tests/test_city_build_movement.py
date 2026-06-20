from pathlib import Path
import sys

import sumolib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import build_network  # noqa: E402
from scripts.inspect_movement_city import (  # noqa: E402
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
    assert 'vehsPerHour="' in content


def test_city_routes_include_internal_origins_and_destinations() -> None:
    network = sumolib.net.readNet(str(GRID_NET), withConnections=True)

    routes = build_network._city_routes(network, route_count=30)
    source_edges = {route[0] for route in routes}
    destination_edges = {route[-1] for route in routes}

    assert any(not edge_id.split('_to_', 1)[0].startswith('S_') for edge_id in source_edges)
    assert any(not edge_id.split('_to_', 1)[1].startswith('S_') for edge_id in destination_edges)


def test_city_additional_file_matches_empty_grid_additional(tmp_path: Path) -> None:
    add_path = tmp_path / 'city.add.xml'

    build_network._write_additional(add_path)
    content = add_path.read_text(encoding='utf-8')

    assert '<additional/>' in content
    assert 'laneAreaDetector' not in content


def test_inspection_suspicion_checks_accept_current_grid_graph() -> None:
    runtime = MovementControlRuntime(cfg_path=GRID_CFG, gui=False, seed=42)
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=GRID_NET)
        network = sumolib.net.readNet(str(GRID_NET), withConnections=True)
        lane_group_warnings = _suspicious_lane_groups(graph=graph, network=network)
        movement_warnings = _suspicious_movements(graph)
    finally:
        runtime.close()

    assert lane_group_warnings == ()
    assert movement_warnings == ()
