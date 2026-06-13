from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import (
    graph_max_pressure_scores_from_features,
    resolve_sumocfg_net_path,
)
from src.movement.extraction import extract_traffic_light_program
from src.movement.features import (
    LaneGroupGeometry,
    MovementControlState,
    VehicleSnapshot,
    build_feature_frame,
    vehicle_snapshots_from_api,
)
from src.movement.graph import build_movement_graph


def test_resolve_sumocfg_net_path_uses_config_directory(tmp_path: Path) -> None:
    cfg = tmp_path / 'nested' / 'grid.sumocfg'
    cfg.parent.mkdir()
    cfg.write_text(
        """<configuration><input><net-file value="grid.net.xml"/></input></configuration>""",
        encoding='utf-8',
    )

    assert resolve_sumocfg_net_path(cfg) == cfg.parent / 'grid.net.xml'


class FakeVehicleApi:
    def getIDList(self) -> list[str]:
        return ['v0', 'v1']

    def getLaneID(self, vehicle_id: str) -> str:
        return {'v0': 'north_in_0', 'v1': 'east_in_0'}[vehicle_id]

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]:
        return {
            'v0': ('north_in', 'south_out'),
            'v1': ('east_in',),
        }[vehicle_id]

    def getRouteIndex(self, vehicle_id: str) -> int:
        return {'v0': 0, 'v1': 0}[vehicle_id]

    def getLanePosition(self, vehicle_id: str) -> float:
        return {'v0': 175.0, 'v1': 50.0}[vehicle_id]

    def getSpeed(self, vehicle_id: str) -> float:
        return {'v0': 12.0, 'v1': 0.0}[vehicle_id]

    def getLength(self, vehicle_id: str) -> float:
        return 5.0


def test_vehicle_snapshots_use_next_route_edge_for_oracle_demand() -> None:
    snapshots = vehicle_snapshots_from_api(FakeVehicleApi())

    assert snapshots[0].vehicle_id == 'v0'
    assert snapshots[0].lane_id == 'north_in_0'
    assert snapshots[0].next_edge_id == 'south_out'
    assert snapshots[0].lane_position_m == 175.0
    assert snapshots[0].speed_mps == 12.0
    assert snapshots[0].length_m == 5.0
    assert snapshots[1].next_edge_id is None


def test_graph_max_pressure_scores_use_visible_lane_group_features() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['Gr', 'rG'],
        controlled_links=[
            [('north_in_0', 'south_out_0', None), ('north_in_1', 'south_out_1', None)],
            [('east_in_0', 'west_out_0', None)],
        ],
    )
    graph = build_movement_graph({'J0': program})
    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            'north_in': ('north_in_0', 'north_in_1'),
            'south_out': ('south_out_0', 'south_out_1'),
            'east_in': ('east_in_0',),
            'west_out': ('west_out_0',),
        },
        lane_geometries={
            'north_in': LaneGroupGeometry(length_m=200.0, num_lanes=2),
            'south_out': LaneGroupGeometry(length_m=200.0, num_lanes=2),
            'east_in': LaneGroupGeometry(length_m=200.0, num_lanes=1),
            'west_out': LaneGroupGeometry(length_m=200.0, num_lanes=1),
        },
        control_state=MovementControlState(),
        vehicles=(
            *_halted_vehicles('north', 'north_in', 9),
            *_halted_vehicles('south', 'south_out', 3),
            *_halted_vehicles('east', 'east_in', 6),
            *_halted_vehicles('west', 'west_out', 2),
        ),
    )

    assert graph_max_pressure_scores_from_features(graph, frame) == (6.0, 4.0)


def _halted_vehicles(prefix: str, edge_id: str, count: int) -> tuple[VehicleSnapshot, ...]:
    return tuple(
        VehicleSnapshot(
            vehicle_id=f'{prefix}_{index}',
            lane_id=f'{edge_id}_0',
            next_edge_id=None,
            lane_position_m=150.0 + index,
            speed_mps=0.0,
            length_m=5.0,
        )
        for index in range(count)
    )
