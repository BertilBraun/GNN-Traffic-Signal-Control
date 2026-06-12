from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import (
    graph_max_pressure_scores_from_features,
    resolve_sumocfg_net_path,
    vehicle_snapshots_from_api,
)
from src.movement.extraction import extract_traffic_light_program
from src.movement.features import (
    LaneGroupGeometry,
    MovementControlState,
    build_feature_frame,
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


def test_vehicle_snapshots_use_next_route_edge_for_oracle_demand() -> None:
    snapshots = vehicle_snapshots_from_api(FakeVehicleApi())

    assert snapshots[0].vehicle_id == 'v0'
    assert snapshots[0].lane_id == 'north_in_0'
    assert snapshots[0].next_lane_id == 'south_out'
    assert snapshots[1].next_lane_id is None


class FakeLaneApi:
    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        return 0

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return {
            'north_in_0': 5,
            'north_in_1': 4,
            'south_out_0': 2,
            'south_out_1': 1,
            'east_in_0': 6,
            'west_out_0': 2,
        }.get(str(lane_id), 0)

    def getLastStepLength(self, lane_id: str) -> float:
        return 0.0

    def getLastStepOccupancy(self, lane_id: str) -> float:
        return 0.0

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        return 0.0


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
        lane_api=FakeLaneApi(),
        control_state=MovementControlState(),
        vehicles=(),
    )

    assert graph_max_pressure_scores_from_features(graph, frame) == (6.0, 4.0)
