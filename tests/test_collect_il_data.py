from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path, vehicle_snapshots_from_api


def test_resolve_sumocfg_net_path_uses_config_directory(tmp_path: Path) -> None:
    cfg = tmp_path / "nested" / "grid.sumocfg"
    cfg.parent.mkdir()
    cfg.write_text(
        """<configuration><input><net-file value="grid.net.xml"/></input></configuration>""",
        encoding="utf-8",
    )

    assert resolve_sumocfg_net_path(cfg) == cfg.parent / "grid.net.xml"


class FakeVehicleApi:
    def getIDList(self) -> list[str]:
        return ["v0", "v1"]

    def getLaneID(self, vehicle_id: str) -> str:
        return {"v0": "north_in_0", "v1": "east_in_0"}[vehicle_id]

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]:
        return {
            "v0": ("north_in", "south_out"),
            "v1": ("east_in",),
        }[vehicle_id]

    def getRouteIndex(self, vehicle_id: str) -> int:
        return {"v0": 0, "v1": 0}[vehicle_id]


def test_vehicle_snapshots_use_next_route_edge_for_oracle_demand() -> None:
    snapshots = vehicle_snapshots_from_api(FakeVehicleApi())

    assert snapshots[0].vehicle_id == "v0"
    assert snapshots[0].lane_id == "north_in_0"
    assert snapshots[0].next_lane_id == "south_out"
    assert snapshots[1].next_lane_id is None
