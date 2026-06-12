from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.extraction import extract_traffic_light_program
from src.movement.features import (
    LaneGroupGeometry,
    MovementControlState,
    VehicleSnapshot,
    build_feature_frame,
    detector_capacity,
    detector_length,
)
from src.movement.graph import build_movement_graph


class FakeLaneApi:
    def __init__(self) -> None:
        self.vehicle_count = {"north_in_0": 6, "north_in_1": 4, "south_out_0": 2}
        self.halting = {"north_in_0": 3, "north_in_1": 1, "south_out_0": 1}
        self.queue_m = {"north_in_0": 50.0, "north_in_1": 30.0, "south_out_0": 10.0}
        self.occupancy = {"north_in_0": 80.0, "north_in_1": 40.0, "south_out_0": 20.0}
        self.speed = {"north_in_0": 4.0, "north_in_1": 8.0, "south_out_0": 10.0}

    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        return self.vehicle_count.get(lane_id, 0)

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return self.halting.get(lane_id, 0)

    def getLastStepLength(self, lane_id: str) -> float:
        return self.queue_m.get(lane_id, 0.0)

    def getLastStepOccupancy(self, lane_id: str) -> float:
        return self.occupancy.get(lane_id, 0.0)

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        return self.speed.get(lane_id, 0.0)


def test_detector_helpers_use_lane_group_length_cap() -> None:
    assert detector_length(lane_group_length=120.0) == 120.0
    assert detector_length(lane_group_length=350.0) == 200.0
    assert detector_capacity(detector_length_m=100.0, num_lanes=2) == 25.0


def test_feature_frame_extracts_lane_group_rows_in_graph_order() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["G"],
        controlled_links=[
            [("north_in_0", "south_out_0", None), ("north_in_1", "south_out_1", None)],
        ],
    )
    graph = build_movement_graph({"J0": program})

    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            "north_in": ("north_in_0", "north_in_1"),
            "south_out": ("south_out_0", "south_out_1"),
        },
        lane_geometries={
            "north_in": LaneGroupGeometry(length_m=150.0, num_lanes=2, speed_limit_mps=15.0),
            "south_out": LaneGroupGeometry(length_m=400.0, num_lanes=2, speed_limit_mps=20.0),
        },
        lane_api=FakeLaneApi(),
        control_state=MovementControlState(
            current_enabled_movement_ids=(0,),
            previous_enabled_movement_ids=(),
            time_since_enabled_s={0: 30.0},
        ),
        vehicles=(),
    )

    north_row = frame.lane_group_rows[graph.lane_group_id_by_edge["north_in"]]
    assert north_row.static.length_m == 150.0
    assert north_row.static.detector_length_m == 150.0
    assert north_row.static.num_lanes == 2
    assert north_row.dynamic.vehicle_count_detector == 10.0
    assert north_row.dynamic.halting_count_detector == 4.0
    assert north_row.dynamic.queue_length_m_detector == 80.0
    assert round(north_row.dynamic.vehicle_count_norm_detector, 3) == 0.267
    assert round(north_row.dynamic.queue_length_norm_detector, 3) == 0.533
    assert north_row.dynamic.detector_saturation == 0.0


def test_feature_frame_extracts_oracle_movement_demand_by_graph_id() -> None:
    program = extract_traffic_light_program(
        tls_id="J0",
        phase_states=["G"],
        controlled_links=[
            [("north_in_0", "south_out_0", None), ("north_in_1", "south_out_1", None)],
        ],
    )
    graph = build_movement_graph({"J0": program})

    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            "north_in": ("north_in_0", "north_in_1"),
            "south_out": ("south_out_0", "south_out_1"),
        },
        lane_geometries={
            "north_in": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            "south_out": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
        },
        lane_api=FakeLaneApi(),
        control_state=MovementControlState(
            current_enabled_movement_ids=(0,),
            previous_enabled_movement_ids=(0,),
            time_since_enabled_s={0: 45.0},
        ),
        vehicles=(
            VehicleSnapshot(vehicle_id="v0", lane_id="north_in_0", next_lane_id="south_out_0"),
            VehicleSnapshot(vehicle_id="v1", lane_id="north_in_1", next_lane_id="south_out_1"),
            VehicleSnapshot(vehicle_id="v2", lane_id="north_in_0", next_lane_id="other_0"),
        ),
    )

    movement_row = frame.movement_rows[0]
    assert movement_row.static.num_underlying_controlled_links == 2
    assert movement_row.dynamic.oracle_movement_demand == 2.0
    assert movement_row.dynamic.oracle_movement_demand_norm == 0.04
    assert movement_row.dynamic.is_currently_enabled == 1.0
    assert movement_row.dynamic.was_enabled_last_decision == 1.0
    assert movement_row.dynamic.time_since_enabled_s == 45.0
