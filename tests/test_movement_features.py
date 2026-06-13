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


def test_detector_helpers_use_lane_group_length_cap() -> None:
    assert detector_length(lane_group_length=120.0) == 120.0
    assert detector_length(lane_group_length=350.0) == 200.0
    assert detector_capacity(detector_length_m=100.0, num_lanes=2) == 25.0


def test_feature_frame_extracts_lane_group_rows_in_graph_order() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['G'],
        controlled_links=[
            [('north_in_0', 'south_out_0', None), ('north_in_1', 'south_out_1', None)],
        ],
    )
    graph = build_movement_graph({'J0': program})

    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            'north_in': ('north_in_0', 'north_in_1'),
            'south_out': ('south_out_0', 'south_out_1'),
        },
        lane_geometries={
            'north_in': LaneGroupGeometry(length_m=150.0, num_lanes=2, speed_limit_mps=15.0),
            'south_out': LaneGroupGeometry(length_m=400.0, num_lanes=2, speed_limit_mps=20.0),
        },
        control_state=MovementControlState(
            current_enabled_movement_ids=(0,),
            previous_enabled_movement_ids=(),
            time_since_enabled_s={0: 30.0},
        ),
        vehicles=(
            VehicleSnapshot('moving_0', 'north_in_0', 'south_out', 130.0, 12.0, 5.0),
            VehicleSnapshot('moving_1', 'north_in_1', 'south_out', 140.0, 8.0, 5.0),
            VehicleSnapshot('halted_0', 'north_in_0', 'south_out', 145.0, 0.0, 5.0),
            VehicleSnapshot('upstream', 'south_out_0', None, 50.0, 10.0, 5.0),
        ),
    )

    north_row = frame.lane_group_rows[graph.lane_group_id_by_edge['north_in']]
    assert north_row.static.length_m == 150.0
    assert north_row.static.detector_length_m == 150.0
    assert north_row.static.num_lanes == 2
    assert north_row.dynamic.vehicle_count_detector == 3.0
    assert north_row.dynamic.moving_count_detector == 2.0
    assert north_row.dynamic.halting_count_detector == 1.0
    assert north_row.dynamic.queue_length_m_detector == 5.0
    assert round(north_row.dynamic.vehicle_count_norm_detector, 3) == 0.08
    assert round(north_row.dynamic.moving_count_norm_detector, 3) == 0.053
    assert round(north_row.dynamic.queue_length_norm_detector, 3) == 0.033
    assert round(north_row.dynamic.mean_speed_detector, 3) == 6.667
    assert north_row.dynamic.detector_saturation == 0.0


def test_feature_frame_excludes_vehicles_upstream_of_detector_range() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['G'],
        controlled_links=[[('north_in_0', 'south_out_0', None)]],
    )
    graph = build_movement_graph({'J0': program})

    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            'north_in': ('north_in_0',),
            'south_out': ('south_out_0',),
        },
        lane_geometries={
            'north_in': LaneGroupGeometry(length_m=400.0, num_lanes=1),
            'south_out': LaneGroupGeometry(length_m=400.0, num_lanes=1),
        },
        control_state=MovementControlState(),
        vehicles=(
            VehicleSnapshot('upstream', 'north_in_0', 'south_out', 199.0, 13.0, 5.0),
            VehicleSnapshot('approaching', 'north_in_0', 'south_out', 250.0, 13.0, 5.0),
            VehicleSnapshot('queued', 'north_in_0', 'south_out', 390.0, 0.0, 5.0),
        ),
    )

    north_row = frame.lane_group_rows[graph.lane_group_id_by_edge['north_in']]
    assert north_row.dynamic.vehicle_count_detector == 2.0
    assert north_row.dynamic.moving_count_detector == 1.0
    assert north_row.dynamic.halting_count_detector == 1.0


def test_feature_frame_extracts_oracle_movement_demand_by_graph_id() -> None:
    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['G'],
        controlled_links=[
            [('north_in_0', 'south_out_0', None), ('north_in_1', 'south_out_1', None)],
        ],
    )
    graph = build_movement_graph({'J0': program})

    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            'north_in': ('north_in_0', 'north_in_1'),
            'south_out': ('south_out_0', 'south_out_1'),
        },
        lane_geometries={
            'north_in': LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            'south_out': LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
        },
        control_state=MovementControlState(
            current_enabled_movement_ids=(0,),
            previous_enabled_movement_ids=(0,),
            time_since_enabled_s={0: 45.0},
        ),
        vehicles=(
            VehicleSnapshot('v0', 'north_in_0', 'south_out', 150.0, 10.0, 5.0),
            VehicleSnapshot('v1', 'north_in_1', 'south_out', 175.0, 8.0, 5.0),
            VehicleSnapshot('v2', 'north_in_0', 'other', 190.0, 5.0, 5.0),
        ),
    )

    movement_row = frame.movement_rows[0]
    assert movement_row.static.num_underlying_controlled_links == 2
    assert movement_row.dynamic.oracle_movement_demand == 2.0
    assert movement_row.dynamic.oracle_movement_demand_norm == 0.04
    assert movement_row.dynamic.is_currently_enabled == 1.0
    assert movement_row.dynamic.was_enabled_last_decision == 1.0
    assert movement_row.dynamic.time_since_enabled_s == 45.0
