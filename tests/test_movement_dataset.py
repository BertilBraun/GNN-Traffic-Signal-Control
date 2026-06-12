from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import (
    MovementDatasetSample,
    build_dataset_sample,
    load_jsonl_samples,
    replay_teacher_selected_phases,
    save_jsonl_samples,
)
from src.movement.extraction import extract_traffic_light_program
from src.movement.features import (
    LaneGroupGeometry,
    MovementControlState,
    build_feature_frame,
)
from src.movement.graph import build_movement_graph
from src.movement.schema import MovementIndex


class FakeLaneApi:
    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        return 0

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        queues = {
            "north_in_0": 9,
            "north_in_1": 9,
            "east_in_0": 5,
            "south_out_0": 1,
            "south_out_1": 1,
            "west_out_0": 2,
        }
        return queues.get(str(lane_id), 0)

    def getLastStepLength(self, lane_id: str) -> float:
        return 0.0

    def getLastStepOccupancy(self, lane_id: str) -> float:
        return 0.0

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        return 0.0


def _program():
    return extract_traffic_light_program(
        tls_id="J0",
        phase_states=["Gr", "rG"],
        controlled_links=[
            [("north_in_0", "south_out_0", None), ("north_in_1", "south_out_1", None)],
            [("east_in_0", "west_out_0", None)],
        ],
    )


def _sample() -> MovementDatasetSample:
    program = _program()
    graph = build_movement_graph({"J0": program})
    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            "north_in": ("north_in_0", "north_in_1"),
            "south_out": ("south_out_0", "south_out_1"),
            "east_in": ("east_in_0",),
            "west_out": ("west_out_0",),
        },
        lane_geometries={
            "north_in": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            "south_out": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            "east_in": LaneGroupGeometry(length_m=200.0, num_lanes=1, speed_limit_mps=10.0),
            "west_out": LaneGroupGeometry(length_m=200.0, num_lanes=1, speed_limit_mps=10.0),
        },
        lane_api=FakeLaneApi(),
        control_state=MovementControlState(),
        vehicles=(),
    )
    return build_dataset_sample(
        graph=graph,
        feature_frame=frame,
        programs={"J0": program},
        teacher_controlled_scores={
            "J0": {
                MovementIndex(0): 8.0,
                MovementIndex(1): 8.0,
                MovementIndex(2): 3.0,
            }
        },
        metadata={"network_id": "unit", "simulation_time_s": 15},
    )


def test_build_dataset_sample_stores_graph_arrays_and_teacher_replay() -> None:
    sample = _sample()

    assert len(sample.x_lane) == 4
    assert len(sample.x_movement) == 2
    assert sample.teacher_movement_scores == (16.0, 3.0)
    assert sample.teacher_selected_phase_by_tls == {"J0": 0}
    assert replay_teacher_selected_phases(sample) == {"J0": 0}
    assert sample.edge_index_dict["input_lane_to_movement"] == ((1, 0), (0, 1))


def test_build_dataset_sample_can_use_graph_level_teacher_scores() -> None:
    program = _program()
    graph = build_movement_graph({"J0": program})
    frame = build_feature_frame(
        graph=graph,
        lane_ids_by_edge={
            "north_in": ("north_in_0", "north_in_1"),
            "south_out": ("south_out_0", "south_out_1"),
            "east_in": ("east_in_0",),
            "west_out": ("west_out_0",),
        },
        lane_geometries={
            "north_in": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            "south_out": LaneGroupGeometry(length_m=200.0, num_lanes=2, speed_limit_mps=10.0),
            "east_in": LaneGroupGeometry(length_m=200.0, num_lanes=1, speed_limit_mps=10.0),
            "west_out": LaneGroupGeometry(length_m=200.0, num_lanes=1, speed_limit_mps=10.0),
        },
        lane_api=FakeLaneApi(),
        control_state=MovementControlState(),
        vehicles=(),
    )

    sample = build_dataset_sample(
        graph=graph,
        feature_frame=frame,
        programs={"J0": program},
        teacher_controlled_scores={"J0": {}},
        teacher_graph_scores=(8.0, 3.0),
        metadata={},
    )

    assert sample.teacher_movement_scores == (8.0, 3.0)
    assert sample.teacher_selected_phase_by_tls == {"J0": 0}


def test_jsonl_samples_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    sample = _sample()

    save_jsonl_samples(path, [sample])
    loaded = load_jsonl_samples(path)

    assert loaded == [sample]
