from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.min_green import MinGreenController


def test_min_green_accepts_initial_targets() -> None:
    controller = MinGreenController(min_green_steps=2)

    accepted = controller.filter_targets({"J0": "Grr"})

    assert accepted == {"J0": "Grr"}


def test_min_green_rejects_switch_before_minimum_decision_steps() -> None:
    controller = MinGreenController(min_green_steps=2)

    assert controller.filter_targets({"J0": "Grr"}) == {"J0": "Grr"}
    assert controller.filter_targets({"J0": "rGr"}) == {"J0": "Grr"}


def test_min_green_allows_switch_after_minimum_decision_steps() -> None:
    controller = MinGreenController(min_green_steps=2)

    assert controller.filter_targets({"J0": "Grr"}) == {"J0": "Grr"}
    assert controller.filter_targets({"J0": "Grr"}) == {"J0": "Grr"}
    assert controller.filter_targets({"J0": "rGr"}) == {"J0": "rGr"}


def test_min_green_tracks_each_traffic_light_independently() -> None:
    controller = MinGreenController(min_green_steps=2)

    assert controller.filter_targets({"J0": "Grr", "J1": "rrG"}) == {
        "J0": "Grr",
        "J1": "rrG",
    }
    assert controller.filter_targets({"J0": "rGr", "J1": "rrG"}) == {
        "J0": "Grr",
        "J1": "rrG",
    }
    assert controller.filter_targets({"J0": "rGr", "J1": "Grr"}) == {
        "J0": "rGr",
        "J1": "Grr",
    }


def test_min_green_can_be_disabled() -> None:
    controller = MinGreenController(min_green_steps=0)

    assert controller.filter_targets({"J0": "Grr"}) == {"J0": "Grr"}
    assert controller.filter_targets({"J0": "rGr"}) == {"J0": "rGr"}
