from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.transition import (
    SignalTransitionController,
    transition_yellow_state,
    yellow_from_green_state,
)


def test_yellow_from_green_state_only_yellows_active_green_links() -> None:
    assert yellow_from_green_state("Ggrr") == "yyrr"
    assert yellow_from_green_state("rrrr") == "rrrr"


def test_transition_yellow_state_keeps_links_green_when_target_stays_green() -> None:
    assert transition_yellow_state("GGrr", "GGGr") == "GGrr"
    assert transition_yellow_state("GGGr", "rGGr") == "yGGr"


def test_transition_controller_holds_green_until_target_changes() -> None:
    controller = SignalTransitionController(yellow_duration=3)

    controller.set_targets({"J0": "Grr"})

    assert controller.current_states() == {"J0": "Grr"}
    controller.advance()
    assert controller.current_states() == {"J0": "Grr"}


def test_transition_controller_outputs_yellow_before_new_green() -> None:
    controller = SignalTransitionController(yellow_duration=3)
    controller.set_targets({"J0": "Grr"})

    controller.set_targets({"J0": "rGr"})

    assert controller.current_states() == {"J0": "yrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "yrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "yrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "rGr"}


def test_transition_controller_keeps_persistent_green_links_active() -> None:
    controller = SignalTransitionController(yellow_duration=3)
    controller.set_targets({"J0": "GGrr"})

    controller.set_targets({"J0": "GGGr"})

    assert controller.current_states() == {"J0": "GGrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "GGrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "GGrr"}
    controller.advance()
    assert controller.current_states() == {"J0": "GGGr"}


def test_transition_controller_switches_directly_when_yellow_duration_is_zero() -> None:
    controller = SignalTransitionController(yellow_duration=0)
    controller.set_targets({"J0": "Grr"})

    controller.set_targets({"J0": "rGr"})

    assert controller.current_states() == {"J0": "rGr"}
