"""Shared phase/action schema for canonical traffic-light control."""
from __future__ import annotations

NUM_APPROACH_SLOTS = 4
MOVEMENT_FEATURES_PER_SLOT = 9
MOVEMENT_FEATURE_DIM = NUM_APPROACH_SLOTS * MOVEMENT_FEATURES_PER_SLOT
NUM_PHASES = 8
PHASE_FEATURE_START = MOVEMENT_FEATURE_DIM
ELAPSED_FEATURE_INDEX = PHASE_FEATURE_START + NUM_PHASES
OBS_DIM = ELAPSED_FEATURE_INDEX + 1

MOVEMENT_ORDER = ("l", "s", "r")

# Signal char: "G" = protected green, "g" = permissive green.
SLOT_DIR_PHASES: dict[tuple[int, str], list[tuple[int, str]]] = {
    (0, "s"): [(0, "G"), (4, "G")],
    (0, "r"): [(0, "G"), (3, "g"), (4, "G")],
    (0, "l"): [(1, "G"), (4, "G")],
    (1, "s"): [(2, "G"), (5, "G")],
    (1, "r"): [(2, "G"), (1, "g"), (5, "G")],
    (1, "l"): [(3, "G"), (5, "G")],
    (2, "s"): [(0, "G"), (6, "G")],
    (2, "r"): [(0, "G"), (3, "g"), (6, "G")],
    (2, "l"): [(1, "G"), (6, "G")],
    (3, "s"): [(2, "G"), (7, "G")],
    (3, "r"): [(2, "G"), (1, "g"), (7, "G")],
    (3, "l"): [(3, "G"), (7, "G")],
}


def phase_indices() -> range:
    """Return all fixed phase indices."""
    return range(NUM_PHASES)
