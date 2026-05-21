import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PhaseSchemaTest(unittest.TestCase):
    def test_eight_phase_schema_dimensions_and_single_slot_phases(self) -> None:
        from src.environment.phase_schema import (
            ELAPSED_FEATURE_INDEX,
            NUM_PHASES,
            OBS_DIM,
            PHASE_FEATURE_START,
            SLOT_DIR_PHASES,
        )

        self.assertEqual(NUM_PHASES, 8)
        self.assertEqual(PHASE_FEATURE_START, 36)
        self.assertEqual(ELAPSED_FEATURE_INDEX, 44)
        self.assertEqual(OBS_DIM, 45)
        self.assertIn((4, "G"), SLOT_DIR_PHASES[(0, "l")])
        self.assertIn((5, "G"), SLOT_DIR_PHASES[(1, "s")])
        self.assertIn((6, "G"), SLOT_DIR_PHASES[(2, "r")])
        self.assertIn((7, "G"), SLOT_DIR_PHASES[(3, "l")])


if __name__ == "__main__":
    unittest.main()
