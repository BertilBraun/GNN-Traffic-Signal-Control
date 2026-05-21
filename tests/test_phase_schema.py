import unittest
import sys
import os
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

    @unittest.skipUnless(os.environ.get("SUMO_HOME"), "SUMO_HOME is required for sumolib")
    def test_junction_info_builds_eight_phases_and_multi_phase_connections(self) -> None:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import sumolib

        from src.environment.junction_info import build_junction_info

        net = sumolib.net.readNet(str(ROOT / "configs" / "grid_4x4" / "grid.net.xml"), withConnections=True)
        infos = [
            build_junction_info(node)
            for node in net.getNodes()
            if node.getType() == "traffic_light"
        ]
        supported = [info for info in infos if info is not None]
        self.assertTrue(supported)

        for info in supported:
            self.assertEqual(len(info.phase_states), 8)
            self.assertEqual(len(info.yellow_states), 8)
            self.assertEqual(len(info.phase_served_lanes), 8)

        self.assertTrue(
            any(
                any(any(phase >= 4 for phase in phases) for phases in info.conn_to_phases.values())
                for info in supported
            )
        )


if __name__ == "__main__":
    unittest.main()
