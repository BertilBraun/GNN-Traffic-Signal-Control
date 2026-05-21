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

    @unittest.skipUnless(os.environ.get("SUMO_HOME"), "SUMO_HOME is required for sumolib")
    def test_network_builder_builds_eight_phase_strings(self) -> None:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import sumolib

        from scripts.build_network import _build_phase_strings

        net = sumolib.net.readNet(str(ROOT / "configs" / "grid_4x4" / "grid.net.xml"), withConnections=True)
        for node in net.getNodes():
            if node.getType() != "traffic_light" or len(node.getIncoming()) not in (3, 4):
                continue
            result = _build_phase_strings(node)
            if result is None:
                continue
            greens, yellows, all_red = result
            self.assertEqual(len(greens), 8)
            self.assertEqual(len(yellows), 8)
            self.assertTrue(all(set(g) <= {"r", "g", "G"} for g in greens))
            self.assertTrue(set(all_red) <= {"r"})
            self.assertTrue(any("G" in greens[phase] for phase in (4, 5, 6, 7)))
            break
        else:
            self.fail("No supported traffic-light node found")

    @unittest.skipUnless(os.environ.get("SUMO_HOME"), "SUMO_HOME is required for sumolib")
    def test_graph_builder_uses_observation_dimension(self) -> None:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import sumolib

        from src.environment.junction_info import build_junction_info
        from src.environment.phase_schema import OBS_DIM
        from src.utils.graph_builder import GraphBuilder

        net = sumolib.net.readNet(str(ROOT / "configs" / "grid_4x4" / "grid.net.xml"), withConnections=True)
        junction_infos = {}
        for node in net.getNodes():
            if node.getType() == "traffic_light":
                info = build_junction_info(node)
                if info is not None:
                    junction_infos[node.getID()] = info

        builder = GraphBuilder(net, junction_infos)
        self.assertEqual(builder.normalizer.dim, OBS_DIM)


if __name__ == "__main__":
    unittest.main()
