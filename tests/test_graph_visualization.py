from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path
from scripts.visualize_movement_graph import generate_visualization
from src.movement.graph import build_movement_graph
from src.movement.graph_visualization import build_graph_visualization
from src.movement.runtime import MovementControlRuntime


CFG_PATH = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'


def test_graph_visualization_distinguishes_gnn_nodes_from_sumo_context() -> None:
    runtime = MovementControlRuntime(cfg_path=CFG_PATH, gui=False, seed=42)
    try:
        runtime.start()
        graph = build_movement_graph(
            runtime.programs,
            net_path=resolve_sumocfg_net_path(CFG_PATH),
        )
        visualization = build_graph_visualization(
            net_path=resolve_sumocfg_net_path(CFG_PATH),
            graph=graph,
            programs=runtime.programs,
        )
    finally:
        runtime.close()

    junctions = {junction.junction_id: junction for junction in visualization.junctions}
    assert junctions['N0_1'].is_signalized is True
    assert junctions['N0_0'].is_signalized is False
    assert len(visualization.lane_groups) == len(graph.lane_groups)
    assert len(visualization.movements) == len(graph.movements)
    assert len(visualization.lane_groups) == 24
    assert any(len(lane_group.edge_ids) == 2 for lane_group in visualization.lane_groups)
    assert all(movement.traffic_light_id in runtime.programs for movement in visualization.movements)
    assert junctions['N1_1'].selectable_phase_count == 17


def test_generate_visualization_writes_self_contained_html(tmp_path: Path) -> None:
    output_path = generate_visualization(
        cfg_path=CFG_PATH,
        output_path=tmp_path / 'movement_graph.html',
    )

    html = output_path.read_text(encoding='utf-8')

    assert '<svg id="graph"' in html
    assert 'Movement GNN' in html
    assert '"junction_id":"N0_0"' in html
    assert '"traffic_light_id":"N1_1"' in html
