"""Generate an interactive HTML visualization of the movement GNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path  # noqa: E402
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_visualization import build_graph_visualization  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402

DEFAULT_CFG = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg'
DEFAULT_OUTPUT = ROOT / 'reports' / 'movement_graph_3x3.html'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize movement and lane-group GNN topology.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', type=Path, default=DEFAULT_CFG, help='SUMO .sumocfg path')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='Output HTML path')
    parser.add_argument('--open', action='store_true', help='Open the generated HTML in the default browser')
    return parser.parse_args()


def generate_visualization(cfg_path: Path, output_path: Path) -> Path:
    """Extract the current graph and write a self-contained HTML viewer."""
    runtime = MovementControlRuntime(cfg_path=cfg_path, gui=False, seed=42)
    try:
        runtime.start()
        net_path = resolve_sumocfg_net_path(cfg_path)
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        visualization = build_graph_visualization(
            net_path=net_path,
            graph=graph,
            programs=runtime.programs,
        )
    finally:
        runtime.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_json = visualization.model_dump_json().replace('</', '<\\/')
    output_path.write_text(_html_document(graph_json), encoding='utf-8')
    return output_path


def _html_document(graph_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Movement GNN Graph</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #101416;
  color: #e8edef;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; overflow: hidden; }}
button, input {{ font: inherit; }}
.app {{ height: 100vh; display: grid; grid-template-columns: 292px 1fr; }}
.sidebar {{
  background: #151a1d;
  border-right: 1px solid #30383c;
  padding: 18px;
  overflow-y: auto;
}}
h1 {{ margin: 0 0 4px; font-size: 18px; letter-spacing: 0; }}
.subtitle {{ color: #9eaaaf; font-size: 12px; margin-bottom: 18px; }}
.section {{ border-top: 1px solid #30383c; padding: 15px 0; }}
.section-title {{ color: #aeb8bc; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; }}
.toggle {{ display: flex; align-items: center; gap: 9px; margin: 9px 0; color: #cbd2d5; font-size: 13px; }}
.toggle input {{ accent-color: #49a9c5; }}
.stats {{ display: grid; grid-template-columns: 1fr auto; gap: 7px 12px; font-size: 12px; }}
.stats span:nth-child(odd) {{ color: #9eaaaf; }}
.legend {{ display: grid; gap: 8px; font-size: 12px; color: #cbd2d5; }}
.legend-row {{ display: flex; gap: 9px; align-items: center; }}
.swatch {{ width: 14px; height: 14px; border-radius: 50%; border: 2px solid transparent; flex: 0 0 auto; }}
.line {{ width: 18px; height: 3px; flex: 0 0 auto; }}
.details {{ font-size: 12px; line-height: 1.55; color: #cbd2d5; min-height: 100px; }}
.details strong {{ display: block; color: #fff; margin-bottom: 5px; }}
.canvas {{ position: relative; background: #0d1113; }}
#graph {{ width: 100%; height: 100%; display: block; cursor: grab; }}
#graph.dragging {{ cursor: grabbing; }}
.toolbar {{
  position: absolute; top: 14px; right: 14px; display: flex; gap: 6px;
  background: #151a1de8; border: 1px solid #30383c; border-radius: 6px; padding: 5px;
}}
.toolbar button {{ width: 32px; height: 30px; border: 0; color: #dce3e6; background: transparent; cursor: pointer; border-radius: 4px; }}
.toolbar button:hover {{ background: #293237; }}
.road {{ stroke: #364044; stroke-width: 2; opacity: .6; }}
.road.non-gnn {{ stroke-dasharray: 5 5; opacity: .38; }}
.message {{ fill: none; opacity: .5; cursor: pointer; pointer-events: stroke; vector-effect: non-scaling-stroke; }}
.message.input {{ stroke: #48b5d6; }}
.message.output {{ stroke: #e7ad52; }}
.message.connector {{ stroke: #8ec07c; }}
.message.selected {{ stroke: #fff; opacity: .95; }}
.message.related {{ opacity: .9; }}
.message.dimmed {{ opacity: .13; }}
.junction-group {{ fill: #1d292d; stroke: #55727b; stroke-width: 1.4; opacity: .75; }}
.junction-group.selected {{ stroke: #fff; stroke-width: 2.4; }}
.junction {{ fill: #0d1113; stroke-width: 2; }}
.junction.signal {{ stroke: #55c28a; }}
.junction.unsignalized {{ stroke: #7d878b; }}
.junction.stub {{ stroke: #556065; stroke-dasharray: 3 3; }}
.lane {{ fill: #89a2ad; stroke: #d4e0e5; stroke-width: 1; }}
.movement {{ fill: #d66767; stroke: #ffd0d0; stroke-width: 1; }}
.node {{ cursor: pointer; }}
.node.selected {{ stroke: #fff; stroke-width: 3; }}
.label {{ fill: #dce4e7; font-size: 9px; paint-order: stroke; stroke: #0d1113; stroke-width: 3px; stroke-linejoin: round; pointer-events: none; }}
.minor-label {{ fill: #9eaaaf; font-size: 7px; paint-order: stroke; stroke: #0d1113; stroke-width: 2px; pointer-events: none; }}
@media (max-width: 800px) {{
  .app {{ grid-template-columns: 230px 1fr; }}
  .sidebar {{ padding: 13px; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>Movement GNN</h1>
    <div class="subtitle" id="network-name"></div>
    <div class="section">
      <div class="section-title">Layers</div>
      <label class="toggle"><input id="show-roads" type="checkbox" checked> SUMO road topology</label>
      <label class="toggle"><input id="show-unsignalized" type="checkbox" checked> Unsignalized junctions</label>
      <label class="toggle"><input id="show-messages" type="checkbox" checked> GNN message edges</label>
      <label class="toggle"><input id="show-labels" type="checkbox" checked> Node labels</label>
    </div>
    <div class="section">
      <div class="section-title">Graph size</div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="section">
      <div class="section-title">Legend</div>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background:#89a2ad"></span> LaneGroup GNN node</div>
        <div class="legend-row"><span class="swatch" style="background:#d66767"></span> Movement GNN node</div>
        <div class="legend-row"><span class="swatch" style="border-color:#55c28a;background:#0d1113"></span> Signalized anchor</div>
        <div class="legend-row"><span class="swatch" style="border-color:#7d878b;background:#0d1113"></span> Unsignalized SUMO junction</div>
        <div class="legend-row"><span class="line" style="background:#48b5d6"></span> Input-lane messages</div>
        <div class="legend-row"><span class="line" style="background:#e7ad52"></span> Output-lane messages</div>
        <div class="legend-row"><span class="line" style="background:#8ec07c"></span> Unsignalized LaneGroup connectors</div>
      </div>
    </div>
    <div class="section">
      <div class="section-title">Selection</div>
      <div class="details" id="details">Select a node or junction.</div>
    </div>
  </aside>
  <main class="canvas">
    <svg id="graph" viewBox="0 0 1200 800" role="img" aria-label="Movement GNN graph"></svg>
    <div class="toolbar">
      <button id="zoom-in" title="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out">−</button>
      <button id="reset-view" title="Reset view">⌂</button>
    </div>
  </main>
</div>
<script id="graph-data" type="application/json">{graph_json}</script>
<script>
const data = JSON.parse(document.getElementById('graph-data').textContent);
const svg = document.getElementById('graph');
const NS = 'http://www.w3.org/2000/svg';
const WIDTH = 1200, HEIGHT = 800, PAD = 70;
let selected = null;
let zoom = 1, panX = 0, panY = 0;
const junctionById = new Map(data.junctions.map(j => [j.junction_id, j]));
const laneById = new Map(data.lane_groups.map(l => [l.lane_group_id, l]));
const movementById = new Map(data.movements.map(m => [m.movement_id, m]));
const outgoingConnectorsByLane = new Map();
const incomingConnectorsByLane = new Map();
const movementEdgesByMovement = new Map();
const messageEdgesById = new Map();
for (const connector of data.lane_connectors) {{
  if (!outgoingConnectorsByLane.has(connector.source_lane_group_id)) outgoingConnectorsByLane.set(connector.source_lane_group_id, []);
  if (!incomingConnectorsByLane.has(connector.target_lane_group_id)) incomingConnectorsByLane.set(connector.target_lane_group_id, []);
  outgoingConnectorsByLane.get(connector.source_lane_group_id).push(connector);
  incomingConnectorsByLane.get(connector.target_lane_group_id).push(connector);
}}
const movementsByTls = new Map();
const componentColors = ['#62b6cb', '#f0b64d', '#8ec07c', '#d3869b', '#b8a0ff', '#e07a5f', '#7bc8a4', '#f2cc8f'];
for (const movement of data.movements) {{
  if (!movementsByTls.has(movement.traffic_light_id)) movementsByTls.set(movement.traffic_light_id, []);
  movementsByTls.get(movement.traffic_light_id).push(movement);
  const inputEdge = {{
    id: `signal-input:${{movement.movement_id}}`,
    type: 'signal-input',
    source_lane_group_id: movement.input_lane_group_id,
    target_movement_id: movement.movement_id,
    traffic_light_id: movement.traffic_light_id
  }};
  const outputEdge = {{
    id: `signal-output:${{movement.movement_id}}`,
    type: 'signal-output',
    source_movement_id: movement.movement_id,
    target_lane_group_id: movement.output_lane_group_id,
    traffic_light_id: movement.traffic_light_id
  }};
  movementEdgesByMovement.set(movement.movement_id, [inputEdge, outputEdge]);
  messageEdgesById.set(inputEdge.id, inputEdge);
  messageEdgesById.set(outputEdge.id, outputEdge);
}}
for (const connector of data.lane_connectors) {{
  const edge = {{
    ...connector,
    id: `connector:${{connector.source_lane_group_id}}:${{connector.target_lane_group_id}}:${{connector.via_junction_id}}`,
    type: 'connector'
  }};
  messageEdgesById.set(edge.id, edge);
}}
const xValues = data.junctions.map(j => j.x);
const yValues = data.junctions.map(j => j.y);
const bounds = {{
  minX: Math.min(...xValues), maxX: Math.max(...xValues),
  minY: Math.min(...yValues), maxY: Math.max(...yValues)
}};
function mapPosition(junction) {{
  const xRange = Math.max(1, bounds.maxX - bounds.minX);
  const yRange = Math.max(1, bounds.maxY - bounds.minY);
  return {{
    x: PAD + (junction.x - bounds.minX) / xRange * (WIDTH - PAD * 2),
    y: HEIGHT - PAD - (junction.y - bounds.minY) / yRange * (HEIGHT - PAD * 2)
  }};
}}
function junctionPosition(id) {{
  return mapPosition(junctionById.get(id));
}}
function lanePosition(lane) {{
  const points = lane.junction_ids.map(junctionPosition);
  const lengths = points.slice(1).map((point, index) => Math.hypot(point.x - points[index].x, point.y - points[index].y));
  const halfLength = lengths.reduce((total, value) => total + value, 0) / 2;
  let travelled = 0;
  for (let index = 0; index < lengths.length; index++) {{
    if (travelled + lengths[index] >= halfLength) {{
      const ratio = lengths[index] > 0 ? (halfLength - travelled) / lengths[index] : 0;
      const from = points[index], to = points[index + 1];
      const dx = to.x - from.x, dy = to.y - from.y;
      const length = Math.max(1, lengths[index]);
      return {{x: from.x + dx * ratio - dy / length * 1, y: from.y + dy * ratio + dx / length * 1}};
    }}
    travelled += lengths[index];
  }}
  return points[0];
}}
function movementPosition(movement) {{
  const center = junctionPosition(movement.traffic_light_id);
  const inputLane = laneById.get(movement.input_lane_group_id);
  const inputPoint = lanePosition(inputLane);
  let dx = inputPoint.x - center.x;
  let dy = inputPoint.y - center.y;
  let length = Math.hypot(dx, dy);
  if (length < 1) {{
    const group = movementsByTls.get(movement.traffic_light_id);
    const index = group.findIndex(item => item.movement_id === movement.movement_id);
    const angle = -Math.PI / 2 + index / Math.max(1, group.length) * Math.PI * 2;
    dx = Math.cos(angle);
    dy = Math.sin(angle);
    length = 1;
  }}
  const unitX = dx / length;
  const unitY = dy / length;
  const sameInput = movementsByTls
    .get(movement.traffic_light_id)
    .filter(item => item.input_lane_group_id === movement.input_lane_group_id);
  const inputIndex = sameInput.findIndex(item => item.movement_id === movement.movement_id);
  const tangentOffset = (inputIndex - (sameInput.length - 1) / 2) * 5;
  const radius = sameInput.length > 3 ? 18 : 15;
  return {{
    x: center.x + unitX * radius - unitY * tangentOffset,
    y: center.y + unitY * radius + unitX * tangentOffset
  }};
}}
function componentColor(componentId) {{
  return componentColors[componentId % componentColors.length];
}}
function element(name, attrs = {{}}) {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}}
function addMarker(defs, id, color, markerSize, reverse = false) {{
  const marker = element('marker', {{id, viewBox: '0 0 10 10', refX: reverse ? 2 : 8, refY: 5, markerWidth: markerSize, markerHeight: markerSize, orient: 'auto-start-reverse'}});
  marker.appendChild(element('path', {{d: reverse ? 'M 10 0 L 0 5 L 10 10 z' : 'M 0 0 L 10 5 L 0 10 z', fill: color}}));
  defs.appendChild(marker);
}}
function details(type, item) {{
  const target = document.getElementById('details');
  if (type === 'lane') {{
    const outgoing = outgoingConnectorsByLane.get(item.lane_group_id) || [];
    const incoming = incomingConnectorsByLane.get(item.lane_group_id) || [];
    const connectorText = [...outgoing.map(c => `out L${{c.target_lane_group_id}} via ${{c.via_junction_id}} ${{c.freeflow_time_s.toFixed(1)}}s ${{c.distance_m.toFixed(1)}}m`), ...incoming.map(c => `in L${{c.source_lane_group_id}} via ${{c.via_junction_id}} ${{c.freeflow_time_s.toFixed(1)}}s ${{c.distance_m.toFixed(1)}}m`)].slice(0, 8).join('<br>') || 'none';
    target.innerHTML = `<strong>LaneGroup L${{item.lane_group_id}}</strong>Component: ${{item.component_id}}<br>SUMO edges: ${{item.edge_ids.join(' -> ')}}<br>Direction: ${{item.from_junction_id}} -> ${{item.to_junction_id}}<br>Length: ${{item.length_m.toFixed(1)}} m<br>Effective lanes: ${{item.effective_lane_count.toFixed(2)}}<br>Effective speed: ${{item.effective_speed_limit_mps.toFixed(2)}} m/s<br>Connector edges:<br>${{connectorText}}`;
  }}
  if (type === 'movement') {{
    const edgeIds = (movementEdgesByMovement.get(item.movement_id) || []).map(edge => edge.id).join('<br>');
    target.innerHTML = `<strong>Movement M${{item.movement_id}}</strong>Component: ${{item.component_id}}<br>Traffic light: ${{item.traffic_light_id}}<br>Input LaneGroup: L${{item.input_lane_group_id}}<br>Output LaneGroup: L${{item.output_lane_group_id}}<br>Controlled links: ${{item.controlled_link_count}}<br>Message edges:<br>${{edgeIds || 'none'}}`;
  }}
  if (type === 'junction') {{
    const count = (movementsByTls.get(item.junction_id) || []).length;
    target.innerHTML = `<strong>Junction ${{item.junction_id}}</strong>SUMO type: ${{item.junction_type}}<br>Signalized: ${{item.is_signalized ? 'yes' : 'no'}}<br>Movement nodes: ${{count}}<br>Selectable phases: ${{item.selectable_phase_count}}<br><br>This anchor is visual context, not a GNN node.`;
  }}
  if (type === 'edge') {{
    if (item.type === 'connector') {{
      target.innerHTML = `<strong>Connector edge</strong>L${{item.source_lane_group_id}} -> L${{item.target_lane_group_id}}<br>Via junction: ${{item.via_junction_id}}<br>Type: ${{item.connector_type}}<br>Freeflow time: ${{item.freeflow_time_s.toFixed(2)}} s<br>Distance context: ${{item.distance_m.toFixed(1)}} m<br>Bottleneck lanes: ${{item.lane_count.toFixed(1)}}`;
    }}
    if (item.type === 'signal-input') {{
      const movement = movementById.get(item.target_movement_id);
      target.innerHTML = `<strong>Signal input edge</strong>L${{item.source_lane_group_id}} -> M${{item.target_movement_id}}<br>Traffic light: ${{item.traffic_light_id}}<br>Movement output: L${{movement.output_lane_group_id}}<br>This is movement demand context into the policy score target.`;
    }}
    if (item.type === 'signal-output') {{
      const movement = movementById.get(item.source_movement_id);
      target.innerHTML = `<strong>Signal output edge</strong>M${{item.source_movement_id}} -> L${{item.target_lane_group_id}}<br>Traffic light: ${{item.traffic_light_id}}<br>Movement input: L${{movement.input_lane_group_id}}<br>This is downstream supply context through a controllable signalized movement.`;
    }}
  }}
}}
function selectNode(type, item, event) {{
  event.stopPropagation();
  selected = `${{type}}:${{type === 'lane' ? item.lane_group_id : type === 'movement' ? item.movement_id : item.junction_id}}`;
  details(type, item);
  render();
}}
function selectEdge(edge, event) {{
  event.stopPropagation();
  selected = `edge:${{edge.id}}`;
  details('edge', edge);
  render();
}}
function curvedPath(a, b, bend) {{
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const dx = b.x - a.x, dy = b.y - a.y, length = Math.max(1, Math.hypot(dx, dy));
  return `M ${{a.x}} ${{a.y}} Q ${{mx - dy / length * bend}} ${{my + dx / length * bend}} ${{b.x}} ${{b.y}}`;
}}
function selectedEdgeId() {{
  return selected && selected.startsWith('edge:') ? selected.slice(5) : null;
}}
function selectedLaneId() {{
  return selected && selected.startsWith('lane:') ? Number(selected.slice(5)) : null;
}}
function selectedMovementId() {{
  return selected && selected.startsWith('movement:') ? Number(selected.slice(9)) : null;
}}
function edgeTouchesLane(edge, laneId) {{
  return edge.source_lane_group_id === laneId || edge.target_lane_group_id === laneId;
}}
function edgeTouchesMovement(edge, movementId) {{
  return edge.source_movement_id === movementId || edge.target_movement_id === movementId;
}}
function messageClass(edge, baseClass) {{
  const activeEdgeId = selectedEdgeId();
  const activeLaneId = selectedLaneId();
  const activeMovementId = selectedMovementId();
  const isSelected = activeEdgeId === edge.id;
  const isRelated =
    (activeLaneId !== null && edgeTouchesLane(edge, activeLaneId)) ||
    (activeMovementId !== null && edgeTouchesMovement(edge, activeMovementId));
  const shouldDim = selected !== null && !isSelected && !isRelated && (activeEdgeId !== null || activeLaneId !== null || activeMovementId !== null);
  return `message ${{baseClass}}${{isSelected ? ' selected' : ''}}${{isRelated ? ' related' : ''}}${{shouldDim ? ' dimmed' : ''}}`;
}}
function render() {{
  svg.replaceChildren();
  const nodeScale = Math.max(.18, Math.min(.85, .85 / zoom));
  const labelScale = Math.max(.16, Math.min(.78, .78 / zoom));
  const edgeStroke = Math.max(5.0, 6.0 / Math.sqrt(zoom));
  const connectorStroke = Math.max(5.0, 6.5 / Math.sqrt(zoom));
  const markerSize = Math.max(3.6, 4.2 / Math.pow(zoom, .12));
  const defs = element('defs');
  addMarker(defs, 'arrow-input', '#48b5d6', markerSize);
  addMarker(defs, 'arrow-output', '#e7ad52', markerSize);
  addMarker(defs, 'arrow-connector', '#8ec07c', markerSize);
  svg.appendChild(defs);
  const viewport = element('g', {{transform: `translate(${{panX}} ${{panY}}) scale(${{zoom}})`}});
  svg.appendChild(viewport);
  const showRoads = document.getElementById('show-roads').checked;
  const showUnsignalized = document.getElementById('show-unsignalized').checked;
  const showMessages = document.getElementById('show-messages').checked;
  const showLabels = document.getElementById('show-labels').checked;
  if (showRoads) {{
    const roadLayer = element('g');
    for (const road of data.roads) {{
      const from = junctionPosition(road.from_junction_id), to = junctionPosition(road.to_junction_id);
      roadLayer.appendChild(element('line', {{x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: `road ${{road.is_lane_group ? '' : 'non-gnn'}}`}}));
    }}
    viewport.appendChild(roadLayer);
  }}
  const groups = element('g');
  for (const [tlsId, movements] of movementsByTls) {{
    const center = junctionPosition(tlsId);
    const radius = (movements.length > 12 ? 33 : 27) * nodeScale;
    const circle = element('circle', {{cx: center.x, cy: center.y, r: radius, class: `junction-group ${{selected === `junction:${{tlsId}}` ? 'selected' : ''}}`}});
    circle.addEventListener('click', event => selectNode('junction', junctionById.get(tlsId), event));
    groups.appendChild(circle);
  }}
  viewport.appendChild(groups);
  if (showMessages) {{
    const edges = element('g');
    for (const connector of data.lane_connectors) {{
      const edge = messageEdgesById.get(`connector:${{connector.source_lane_group_id}}:${{connector.target_lane_group_id}}:${{connector.via_junction_id}}`);
      const sourcePoint = lanePosition(laneById.get(connector.source_lane_group_id));
      const targetPoint = lanePosition(laneById.get(connector.target_lane_group_id));
      const path = element('path', {{d: curvedPath(sourcePoint, targetPoint, 0), class: messageClass(edge, 'connector'), 'marker-end': 'url(#arrow-connector)', style: `stroke-width:${{connectorStroke}}px`}});
      path.addEventListener('click', event => selectEdge(edge, event));
      edges.appendChild(path);
    }}
    for (const movement of data.movements) {{
      const movementPoint = movementPosition(movement);
      const inputPoint = lanePosition(laneById.get(movement.input_lane_group_id));
      const outputPoint = lanePosition(laneById.get(movement.output_lane_group_id));
      const inputEdge = messageEdgesById.get(`signal-input:${{movement.movement_id}}`);
      const outputEdge = messageEdgesById.get(`signal-output:${{movement.movement_id}}`);
      const inputPath = element('path', {{d: curvedPath(inputPoint, movementPoint, 7), class: messageClass(inputEdge, 'input'), 'marker-end': 'url(#arrow-input)', style: `stroke-width:${{edgeStroke}}px`}});
      const outputPath = element('path', {{d: curvedPath(movementPoint, outputPoint, -7), class: messageClass(outputEdge, 'output'), 'marker-end': 'url(#arrow-output)', style: `stroke-width:${{edgeStroke}}px`}});
      inputPath.addEventListener('click', event => selectEdge(inputEdge, event));
      outputPath.addEventListener('click', event => selectEdge(outputEdge, event));
      edges.appendChild(inputPath);
      edges.appendChild(outputPath);
    }}
    viewport.appendChild(edges);
  }}
  const junctionLayer = element('g');
  for (const junction of data.junctions) {{
    if (!junction.is_signalized && !showUnsignalized) continue;
    const point = junctionPosition(junction.junction_id);
    const stub = junction.junction_type === 'dead_end';
    const nodeRadius = (junction.is_signalized ? 6 : 4.5) * nodeScale;
    const node = element('circle', {{cx: point.x, cy: point.y, r: nodeRadius, class: `node junction ${{junction.is_signalized ? 'signal' : stub ? 'stub' : 'unsignalized'}} ${{selected === `junction:${{junction.junction_id}}` ? 'selected' : ''}}`}});
    node.addEventListener('click', event => selectNode('junction', junction, event));
    junctionLayer.appendChild(node);
    if (showLabels) {{
      const label = element('text', {{x: point.x + 7 * nodeScale, y: point.y - 7 * nodeScale, class: 'label', style: `font-size:${{9 * labelScale}}px`}});
      label.textContent = junction.selectable_phase_count > 0
        ? `${{junction.junction_id}} · ${{junction.selectable_phase_count}} phases`
        : junction.junction_id;
      junctionLayer.appendChild(label);
    }}
  }}
  viewport.appendChild(junctionLayer);
  const laneLayer = element('g');
  for (const lane of data.lane_groups) {{
    const point = lanePosition(lane);
    const markerSize = 7 * nodeScale;
    const node = element('rect', {{x: point.x - markerSize / 2, y: point.y - markerSize / 2, width: markerSize, height: markerSize, transform: `rotate(45 ${{point.x}} ${{point.y}})`, class: `node lane ${{selected === `lane:${{lane.lane_group_id}}` ? 'selected' : ''}}`, style: `fill:${{componentColor(lane.component_id)}}`}});
    node.addEventListener('click', event => selectNode('lane', lane, event));
    laneLayer.appendChild(node);
    if (showLabels) {{
      const label = element('text', {{x: point.x + 5 * nodeScale, y: point.y + 2 * nodeScale, class: 'minor-label', style: `font-size:${{7 * labelScale}}px`}});
      label.textContent = `L${{lane.lane_group_id}}`;
      laneLayer.appendChild(label);
    }}
  }}
  viewport.appendChild(laneLayer);
  const movementLayer = element('g');
  for (const movement of data.movements) {{
    const point = movementPosition(movement);
    const node = element('circle', {{cx: point.x, cy: point.y, r: 3.4 * nodeScale, class: `node movement ${{selected === `movement:${{movement.movement_id}}` ? 'selected' : ''}}`, style: `fill:${{componentColor(movement.component_id)}}`}});
    node.addEventListener('click', event => selectNode('movement', movement, event));
    movementLayer.appendChild(node);
    if (showLabels && movementsByTls.get(movement.traffic_light_id).length <= 12) {{
      const label = element('text', {{x: point.x + 4 * nodeScale, y: point.y + 2 * nodeScale, class: 'minor-label', style: `font-size:${{7 * labelScale}}px`}});
      label.textContent = `M${{movement.movement_id}}`;
      movementLayer.appendChild(label);
    }}
  }}
  viewport.appendChild(movementLayer);
}}
let panStart = null;
svg.addEventListener('pointerdown', event => {{
  if (event.target === svg) panStart = {{x: event.clientX, y: event.clientY, panX, panY}};
}});
svg.addEventListener('pointermove', event => {{
  if (!panStart) return;
  panX = panStart.panX + event.clientX - panStart.x;
  panY = panStart.panY + event.clientY - panStart.y;
  render();
}});
svg.addEventListener('pointerup', () => panStart = null);
svg.addEventListener('wheel', event => {{
  event.preventDefault();
  const rect = svg.getBoundingClientRect();
  const cursorX = (event.clientX - rect.left) / rect.width * WIDTH;
  const cursorY = (event.clientY - rect.top) / rect.height * HEIGHT;
  const graphX = (cursorX - panX) / zoom;
  const graphY = (cursorY - panY) / zoom;
  zoom = Math.max(.35, Math.min(10, zoom * (event.deltaY < 0 ? 1.18 : .85)));
  panX = cursorX - graphX * zoom;
  panY = cursorY - graphY * zoom;
  render();
}}, {{passive: false}});
for (const id of ['show-roads', 'show-unsignalized', 'show-messages', 'show-labels']) document.getElementById(id).onchange = render;
function zoomAtCenter(factor) {{
  const centerX = WIDTH / 2, centerY = HEIGHT / 2;
  const graphX = (centerX - panX) / zoom;
  const graphY = (centerY - panY) / zoom;
  zoom = Math.max(.35, Math.min(10, zoom * factor));
  panX = centerX - graphX * zoom;
  panY = centerY - graphY * zoom;
  render();
}}
document.getElementById('zoom-in').onclick = () => zoomAtCenter(1.25);
document.getElementById('zoom-out').onclick = () => zoomAtCenter(.8);
document.getElementById('reset-view').onclick = () => {{ zoom = 1; panX = 0; panY = 0; render(); }};
document.getElementById('network-name').textContent = data.network_name;
const signalized = data.junctions.filter(j => j.is_signalized).length;
const unsignalized = data.junctions.length - signalized;
document.getElementById('stats').innerHTML = `
  <span>LaneGroup nodes</span><b>${{data.lane_groups.length}}</b>
  <span>Movement nodes</span><b>${{data.movements.length}}</b>
  <span>Lane connectors</span><b>${{data.lane_connectors.length}}</b>
  <span>GNN components</span><b>${{new Set(data.lane_groups.map(lane => lane.component_id)).size}}</b>
  <span>Signalized anchors</span><b>${{signalized}}</b>
  <span>Other SUMO junctions</span><b>${{unsignalized}}</b>
  <span>Typed message edges</span><b>${{data.movements.length * 4 + data.lane_connectors.length}}</b>`;
svg.addEventListener('click', () => {{ selected = null; document.getElementById('details').textContent = 'Select a node or junction.'; render(); }});
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_path = generate_visualization(cfg_path=args.cfg, output_path=args.out)
    print(f'Wrote {output_path.resolve()}')
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == '__main__':
    main()
