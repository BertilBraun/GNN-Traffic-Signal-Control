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
.segmented {{ display: grid; grid-template-columns: 1fr 1fr; border: 1px solid #3a4448; border-radius: 6px; overflow: hidden; }}
.segmented button {{ border: 0; padding: 8px; color: #b8c1c5; background: #1b2225; cursor: pointer; }}
.segmented button.active {{ color: #fff; background: #365b68; }}
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
.message {{ fill: none; stroke-width: 1.4; opacity: .58; }}
.message.input {{ stroke: #48b5d6; }}
.message.output {{ stroke: #e7ad52; }}
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
      <div class="section-title">Layout</div>
      <div class="segmented">
        <button id="map-layout" class="active">SUMO map</button>
        <button id="force-layout">Relaxed</button>
      </div>
    </div>
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
let layout = 'map', selected = null;
let zoom = 1, panX = 0, panY = 0;
let relaxed = new Map();
const junctionById = new Map(data.junctions.map(j => [j.junction_id, j]));
const laneById = new Map(data.lane_groups.map(l => [l.lane_group_id, l]));
const movementsByTls = new Map();
const componentColors = ['#62b6cb', '#f0b64d', '#8ec07c', '#d3869b', '#b8a0ff', '#e07a5f', '#7bc8a4', '#f2cc8f'];
for (const movement of data.movements) {{
  if (!movementsByTls.has(movement.traffic_light_id)) movementsByTls.set(movement.traffic_light_id, []);
  movementsByTls.get(movement.traffic_light_id).push(movement);
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
for (const junction of data.junctions) relaxed.set(junction.junction_id, {{...mapPosition(junction), vx: 0, vy: 0}});
function junctionPosition(id) {{
  return layout === 'map' ? mapPosition(junctionById.get(id)) : relaxed.get(id);
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
      return {{x: from.x + dx * ratio - dy / length * 9, y: from.y + dy * ratio + dx / length * 9}};
    }}
    travelled += lengths[index];
  }}
  return points[0];
}}
function movementPosition(movement) {{
  const center = junctionPosition(movement.traffic_light_id);
  const group = movementsByTls.get(movement.traffic_light_id);
  const index = group.findIndex(item => item.movement_id === movement.movement_id);
  const radius = group.length > 12 ? 36 : 29;
  const angle = -Math.PI / 2 + index / group.length * Math.PI * 2;
  return {{x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius}};
}}
function componentColor(componentId) {{
  return componentColors[componentId % componentColors.length];
}}
function element(name, attrs = {{}}) {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}}
function addMarker(defs, id, color, reverse = false) {{
  const marker = element('marker', {{id, viewBox: '0 0 10 10', refX: reverse ? 2 : 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'}});
  marker.appendChild(element('path', {{d: reverse ? 'M 10 0 L 0 5 L 10 10 z' : 'M 0 0 L 10 5 L 0 10 z', fill: color}}));
  defs.appendChild(marker);
}}
function details(type, item) {{
  const target = document.getElementById('details');
  if (type === 'lane') target.innerHTML = `<strong>LaneGroup L${{item.lane_group_id}}</strong>Component: ${{item.component_id}}<br>SUMO edges: ${{item.edge_ids.join(' -> ')}}<br>Direction: ${{item.from_junction_id}} -> ${{item.to_junction_id}}<br>Length: ${{item.length_m.toFixed(1)}} m<br>Effective lanes: ${{item.effective_lane_count.toFixed(2)}}<br>Effective speed: ${{item.effective_speed_limit_mps.toFixed(2)}} m/s<br><br>This is a real GNN node.`;
  if (type === 'movement') target.innerHTML = `<strong>Movement M${{item.movement_id}}</strong>Component: ${{item.component_id}}<br>Traffic light: ${{item.traffic_light_id}}<br>Input: L${{item.input_lane_group_id}}<br>Output: L${{item.output_lane_group_id}}<br>Controlled links: ${{item.controlled_link_count}}<br><br>This is a real GNN node.`;
  if (type === 'junction') {{
    const count = (movementsByTls.get(item.junction_id) || []).length;
    target.innerHTML = `<strong>Junction ${{item.junction_id}}</strong>SUMO type: ${{item.junction_type}}<br>Signalized: ${{item.is_signalized ? 'yes' : 'no'}}<br>Movement nodes: ${{count}}<br>Selectable phases: ${{item.selectable_phase_count}}<br><br>This anchor is visual context, not a GNN node.`;
  }}
}}
function selectNode(type, item, event) {{
  event.stopPropagation();
  selected = `${{type}}:${{type === 'lane' ? item.lane_group_id : type === 'movement' ? item.movement_id : item.junction_id}}`;
  details(type, item);
  render();
}}
function curvedPath(a, b, bend) {{
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const dx = b.x - a.x, dy = b.y - a.y, length = Math.max(1, Math.hypot(dx, dy));
  return `M ${{a.x}} ${{a.y}} Q ${{mx - dy / length * bend}} ${{my + dx / length * bend}} ${{b.x}} ${{b.y}}`;
}}
function render() {{
  svg.replaceChildren();
  const nodeScale = Math.max(.18, Math.min(.85, .85 / zoom));
  const labelScale = Math.max(.16, Math.min(.78, .78 / zoom));
  const defs = element('defs');
  addMarker(defs, 'arrow-input', '#48b5d6');
  addMarker(defs, 'arrow-output', '#e7ad52');
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
    const radius = (movements.length > 12 ? 48 : 40) * nodeScale;
    const circle = element('circle', {{cx: center.x, cy: center.y, r: radius, class: `junction-group ${{selected === `junction:${{tlsId}}` ? 'selected' : ''}}`}});
    circle.addEventListener('click', event => selectNode('junction', junctionById.get(tlsId), event));
    groups.appendChild(circle);
  }}
  viewport.appendChild(groups);
  if (showMessages) {{
    const edges = element('g');
    for (const movement of data.movements) {{
      const movementPoint = movementPosition(movement);
      const inputPoint = lanePosition(laneById.get(movement.input_lane_group_id));
      const outputPoint = lanePosition(laneById.get(movement.output_lane_group_id));
      edges.appendChild(element('path', {{d: curvedPath(inputPoint, movementPoint, 7), class: 'message input', 'marker-start': 'url(#arrow-input)', 'marker-end': 'url(#arrow-input)'}}));
      edges.appendChild(element('path', {{d: curvedPath(outputPoint, movementPoint, -7), class: 'message output', 'marker-start': 'url(#arrow-output)', 'marker-end': 'url(#arrow-output)'}}));
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
    if (layout === 'relaxed') node.addEventListener('pointerdown', event => beginAnchorDrag(junction.junction_id, event));
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
function relax() {{
  const nodes = data.junctions;
  const positions = relaxed;
  for (let iteration = 0; iteration < 320; iteration++) {{
    for (const junction of nodes) {{
      const point = positions.get(junction.junction_id);
      point.vx *= .76; point.vy *= .76;
    }}
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {{
      const a = positions.get(nodes[i].junction_id), b = positions.get(nodes[j].junction_id);
      let dx = b.x - a.x, dy = b.y - a.y, d2 = Math.max(100, dx * dx + dy * dy);
      const force = 9000 / d2;
      const length = Math.sqrt(d2);
      dx /= length; dy /= length;
      a.vx -= dx * force; a.vy -= dy * force; b.vx += dx * force; b.vy += dy * force;
    }}
    for (const road of data.roads) {{
      const a = positions.get(road.from_junction_id), b = positions.get(road.to_junction_id);
      const dx = b.x - a.x, dy = b.y - a.y, length = Math.max(1, Math.hypot(dx, dy));
      const force = (length - 135) * .0025;
      a.vx += dx / length * force; a.vy += dy / length * force;
      b.vx -= dx / length * force; b.vy -= dy / length * force;
    }}
    for (const junction of nodes) {{
      const point = positions.get(junction.junction_id);
      point.vx += (WIDTH / 2 - point.x) * .00045;
      point.vy += (HEIGHT / 2 - point.y) * .00045;
      point.x = Math.max(PAD, Math.min(WIDTH - PAD, point.x + point.vx));
      point.y = Math.max(PAD, Math.min(HEIGHT - PAD, point.y + point.vy));
    }}
  }}
}}
function setLayout(next) {{
  layout = next;
  document.getElementById('map-layout').classList.toggle('active', next === 'map');
  document.getElementById('force-layout').classList.toggle('active', next === 'relaxed');
  if (next === 'relaxed') relax();
  render();
}}
let dragAnchor = null;
function beginAnchorDrag(id, event) {{
  dragAnchor = id;
  event.stopPropagation();
  svg.setPointerCapture(event.pointerId);
}}
svg.addEventListener('pointermove', event => {{
  if (!dragAnchor) return;
  const rect = svg.getBoundingClientRect();
  const point = relaxed.get(dragAnchor);
  point.x = ((event.clientX - rect.left) / rect.width * WIDTH - panX) / zoom;
  point.y = ((event.clientY - rect.top) / rect.height * HEIGHT - panY) / zoom;
  render();
}});
svg.addEventListener('pointerup', () => dragAnchor = null);
let panStart = null;
svg.addEventListener('pointerdown', event => {{
  if (event.target === svg) panStart = {{x: event.clientX, y: event.clientY, panX, panY}};
}});
svg.addEventListener('pointermove', event => {{
  if (!panStart || dragAnchor) return;
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
document.getElementById('map-layout').onclick = () => setLayout('map');
document.getElementById('force-layout').onclick = () => setLayout('relaxed');
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
  <span>GNN components</span><b>${{new Set(data.lane_groups.map(lane => lane.component_id)).size}}</b>
  <span>Signalized anchors</span><b>${{signalized}}</b>
  <span>Other SUMO junctions</span><b>${{unsignalized}}</b>
  <span>Typed message edges</span><b>${{data.movements.length * 4}}</b>`;
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
