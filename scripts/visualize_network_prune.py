"""Generate an interactive HTML editor for SUMO network prune recipes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlparse
import webbrowser

from pydantic import BaseModel, ConfigDict
import sumolib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_network import PruneRecipe  # noqa: E402

DEFAULT_NET = ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.net.xml'
DEFAULT_OUTPUT = ROOT / 'reports' / 'network_prune.html'


class PruneJunctionVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    junction_id: str
    x: float
    y: float
    junction_type: str
    incoming_edge_count: int
    outgoing_edge_count: int


class PruneEdgeVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_junction_id: str
    to_junction_id: str
    lane_count: int
    length_m: float
    speed_limit_mps: float
    shape: tuple[tuple[float, float], ...]


class NetworkPruneVisualization(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_name: str
    junctions: tuple[PruneJunctionVisualization, ...]
    edges: tuple[PruneEdgeVisualization, ...]
    existing_prune_recipe: PruneRecipe


class PruneEditorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    save_url: str | None
    rebuild_url: str | None
    finish_url: str | None
    save_path: str
    rebuild_command: str


@dataclass(frozen=True)
class PruneEditorServerContext:
    html_document: str
    net_path: Path
    save_path: Path
    rebuild_command: tuple[str, ...]


@dataclass(frozen=True)
class RebuildCommand:
    args: tuple[str, ...]
    display: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a self-contained HTML prune editor for a SUMO network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--net', type=Path, default=DEFAULT_NET, help='SUMO .net.xml path')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='Output HTML path')
    parser.add_argument('--prune', type=Path, default=None, help='Existing prune recipe to pre-load')
    parser.add_argument('--save-prune', type=Path, default=None, help='Prune recipe path used by local save mode')
    parser.add_argument('--serve', action='store_true', help='Serve the editor locally and enable direct recipe saving')
    parser.add_argument('--host', default='127.0.0.1', help='Host for --serve mode')
    parser.add_argument('--port', type=int, default=8765, help='Port for --serve mode')
    parser.add_argument('--open', action='store_true', help='Open the generated HTML in the default browser')
    return parser.parse_args()


def generate_prune_editor(
    net_path: Path,
    output_path: Path,
    prune_path: Path | None,
    save_prune_path: Path | None = None,
    save_url: str | None = None,
    rebuild_url: str | None = None,
) -> Path:
    save_path = _resolve_save_prune_path(net_path=net_path, prune_path=prune_path, save_prune_path=save_prune_path)
    visualization = build_prune_visualization(
        net_path=net_path,
        prune_recipe=_load_editor_prune_recipe(prune_path=prune_path, save_prune_path=save_path),
    )
    editor_config = build_editor_config(
        net_path=net_path,
        save_prune_path=save_path,
        save_url=save_url,
        rebuild_url=rebuild_url,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_json = visualization.model_dump_json().replace('</', '<\\/')
    editor_config_json = editor_config.model_dump_json().replace('</', '<\\/')
    output_path.write_text(_html_document(visualization_json, editor_config_json), encoding='utf-8')
    return output_path


def build_editor_config(
    net_path: Path,
    save_prune_path: Path,
    save_url: str | None,
    rebuild_url: str | None = None,
    finish_url: str | None = None,
) -> PruneEditorConfig:
    rebuild_command = _rebuild_command(net_path=net_path, prune_path=save_prune_path)
    return PruneEditorConfig(
        save_url=save_url,
        rebuild_url=rebuild_url,
        finish_url=finish_url,
        save_path=_display_path(save_prune_path),
        rebuild_command=rebuild_command.display,
    )


def build_prune_visualization(net_path: Path, prune_recipe: PruneRecipe) -> NetworkPruneVisualization:
    network = sumolib.net.readNet(str(net_path), withConnections=True)
    normal_edges = tuple(
        edge for edge in network.getEdges() if not str(edge.getID()).startswith(':') and str(edge.getFunction()) == ''
    )
    editable_junction_ids = {str(edge.getFromNode().getID()) for edge in normal_edges} | {
        str(edge.getToNode().getID()) for edge in normal_edges
    }
    junctions = tuple(
        PruneJunctionVisualization(
            junction_id=str(node.getID()),
            x=float(node.getCoord()[0]),
            y=float(node.getCoord()[1]),
            junction_type=str(node.getType()),
            incoming_edge_count=len(
                [edge for edge in normal_edges if str(edge.getToNode().getID()) == str(node.getID())]
            ),
            outgoing_edge_count=len(
                [edge for edge in normal_edges if str(edge.getFromNode().getID()) == str(node.getID())]
            ),
        )
        for node in sorted(network.getNodes(), key=lambda item: str(item.getID()))
        if str(node.getID()) in editable_junction_ids
    )
    edges = tuple(
        PruneEdgeVisualization(
            edge_id=str(edge.getID()),
            from_junction_id=str(edge.getFromNode().getID()),
            to_junction_id=str(edge.getToNode().getID()),
            lane_count=int(edge.getLaneNumber()),
            length_m=float(edge.getLength()),
            speed_limit_mps=float(edge.getSpeed()),
            shape=tuple((float(x_coordinate), float(y_coordinate)) for x_coordinate, y_coordinate in edge.getShape()),
        )
        for edge in sorted(normal_edges, key=lambda item: str(item.getID()))
    )
    return NetworkPruneVisualization(
        network_name=net_path.parent.name,
        junctions=junctions,
        edges=edges,
        existing_prune_recipe=prune_recipe,
    )


def _load_existing_prune_recipe(prune_path: Path | None) -> PruneRecipe:
    if prune_path is None or not prune_path.exists():
        return PruneRecipe(delete_junctions=(), delete_edges=(), keep_junctions=(), notes=())
    return PruneRecipe.model_validate_json(prune_path.read_text(encoding='utf-8-sig'))


def _load_editor_prune_recipe(prune_path: Path | None, save_prune_path: Path) -> PruneRecipe:
    if prune_path is not None:
        return _load_existing_prune_recipe(prune_path)
    if save_prune_path.exists():
        return _load_existing_prune_recipe(save_prune_path)
    return _load_existing_prune_recipe(None)


def _resolve_save_prune_path(net_path: Path, prune_path: Path | None, save_prune_path: Path | None) -> Path:
    if save_prune_path is not None:
        return save_prune_path
    if prune_path is not None:
        return prune_path
    return _default_prune_path(net_path)


def _default_prune_path(net_path: Path) -> Path:
    net_name = net_path.name
    if net_name.endswith('.net.xml'):
        return net_path.with_name(f'{net_name.removesuffix(".net.xml")}.prune.json')
    return net_path.with_suffix('.prune.json')


def _rebuild_command(net_path: Path, prune_path: Path) -> RebuildCommand:
    out_dir = net_path.parent.resolve()
    name = net_path.name.removesuffix('.net.xml') if net_path.name.endswith('.net.xml') else net_path.stem
    osm_path = out_dir / f'{name}.osm'
    if not osm_path.exists():
        return RebuildCommand(
            args=(),
            display=(
                'Cannot infer rebuild command without sibling OSM file. '
                f'Expected {_display_path(osm_path)}; rebuild manually with --bbox.'
            ),
        )
    build_script = ROOT / 'scripts' / 'build_network.py'
    args = (
        sys.executable,
        str(build_script),
        '--osm',
        str(osm_path),
        '--out-dir',
        str(out_dir),
        '--name',
        name,
        '--prune',
        str(prune_path.resolve()),
    )
    display = (
        f'python scripts\\build_network.py --osm {_display_path(osm_path)} '
        f'--out-dir {_display_path(out_dir)} --name {name} --prune {_display_path(prune_path)}'
    )
    return RebuildCommand(args=args, display=display)


def _display_path(path: Path) -> str:
    absolute_path = path.resolve()
    try:
        return str(absolute_path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _serve_prune_editor(
    net_path: Path,
    prune_path: Path | None,
    save_prune_path: Path | None,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    save_path = _resolve_save_prune_path(net_path=net_path, prune_path=prune_path, save_prune_path=save_prune_path)
    visualization = build_prune_visualization(
        net_path=net_path,
        prune_recipe=_load_editor_prune_recipe(prune_path=prune_path, save_prune_path=save_path),
    )
    editor_config = build_editor_config(
        net_path=net_path,
        save_prune_path=save_path,
        save_url='/prune',
        rebuild_url='/rebuild',
        finish_url='/finish',
    )
    rebuild_command = _rebuild_command(net_path=net_path, prune_path=save_path)
    html_document = _html_document(
        visualization.model_dump_json().replace('</', '<\\/'),
        editor_config.model_dump_json().replace('</', '<\\/'),
    )
    context = PruneEditorServerContext(
        html_document=html_document,
        net_path=net_path,
        save_path=save_path,
        rebuild_command=rebuild_command.args,
    )
    server = HTTPServer((host, port), _handler_class(context))
    url = f'http://{host}:{server.server_port}/'
    print(f'Serving prune editor at {url}')
    print(f'Saving prune recipe to {save_path.resolve()}')
    print(f'Rebuild command: {editor_config.rebuild_command}')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped prune editor server.')
    finally:
        server.server_close()


def _handler_class(context: PruneEditorServerContext) -> type[BaseHTTPRequestHandler]:
    def _fresh_visualization_json() -> str:
        prune_recipe = _load_existing_prune_recipe(context.save_path if context.save_path.exists() else None)
        visualization = build_prune_visualization(net_path=context.net_path, prune_recipe=prune_recipe)
        return visualization.model_dump_json()

    class PruneEditorRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path == '/data':
                self._write_json_response(_fresh_visualization_json())
                return
            if parsed_url.path != '/':
                self.send_error(404)
                return
            encoded_html = context.html_document.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(encoded_html)))
            self.end_headers()
            self.wfile.write(encoded_html)

        def do_POST(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path not in {'/prune', '/rebuild', '/finish'}:
                self.send_error(404)
                return
            self._save_recipe_body()
            if parsed_url.path == '/rebuild':
                response_payload = self._run_rebuild()
            elif parsed_url.path == '/finish':
                response_payload = {'saved_path': _display_path(context.save_path), 'finished': True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                response_payload = {'saved_path': _display_path(context.save_path)}
            response = json.dumps(response_payload).encode('utf-8')
            self._write_json_response(response.decode('utf-8'))

        def _write_json_response(self, content: str) -> None:
            response = content.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def _save_recipe_body(self) -> None:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            PruneRecipe.model_validate_json(body.decode('utf-8'))
            context.save_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_json = json.loads(body.decode('utf-8'))
            context.save_path.write_text(json.dumps(parsed_json, indent=2) + '\n', encoding='utf-8')

        def _run_rebuild(self) -> dict[str, object]:
            if not context.rebuild_command:
                return {
                    'saved_path': _display_path(context.save_path),
                    'returncode': 2,
                    'stdout': '',
                    'stderr': 'Cannot rebuild automatically without a sibling OSM file.',
                }
            completed = subprocess.run(
                context.rebuild_command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            return {
                'saved_path': _display_path(context.save_path),
                'returncode': completed.returncode,
                'stdout': completed.stdout[-6000:],
                'stderr': completed.stderr[-6000:],
            }

        def log_message(self, format: str, *args: object) -> None:
            return

    return PruneEditorRequestHandler


def _html_document(visualization_json: str, editor_config_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SUMO Network Prune Editor</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #0f1314;
  color: #e8edef;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; overflow: hidden; }}
button, textarea {{ font: inherit; }}
.app {{ height: 100vh; display: grid; grid-template-columns: 320px 1fr; }}
.sidebar {{
  background: #151a1d;
  border-right: 1px solid #30383c;
  padding: 16px;
  overflow-y: auto;
}}
h1 {{ margin: 0 0 4px; font-size: 18px; letter-spacing: 0; }}
.subtitle {{ color: #9eaaaf; font-size: 12px; margin-bottom: 16px; }}
.section {{ border-top: 1px solid #30383c; padding: 14px 0; }}
.section-title {{ color: #aeb8bc; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; }}
.stats {{ display: grid; grid-template-columns: 1fr auto; gap: 7px 12px; font-size: 12px; }}
.stats span:nth-child(odd) {{ color: #9eaaaf; }}
.button-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }}
.button-row.three {{ grid-template-columns: 1fr 1fr 1fr; }}
button {{
  border: 1px solid #3a454a;
  background: #20282c;
  color: #e4ebee;
  border-radius: 6px;
  padding: 7px 9px;
  cursor: pointer;
}}
button:hover {{ background: #2b353a; }}
button.active {{ border-color: #64b6cf; background: #17313a; }}
button.warn {{ border-color: #b46c6c; }}
button:disabled {{ opacity: .45; cursor: default; }}
textarea {{
  width: 100%;
  min-height: 132px;
  resize: vertical;
  border: 1px solid #30383c;
  border-radius: 6px;
  background: #0f1314;
  color: #dce5e8;
  padding: 8px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 11px;
  line-height: 1.4;
}}
.details {{ font-size: 12px; line-height: 1.55; color: #cbd2d5; min-height: 92px; }}
.details strong {{ display: block; color: #fff; margin-bottom: 5px; }}
.canvas {{ position: relative; background: #0c1011; }}
#network {{ width: 100%; height: 100%; display: block; cursor: grab; }}
#network.dragging {{ cursor: grabbing; }}
.toolbar {{
  position: absolute; top: 14px; right: 14px; display: flex; gap: 6px;
  background: #151a1de8; border: 1px solid #30383c; border-radius: 6px; padding: 5px;
}}
.toolbar button {{ width: 32px; height: 30px; border: 0; padding: 0; border-radius: 4px; }}
.edge {{ fill: none; stroke: #53626a; stroke-width: 3; opacity: .8; cursor: pointer; pointer-events: stroke; vector-effect: non-scaling-stroke; }}
.edge:hover {{ stroke: #9fb0b8; opacity: 1; }}
.edge.delete {{ stroke: #e07272; opacity: .95; }}
.edge.incident {{ stroke: #b86a6a; stroke-dasharray: 8 6; opacity: .8; }}
.edge.selected {{ stroke: #fff; stroke-width: 5; }}
.selection-box {{ fill: #64b6cf22; stroke: #64b6cf; stroke-width: 1.5; stroke-dasharray: 7 5; pointer-events: none; }}
.junction {{ fill: #101719; stroke-width: 2; cursor: pointer; }}
.junction.traffic {{ stroke: #54c189; }}
.junction.other {{ stroke: #8d9aa0; }}
.junction.delete {{ fill: #5b2020; stroke: #f08b8b; }}
.junction.selected {{ stroke: #fff; stroke-width: 3; }}
.label {{ fill: #dce4e7; font-size: 8px; paint-order: stroke; stroke: #0c1011; stroke-width: 3px; pointer-events: none; }}
.hidden {{ display: none; }}
@media (max-width: 820px) {{
  .app {{ grid-template-columns: 250px 1fr; }}
  .sidebar {{ padding: 12px; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>Network Prune</h1>
    <div class="subtitle" id="network-name"></div>
    <div class="section">
      <div class="section-title">View</div>
      <div class="button-row">
        <button id="toggle-labels">Labels</button>
        <button id="show-apply">Apply</button>
      </div>
    </div>
    <div class="section">
      <div class="section-title">Selection</div>
      <div class="details" id="details">Select a junction or edge.</div>
    </div>
    <div class="section">
      <div class="section-title">Recipe</div>
      <div class="stats" id="stats"></div>
      <div class="button-row">
        <button id="undo-last" disabled>Undo</button>
        <button id="rebuild-network">Rebuild Network</button>
      </div>
      <div class="button-row">
        <button id="finish-pruning">Finish Pruning</button>
      </div>
      <div class="details" id="apply-details"></div>
      <textarea id="recipe-text" spellcheck="false"></textarea>
    </div>
  </aside>
  <main class="canvas">
    <svg id="network" viewBox="0 0 1200 800" role="img" aria-label="SUMO network prune editor"></svg>
    <div class="toolbar">
      <button id="zoom-in" title="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out">-</button>
      <button id="reset-view" title="Reset view">R</button>
    </div>
  </main>
</div>
<script id="network-data" type="application/json">{visualization_json}</script>
<script id="editor-config" type="application/json">{editor_config_json}</script>
<script>
let data = JSON.parse(document.getElementById('network-data').textContent);
const editorConfig = JSON.parse(document.getElementById('editor-config').textContent);
const svg = document.getElementById('network');
const NS = 'http://www.w3.org/2000/svg';
const WIDTH = 1200, HEIGHT = 800, PAD = 70;
let selected = null;
let showLabels = false;
let zoom = 1, panX = 0, panY = 0;
let boxStart = null;
let boxEnd = null;
let suppressNextClick = false;
let isRebuilding = false;
const deleteJunctions = new Set();
const deleteEdges = new Set();
const notes = new Map();
const undoStack = [];
let junctionById = new Map();
let edgeById = new Map();
let xValues = data.junctions.map(junction => junction.x);
let yValues = data.junctions.map(junction => junction.y);
let bounds = {{
  minX: Math.min(...xValues), maxX: Math.max(...xValues),
  minY: Math.min(...yValues), maxY: Math.max(...yValues)
}};
function resetSet(target, values) {{
  target.clear();
  for (const value of values) target.add(value);
}}
function resetMap(target, entries) {{
  target.clear();
  for (const [key, value] of entries) target.set(key, value);
}}
function refreshDerivedState() {{
  junctionById = new Map(data.junctions.map(junction => [junction.junction_id, junction]));
  edgeById = new Map(data.edges.map(edge => [edge.edge_id, edge]));
  xValues = data.junctions.map(junction => junction.x);
  yValues = data.junctions.map(junction => junction.y);
  bounds = {{
    minX: Math.min(...xValues), maxX: Math.max(...xValues),
    minY: Math.min(...yValues), maxY: Math.max(...yValues)
  }};
}}
function loadRecipeState(pruneRecipe, clearUndo = true) {{
  resetSet(deleteJunctions, pruneRecipe.delete_junctions);
  resetSet(deleteEdges, pruneRecipe.delete_edges);
  resetMap(notes, pruneRecipe.notes.map(note => [note.target_id, note.text]));
  if (clearUndo) undoStack.length = 0;
  updateUndoButton();
}}
function deleteSnapshot() {{
  return {{
    junctions: Array.from(deleteJunctions),
    edges: Array.from(deleteEdges)
  }};
}}
function sameDeleteState(snapshot) {{
  if (snapshot.junctions.length !== deleteJunctions.size || snapshot.edges.length !== deleteEdges.size) return false;
  return snapshot.junctions.every(id => deleteJunctions.has(id)) && snapshot.edges.every(id => deleteEdges.has(id));
}}
function pushUndoSnapshot(snapshot) {{
  if (sameDeleteState(snapshot)) return;
  undoStack.push(snapshot);
  if (undoStack.length > 100) undoStack.shift();
  updateUndoButton();
}}
function restoreDeleteSnapshot(snapshot) {{
  resetSet(deleteJunctions, snapshot.junctions);
  resetSet(deleteEdges, snapshot.edges);
  updateRecipeText();
  render();
}}
function updateUndoButton() {{
  const button = document.getElementById('undo-last');
  if (!button) return;
  button.disabled = undoStack.length === 0 || isRebuilding;
  const finishButton = document.getElementById('finish-pruning');
  if (finishButton) finishButton.disabled = isRebuilding;
}}
function undoLast() {{
  if (isRebuilding) return;
  const snapshot = undoStack.pop();
  if (!snapshot) return;
  restoreDeleteSnapshot(snapshot);
  updateUndoButton();
  document.getElementById('details').textContent = 'Undid the last edit.';
  rebuildNetwork('Undo saved. Rebuild running...');
}}
function element(name, attrs = {{}}) {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}}
function pointPosition(point) {{
  const xRange = Math.max(1, bounds.maxX - bounds.minX);
  const yRange = Math.max(1, bounds.maxY - bounds.minY);
  return {{
    x: PAD + (point[0] - bounds.minX) / xRange * (WIDTH - PAD * 2),
    y: HEIGHT - PAD - (point[1] - bounds.minY) / yRange * (HEIGHT - PAD * 2)
  }};
}}
function junctionPosition(junction) {{
  return pointPosition([junction.x, junction.y]);
}}
function edgePath(edge) {{
  const points = edge.shape.length > 0
    ? edge.shape.map(pointPosition)
    : [junctionPosition(junctionById.get(edge.from_junction_id)), junctionPosition(junctionById.get(edge.to_junction_id))];
  return points.map((point, index) => `${{index === 0 ? 'M' : 'L'}} ${{point.x}} ${{point.y}}`).join(' ');
}}
function svgPointFromEvent(event) {{
  const rect = svg.getBoundingClientRect();
  return {{
    x: (event.clientX - rect.left) / rect.width * WIDTH,
    y: (event.clientY - rect.top) / rect.height * HEIGHT
  }};
}}
function graphPointFromEvent(event) {{
  const point = svgPointFromEvent(event);
  return {{
    x: (point.x - panX) / zoom,
    y: (point.y - panY) / zoom
  }};
}}
function normalizedBox() {{
  if (!boxStart || !boxEnd) return null;
  return {{
    minX: Math.min(boxStart.x, boxEnd.x),
    maxX: Math.max(boxStart.x, boxEnd.x),
    minY: Math.min(boxStart.y, boxEnd.y),
    maxY: Math.max(boxStart.y, boxEnd.y)
  }};
}}
function pointInBox(point, box) {{
  return point.x >= box.minX && point.x <= box.maxX && point.y >= box.minY && point.y <= box.maxY;
}}
function edgeMidpoint(edge) {{
  const points = edge.shape.length > 0
    ? edge.shape.map(pointPosition)
    : [junctionPosition(junctionById.get(edge.from_junction_id)), junctionPosition(junctionById.get(edge.to_junction_id))];
  const middleIndex = Math.floor(points.length / 2);
  if (points.length % 2 === 1) return points[middleIndex];
  const first = points[middleIndex - 1];
  const second = points[middleIndex];
  return {{x: (first.x + second.x) / 2, y: (first.y + second.y) / 2}};
}}
function edgeSelectedByBox(edge, box) {{
  return pointInBox(edgeMidpoint(edge), box);
}}
function incidentDeleted(edge) {{
  return deleteJunctions.has(edge.from_junction_id) || deleteJunctions.has(edge.to_junction_id);
}}
function recipe() {{
  return {{
    delete_junctions: Array.from(deleteJunctions).sort(),
    delete_edges: Array.from(deleteEdges).sort(),
    keep_junctions: [],
    notes: Array.from(notes.entries()).sort((a, b) => a[0].localeCompare(b[0])).map(([target_id, text]) => ({{target_id, text}}))
  }};
}}
function updateRecipeText() {{
  const currentRecipe = recipe();
  document.getElementById('recipe-text').value = JSON.stringify(currentRecipe, null, 2);
  document.getElementById('stats').innerHTML = `
    <span>Delete junctions</span><b>${{currentRecipe.delete_junctions.length}}</b>
    <span>Delete edges</span><b>${{currentRecipe.delete_edges.length}}</b>
    <span>Incident edges</span><b>${{data.edges.filter(incidentDeleted).length}}</b>`;
  updateUndoButton();
}}
function updateApplyDetails(message = null) {{
  const saveMode = editorConfig.save_url ? `Rebuild saves to ${{editorConfig.save_path}} before running.` : `Static mode cannot rebuild directly; use --serve for city iteration.`;
  const prefix = message ? `${{message}}<br><br>` : '';
  document.getElementById('apply-details').innerHTML = `${{prefix}}${{saveMode}}<br><br>Rebuild:<br><code>${{editorConfig.rebuild_command}}</code>`;
}}
async function reloadVisualization(clearUndo = false) {{
  if (!editorConfig.rebuild_url) return;
  const response = await fetch('/data');
  if (!response.ok) {{
    updateApplyDetails(`Rebuild succeeded, but reload failed: ${{response.status}}`);
    return;
  }}
  data = await response.json();
  refreshDerivedState();
  loadRecipeState(data.existing_prune_recipe, clearUndo);
  selected = null;
  document.getElementById('network-name').textContent = data.network_name;
  document.getElementById('details').textContent = 'Network reloaded after rebuild.';
  updateRecipeText();
  render();
}}
function selectObject(type, item, event) {{
  event.stopPropagation();
  if (isRebuilding) return;
  selected = `${{type}}:${{type === 'junction' ? item.junction_id : item.edge_id}}`;
  const before = deleteSnapshot();
  if (type === 'junction') deleteJunctions.add(item.junction_id);
  else deleteEdges.add(item.edge_id);
  pushUndoSnapshot(before);
  showDetails(type, item);
  updateRecipeText();
  render();
  if (!sameDeleteState(before)) rebuildNetwork('Selection saved. Rebuild running...');
}}
function showDetails(type, item) {{
  const target = document.getElementById('details');
  if (type === 'junction') {{
    const noteText = notes.get(item.junction_id) || '';
    target.innerHTML = `<strong>Junction ${{item.junction_id}}</strong>Type: ${{item.junction_type}}<br>Incoming edges: ${{item.incoming_edge_count}}<br>Outgoing edges: ${{item.outgoing_edge_count}}<br>Status: ${{deleteJunctions.has(item.junction_id) ? 'delete' : 'unchanged'}}<br><br><textarea id="note-input" placeholder="Optional note">${{noteText}}</textarea>`;
    document.getElementById('note-input').addEventListener('input', event => {{
      const text = event.target.value.trim();
      if (text) notes.set(item.junction_id, text);
      else notes.delete(item.junction_id);
      updateRecipeText();
    }});
  }} else {{
    target.innerHTML = `<strong>Edge ${{item.edge_id}}</strong>${{item.from_junction_id}} -> ${{item.to_junction_id}}<br>Length: ${{item.length_m.toFixed(1)}} m<br>Lanes: ${{item.lane_count}}<br>Speed: ${{item.speed_limit_mps.toFixed(1)}} m/s<br>Status: ${{deleteEdges.has(item.edge_id) || incidentDeleted(item) ? 'delete' : 'unchanged'}}`;
  }}
}}
function applyBoxSelection() {{
  const box = normalizedBox();
  if (!box || isRebuilding) return;
  const selectedJunctions = data.junctions.filter(junction => pointInBox(junctionPosition(junction), box));
  const selectedEdges = data.edges.filter(edge => edgeSelectedByBox(edge, box));
  const before = deleteSnapshot();
  for (const junction of selectedJunctions) {{
    deleteJunctions.add(junction.junction_id);
  }}
  for (const edge of selectedEdges) deleteEdges.add(edge.edge_id);
  pushUndoSnapshot(before);
  document.getElementById('details').textContent = `Box selected ${{selectedJunctions.length}} junctions and ${{selectedEdges.length}} edges.`;
  updateRecipeText();
  if (!sameDeleteState(before)) rebuildNetwork('Selection saved. Rebuild running...');
}}
function render() {{
  svg.replaceChildren();
  const viewport = element('g', {{transform: `translate(${{panX}} ${{panY}}) scale(${{zoom}})`}});
  svg.appendChild(viewport);
  const edgeStroke = Math.max(5, 6 / Math.sqrt(zoom));
  for (const edge of data.edges) {{
    const classNames = ['edge'];
    if (deleteEdges.has(edge.edge_id)) classNames.push('delete');
    if (incidentDeleted(edge)) classNames.push('incident');
    if (selected === `edge:${{edge.edge_id}}`) classNames.push('selected');
    const path = element('path', {{d: edgePath(edge), class: classNames.join(' '), style: `stroke-width:${{edgeStroke}}px`}});
    path.addEventListener('click', event => selectObject('edge', edge, event));
    viewport.appendChild(path);
  }}
  const nodeScale = Math.max(.35, Math.min(1, .9 / Math.sqrt(zoom)));
  const labelScale = Math.max(.2, Math.min(.85, .85 / zoom));
  for (const junction of data.junctions) {{
    const point = junctionPosition(junction);
    const classNames = ['junction', junction.junction_type.includes('traffic_light') ? 'traffic' : 'other'];
    if (deleteJunctions.has(junction.junction_id)) classNames.push('delete');
    if (selected === `junction:${{junction.junction_id}}`) classNames.push('selected');
    const radius = (junction.junction_type.includes('traffic_light') ? 6.4 : 4.8) * nodeScale;
    const node = element('circle', {{cx: point.x, cy: point.y, r: radius, class: classNames.join(' ')}});
    node.addEventListener('click', event => selectObject('junction', junction, event));
    viewport.appendChild(node);
    if (showLabels) {{
      const label = element('text', {{x: point.x + 7 * nodeScale, y: point.y - 7 * nodeScale, class: 'label', style: `font-size:${{8 * labelScale}}px`}});
      label.textContent = junction.junction_id;
      viewport.appendChild(label);
    }}
  }}
  const box = normalizedBox();
  if (box) {{
    viewport.appendChild(element('rect', {{
      x: box.minX,
      y: box.minY,
      width: box.maxX - box.minX,
      height: box.maxY - box.minY,
      class: 'selection-box'
    }}));
  }}
}}
let panStart = null;
svg.addEventListener('pointerdown', event => {{
  if (isRebuilding) return;
  if (event.button === 1) {{
    event.preventDefault();
    panStart = {{x: event.clientX, y: event.clientY, panX, panY}};
    return;
  }}
  if (event.target !== svg || event.button !== 0) return;
  {{
    boxStart = graphPointFromEvent(event);
    boxEnd = boxStart;
    suppressNextClick = true;
    render();
    return;
  }}
}});
svg.addEventListener('pointermove', event => {{
  if (boxStart) {{
    boxEnd = graphPointFromEvent(event);
    render();
    return;
  }}
  if (!panStart) return;
  panX = panStart.panX + event.clientX - panStart.x;
  panY = panStart.panY + event.clientY - panStart.y;
  render();
}});
svg.addEventListener('pointerup', () => {{
  if (boxStart) {{
    applyBoxSelection();
    boxStart = null;
    boxEnd = null;
    render();
    return;
  }}
  panStart = null;
}});
svg.addEventListener('wheel', event => {{
  event.preventDefault();
  const rect = svg.getBoundingClientRect();
  const cursorX = (event.clientX - rect.left) / rect.width * WIDTH;
  const cursorY = (event.clientY - rect.top) / rect.height * HEIGHT;
  const graphX = (cursorX - panX) / zoom;
  const graphY = (cursorY - panY) / zoom;
  zoom = Math.max(.25, Math.min(14, zoom * (event.deltaY < 0 ? 1.18 : .85)));
  panX = cursorX - graphX * zoom;
  panY = cursorY - graphY * zoom;
  render();
}}, {{passive: false}});
function zoomAtCenter(factor) {{
  const graphX = (WIDTH / 2 - panX) / zoom;
  const graphY = (HEIGHT / 2 - panY) / zoom;
  zoom = Math.max(.25, Math.min(14, zoom * factor));
  panX = WIDTH / 2 - graphX * zoom;
  panY = HEIGHT / 2 - graphY * zoom;
  render();
}}
document.getElementById('undo-last').onclick = () => undoLast();
document.getElementById('toggle-labels').onclick = event => {{
  showLabels = !showLabels;
  event.target.classList.toggle('active', showLabels);
  render();
}};
document.getElementById('show-apply').onclick = () => updateApplyDetails();
async function rebuildNetwork(message = 'Rebuild running...') {{
  if (!editorConfig.rebuild_url) {{
    updateApplyDetails('Rebuild is only available in --serve mode.');
    return;
  }}
  if (isRebuilding) return;
  isRebuilding = true;
  updateUndoButton();
  document.getElementById('rebuild-network').disabled = true;
  updateApplyDetails(message);
  try {{
    const response = await fetch(editorConfig.rebuild_url, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(recipe())
    }});
    if (!response.ok) {{
      updateApplyDetails(`Rebuild request failed: ${{response.status}}`);
      return;
    }}
    const payload = await response.json();
    const output = [payload.stdout, payload.stderr].filter(Boolean).join('\\n');
    const escapedOutput = output.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    updateApplyDetails(`Rebuild finished with exit code ${{payload.returncode}}. Recipe saved to ${{payload.saved_path}}.<br><br><pre>${{escapedOutput}}</pre>`);
    if (payload.returncode === 0) await reloadVisualization(false);
  }} finally {{
    isRebuilding = false;
    document.getElementById('rebuild-network').disabled = false;
    updateUndoButton();
  }}
}}
document.getElementById('rebuild-network').onclick = () => rebuildNetwork();
async function finishPruning() {{
  if (!editorConfig.finish_url) {{
    updateApplyDetails('Finish is only available in --serve mode.');
    return;
  }}
  if (isRebuilding) return;
  isRebuilding = true;
  updateUndoButton();
  document.getElementById('rebuild-network').disabled = true;
  updateApplyDetails('Saving prune JSON and finishing...');
  try {{
    const response = await fetch(editorConfig.finish_url, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(recipe())
    }});
    if (!response.ok) {{
      updateApplyDetails(`Finish request failed: ${{response.status}}`);
      return;
    }}
    const payload = await response.json();
    updateApplyDetails(`Prune JSON saved to ${{payload.saved_path}}. You can close this tab.`);
    document.getElementById('details').textContent = 'Finished pruning. The workbench will continue in the terminal.';
  }} finally {{
    isRebuilding = false;
    document.getElementById('rebuild-network').disabled = false;
    updateUndoButton();
  }}
}}
document.getElementById('finish-pruning').onclick = () => finishPruning();
document.getElementById('zoom-in').onclick = () => zoomAtCenter(1.25);
document.getElementById('zoom-out').onclick = () => zoomAtCenter(.8);
document.getElementById('reset-view').onclick = () => {{ zoom = 1; panX = 0; panY = 0; render(); }};
document.addEventListener('keydown', event => {{
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {{
    event.preventDefault();
    undoLast();
  }}
}});
svg.addEventListener('click', () => {{
  if (suppressNextClick) {{
    suppressNextClick = false;
    return;
  }}
  selected = null;
  document.getElementById('details').textContent = 'Select a junction or edge.';
  render();
}});
document.getElementById('network-name').textContent = data.network_name;
refreshDerivedState();
loadRecipeState(data.existing_prune_recipe);
updateRecipeText();
updateApplyDetails();
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.serve:
        _serve_prune_editor(
            net_path=args.net,
            prune_path=args.prune,
            save_prune_path=args.save_prune,
            host=args.host,
            port=args.port,
            open_browser=args.open,
        )
        return
    output_path = generate_prune_editor(
        net_path=args.net,
        output_path=args.out,
        prune_path=args.prune,
        save_prune_path=args.save_prune,
        save_url=None,
    )
    print(f'Wrote {output_path.resolve()}')
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == '__main__':
    main()
