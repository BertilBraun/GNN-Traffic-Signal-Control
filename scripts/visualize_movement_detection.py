"""Generate an HTML viewer for movement LaneGroup detector windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import webbrowser

import traci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import lane_inputs_from_net, resolve_sumocfg_net_path  # noqa: E402
from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale  # noqa: E402
from src.movement.features import (  # noqa: E402
    HALTING_SPEED_THRESHOLD_MPS,
    MovementControlState,
    VehicleSnapshotCollector,
    build_feature_frame,
    detector_length,
)
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.graph_schema import LaneGroupNode, MovementGraph  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402

DEFAULT_OUTPUT = ROOT / 'reports' / 'movement_detection.html'


def generate_visualization(
    cfg_path: Path,
    output_path: Path,
    steps: int,
    sample_every: int,
    demand_scale: float,
    seed: int,
    time_to_teleport: int,
) -> Path:
    """Record detector samples and write an interactive HTML viewer."""
    net_path = resolve_sumocfg_net_path(cfg_path)
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(net_path)
    demand_route_files = route_files_for_demand_scale(cfg_path=cfg_path, demand_scale=demand_scale)
    runtime = MovementControlRuntime(
        cfg_path=cfg_path,
        gui=False,
        seed=seed,
        time_to_teleport=time_to_teleport,
        additional_sumo_args=route_file_sumo_args(demand_route_files.route_files),
    )
    try:
        runtime.start()
        graph = build_movement_graph(runtime.programs, net_path=net_path)
        vehicle_snapshot_collector = VehicleSnapshotCollector(traci.vehicle)
        static_payload = _static_payload(graph=graph, lane_ids_by_edge=lane_ids_by_edge)
        samples = []
        for step in range(steps + 1):
            if step % sample_every == 0:
                vehicles = vehicle_snapshot_collector.capture()
                feature_frame = build_feature_frame(
                    graph=graph,
                    lane_ids_by_edge=lane_ids_by_edge,
                    lane_geometries=lane_geometries,
                    control_state=MovementControlState(),
                    vehicles=vehicles,
                )
                samples.append(
                    _sample_payload(
                        step=step,
                        graph=graph,
                        feature_frame=feature_frame,
                        lane_ids_by_edge=lane_ids_by_edge,
                    )
                )
            if step < steps:
                runtime.step()
            if not runtime.is_running():
                break
    finally:
        runtime.close()
        demand_route_files.cleanup()

    payload = {
        **static_payload,
        'cfg': str(cfg_path),
        'steps': steps,
        'sampleEvery': sample_every,
        'samples': samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(json.dumps(payload).replace('</', '<\\/')), encoding='utf-8')
    return output_path


def _static_payload(
    graph: MovementGraph,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    roads = []
    lane_groups = []
    all_points: list[tuple[float, float]] = []
    edge_shape_by_id = {}
    edge_lane_ids = {}
    for edge_id, lane_ids in lane_ids_by_edge.items():
        if not lane_ids:
            continue
        shape = tuple((float(x), float(y)) for x, y in traci.lane.getShape(lane_ids[0]))
        edge_shape_by_id[edge_id] = shape
        edge_lane_ids[edge_id] = lane_ids
        all_points.extend(shape)
        roads.append({'edgeId': edge_id, 'points': shape})

    for lane_group in graph.lane_groups:
        points = _lane_group_shape(lane_group=lane_group, edge_shape_by_id=edge_shape_by_id)
        detector_points = _tail_segment(points, detector_length(_polyline_length(points)))
        lane_groups.append(
            {
                'id': int(lane_group.lane_group_id),
                'edgeIds': [str(edge_id) for edge_id in lane_group.edge_ids],
                'points': points,
                'detectorPoints': detector_points,
            }
        )
    return {
        'bounds': _bounds(all_points),
        'roads': roads,
        'laneGroups': lane_groups,
        'movements': [
            {
                'id': int(movement.movement_id),
                'tls': str(movement.traffic_light_id),
                'inputLaneGroupId': int(movement.input_lane_group_id),
                'outputLaneGroupId': int(movement.output_lane_group_id),
            }
            for movement in graph.movements
        ],
    }


def _sample_payload(
    step: int,
    graph: MovementGraph,
    feature_frame,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    counts = {
        int(row.lane_group_id): {
            'featureCount': round(row.dynamic.vehicle_count_detector),
            'featureHalting': round(row.dynamic.halting_count_detector),
        }
        for row in feature_frame.lane_group_rows
    }
    raw_counts = _raw_detector_counts(graph=graph, lane_ids_by_edge=lane_ids_by_edge)
    for lane_group_id, raw in raw_counts.items():
        counts.setdefault(lane_group_id, {'featureCount': 0, 'featureHalting': 0})
        counts[lane_group_id].update(raw)

    lane_group_by_edge = {
        str(edge_id): int(lane_group_id) for edge_id, lane_group_id in graph.lane_group_id_by_edge.items()
    }
    vehicles = []
    for vehicle_id in traci.vehicle.getIDList():
        lane_id = str(traci.vehicle.getLaneID(vehicle_id))
        edge_id = _edge_from_lane(lane_id)
        if edge_id.startswith(':'):
            continue
        shape = tuple((float(x), float(y)) for x, y in traci.lane.getShape(lane_id))
        x, y = _point_at_distance(shape, float(traci.vehicle.getLanePosition(vehicle_id)))
        lane_group_id = lane_group_by_edge.get(edge_id)
        vehicles.append(
            {
                'id': str(vehicle_id),
                'edgeId': edge_id,
                'laneId': lane_id,
                'laneGroupId': lane_group_id,
                'x': x,
                'y': y,
                'speed': float(traci.vehicle.getSpeed(vehicle_id)),
                'halting': float(traci.vehicle.getSpeed(vehicle_id)) <= HALTING_SPEED_THRESHOLD_MPS,
                'inDetector': bool(lane_group_id is not None and vehicle_id in raw_counts[lane_group_id]['vehicleIds']),
            }
        )
    return {'step': step, 'counts': counts, 'vehicles': vehicles}


def _raw_detector_counts(
    graph: MovementGraph,
    lane_ids_by_edge: dict[str, tuple[str, ...]],
) -> dict[int, dict[str, object]]:
    counts = {}
    for lane_group in graph.lane_groups:
        vehicle_ids = []
        halting_ids = []
        total_length = sum(
            float(traci.lane.getLength(lane_ids_by_edge[str(edge_id)][0])) for edge_id in lane_group.edge_ids
        )
        detector_start = max(0.0, total_length - detector_length(total_length))
        offset = 0.0
        for edge_id in lane_group.edge_ids:
            edge_text = str(edge_id)
            edge_length = float(traci.lane.getLength(lane_ids_by_edge[edge_text][0]))
            for lane_id in lane_ids_by_edge[edge_text]:
                for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    vehicle_text = str(vehicle_id)
                    position = offset + float(traci.vehicle.getLanePosition(vehicle_text))
                    if detector_start <= position <= total_length:
                        vehicle_ids.append(vehicle_text)
                        if float(traci.vehicle.getSpeed(vehicle_text)) <= HALTING_SPEED_THRESHOLD_MPS:
                            halting_ids.append(vehicle_text)
            offset += edge_length
        counts[int(lane_group.lane_group_id)] = {
            'rawCount': len(vehicle_ids),
            'rawHalting': len(halting_ids),
            'vehicleIds': sorted(vehicle_ids),
        }
    return counts


def _lane_group_shape(
    lane_group: LaneGroupNode,
    edge_shape_by_id: dict[str, tuple[tuple[float, float], ...]],
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for edge_id in lane_group.edge_ids:
        shape = list(edge_shape_by_id[str(edge_id)])
        if points and shape and points[-1] == shape[0]:
            points.extend(shape[1:])
        else:
            points.extend(shape)
    return points


def _tail_segment(points: list[tuple[float, float]], tail_length: float) -> list[tuple[float, float]]:
    if not points or tail_length <= 0.0:
        return []
    total = _polyline_length(points)
    return _subline(points, max(0.0, total - tail_length), total)


def _subline(
    points: list[tuple[float, float]],
    start_m: float,
    end_m: float,
) -> list[tuple[float, float]]:
    output = []
    distance = 0.0
    for first, second in zip(points, points[1:]):
        segment_length = _distance(first, second)
        segment_start = distance
        segment_end = distance + segment_length
        if segment_end >= start_m and segment_start <= end_m:
            local_start = max(0.0, start_m - segment_start)
            local_end = min(segment_length, end_m - segment_start)
            start_point = _interpolate(first, second, local_start / segment_length if segment_length else 0.0)
            end_point = _interpolate(first, second, local_end / segment_length if segment_length else 0.0)
            if not output or output[-1] != start_point:
                output.append(start_point)
            output.append(end_point)
        distance = segment_end
    return output


def _point_at_distance(points: tuple[tuple[float, float], ...], target_m: float) -> tuple[float, float]:
    distance = 0.0
    for first, second in zip(points, points[1:]):
        segment_length = _distance(first, second)
        if distance + segment_length >= target_m:
            ratio = (target_m - distance) / segment_length if segment_length else 0.0
            return _interpolate(first, second, ratio)
        distance += segment_length
    return points[-1] if points else (0.0, 0.0)


def _polyline_length(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> float:
    return sum(_distance(first, second) for first, second in zip(points, points[1:]))


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5


def _interpolate(
    first: tuple[float, float],
    second: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    return (
        first[0] + (second[0] - first[0]) * ratio,
        first[1] + (second[1] - first[1]) * ratio,
    )


def _edge_from_lane(lane_id: str) -> str:
    edge_id, separator, lane_index = lane_id.rpartition('_')
    return edge_id if separator and lane_index.isdigit() and edge_id else lane_id


def _bounds(points: list[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {'minX': 0.0, 'maxX': 1.0, 'minY': 0.0, 'maxY': 1.0}
    return {
        'minX': min(point[0] for point in points),
        'maxX': max(point[0] for point in points),
        'minY': min(point[1] for point in points),
        'maxY': max(point[1] for point in points),
    }


def _html_document(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Movement Detector Windows</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; background: #101416; color: #e8edef; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; overflow: hidden; }}
.app {{ height: 100vh; display: grid; grid-template-columns: 320px 1fr; }}
.sidebar {{ background: #151a1d; border-right: 1px solid #30383c; padding: 16px; overflow-y: auto; }}
h1 {{ margin: 0 0 4px; font-size: 18px; }}
.sub {{ color: #9eaaaf; font-size: 12px; margin-bottom: 16px; }}
.section {{ border-top: 1px solid #30383c; padding: 14px 0; }}
.section-title {{ color: #aeb8bc; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 9px; }}
input[type=range] {{ width: 100%; }}
.stats {{ display: grid; grid-template-columns: 1fr auto; gap: 7px 10px; font-size: 12px; }}
.details {{ font-size: 12px; line-height: 1.5; color: #cbd2d5; white-space: normal; overflow-wrap: anywhere; }}
.details strong {{ color: #fff; display: block; margin-bottom: 5px; }}
.canvas {{ position: relative; background: #0d1113; }}
#map {{ width: 100%; height: 100%; display: block; cursor: grab; }}
#map.dragging {{ cursor: grabbing; }}
.toolbar {{ position: absolute; top: 14px; right: 14px; display: flex; gap: 6px; background: #151a1de8; border: 1px solid #30383c; border-radius: 6px; padding: 5px; }}
.toolbar button {{ width: 32px; height: 30px; border: 0; color: #dce3e6; background: transparent; cursor: pointer; border-radius: 4px; }}
.toolbar button:hover {{ background: #293237; }}
.road {{ fill: none; stroke: #3a4448; stroke-width: 1.5; opacity: .55; }}
.lane-group {{ fill: none; stroke: #6f858d; stroke-width: 2.2; opacity: .9; cursor: pointer; }}
.lane-group.active {{ stroke: #6fc48f; stroke-width: 3.2; }}
.lane-group.mismatch {{ stroke: #e06969; stroke-width: 4; }}
.detector {{ fill: none; stroke: #f0b64d; stroke-width: 6; stroke-linecap: round; opacity: .82; cursor: pointer; }}
.vehicle {{ stroke: #0d1113; stroke-width: 1.2; }}
.vehicle.moving {{ fill: #62c6e8; }}
.vehicle.halting {{ fill: #f06969; }}
.vehicle.detector {{ stroke: #fff; stroke-width: 2; }}
.label {{ fill: #dce4e7; paint-order: stroke; stroke: #0d1113; stroke-width: 3px; pointer-events: none; }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>Detector Windows</h1>
    <div class="sub" id="cfg"></div>
    <div class="section">
      <div class="section-title">Timeline</div>
      <input id="timeline" type="range" min="0" max="0" value="0">
      <div class="stats" id="stats"></div>
    </div>
    <div class="section">
      <div class="section-title">Selection</div>
      <div class="details" id="details">Select a detector window, LaneGroup, or vehicle.</div>
    </div>
  </aside>
  <main class="canvas">
    <svg id="map" viewBox="0 0 1200 800"></svg>
    <div class="toolbar">
      <button id="zoom-in" title="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out">-</button>
      <button id="reset-view" title="Reset view">R</button>
    </div>
  </main>
</div>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const svg = document.getElementById('map');
const NS = 'http://www.w3.org/2000/svg';
const WIDTH = 1200, HEIGHT = 800, PAD = 55;
let zoom = 1, panX = 0, panY = 0, selected = null;
const laneGroups = new Map(data.laneGroups.map(item => [item.id, item]));
const vehicleById = new Map();
function xy(point) {{
  const b = data.bounds;
  const xRange = Math.max(1, b.maxX - b.minX);
  const yRange = Math.max(1, b.maxY - b.minY);
  return {{
    x: PAD + (point[0] - b.minX) / xRange * (WIDTH - PAD * 2),
    y: HEIGHT - PAD - (point[1] - b.minY) / yRange * (HEIGHT - PAD * 2)
  }};
}}
function path(points) {{
  if (!points.length) return '';
  return points.map((point, index) => {{
    const mapped = xy(point);
    return `${{index === 0 ? 'M' : 'L'}} ${{mapped.x}} ${{mapped.y}}`;
  }}).join(' ');
}}
function element(name, attrs = {{}}) {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}}
function currentSample() {{
  return data.samples[Number(document.getElementById('timeline').value)];
}}
function countFor(laneGroupId) {{
  const counts = currentSample().counts[String(laneGroupId)] || currentSample().counts[laneGroupId] || {{}};
  return counts;
}}
function hasMismatch(counts) {{
  return counts.featureCount !== counts.rawCount || counts.featureHalting !== counts.rawHalting;
}}
function render() {{
  svg.replaceChildren();
  vehicleById.clear();
  const viewport = element('g', {{transform: `translate(${{panX}} ${{panY}}) scale(${{zoom}})`}});
  svg.appendChild(viewport);
  const nodeScale = Math.max(.18, Math.min(.9, .9 / zoom));
  const labelScale = Math.max(.15, Math.min(.8, .8 / zoom));
  const roads = element('g');
  for (const road of data.roads) roads.appendChild(element('path', {{d: path(road.points), class: 'road'}}));
  viewport.appendChild(roads);
  const lanes = element('g');
  for (const laneGroup of data.laneGroups) {{
    const counts = countFor(laneGroup.id);
    const cls = hasMismatch(counts) ? 'lane-group mismatch' : counts.rawCount > 0 ? 'lane-group active' : 'lane-group';
    const line = element('path', {{d: path(laneGroup.points), class: cls}});
    line.addEventListener('click', event => selectLaneGroup(laneGroup, event));
    lanes.appendChild(line);
  }}
  viewport.appendChild(lanes);
  const detectors = element('g');
  for (const laneGroup of data.laneGroups) {{
    const line = element('path', {{d: path(laneGroup.detectorPoints), class: 'detector'}});
    line.addEventListener('click', event => selectLaneGroup(laneGroup, event));
    detectors.appendChild(line);
  }}
  viewport.appendChild(detectors);
  const vehicles = element('g');
  for (const vehicle of currentSample().vehicles) {{
    vehicleById.set(vehicle.id, vehicle);
    const point = xy([vehicle.x, vehicle.y]);
    const cls = `vehicle ${{vehicle.halting ? 'halting' : 'moving'}} ${{vehicle.inDetector ? 'detector' : ''}}`;
    const node = element('circle', {{cx: point.x, cy: point.y, r: 4.5 * nodeScale, class: cls}});
    node.addEventListener('click', event => selectVehicle(vehicle, event));
    vehicles.appendChild(node);
    if (vehicle.inDetector) {{
      const label = element('text', {{x: point.x + 6 * nodeScale, y: point.y - 6 * nodeScale, class: 'label', style: `font-size:${{8 * labelScale}}px`}});
      label.textContent = vehicle.id;
      vehicles.appendChild(label);
    }}
  }}
  viewport.appendChild(vehicles);
  updateStats();
  refreshSelectionDetails();
}}
function selectLaneGroup(laneGroup, event) {{
  event.stopPropagation();
  selected = {{type: 'laneGroup', id: laneGroup.id}};
  updateLaneGroupDetails(laneGroup);
}}
function updateLaneGroupDetails(laneGroup) {{
  const counts = countFor(laneGroup.id);
  const movements = data.movements.filter(m => m.inputLaneGroupId === laneGroup.id || m.outputLaneGroupId === laneGroup.id);
  const movementText = movements.slice(0, 12).map(m => `M${{m.id}} ${{m.inputLaneGroupId === laneGroup.id ? 'input' : 'output'}} @ ${{m.tls}}`).join('<br>');
  document.getElementById('details').innerHTML =
    `<strong>LaneGroup L${{laneGroup.id}}</strong>` +
    `Edges: ${{laneGroup.edgeIds.join(' -> ')}}<br>` +
    `Feature/raw count: ${{counts.featureCount ?? 0}} / ${{counts.rawCount ?? 0}}<br>` +
    `Feature/raw halting: ${{counts.featureHalting ?? 0}} / ${{counts.rawHalting ?? 0}}<br>` +
    `Vehicles in detector: ${{(counts.vehicleIds || []).join(', ') || 'none'}}<br><br>` +
    `${{movementText || 'No controlled movement uses this LaneGroup in the current graph.'}}`;
}}
function selectVehicle(vehicle, event) {{
  event.stopPropagation();
  selected = {{type: 'vehicle', id: vehicle.id}};
  updateVehicleDetails(vehicle);
}}
function updateVehicleDetails(vehicle) {{
  document.getElementById('details').innerHTML =
    `<strong>Vehicle ${{vehicle.id}}</strong>` +
    `Lane: ${{vehicle.laneId}}<br>` +
    `Edge: ${{vehicle.edgeId}}<br>` +
    `LaneGroup: ${{vehicle.laneGroupId === null ? 'none' : `L${{vehicle.laneGroupId}}`}}<br>` +
    `Speed: ${{vehicle.speed.toFixed(2)}} m/s<br>` +
    `In detector window: ${{vehicle.inDetector ? 'yes' : 'no'}}`;
}}
function refreshSelectionDetails() {{
  if (!selected) return;
  if (selected.type === 'laneGroup') {{
    const laneGroup = laneGroups.get(selected.id);
    if (laneGroup) updateLaneGroupDetails(laneGroup);
    return;
  }}
  if (selected.type === 'vehicle') {{
    const vehicle = vehicleById.get(selected.id);
    if (vehicle) {{
      updateVehicleDetails(vehicle);
    }} else {{
      document.getElementById('details').innerHTML =
        `<strong>Vehicle ${{selected.id}}</strong>` +
        `Not present in the current timestep.`;
    }}
  }}
}}
function updateStats() {{
  const sample = currentSample();
  const totalDetectorVehicles = Object.values(sample.counts).reduce((total, counts) => total + (counts.rawCount || 0), 0);
  const mismatches = Object.values(sample.counts).filter(hasMismatch).length;
  document.getElementById('stats').innerHTML =
    `<span>Step</span><b>${{sample.step}}s</b>` +
    `<span>Vehicles</span><b>${{sample.vehicles.length}}</b>` +
    `<span>In detector windows</span><b>${{totalDetectorVehicles}}</b>` +
    `<span>Count mismatches</span><b>${{mismatches}}</b>`;
}}
function zoomAt(screenX, screenY, factor) {{
  const graphX = (screenX - panX) / zoom;
  const graphY = (screenY - panY) / zoom;
  zoom = Math.max(.35, Math.min(12, zoom * factor));
  panX = screenX - graphX * zoom;
  panY = screenY - graphY * zoom;
  render();
}}
document.getElementById('timeline').max = Math.max(0, data.samples.length - 1);
document.getElementById('timeline').oninput = render;
document.getElementById('cfg').textContent = data.cfg;
document.getElementById('zoom-in').onclick = () => zoomAt(WIDTH / 2, HEIGHT / 2, 1.25);
document.getElementById('zoom-out').onclick = () => zoomAt(WIDTH / 2, HEIGHT / 2, .8);
document.getElementById('reset-view').onclick = () => {{ zoom = 1; panX = 0; panY = 0; render(); }};
svg.addEventListener('wheel', event => {{
  event.preventDefault();
  const rect = svg.getBoundingClientRect();
  zoomAt((event.clientX - rect.left) / rect.width * WIDTH, (event.clientY - rect.top) / rect.height * HEIGHT, event.deltaY < 0 ? 1.18 : .85);
}}, {{passive: false}});
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
svg.addEventListener('click', () => {{
  selected = null;
  document.getElementById('details').textContent = 'Select a detector window, LaneGroup, or vehicle.';
}});
render();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize movement detector windows and sampled vehicle counts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', type=Path, required=True, help='SUMO .sumocfg path')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='Output HTML path')
    parser.add_argument('--steps', type=int, default=120, help='Simulation seconds to record')
    parser.add_argument('--sample-every', type=int, default=1, help='Seconds between visual samples')
    parser.add_argument('--demand-scale', type=float, default=1.0, help='Runtime demand scale')
    parser.add_argument('--seed', type=int, default=42, help='SUMO random seed')
    parser.add_argument(
        '--time-to-teleport',
        type=int,
        default=-1,
        help='SUMO gridlock teleport timeout in seconds; use -1 to disable gridlock teleporting',
    )
    parser.add_argument('--open', action='store_true', help='Open generated HTML in the default browser')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_visualization(
        cfg_path=args.cfg,
        output_path=args.out,
        steps=args.steps,
        sample_every=args.sample_every,
        demand_scale=args.demand_scale,
        seed=args.seed,
        time_to_teleport=args.time_to_teleport,
    )
    print(f'Wrote {output_path.resolve()}')
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == '__main__':
    main()
