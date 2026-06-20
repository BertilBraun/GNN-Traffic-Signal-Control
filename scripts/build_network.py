"""Build a SUMO network from OSM for movement-based signal control.

Pipeline
--------
1. Download OSM via Overpass API (--bbox) or use an existing file (--osm)
2. netconvert  — imports OSM with sane defaults
3. Plain-XML round-trip — aggressively merge pass-through nodes (both
   one-way 1-in/1-out and two-way 2-in/2-out cases), which netconvert's
   --geometry.remove refuses to merge when adjacent edges differ in lane
   count / speed / name
4. Optionally promote 3+arm junctions to traffic_light via --tls.set
5. Audit       — junction-arm-count table
6. TLL         — movement-safe conflict-synthesized signal programs
7. Additional  — empty file, matching generated grids
8. sumocfg     — ties everything together
9. Verify      — launches SUMO-GUI with the movement max-pressure runner

Usage
-----
    python scripts/build_network.py ^
        --bbox 48.147,11.568,48.155,11.581 ^
        --out-dir configs/city ^
        --verify

    python scripts/build_network.py ^
        --osm configs/city/city.osm ^
        --out-dir configs/city ^
        --verify
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

import sumolib  # noqa: E402  (needs SUMO_HOME on path first)

from src.movement.phase_synthesis import (  # noqa: E402
    TrafficLightLinkSpec,
    synthesize_traffic_light_program,
)
from src.movement.schema import LaneId, TrafficLightId  # noqa: E402

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

YELLOW_DUR = 3
ALLRED_DUR = 2
GREEN_DUR = 25
DEFAULT_ROUTE_COUNT = 300
DEFAULT_DEMAND_VEHICLES_PER_HOUR = 900.0
ROUTE_SAMPLE_ATTEMPTS_PER_REQUESTED_ROUTE = 30

TL_TYPES = {'traffic_light', 'traffic_light_unregulated', 'traffic_light_right_on_red'}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Build a SUMO network from OSM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        '--bbox',
        metavar='S,W,N,E',
        help='Bounding box to download from Overpass (south,west,north,east)',
    )
    src.add_argument(
        '--osm',
        metavar='FILE',
        help='Path to an existing .osm file',
    )
    p.add_argument(
        '--out-dir',
        required=True,
        metavar='DIR',
        help='Output directory (will be created if needed)',
    )
    p.add_argument(
        '--name',
        default=None,
        metavar='NAME',
        help='Base filename for outputs (default: out-dir folder name)',
    )
    p.add_argument(
        '--join-dist',
        type=float,
        default=40.0,
        metavar='M',
        help='Join junctions within this distance (metres)',
    )
    p.add_argument(
        '--verify',
        action='store_true',
        help='Launch SUMO-GUI with movement max-pressure after build',
    )
    p.add_argument(
        '--promote-all-junctions-to-tl',
        action='store_true',
        help='Promote every 3+-arm road junction to traffic_light instead of using OSM/netconvert signals only',
    )
    p.add_argument(
        '--route-count',
        type=int,
        default=DEFAULT_ROUTE_COUNT,
        metavar='N',
        help='Maximum deterministic city O-D routes written to the route file',
    )
    p.add_argument(
        '--demand-vehicles-per-hour',
        type=float,
        default=DEFAULT_DEMAND_VEHICLES_PER_HOUR,
        metavar='VPH',
        help='Total background demand distributed across generated city O-D routes',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — OSM download
# ---------------------------------------------------------------------------


def _download_osm(bbox: str, out_path: Path) -> None:
    south, west, north, east = bbox.split(',')
    query = f"""
[out:xml][timeout:90];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street)$"]({south},{west},{north},{east});
  node(w)({south},{west},{north},{east});
);
out body;
>;
out skel qt;
""".strip()

    print(f'  Querying Overpass API for bbox {bbox} ...')
    payload = urllib.parse.urlencode({'data': query}).encode('utf-8')
    req = urllib.request.Request(OVERPASS_URL, data=payload, method='POST')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    req.add_header('User-Agent', 'SUMO-GNN-research/1.0')
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
    out_path.write_bytes(content)
    print(f'  Saved {out_path}  ({out_path.stat().st_size // 1024} kB)')


# ---------------------------------------------------------------------------
# Step 2 — netconvert + plain-XML cleanup
# ---------------------------------------------------------------------------


def _run_netconvert(
    osm_path: Path,
    net_path: Path,
    join_dist: float,
    promote_all_junctions_to_tl: bool,
) -> None:
    netconvert = os.path.join(SUMO_HOME, 'bin', 'netconvert')
    _netconvert_from_osm(osm_path, net_path, join_dist, netconvert)
    _plain_xml_cleanup(net_path, join_dist, netconvert)
    if promote_all_junctions_to_tl:
        _promote_junctions_to_tl(net_path, netconvert, join_dist)
        # TL promotion + tls.join in the rebuild may surface new phantom TLs and pass-throughs.
        _plain_xml_cleanup(net_path, join_dist, netconvert)
    _final_phantom_tl_demote(net_path)
    print(f'  Wrote {net_path}')


def _netconvert_from_osm(osm_path: Path, net_path: Path, join_dist: float, netconvert: str) -> None:
    cmd = [
        netconvert,
        '--osm-files',
        str(osm_path),
        '--output-file',
        str(net_path),
        '--no-turnarounds',
        'true',
        '--geometry.remove',
        'true',
        '--junctions.join',
        'true',
        '--junctions.join-dist',
        str(join_dist),
        '--tls.join',
        'true',
        '--tls.join-dist',
        str(join_dist),
        '--tls.guess-signals',
        'true',
        '--remove-edges.by-vclass',
        'pedestrian',
        '--osm.sidewalks',
        'false',
        '--osm.crossings',
        'false',
    ]
    print('  Running netconvert (OSM import) ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f'netconvert OSM import failed (exit {result.returncode})')


def _plain_xml_cleanup(net_path: Path, join_dist: float, netconvert: str) -> None:
    """Round-trip through plain XML to forcibly merge pass-through nodes.

    netconvert's --geometry.remove preserves nodes when adjacent edges have
    different attributes (lane count, speed, priority, name). For our use
    case those distinctions are noise — we want a clean graph of real
    intersections, so we merge the nodes ourselves at the plain-XML level
    (which has no internal junctions or TL link indices to corrupt) and
    then let netconvert regenerate connections from scratch.
    """
    tmpdir = net_path.parent / '_plain_tmp'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir()
    try:
        prefix = tmpdir / 'plain'
        _export_plain_xml(net_path, prefix, netconvert)

        nod_file = Path(f'{prefix}.nod.xml')
        edg_file = Path(f'{prefix}.edg.xml')
        con_file = Path(f'{prefix}.con.xml')
        tll_file = Path(f'{prefix}.tll.xml')

        # Drop pre-computed connections + tlLogic; netconvert regenerates
        # them from edge geometry on rebuild, which is what we want after
        # merging arbitrary node pairs.
        for f in (con_file, tll_file):
            if f.exists():
                f.unlink()

        stripped = _strip_tl_join_metadata_in_plain(nod_file)
        if stripped:
            print(f'  Stripped shared-TL metadata from {stripped} nodes')

        demoted = _demote_phantom_tls_in_plain(nod_file, edg_file)
        if demoted:
            print(f'  Demoted {demoted} phantom TL nodes to priority')

        roundabouts = _remove_roundabouts_in_plain(edg_file)
        if roundabouts:
            print(f'  Removed {roundabouts} stale roundabout metadata entries')

        merged_total, loop_total, spur_total = 0, 0, 0
        for _ in range(20):
            spurs = _remove_dead_end_spurs_in_plain(nod_file, edg_file)
            merged = _merge_passthroughs_in_plain(nod_file, edg_file)
            loops = _remove_loop_stubs_in_plain(nod_file, edg_file)
            merged_total += merged
            loop_total += loops
            spur_total += spurs
            if merged == 0 and loops == 0 and spurs == 0:
                break
        if merged_total:
            print(f'  Merged {merged_total} pass-through nodes')
        if loop_total:
            print(f'  Removed {loop_total} loop-stub nodes')
        if spur_total:
            print(f'  Removed {spur_total} dead-end spur edges')

        joined = _add_join_directives_for_short_tl_edges(nod_file, edg_file)
        if joined:
            print(f'  Wrote <join> directives covering {joined} TL nodes (short-edge cluster)')

        _rebuild_from_plain(prefix, net_path, join_dist, netconvert)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _export_plain_xml(net_path: Path, prefix: Path, netconvert: str) -> None:
    cmd = [
        netconvert,
        '--sumo-net-file',
        str(net_path),
        '--plain-output-prefix',
        str(prefix),
    ]
    print('  Exporting net.xml to plain XML ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f'plain-XML export failed (exit {result.returncode})')


def _rebuild_from_plain(prefix: Path, net_path: Path, join_dist: float, netconvert: str) -> None:
    nod_file = Path(f'{prefix}.nod.xml')
    edg_file = Path(f'{prefix}.edg.xml')
    cmd = [
        netconvert,
        '--node-files',
        str(nod_file),
        '--edge-files',
        str(edg_file),
        '--output-file',
        str(net_path),
        '--no-turnarounds',
        'true',
        '--geometry.remove',
        'true',
        '--junctions.join',
        'true',
        '--junctions.join-dist',
        str(join_dist),
    ]
    print('  Running netconvert (rebuild from cleaned plain XML) ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f'netconvert rebuild failed (exit {result.returncode})')


def _remove_roundabouts_in_plain(edg_file: Path) -> int:
    """Remove plain-XML roundabout metadata before topology rewrites.

    Plain export stores roundabouts as edge-id lists inside ``*.edg.xml``.
    After pass-through merging, spur removal, or loop-stub removal, those
    lists can reference edge IDs that no longer exist, causing netconvert's
    rebuild to fail with "Unknown edge ... in roundabout".  The rebuilt net
    does not rely on these hints for our signal-control workflow.
    """
    edges_tree = ET.parse(str(edg_file))
    edges_root = edges_tree.getroot()
    removed = 0
    for elem in list(edges_root.findall('roundabout')):
        edges_root.remove(elem)
        removed += 1
    if removed:
        edges_tree.write(str(edg_file), encoding='utf-8', xml_declaration=True)
    return removed


def _strip_tl_join_metadata_in_plain(nod_file: Path) -> int:
    """Remove `tl` and `controlledInner` attributes from TL nodes.

    These hold shared program IDs from a previous --tls.join run (e.g.
    `tl="joinedS_A_B"`). If we leave them in place, netconvert recreates the
    shared program on rebuild, and our per-junction TLL builder writes
    programs SUMO can't match — junction A's tl reference is the joined ID,
    not 'A', so the additional-files load fails with 'no initial signal plan'.
    """
    nodes_tree = ET.parse(str(nod_file))
    nodes_root = nodes_tree.getroot()
    stripped = 0
    for node in nodes_root.findall('node'):
        if node.get('type') not in TL_TYPES:
            continue
        changed = False
        for attr in ('tl', 'controlledInner'):
            if attr in node.attrib:
                del node.attrib[attr]
                changed = True
        if changed:
            stripped += 1
    if stripped:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
    return stripped


def _demote_phantom_tls_in_plain(nod_file: Path, edg_file: Path) -> int:
    """Demote TL nodes with ≤ 2 unique neighbors. They're road points,
    not real intersections, regardless of any OSM signal tag."""
    nodes_tree = ET.parse(str(nod_file))
    edges_tree = ET.parse(str(edg_file))
    nodes_root = nodes_tree.getroot()

    neighbors_by_node: dict = {n.get('id'): set() for n in nodes_root.findall('node')}
    for e in edges_tree.getroot().findall('edge'):
        fr, to = e.get('from'), e.get('to')
        if to in neighbors_by_node:
            neighbors_by_node[to].add(fr)
        if fr in neighbors_by_node:
            neighbors_by_node[fr].add(to)

    demoted = 0
    for node in nodes_root.findall('node'):
        if node.get('type') not in TL_TYPES:
            continue
        nid = node.get('id')
        unique = neighbors_by_node[nid] - {nid}
        if len(unique) <= 2:
            node.set('type', 'priority')
            # Drop tlLogic reference so the rebuild doesn't re-promote it
            if 'tl' in node.attrib:
                del node.attrib['tl']
            demoted += 1

    if demoted:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
    return demoted


def _merge_passthroughs_in_plain(nod_file: Path, edg_file: Path) -> int:
    """Merge pass-through nodes: a node where connected edges form one
    continuous road, not a real intersection.

    Cases handled:
    - One-way through-road: 1 in, 1 out, two distinct neighbors
    - Two-way through-road: 2 in, 2 out, exactly two distinct neighbors,
      each neighbor contributing one incoming and one outgoing edge
    """
    nodes_tree = ET.parse(str(nod_file))
    edges_tree = ET.parse(str(edg_file))
    nodes_root = nodes_tree.getroot()
    edges_root = edges_tree.getroot()

    nodes_by_id = {n.get('id'): n for n in nodes_root.findall('node')}
    in_by_node: dict = {nid: [] for nid in nodes_by_id}
    out_by_node: dict = {nid: [] for nid in nodes_by_id}
    for e in edges_root.findall('edge'):
        if e.get('to') in in_by_node:
            in_by_node[e.get('to')].append(e)
        if e.get('from') in out_by_node:
            out_by_node[e.get('from')].append(e)

    merged = 0
    for nid, node in list(nodes_by_id.items()):
        if node.get('type') in TL_TYPES:
            continue
        ins = in_by_node[nid]
        outs = out_by_node[nid]
        if not ins or not outs:
            continue

        neighbors = ({e.get('from') for e in ins} | {e.get('to') for e in outs}) - {nid}
        if len(neighbors) != 2:
            continue
        a, b = sorted(neighbors)

        in_a = [e for e in ins if e.get('from') == a]
        in_b = [e for e in ins if e.get('from') == b]
        out_a = [e for e in outs if e.get('to') == a]
        out_b = [e for e in outs if e.get('to') == b]

        # Each direction's incoming count must match its outgoing count, so
        # every through-flow can be paired. This handles both 1-each (clean
        # 2-way road) and parallel-edge cases (multi-edge bidirectional)
        # while still rejecting asymmetric Y-merges/Y-splits.
        if len(in_a) != len(out_b) or len(in_b) != len(out_a):
            continue
        if len(in_a) + len(in_b) != len(ins):
            continue
        if len(out_a) + len(out_b) != len(outs):
            continue
        if not ins and not outs:
            continue

        # Pair edges arbitrarily within each direction. The graph is
        # topologically equivalent regardless of pairing.
        pairs: list[tuple] = list(zip(in_a, out_b)) + list(zip(in_b, out_a))
        if not pairs:
            continue

        for in_edge, out_edge in pairs:
            new_to = out_edge.get('to')
            in_edge.set('to', new_to)

            shape_in = (in_edge.get('shape') or '').strip()
            shape_out = (out_edge.get('shape') or '').strip()
            if shape_in and shape_out:
                pts_in = shape_in.split()
                pts_out = shape_out.split()
                merged_pts = pts_in + pts_out[1:]
                in_edge.set('shape', ' '.join(merged_pts))
            elif shape_out:
                in_edge.set('shape', shape_out)

            try:
                in_lanes = int(in_edge.get('numLanes', '1'))
                out_lanes = int(out_edge.get('numLanes', '1'))
                if out_lanes > in_lanes:
                    in_edge.set('numLanes', str(out_lanes))
            except ValueError:
                pass

            edges_root.remove(out_edge)
            if new_to in in_by_node:
                in_by_node[new_to] = [e for e in in_by_node[new_to] if e is not out_edge]
                in_by_node[new_to].append(in_edge)

        nodes_root.remove(node)
        del nodes_by_id[nid]
        del in_by_node[nid]
        del out_by_node[nid]
        merged += 1

    if merged:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
        edges_tree.write(str(edg_file), encoding='utf-8', xml_declaration=True)
    return merged


def _remove_loop_stubs_in_plain(nod_file: Path, edg_file: Path) -> int:
    """Remove A→N→A loop stubs left by junction joining."""
    nodes_tree = ET.parse(str(nod_file))
    edges_tree = ET.parse(str(edg_file))
    nodes_root = nodes_tree.getroot()
    edges_root = edges_tree.getroot()

    nodes_by_id = {n.get('id'): n for n in nodes_root.findall('node')}
    in_by_node: dict = {nid: [] for nid in nodes_by_id}
    out_by_node: dict = {nid: [] for nid in nodes_by_id}
    for e in edges_root.findall('edge'):
        if e.get('to') in in_by_node:
            in_by_node[e.get('to')].append(e)
        if e.get('from') in out_by_node:
            out_by_node[e.get('from')].append(e)

    removed = 0
    deleted_edges: set[int] = set()
    deleted_nodes: set[int] = set()
    for nid, node in list(nodes_by_id.items()):
        ins = in_by_node[nid]
        outs = out_by_node[nid]
        if len(ins) != 1 or len(outs) != 1:
            continue
        e_in, e_out = ins[0], outs[0]
        if e_in.get('from') != e_out.get('to') or e_in.get('from') == nid:
            continue
        if id(node) in deleted_nodes:
            continue
        for e in (e_in, e_out):
            if id(e) not in deleted_edges:
                edges_root.remove(e)
                deleted_edges.add(id(e))
        nodes_root.remove(node)
        deleted_nodes.add(id(node))
        del nodes_by_id[nid]
        removed += 1

    if removed:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
        edges_tree.write(str(edg_file), encoding='utf-8', xml_declaration=True)
    return removed


def _edge_length(edge_elem, nodes_by_id: dict) -> float:
    shape = (edge_elem.get('shape') or '').strip()
    if shape:
        try:
            pts = [tuple(map(float, p.split(','))) for p in shape.split()]
        except ValueError:
            return float('inf')
    else:
        fr = nodes_by_id.get(edge_elem.get('from'))
        to = nodes_by_id.get(edge_elem.get('to'))
        if fr is None or to is None:
            return float('inf')
        try:
            pts = [
                (float(fr.get('x')), float(fr.get('y'))),
                (float(to.get('x')), float(to.get('y'))),
            ]
        except (TypeError, ValueError):
            return float('inf')
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _add_join_directives_for_short_tl_edges(
    nod_file: Path,
    edg_file: Path,
    max_edge_len: float = 10.0,
    max_centroid_dist: float = 30.0,
) -> int:
    """Emit <join nodes="A B ..."/> elements for clusters of TL nodes linked
    by very short edges, so netconvert's own clustering merges them with
    correct geometry.

    Joins only when both the connecting edge AND the junction centroids are
    close. A 3 m edge between two centroids 58 m apart means the junctions
    each cover wide areas in opposite directions — joining them produces a
    100m+ elongated junction box with absurdly long internal lanes.
    """
    nodes_tree = ET.parse(str(nod_file))
    edges_tree = ET.parse(str(edg_file))
    nodes_root = nodes_tree.getroot()
    edges_root = edges_tree.getroot()

    nodes_by_id = {n.get('id'): n for n in nodes_root.findall('node')}

    def _centroid_distance(a_node, b_node) -> float:
        try:
            ax, ay = float(a_node.get('x')), float(a_node.get('y'))
            bx, by = float(b_node.get('x')), float(b_node.get('y'))
        except (TypeError, ValueError):
            return float('inf')
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    parent: dict = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(a, b):
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges_root.findall('edge'):
        if _edge_length(e, nodes_by_id) > max_edge_len:
            continue
        fr = nodes_by_id.get(e.get('from'))
        to = nodes_by_id.get(e.get('to'))
        if fr is None or to is None:
            continue
        if fr.get('type') not in TL_TYPES or to.get('type') not in TL_TYPES:
            continue
        if _centroid_distance(fr, to) > max_centroid_dist:
            continue
        union(e.get('from'), e.get('to'))

    if not parent:
        return 0

    clusters: dict = {}
    for nid in parent:
        clusters.setdefault(find(nid), []).append(nid)

    covered = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        join_elem = ET.SubElement(nodes_root, 'join', nodes=' '.join(sorted(members)))
        join_elem.tail = '\n'
        covered += len(members)

    if covered:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
    return covered


def _remove_dead_end_spurs_in_plain(nod_file: Path, edg_file: Path, max_len: float = 8.0) -> int:
    """Remove short edges ending at a node with no outgoing edges.

    These short spurs (driveways, OSM artefacts) confuse the pass-through
    detector: a node with one incoming road and two outgoing edges (one real,
    one spur to a dead-end) looks like a Y-split instead of a clean
    pass-through, so the merge skips it.
    """
    nodes_tree = ET.parse(str(nod_file))
    edges_tree = ET.parse(str(edg_file))
    nodes_root = nodes_tree.getroot()
    edges_root = edges_tree.getroot()

    nodes_by_id = {n.get('id'): n for n in nodes_root.findall('node')}
    in_by_node: dict = {nid: [] for nid in nodes_by_id}
    out_by_node: dict = {nid: [] for nid in nodes_by_id}
    for e in edges_root.findall('edge'):
        if e.get('to') in in_by_node:
            in_by_node[e.get('to')].append(e)
        if e.get('from') in out_by_node:
            out_by_node[e.get('from')].append(e)

    removed = 0
    for nid, node in list(nodes_by_id.items()):
        if out_by_node[nid]:
            continue  # has onward connectivity — not a dead end
        ins = in_by_node[nid]
        if len(ins) != 1:
            continue  # multiple incoming dead-ends are unusual; leave alone
        e_in = ins[0]
        if _edge_length(e_in, nodes_by_id) > max_len:
            continue
        edges_root.remove(e_in)
        nodes_root.remove(node)
        del nodes_by_id[nid]
        removed += 1

    if removed:
        nodes_tree.write(str(nod_file), encoding='utf-8', xml_declaration=True)
        edges_tree.write(str(edg_file), encoding='utf-8', xml_declaration=True)
    return removed


def _promote_junctions_to_tl(net_path: Path, netconvert: str, join_dist: float) -> None:
    """Promote junctions with 3+ unique neighbors to traffic_light via --tls.set.

    Uses unique-neighbor count rather than edge count: a T-intersection of
    one-way roads has 2 incoming + 2 outgoing edges but 3 distinct neighbors,
    and netconvert sometimes types these as `right_before_left` instead of
    `priority` — both must be promoted.
    """
    net = sumolib.net.readNet(str(net_path), withConnections=True)
    to_promote: list[str] = []
    for node in net.getNodes():
        if node.getType() in TL_TYPES:
            continue
        neighbors = {e.getFromNode().getID() for e in node.getIncoming()} | {
            e.getToNode().getID() for e in node.getOutgoing()
        }
        neighbors.discard(node.getID())
        if len(neighbors) >= 3:
            to_promote.append(node.getID())

    if not to_promote:
        print('  All 3+-arm junctions already TL-controlled')
        return

    print(f'  Promoting {len(to_promote)} junctions to traffic_light via --tls.set ...')
    cmd = [
        netconvert,
        '--sumo-net-file',
        str(net_path),
        '--output-file',
        str(net_path),
        '--tls.set',
        ','.join(to_promote),
        '--tls.join',
        'true',
        '--tls.join-dist',
        str(join_dist),
        '--no-turnarounds',
        'true',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f'netconvert TL promotion failed (exit {result.returncode})')


def _final_phantom_tl_demote(net_path: Path) -> None:
    """Defensive: demote any TL junction with <3 arms in the final net.xml.

    The plain-XML round-trip should leave the network clean, but this catches
    any TL that slips through (e.g., a junction whose arm count changed in
    the rebuild). Pure text substitution preserves file formatting.
    """
    net = sumolib.net.readNet(str(net_path), withConnections=True)
    content = net_path.read_text(encoding='utf-8')

    demoted: list[str] = []
    for node in net.getNodes():
        if node.getType() not in TL_TYPES:
            continue
        neighbors = {e.getFromNode().getID() for e in node.getIncoming()} | {
            e.getToNode().getID() for e in node.getOutgoing()
        }
        neighbors.discard(node.getID())
        # Demote only if not a real intersection (fewer than 3 distinct
        # neighbors). Edge count alone is misleading — a T-junction of
        # one-way roads has 2 in + 2 out but 3 distinct neighbors.
        if len(neighbors) < 3:
            demoted.append(node.getID())

    for jid in demoted:
        content = re.sub(
            rf'(<junction\b(?=[^>]*\bid="{re.escape(jid)}")[^>]*)type="traffic_light[^"]*"',
            r'\1type="priority"',
            content,
        )

    if demoted:
        net_path.write_text(content, encoding='utf-8')
        print(f'  Final demotion: {len(demoted)} junctions back to priority')


# ---------------------------------------------------------------------------
# Step 3 — audit
# ---------------------------------------------------------------------------


def _audit(net) -> tuple[list[str], list[str], list[tuple[str, int]]]:
    three_way, four_way, other = [], [], []
    for node in sorted(net.getNodes(), key=lambda n: n.getID()):
        if node.getType() != 'traffic_light':
            continue
        n_in = len(node.getIncoming())
        if n_in == 3:
            three_way.append(node.getID())
        elif n_in == 4:
            four_way.append(node.getID())
        else:
            other.append((node.getID(), n_in))

    print(f'  3-way TL junctions : {len(three_way)}')
    print(f'  4-way TL junctions : {len(four_way)}')
    print(f'  Other TL arm counts: {len(other)}')
    for jid, n in other:
        print(f'    {jid}  ({n} arms)')
    return three_way, four_way, other


# ---------------------------------------------------------------------------
# Step 4 — movement TLL builder
# ---------------------------------------------------------------------------


def _all_tl_junction_ids(net_path: Path) -> set[str]:
    """Parse net.xml directly for every junction with type traffic_light*."""
    ids = set()
    for _, elem in ET.iterparse(str(net_path), events=('start',)):
        if elem.tag == 'junction' and 'traffic_light' in elem.get('type', ''):
            ids.add(elem.get('id'))
    return ids


def _link_count(node) -> int:
    indices = {
        conn.getTLLinkIndex()
        for in_edge in node.getIncoming()
        for out_edge in in_edge.getOutgoing()
        for conn in in_edge.getConnections(out_edge)
        if conn.getTLLinkIndex() >= 0
    }
    return max(indices) + 1 if indices else 0


def _build_tll(net, net_path: Path, tll_path: Path) -> int:
    root = ET.Element('additional')
    written = 0
    skipped: list[tuple[str, str]] = []

    for node in tqdm.tqdm(sorted(net.getNodes(), key=lambda n: n.getID()), total=len(net.getNodes())):
        if node.getType() not in TL_TYPES:
            continue
        jid = node.getID()
        link_specs = _movement_link_specs(node)
        if not link_specs:
            skipped.append((jid, 'no SUMO controlled links'))
            continue
        duplicate_signal_indices = _duplicate_signal_indices(link_specs)
        if duplicate_signal_indices:
            skipped.append(
                (
                    jid,
                    'shared traffic-light link indices are not supported by city TLL synthesis: '
                    f'{duplicate_signal_indices}',
                )
            )
            continue
        try:
            synthesized_program = synthesize_traffic_light_program(
                traffic_light_id=TrafficLightId(jid),
                links=link_specs,
                are_foes=node.areFoes,
            )
        except ValueError as exc:
            skipped.append((jid, str(exc)))
            continue
        if not synthesized_program.selectable_phases:
            skipped.append((jid, 'movement synthesis produced no selectable phases'))
            continue

        n_links = max(link.traffic_light_link_index for link in link_specs) + 1
        allred = 'r' * n_links
        tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='movement_safe', offset='0')
        for phase in synthesized_program.selectable_phases:
            green = str(phase.state)
            ET.SubElement(tll, 'phase', duration=str(GREEN_DUR), state=green)
            ET.SubElement(tll, 'phase', duration=str(YELLOW_DUR), state=_yellow_state(green))
            ET.SubElement(tll, 'phase', duration=str(ALLRED_DUR), state=allred)
        written += 1

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
    xml_str = '\n'.join(xml_str.split('\n')[1:])
    tll_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str, encoding='utf-8')
    print(f'  Wrote {written} movement-safe traffic-light programs -> {tll_path}')
    for jid, reason in skipped:
        print(f'  SKIP {jid}: {reason}')
    missing = sorted(_all_tl_junction_ids(net_path) - {str(logic.get('id')) for logic in root.findall('tlLogic')})
    if missing:
        print(f'  {len(missing)} traffic lights rely on net.xml default programs until inspected/demoted')
    return written


def _movement_link_specs(node) -> list[TrafficLightLinkSpec]:
    specs: list[TrafficLightLinkSpec] = []
    for incoming in node.getIncoming():
        for outgoing in incoming.getOutgoing():
            for connection in incoming.getConnections(outgoing):
                link_index = int(connection.getTLLinkIndex())
                if link_index < 0:
                    continue
                request_index = int(connection.getJunctionIndex())
                specs.append(
                    TrafficLightLinkSpec(
                        traffic_light_link_index=link_index,
                        incoming_lane_id=LaneId(str(connection.getFromLane().getID())),
                        outgoing_lane_id=LaneId(str(connection.getToLane().getID())),
                        outgoing_edge_id=str(connection.getTo().getID()),
                        request_index=request_index if request_index >= 0 else None,
                    )
                )
    return sorted(specs, key=lambda spec: spec.traffic_light_link_index)


def _duplicate_signal_indices(link_specs: list[TrafficLightLinkSpec]) -> tuple[int, ...]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for link_spec in link_specs:
        if link_spec.traffic_light_link_index in seen:
            duplicates.add(link_spec.traffic_light_link_index)
        seen.add(link_spec.traffic_light_link_index)
    return tuple(sorted(duplicates))


def _yellow_state(green: str) -> str:
    return ''.join('y' if char in {'G', 'g'} else 'r' for char in green)


# ---------------------------------------------------------------------------
# Step 5 — additional file
# ---------------------------------------------------------------------------


def _write_additional(add_path: Path) -> None:
    _write_xml(add_path, ET.Element('additional'))
    print(f'  Wrote empty additional file -> {add_path}')


def _write_xml(path: Path, root: ET.Element) -> None:
    xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
    path.write_text(xml, encoding='utf-8')


# ---------------------------------------------------------------------------
# Step 6 — routes
# ---------------------------------------------------------------------------


def _write_routes(
    net,
    rou_path: Path,
    route_count: int,
    demand_vehicles_per_hour: float,
) -> int:
    routes = _city_routes(net, route_count=route_count)
    root = ET.Element('routes')
    ET.SubElement(
        root,
        'vType',
        {
            'id': 'car',
            'accel': '2.6',
            'decel': '4.5',
            'length': '5.0',
            'minGap': '2.5',
            'maxSpeed': '13.89',
        },
    )
    for route_index, edge_ids in enumerate(routes):
        route_id = f'city_route_{route_index}'
        ET.SubElement(root, 'route', {'id': route_id, 'edges': ' '.join(edge_ids)})
        ET.SubElement(
            root,
            'flow',
            {
                'id': f'city_flow_{route_index}',
                'type': 'car',
                'route': route_id,
                'begin': '0',
                'end': '3600',
                'vehsPerHour': f'{demand_vehicles_per_hour / max(len(routes), 1):.3f}',
                'departLane': 'best',
                'departSpeed': 'random',
            },
        )
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
    xml_str = '\n'.join(xml_str.split('\n')[1:])
    rou_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str, encoding='utf-8')
    print(f'  Wrote {len(routes)} city O-D flows -> {rou_path}')
    return len(routes)


def _city_routes(net, route_count: int) -> tuple[tuple[str, ...], ...]:
    if route_count <= 0:
        raise ValueError('route_count must be positive.')
    candidate_edges = _normal_edges(net)
    if not candidate_edges:
        raise ValueError('Network has no normal drivable edges for route generation.')
    routes: list[tuple[str, ...]] = []
    route_set: set[tuple[str, ...]] = set()
    edge_weights = tuple(_edge_storage_capacity(edge) for edge in candidate_edges)
    random_generator = random.Random(42)
    max_attempts = route_count * ROUTE_SAMPLE_ATTEMPTS_PER_REQUESTED_ROUTE
    for _attempt in range(max_attempts):
        source = random_generator.choices(candidate_edges, weights=edge_weights, k=1)[0]
        sink = random_generator.choices(candidate_edges, weights=edge_weights, k=1)[0]
        if source == sink:
            continue
        route = _shortest_route(net, source, sink)
        if route is None or route in route_set:
            continue
        route_set.add(route)
        routes.append(route)
        if len(routes) >= route_count:
            break
    if not routes:
        raise RuntimeError('No valid city O-D routes found. Inspect network connectivity before running policies.')
    return tuple(sorted(routes, key=lambda edge_ids: (edge_ids[0], edge_ids[-1], len(edge_ids), edge_ids)))


def _normal_edges(net) -> tuple[object, ...]:
    return tuple(
        edge
        for edge in net.getEdges()
        if not str(edge.getID()).startswith(':')
        and str(edge.getFunction()) == ''
        and float(edge.getLength()) > 0.0
        and int(edge.getLaneNumber()) > 0
    )


def _edge_storage_capacity(edge: object) -> float:
    return float(edge.getLength()) * float(edge.getLaneNumber())


def _shortest_route(net, source: object, sink: object) -> tuple[str, ...] | None:
    path, _cost = net.getOptimalPath(
        source,
        sink,
        fastest=False,
    )
    if path is None or len(path) < 2:
        return None
    return tuple(str(edge.getID()) for edge in path)


# ---------------------------------------------------------------------------
# Step 7 — sumocfg
# ---------------------------------------------------------------------------


def _write_sumocfg(cfg_path: Path, name: str) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{name}.net.xml"/>
        <route-files value="{name}.rou.xml"/>
        <additional-files value="{name}.tll.xml,{name}.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <report>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    cfg_path.write_text(xml, encoding='utf-8')
    print(f'  Wrote {cfg_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or out_dir.name

    osm_path = out_dir / f'{name}.osm'
    net_path = out_dir / f'{name}.net.xml'
    tll_path = out_dir / f'{name}.tll.xml'
    add_path = out_dir / f'{name}.add.xml'
    cfg_path = out_dir / f'{name}.sumocfg'
    rou_path = out_dir / f'{name}.rou.xml'

    print(f'\n{"=" * 62}')
    print(f'  Network : {name}')
    print(f'  Out dir : {out_dir}')
    print(f'{"=" * 62}\n')

    print('[1/7] OSM')
    if args.bbox:
        _download_osm(args.bbox, osm_path)
    else:
        osm_path = Path(args.osm)
        print(f'  Using existing {osm_path}')

    print('\n[2/7] netconvert')
    _run_netconvert(
        osm_path=osm_path,
        net_path=net_path,
        join_dist=args.join_dist,
        promote_all_junctions_to_tl=args.promote_all_junctions_to_tl,
    )

    print('\n[3/7] Audit')
    net = sumolib.net.readNet(str(net_path), withConnections=True, withFoes=True)
    _audit(net)
    usable = len([node for node in net.getNodes() if node.getType() in TL_TYPES])
    if usable == 0:
        raise RuntimeError('No traffic-light junctions found — check bbox or OSM data.')

    print('\n[4/7] Traffic light programs')
    _build_tll(net, net_path, tll_path)

    print('\n[5/7] Additional file')
    _write_additional(add_path)

    print('\n[6/7] Routes')
    _write_routes(
        net=net,
        rou_path=rou_path,
        route_count=args.route_count,
        demand_vehicles_per_hour=args.demand_vehicles_per_hour,
    )

    print('\n[7/7] Config')
    _write_sumocfg(cfg_path, name)

    print(f'\n{"=" * 62}')
    print(f'  Build complete.  Usable TL junctions: {usable}')
    print(f'  Config: {cfg_path}')
    print(f'{"=" * 62}\n')

    if args.verify:
        print('Launching SUMO-GUI with movement max-pressure ...\n')
        verify_cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'run.py'),
            '--cfg',
            str(cfg_path),
            '--method',
            'max-pressure',
            '--gui',
        ]
        subprocess.run(verify_cmd)


if __name__ == '__main__':
    main()
