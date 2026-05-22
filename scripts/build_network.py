"""Build a SUMO network from OSM and verify it visually with the greedy expert.

Pipeline
--------
1. Download OSM via Overpass API (--bbox) or use an existing file (--osm)
2. netconvert  — imports OSM with sane defaults
3. Plain-XML round-trip — aggressively merge pass-through nodes (both
   one-way 1-in/1-out and two-way 2-in/2-out cases), which netconvert's
   --geometry.remove refuses to merge when adjacent edges differ in lane
   count / speed / name
4. Promote 3+arm junctions to traffic_light via --tls.set
5. Audit       — junction-arm-count table
6. TLL         — canonical 8-phase signal programs
7. Detectors   — E2 laneAreaDetectors on incoming TL lanes (≤200 m)
8. sumocfg     — ties everything together
9. Verify      — launches SUMO-GUI with the greedy expert (--verify)

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
import math
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

import sumolib  # noqa: E402  (needs SUMO_HOME on path first)

from src.environment.phase_schema import NUM_PHASES, SLOT_DIR_PHASES  # noqa: E402

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

YELLOW_DUR = 3
ALLRED_DUR = 2
GREEN_DUR = 25
DET_NOMINAL = 200.0

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
        help='Launch SUMO-GUI with greedy expert after build',
    )
    p.add_argument(
        '--flow-range',
        nargs=2,
        type=int,
        default=[700, 1200],
        metavar=('MIN', 'MAX'),
        help='Traffic demand range for verify step (veh/h)',
    )
    p.add_argument(
        '--demand-min-rate',
        type=float,
        default=1.0,
        metavar='R',
        help='Min veh/h per O-D pair for verify step',
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


def _run_netconvert(osm_path: Path, net_path: Path, join_dist: float) -> None:
    netconvert = os.path.join(SUMO_HOME, 'bin', 'netconvert')
    _netconvert_from_osm(osm_path, net_path, join_dist, netconvert)
    _plain_xml_cleanup(net_path, join_dist, netconvert)
    _promote_junctions_to_tl(net_path, netconvert, join_dist)
    # Second cleanup pass: TL promotion + tls.join in the rebuild may surface
    # new phantom TLs and pass-throughs (e.g., TLs whose neighbors got merged).
    _plain_xml_cleanup(net_path, join_dist, netconvert)
    _final_phantom_tl_demote(net_path)
    _final_unsupported_tl_demote(net_path)
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
    import re

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


def _unsupported_tl_ids_by_incoming_count(net) -> list[str]:
    """Return TL node IDs that the current 3/4-arm representation cannot manage."""
    unsupported: list[str] = []
    for node in net.getNodes():
        if node.getType() not in TL_TYPES:
            continue
        if len(node.getIncoming()) not in (3, 4):
            unsupported.append(node.getID())
    return sorted(unsupported)


def _final_unsupported_tl_demote(net_path: Path) -> None:
    """Demote TLs that cannot be represented by TrafficEnv.

    Leaving 1/2/5+-incoming-arm TLs as signalized nodes means SUMO runs them
    with stub programs while the model and expert ignore them.  That creates
    unmanaged signal behavior in evaluation and training.  If the current
    representation cannot control a junction, keep it as priority traffic.
    """
    import re

    net = sumolib.net.readNet(str(net_path), withConnections=True)
    demoted = _unsupported_tl_ids_by_incoming_count(net)
    if not demoted:
        return

    content = net_path.read_text(encoding='utf-8')
    for jid in demoted:
        content = re.sub(
            rf'(<junction\b(?=[^>]*\bid="{re.escape(jid)}")[^>]*)type="traffic_light[^"]*"',
            r'\1type="priority"',
            content,
        )
    net_path.write_text(content, encoding='utf-8')
    print(f'  Final unsupported demotion: {len(demoted)} TL junctions back to priority')


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
    print(f'  Other (skipped)    : {len(other)}')
    for jid, n in other:
        print(f'    {jid}  ({n} arms)')
    return three_way, four_way, other


# ---------------------------------------------------------------------------
# Step 4 — TLL builder
# ---------------------------------------------------------------------------


def _bearing(from_xy: tuple, to_xy: tuple) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def _approach_bearing(edge) -> float:
    shape = edge.getShape()
    if len(shape) < 2:
        return 0.0
    p1, p2 = shape[-2], shape[-1]
    return _bearing(p2, p1)


def _bearing_diff(b1: float, b2: float) -> float:
    d = abs(b1 - b2) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _slot_order(node) -> list:
    incoming = list(node.getIncoming())
    n = len(incoming)
    if n == 4:
        return sorted(incoming, key=_approach_bearing)
    # 3-way: find the pair closest to 180° apart -> slots 0/2; stem -> slot 1
    bearings = [(_approach_bearing(e), e) for e in incoming]
    best_diff = float('inf')
    best_pair = (0, 1)
    for i in range(3):
        for j in range(i + 1, 3):
            d = _bearing_diff(bearings[i][0], bearings[j][0])
            diff = abs(d - 180.0)
            if diff < best_diff:
                best_diff = diff
                best_pair = (i, j)
    i, j = best_pair
    stem_idx = next(k for k in range(3) if k not in best_pair)
    b_i, b_j = bearings[i][0], bearings[j][0]
    if b_i <= b_j:
        slot0_edge, slot2_edge = bearings[i][1], bearings[j][1]
    else:
        slot0_edge, slot2_edge = bearings[j][1], bearings[i][1]
    return [slot0_edge, bearings[stem_idx][1], slot2_edge]


def _build_phase_strings(node) -> tuple[list[str], list[str], str] | None:
    incoming = list(node.getIncoming())
    if len(incoming) not in (3, 4):
        return None
    slots = _slot_order(node)
    slot_map = {e.getID(): i for i, e in enumerate(slots)}

    link_info: list[tuple[int, str, str]] = []
    for in_edge in node.getIncoming():
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                tl_idx = conn.getTLLinkIndex()
                if tl_idx < 0:
                    continue
                link_info.append((tl_idx, in_edge.getID(), conn.getDirection()))

    if not link_info:
        return None

    n_links = max(t[0] for t in link_info) + 1
    phase_states = [['r'] * n_links for _ in range(NUM_PHASES)]

    for tl_idx, from_edge, direction in link_info:
        slot_idx = slot_map.get(from_edge)
        if slot_idx is None:
            continue
        for ph, ch in SLOT_DIR_PHASES.get((slot_idx, direction.lower()), []):
            existing = phase_states[ph][tl_idx]
            if existing == 'r' or (existing == 'g' and ch == 'G'):
                phase_states[ph][tl_idx] = ch

    green_states = [''.join(s) for s in phase_states]
    yellow_states = [''.join('y' if c in 'Gg' else 'r' for c in s) for s in green_states]
    return green_states, yellow_states, 'r' * n_links


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


def _write_stub(root: ET.Element, jid: str, n_links: int) -> None:
    tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='canonical', offset='0')
    ET.SubElement(tll, 'phase', duration='30', state='G' * max(n_links, 1))
    ET.SubElement(tll, 'phase', duration='3', state='y' * max(n_links, 1))


def _build_tll(net, net_path: Path, tll_path: Path) -> int:
    root = ET.Element('additional')
    written = 0
    covered: set[str] = set()

    node_map = {n.getID(): n for n in net.getNodes()}

    for node in sorted(net.getNodes(), key=lambda n: n.getID()):
        if node.getType() not in TL_TYPES:
            continue
        jid = node.getID()
        n_in = len(node.getIncoming())

        if n_in not in (3, 4):
            _write_stub(root, jid, _link_count(node))
            covered.add(jid)
            continue

        result = _build_phase_strings(node)
        if result is None:
            _write_stub(root, jid, _link_count(node))
            covered.add(jid)
            print(f'  STUB {jid} (3/4-way, no controlled links)')
            continue

        greens, yellows, allred = result
        tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='canonical', offset='0')
        for g, y in zip(greens, yellows):
            ET.SubElement(tll, 'phase', duration=str(GREEN_DUR), state=g)
            ET.SubElement(tll, 'phase', duration=str(YELLOW_DUR), state=y)
            ET.SubElement(tll, 'phase', duration=str(ALLRED_DUR), state=allred)
        covered.add(jid)
        written += 1

    for jid in sorted(_all_tl_junction_ids(net_path) - covered):
        node = node_map.get(jid)
        n_links = _link_count(node) if node else 1
        _write_stub(root, jid, n_links)
        print(f'  STUB {jid} (gap-fill)')

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
    xml_str = '\n'.join(xml_str.split('\n')[1:])
    tll_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str, encoding='utf-8')
    print(f'  Wrote {written} canonical + stubs -> {tll_path}')
    return written


# ---------------------------------------------------------------------------
# Step 5 — detectors
# ---------------------------------------------------------------------------


def _build_detectors(net, add_path: Path) -> int:
    root = ET.Element('additional')
    n_det = 0
    for node in sorted(net.getNodes(), key=lambda n: n.getID()):
        if node.getType() != 'traffic_light':
            continue
        for edge in node.getIncoming():
            for lane in edge.getLanes():
                lane_len = lane.getLength()
                det_len = min(DET_NOMINAL, lane_len)
                pos = max(0.0, lane_len - det_len)
                ET.SubElement(
                    root,
                    'laneAreaDetector',
                    id=f'det_{lane.getID()}',
                    lane=lane.getID(),
                    pos=f'{pos:.2f}',
                    length=f'{det_len:.2f}',
                    freq='3600',
                    file='detector_out.xml',
                    friendlyPos='true',
                )
                n_det += 1
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
    xml_str = '\n'.join(xml_str.split('\n')[1:])
    add_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str, encoding='utf-8')
    print(f'  Wrote {n_det} detectors -> {add_path}')
    return n_det


# ---------------------------------------------------------------------------
# Step 6 — sumocfg
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

    print('[1/6] OSM')
    if args.bbox:
        _download_osm(args.bbox, osm_path)
    else:
        osm_path = Path(args.osm)
        print(f'  Using existing {osm_path}')

    print('\n[2/6] netconvert')
    _run_netconvert(osm_path, net_path, args.join_dist)

    print('\n[3/6] Audit')
    net = sumolib.net.readNet(str(net_path), withConnections=True)
    three_way, four_way, _ = _audit(net)
    usable = len(three_way) + len(four_way)
    if usable == 0:
        raise RuntimeError('No usable 3/4-way TL junctions found — check bbox or OSM data.')

    print('\n[4/6] Traffic light programs')
    _build_tll(net, net_path, tll_path)

    print('\n[5/6] Detectors')
    _build_detectors(net, add_path)

    print('\n[6/6] Config')
    _write_sumocfg(cfg_path, name)
    rou_path.write_text('<routes/>\n', encoding='utf-8')
    print(f'  Wrote placeholder {rou_path}')

    print(f'\n{"=" * 62}')
    print(f'  Build complete.  Usable TL junctions: {usable}')
    print(f'  Config: {cfg_path}')
    print(f'{"=" * 62}\n')

    if args.verify:
        print('Launching SUMO-GUI with greedy expert ...\n')
        verify_cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'run_grid.py'),
            '--mode',
            'expert',
            '--cfg',
            str(cfg_path),
            '--gui',
            '--flow-range',
            str(args.flow_range[0]),
            str(args.flow_range[1]),
            '--demand-min-rate',
            str(args.demand_min_rate),
        ]
        subprocess.run(verify_cmd)


if __name__ == '__main__':
    main()
