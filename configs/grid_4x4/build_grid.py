"""Build an irregular grid network for GNN traffic control experiments.

Usage
-----
    python build_grid.py                          # default 4x4 config
    python build_grid.py --size 5 --length 200    # 5x5 variant

Strategy
--------
Start from a regular NxN grid (all junctions signalised), then remove a
small set of *bidirectional road segments* (not whole nodes).  This turns
selected 4-way junctions into 3-way junctions while preserving:
  - Full network connectivity
  - Every junction remaining signalised (no uncontrolled pass-throughs)
  - A minimum of 3 incoming edges per junction

Rule: only remove an edge (A, B) when both A and B currently have > 3
incoming edges, so neither drops below 3 after removal.

Default removals for the 4x4 grid (12 TL junctions)
-----------------------------------------------------
  B1 <-> C1  (one vertical interior cut)

Result: 2 × 4-way (B2, C2) + 10 × 3-way  —  ~4 % of directed edges removed.

This script is importable; call build() directly for programmatic use.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

HERE = Path(__file__).parent
NETGEN = sumolib.checkBinary('netgenerate')
NETCON = sumolib.checkBinary('netconvert')


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------


def _is_connected(adj: dict[str, set[str]]) -> tuple[bool, set[str]]:
    """Return (connected, unreachable_nodes)."""
    if not adj:
        return True, set()
    start = next(iter(adj))
    seen, queue = {start}, [start]
    while queue:
        n = queue.pop()
        for nb in adj[n]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    unreachable = adj.keys() - seen
    return not unreachable, unreachable


def _build_adj(edge_map: dict[str, tuple[str, str]], nodes: set[str]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for frm, to in edge_map.values():
        if frm in nodes and to in nodes:
            adj[frm].add(to)
            adj[to].add(frm)
    return adj


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build(
    size: int = 4,
    length: float = 200.0,
    lanes: int = 2,
    speed: float = 13.89,
    remove_nodes: list[str] | None = None,
    remove_segments: list[tuple[str, str]] | None = None,
    out_dir: Path = HERE,
    net_name: str = 'grid.net.xml',
    verbose: bool = True,
) -> Path:
    """Build the irregular grid .net.xml and return its path.

    Parameters
    ----------
    size             : Nodes per side (4 → 4×4, 16 total nodes).
    length           : Road segment length in metres.
    lanes            : Number of lanes per direction.
    speed            : Speed limit m/s (~50 km/h = 13.89).
    remove_nodes     : Node IDs to delete (all incident edges removed too).
    remove_segments  : Undirected (nodeA, nodeB) pairs to cut.
    out_dir          : Directory for intermediate and output files.
    net_name         : Filename for the final .net.xml.
    verbose          : Print topology summary.
    """
    if remove_nodes is None:
        remove_nodes = []
    if remove_segments is None:
        remove_segments = []

    out_dir = Path(out_dir)
    prefix = out_dir / 'plain'
    out_net = out_dir / net_name

    # 1. Regular grid plain XML -------------------------------------------
    subprocess.run(
        [
            NETGEN,
            '--grid',
            f'--grid.number={size}',
            f'--grid.length={length}',
            f'--default.lanenumber={lanes}',
            f'--default.speed={speed}',
            '--no-turnarounds',
            '--tls.guess=true',
            '--plain-output-prefix',
            str(prefix),
        ],
        check=True,
        capture_output=True,
    )

    # 2. Load edge + node data --------------------------------------------
    edg_tree = ET.parse(out_dir / 'plain.edg.xml')
    edg_root = edg_tree.getroot()
    nod_tree = ET.parse(out_dir / 'plain.nod.xml')
    nod_root = nod_tree.getroot()

    edge_map: dict[str, tuple[str, str]] = {el.get('id'): (el.get('from'), el.get('to')) for el in edg_root}

    # 3. Collect all edge IDs to remove -----------------------------------
    remove_node_set = set(remove_nodes)
    remove_ids: set[str] = set()

    # Edges incident to removed nodes
    for eid, (frm, to) in edge_map.items():
        if frm in remove_node_set or to in remove_node_set:
            remove_ids.add(eid)

    # Explicit segment cuts (both directions)
    for a, b in remove_segments:
        found = {eid for eid, (frm, to) in edge_map.items() if (frm == a and to == b) or (frm == b and to == a)}
        if not found:
            print(f'  WARNING: no edge found for {a}<->{b}; skipping')
        remove_ids |= found

    for eid in sorted(remove_ids):
        frm, to = edge_map[eid]
        if verbose:
            print(f'  Removed edge: {eid}  ({frm} -> {to})')

    for nid in sorted(remove_node_set):
        if verbose:
            print(f'  Removed node: {nid}')

    # 4. Write filtered node + edge files ---------------------------------
    for el in list(nod_root):
        if el.get('id') in remove_node_set:
            nod_root.remove(el)
    for el in list(edg_root):
        if el.get('id') in remove_ids:
            edg_root.remove(el)

    # Count incoming edges in the remaining network
    remaining = {eid: ft for eid, ft in edge_map.items() if eid not in remove_ids}
    remaining_in: dict[str, int] = {}
    for eid, (frm, to) in remaining.items():
        remaining_in[to] = remaining_in.get(to, 0) + 1

    # Demote any TL node with < 3 remaining incoming edges to priority
    for node_el in nod_root:
        nid = node_el.get('id')
        if node_el.get('type') == 'traffic_light' and remaining_in.get(nid, 0) < 3:
            node_el.set('type', 'priority')
            node_el.attrib.pop('tl', None)
            if verbose:
                print(f'  Demoted {nid} to priority ({remaining_in.get(nid, 0)} connections)')

    irr_nod = out_dir / 'irregular.nod.xml'
    irr_edg = out_dir / 'irregular.edg.xml'
    nod_tree.write(irr_nod)
    edg_tree.write(irr_edg)

    # 5. Connectivity check on remaining TL nodes -------------------------
    tl_nodes = {el.get('id') for el in nod_root if el.get('type') == 'traffic_light'}
    adj = _build_adj(remaining, tl_nodes)
    ok, unreachable = _is_connected(adj)
    if not ok:
        print(f'  WARNING: TL sub-graph disconnected — unreachable: {unreachable}')

    # 6. Reconvert --------------------------------------------------------
    subprocess.run(
        [
            NETCON,
            '--node-files',
            str(irr_nod),
            '--edge-files',
            str(irr_edg),
            '--output-file',
            str(out_net),
            '--no-turnarounds',
        ],
        check=True,
        capture_output=True,
    )

    # 6. Summary ----------------------------------------------------------
    if verbose:
        net = sumolib.net.readNet(str(out_net))
        tls = sorted([n for n in net.getNodes() if n.getType() == 'traffic_light'], key=lambda n: n.getID())
        ways = {3: 0, 4: 0}
        for n in tls:
            k = len(n.getIncoming())
            ways[k] = ways.get(k, 0) + 1

        pct = 100 * len(remove_ids) / max(len(edge_map), 1)
        print(
            f'\nFinal: {out_net.name}  |  {len(tls)} TL junctions  '
            f'|  removed {len(remove_ids)}/{len(edge_map)} edges ({pct:.1f} %)'
        )
        print(f'  4-way: {ways.get(4, 0)}   3-way: {ways.get(3, 0)}')
        for n in tls:
            inc = len(n.getIncoming())
            xy = tuple(round(v) for v in n.getCoord())
            print(f'    {n.getID():6s}  {inc}-way  {xy}')

    return out_net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build irregular grid network')
    parser.add_argument('--size', type=int, default=4, help='Nodes per side')
    parser.add_argument('--length', type=float, default=200.0, help='Segment length (m)')
    parser.add_argument('--lanes', type=int, default=2, help='Lanes per direction')
    parser.add_argument('--speed', type=float, default=13.89, help='Speed limit (m/s)')
    args = parser.parse_args()

    build(
        size=args.size,
        length=args.length,
        lanes=args.lanes,
        speed=args.speed,
        # Remove node B1, edge C2<->D2, and upper-right corner stub (A3 + its roads)
        remove_nodes=['B1', 'A3'],
        remove_segments=[('C2', 'D2')],
    )
