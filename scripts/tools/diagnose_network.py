"""Diagnose suspicious junctions/edges in a built SUMO network.

Prints IDs and reasons for everything that looks like a merge or promotion
went wrong, so you can copy IDs straight into a follow-up fix.

Usage
-----
    python scripts/diagnose_network.py configs/city/city.net.xml
    python scripts/diagnose_network.py configs/city/city.net.xml --short-edge 40
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

import sumolib  # noqa: E402

TL_TYPES = {'traffic_light', 'traffic_light_unregulated', 'traffic_light_right_on_red'}


def _unique_neighbors(node) -> set[str]:
    neighbors: set[str] = set()
    for e in node.getIncoming():
        neighbors.add(e.getFromNode().getID())
    for e in node.getOutgoing():
        neighbors.add(e.getToNode().getID())
    neighbors.discard(node.getID())
    return neighbors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('net_file')
    parser.add_argument('--short-edge', type=float, default=40.0, help='Flag edges shorter than this (m)')
    args = parser.parse_args()

    net = sumolib.net.readNet(args.net_file, withConnections=True)

    print(f'\nDiagnosing {args.net_file}')
    print('=' * 70)

    # 1. Phantom TLs: TL nodes that don't look like real intersections
    print('\n[1] Phantom traffic-light junctions (<=2 unique neighbors)')
    phantom_tls = []
    for node in net.getNodes():
        if node.getType() not in TL_TYPES:
            continue
        if len(_unique_neighbors(node)) <= 2:
            phantom_tls.append(node)
    if not phantom_tls:
        print('  (none)')
    for node in phantom_tls:
        neighbors = _unique_neighbors(node)
        print(
            f'  {node.getID():40s} type={node.getType():15s} '
            f'in={len(node.getIncoming())} out={len(node.getOutgoing())} '
            f'unique_neighbors={len(neighbors)}'
        )

    # 2. Pass-through nodes that should have been merged but weren't
    print('\n[2] Pass-through priority nodes (2 unique neighbors, not a real junction)')
    leftover_passthroughs = []
    for node in net.getNodes():
        if node.getType() in TL_TYPES:
            continue
        neighbors = _unique_neighbors(node)
        if len(neighbors) == 2:
            leftover_passthroughs.append(node)
    if not leftover_passthroughs:
        print('  (none)')
    for node in leftover_passthroughs:
        ins, outs = node.getIncoming(), node.getOutgoing()
        print(
            f'  {node.getID():40s} type={node.getType():15s} '
            f'in={len(ins)} out={len(outs)} neighbors={sorted(_unique_neighbors(node))}'
        )

    # 3. Priority junctions with >=3 arms — should have been promoted to TL
    print('\n[3] Priority junctions with >=3 arms (missed TL promotion)')
    missed_promotions = []
    for node in net.getNodes():
        if node.getType() in TL_TYPES:
            continue
        arms = max(len(node.getIncoming()), len(node.getOutgoing()))
        if arms >= 3:
            missed_promotions.append(node)
    if not missed_promotions:
        print('  (none)')
    for node in missed_promotions:
        print(
            f'  {node.getID():40s} type={node.getType():15s} in={len(node.getIncoming())} out={len(node.getOutgoing())}'
        )

    # 4. Very short edges — likely remnants of a missed merge
    print(f'\n[4] Edges shorter than {args.short_edge:.0f} m (potential unmerged remnants)')
    short_edges = []
    for edge in net.getEdges():
        if edge.getLength() < args.short_edge:
            short_edges.append(edge)
    if not short_edges:
        print('  (none)')
    short_edges.sort(key=lambda e: e.getLength())
    for edge in short_edges[:30]:
        fn = edge.getFromNode()
        tn = edge.getToNode()
        print(
            f'  {edge.getID():40s} len={edge.getLength():6.1f}m '
            f'from={fn.getID()}({fn.getType()})  to={tn.getID()}({tn.getType()})'
        )
    if len(short_edges) > 30:
        print(f'  ... and {len(short_edges) - 30} more')

    # 5. TL junctions where some incoming edge has no lanes (no detectors will be written)
    print('\n[5] TL junctions with zero incoming lanes (no detectors generated)')
    zero_lane_tls = []
    for node in net.getNodes():
        if node.getType() not in TL_TYPES:
            continue
        n_lanes = sum(len(e.getLanes()) for e in node.getIncoming())
        if n_lanes == 0:
            zero_lane_tls.append(node)
    if not zero_lane_tls:
        print('  (none)')
    for node in zero_lane_tls:
        print(f'  {node.getID():40s} in_edges={len(node.getIncoming())}')

    print('\n' + '=' * 70)
    print(
        f'Summary: {len(phantom_tls)} phantom TLs, {len(leftover_passthroughs)} pass-throughs, '
        f'{len(missed_promotions)} missed promotions, {len(short_edges)} short edges'
    )


if __name__ == '__main__':
    main()
