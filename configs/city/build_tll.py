"""Generate canonical 4-phase tll.xml for all TL junctions in city.net.xml.

Same logic as configs/grid_4x4/build_tll.py adapted for the city network.
Junctions with 1, 2, or 5+ arms are skipped (same rule as build_junction_info).
"""

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

sys.path.append(os.path.join(os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo'), 'tools'))
import sumolib

HERE = Path(__file__).parent
NET = sumolib.net.readNet(str(HERE / 'city.net.xml'), withConnections=True)

YELLOW_DUR = 3
ALLRED_DUR = 2
GREEN_DUR = 25


def bearing(from_xy: tuple, to_xy: tuple) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def approach_bearing(edge) -> float:
    shape = edge.getShape()
    if len(shape) < 2:
        return 0.0
    p1, p2 = shape[-2], shape[-1]
    return bearing(p2, p1)


def bearing_diff(b1: float, b2: float) -> float:
    d = abs(b1 - b2) % 360.0
    return d if d <= 180.0 else 360.0 - d


def slot_order(node) -> list:
    """Return incoming edges in canonical slot order.

    4-way: sort clockwise by approach bearing.
    3-way: straight-through pair -> slots 0/2, stem -> slot 1.
    Returns a list of edges (length 3 or 4).
    """
    incoming = list(node.getIncoming())
    n = len(incoming)

    if n == 4:
        return sorted(incoming, key=approach_bearing)

    # 3-way: find the pair closest to 180 degrees apart
    bearings = [(approach_bearing(e), e) for e in incoming]
    best_diff = float('inf')
    best_pair = (0, 1)
    for i in range(3):
        for j in range(i + 1, 3):
            d = bearing_diff(bearings[i][0], bearings[j][0])
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


def phases_for(slot_idx: int, direction: str) -> list[tuple[int, str]]:
    d = direction.lower()
    if slot_idx == 0:
        if d == 's':
            return [(0, 'G')]
        if d == 'r':
            return [(0, 'G'), (3, 'g')]
        if d == 'l':
            return [(1, 'G')]
    elif slot_idx == 1:
        if d == 's':
            return [(2, 'G')]
        if d == 'r':
            return [(2, 'G'), (1, 'g')]
        if d == 'l':
            return [(3, 'G')]
    elif slot_idx == 2:
        if d == 's':
            return [(0, 'G')]
        if d == 'r':
            return [(0, 'G'), (3, 'g')]
        if d == 'l':
            return [(1, 'G')]
    elif slot_idx == 3:
        if d == 's':
            return [(2, 'G')]
        if d == 'r':
            return [(2, 'G'), (1, 'g')]
        if d == 'l':
            return [(3, 'G')]
    return []


def build_phase_strings(node) -> tuple | None:
    incoming = list(node.getIncoming())
    n = len(incoming)
    if n not in (3, 4):
        return None

    slots = slot_order(node)
    slot_map = {e.getID(): i for i, e in enumerate(slots)}

    link_info: list[tuple] = []
    for in_edge in node.getIncoming():
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                tl_idx = conn.getTLLinkIndex()
                if tl_idx < 0:
                    continue
                link_info.append(
                    (
                        tl_idx,
                        in_edge.getID(),
                        conn.getDirection(),
                    )
                )

    if not link_info:
        return None

    n_links = max(t[0] for t in link_info) + 1
    phase_states = [['r'] * n_links for _ in range(4)]

    for tl_idx, from_edge, direction in link_info:
        slot_idx = slot_map.get(from_edge)
        if slot_idx is None:
            continue
        for ph, ch in phases_for(slot_idx, direction):
            existing = phase_states[ph][tl_idx]
            if existing == 'r' or (existing == 'g' and ch == 'G'):
                phase_states[ph][tl_idx] = ch

    green_states = [''.join(s) for s in phase_states]
    yellow_states = [''.join('y' if c in 'Gg' else 'r' for c in s) for s in green_states]
    all_red = 'r' * n_links
    return green_states, yellow_states, all_red


root = ET.Element('additional')
skipped = 0
written = 0

for node in sorted(NET.getNodes(), key=lambda n: n.getID()):
    if node.getType() != 'traffic_light':
        continue
    jid = node.getID()
    n_in = len(node.getIncoming())
    if n_in not in (3, 4):
        # Emit a simple all-green stub so SUMO has a valid program.
        # These junctions are excluded from GNN control but SUMO still needs a plan.
        link_indices = set()
        for in_edge in node.getIncoming():
            for out_edge in in_edge.getOutgoing():
                for conn in in_edge.getConnections(out_edge):
                    tl_idx = conn.getTLLinkIndex()
                    if tl_idx >= 0:
                        link_indices.add(tl_idx)
        if link_indices:
            n_links = max(link_indices) + 1
            all_green = 'G' * n_links
            yellow = 'y' * n_links
            tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='canonical', offset='0')
            ET.SubElement(tll, 'phase', duration='30', state=all_green)
            ET.SubElement(tll, 'phase', duration='3', state=yellow)
            print(f'  STUB {jid}  ({n_in} arms, {n_links} links)')
        else:
            print(f'  SKIP {jid}  ({n_in} arms, no links)')
        skipped += 1
        continue

    result = build_phase_strings(node)
    if result is None:
        print(f'  SKIP {jid}  (no controlled links)')
        skipped += 1
        continue

    greens, yellows, allred = result
    print(f'  {jid}  {n_in}-way  {len(greens[0])} links')
    for i, (g, y) in enumerate(zip(greens, yellows)):
        print(f'    phase {i}  G={g}  Y={y}')

    tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='canonical', offset='0')
    for g, y in zip(greens, yellows):
        ET.SubElement(tll, 'phase', duration=str(GREEN_DUR), state=g)
        ET.SubElement(tll, 'phase', duration=str(YELLOW_DUR), state=y)
        ET.SubElement(tll, 'phase', duration=str(ALLRED_DUR), state=allred)
    written += 1

xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
xml_str = '\n'.join(xml_str.split('\n')[1:])
out = HERE / 'city.tll.xml'
out.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str)
print(f'\nWrote {written} TL programs -> {out}  (skipped {skipped})')
