"""Generate canonical 4-phase tll.xml for all TL junctions in the grid.

For each junction:
  1. Sort incoming edges clockwise by approach bearing.
  2. Assign slot 0-3 (slot 3 zero-absent for 3-way).
  3. Build 16-character (or variable) SUMO state strings per the canonical
     phase definitions in PLAN §4, then write .tll.xml.

Phase definitions (against canonical slot indices):
  Phase 0: slot0 through+right,  slot2 through+right
  Phase 1: slot0 left,           slot2 left,  slot1 right, slot3 right
  Phase 2: slot1 through+right,  slot3 through+right
  Phase 3: slot1 left,           slot3 left,  slot0 right, slot2 right
"""

import os
import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

HERE = Path(__file__).parent
NET = sumolib.net.readNet(str(HERE / 'grid.net.xml'), withConnections=True)

YELLOW_DUR = 3
ALLRED_DUR = 2
GREEN_DUR = 25


def bearing(from_xy, to_xy) -> float:
    """Compass bearing (degrees, 0=N, clockwise) from from_xy toward to_xy."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    angle = math.degrees(math.atan2(dx, dy))  # N=0, E=90, S=180, W=270
    return angle % 360


def slot_order(junction_id: str) -> list:
    """Return incoming edge IDs sorted clockwise by approach bearing (slot 0-3).
    For 3-way junctions the list has 3 entries; slot 3 is implicitly absent.
    """
    node = NET.getNode(junction_id)
    edges = node.getIncoming()

    # bearing = direction road segment points AWAY from junction (outward bearing)
    def edge_bearing(e):
        shape = e.getShape()
        # last two points of the shape approach the junction
        p1, p2 = shape[-2], shape[-1]
        return bearing(p2, p1)  # outward: from junction end back along road

    return sorted(edges, key=edge_bearing)


def build_phase_strings(junction_id: str):
    """Return (green_states, yellow_states, all_red) for the 4 canonical phases."""
    node = NET.getNode(junction_id)
    slots = slot_order(junction_id)  # list of incoming edges, slot 0..N-1

    # Map (from_edge_id, to_edge_id, dir) -> tl_link_index
    link_info: list[tuple] = []  # (tl_idx, from_edge, to_edge, from_lane, dir)
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
                        out_edge.getID(),
                        conn.getFromLane().getID(),
                        conn.getDirection(),
                    )
                )
    if not link_info:
        return None
    n_links = max(t[0] for t in link_info) + 1

    # slot_id -> index in slots list (0-based)
    slot_map = {e.getID(): i for i, e in enumerate(slots)}

    # For each link decide: which slot, which movement, which phases serve it.
    def phases_for(slot_idx: int, direction: str) -> list[tuple[int, str]]:
        """Return list of (phase_index, 'G'|'g') for this movement."""
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

    # Build per-phase state arrays
    phase_states = [['r'] * n_links for _ in range(4)]
    for tl_idx, from_edge, to_edge, from_lane, direction in link_info:
        slot_idx = slot_map.get(from_edge)
        if slot_idx is None:
            continue
        for ph, ch in phases_for(slot_idx, direction):
            # Only overwrite if not already set to 'G' (G > g > r)
            existing = phase_states[ph][tl_idx]
            if existing == 'r' or (existing == 'g' and ch == 'G'):
                phase_states[ph][tl_idx] = ch

    green_states = [''.join(s) for s in phase_states]
    yellow_states = [''.join('y' if c in 'Gg' else 'r' for c in s) for s in green_states]
    all_red = 'r' * n_links
    return green_states, yellow_states, all_red


# Build XML
root = ET.Element('additional')
for node in sorted(NET.getNodes(), key=lambda n: n.getID()):
    if node.getType() != 'traffic_light':
        continue
    jid = node.getID()
    result = build_phase_strings(jid)
    if result is None:
        print(f'  SKIP {jid} — no controlled links')
        continue
    greens, yellows, allred = result
    print(f'  {jid}:  {len(greens[0])} links')
    for i, (g, y) in enumerate(zip(greens, yellows)):
        print(f'    phase {i}  G={g}  Y={y}')

    tll = ET.SubElement(root, 'tlLogic', id=jid, type='static', programID='canonical', offset='0')
    for i, (g, y) in enumerate(zip(greens, yellows)):
        ET.SubElement(tll, 'phase', duration=str(GREEN_DUR), state=g)
        ET.SubElement(tll, 'phase', duration=str(YELLOW_DUR), state=y)
        ET.SubElement(tll, 'phase', duration=str(ALLRED_DUR), state=allred)

# Pretty-print
xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
xml_str = '\n'.join(xml_str.split('\n')[1:])  # strip xml declaration
out = HERE / 'grid.tll.xml'
out.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str)
print(f'\nWrote {out}')
