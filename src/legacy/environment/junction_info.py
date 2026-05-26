"""Canonical junction representation for GNN traffic control.

Parses SUMO network nodes into a fixed, geometry-derived representation:

  - Slot ordering (§2):  4-way → sort incoming edges clockwise by approach
    bearing; 3-way → straight-through pair → slots 0/2, stem → slot 1,
    slot 3 absent.
  - Movement decomposition (§2): per slot: left ('l'), through ('s'), right ('r').
  - Phase string builder (§4): 8 canonical green/yellow/all-red SUMO state
    strings derived from the network connection table.
  - Feature helpers: for each (slot, movement) pair, which incoming lane IDs
    and detector lengths serve it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .phase_schema import NUM_PHASES, SLOT_DIR_PHASES

# ---------------------------------------------------------------------------
# Bearing helpers
# ---------------------------------------------------------------------------

def _bearing(from_xy: tuple, to_xy: tuple) -> float:
    """Compass bearing (degrees, 0 = N, clockwise) from *from_xy* toward *to_xy*."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _approach_bearing(edge) -> float:
    """Approach bearing of *edge* arriving at its target junction.

    We take the last two shape points (closest to the stop line) and return
    the direction FROM the junction back along the road — i.e. the compass
    heading of the road segment as seen by approaching traffic.
    """
    shape = edge.getShape()
    if len(shape) < 2:
        return 0.0
    p1, p2 = shape[-2], shape[-1]   # p2 is the stop-line end
    return _bearing(p2, p1)          # outward from junction = approach direction


def _bearing_diff(b1: float, b2: float) -> float:
    """Absolute angular difference (0–180°) between two bearings."""
    d = abs(b1 - b2) % 360.0
    return d if d <= 180.0 else 360.0 - d

# ---------------------------------------------------------------------------
# Detector length helper
# ---------------------------------------------------------------------------

NOMINAL_DET_LEN = 200.0   # metres


def _det_len(lane) -> float:
    """Detector length for *lane*: min(200 m, actual lane length)."""
    return min(NOMINAL_DET_LEN, lane.getLength())

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MovementInfo:
    """One movement direction (left/through/right) on a single slot."""
    direction: str                   # 'l', 's', 'r'
    lane_ids:    list[str] = field(default_factory=list)
    det_lengths: list[float] = field(default_factory=list)

    @property
    def total_det_length(self) -> float:
        """Sum of detector lengths; returns 1.0 to avoid division by zero."""
        return sum(self.det_lengths) or 1.0


@dataclass
class SlotInfo:
    """One canonical approach slot at a junction."""
    slot_idx:   int
    edge_id:    str
    bearing:    float                         # approach bearing (degrees)
    movements:  dict[str, MovementInfo]       # 'l' / 's' / 'r'
    all_lane_ids:    list[str] = field(default_factory=list)
    all_det_lengths: list[float] = field(default_factory=list)


@dataclass
class JunctionInfo:
    """Everything the environment needs to know about one TL junction."""
    junction_id:   str
    n_slots:       int                        # 3 or 4
    slots:         list[Optional[SlotInfo]]   # always length 4; [3] may be None
    phase_states:  list[str]                  # 8 green state strings
    yellow_states: list[str]                  # 8 yellow state strings
    all_red:       str
    # (lane_id, det_len) for every incoming lane — used for reward computation
    all_lane_det:  list[tuple[str, float]] = field(default_factory=list)
    # For each phase: set of incoming lane IDs with protected green ('G')
    phase_served_lanes: list[set] = field(default_factory=list)
    # (from_edge_id, to_edge_id) -> phase indices giving protected green ('G')
    # Used by the expert to attribute each vehicle's wait to all serving phases.
    conn_to_phases: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Slot ordering
# ---------------------------------------------------------------------------

def _order_slots_4way(incoming_edges: list) -> list[Optional[SlotInfo]]:
    """Sort 4 incoming edges clockwise by approach bearing → slots 0–3."""
    sorted_edges = sorted(incoming_edges, key=_approach_bearing)
    slots: list[Optional[SlotInfo]] = []
    for idx, edge in enumerate(sorted_edges):
        slots.append(_make_slot(idx, edge))
    return slots   # length 4


def _order_slots_3way(incoming_edges: list) -> list[Optional[SlotInfo]]:
    """Assign slots for a 3-way junction.

    Find the pair of arms whose bearings are closest to 180° apart
    (straight-through pair) → slots 0 and 2.
    The remaining arm → slot 1.  Slot 3 is None.
    """
    bearings = [(_approach_bearing(e), e) for e in incoming_edges]

    best_diff = float("inf")
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

    # Within the straight-through pair, assign slot 0 to the lower bearing.
    b_i, b_j = bearings[i][0], bearings[j][0]
    if b_i <= b_j:
        slot0_edge, slot2_edge = bearings[i][1], bearings[j][1]
    else:
        slot0_edge, slot2_edge = bearings[j][1], bearings[i][1]

    stem_edge = bearings[stem_idx][1]

    return [
        _make_slot(0, slot0_edge),
        _make_slot(1, stem_edge),
        _make_slot(2, slot2_edge),
        None,                          # slot 3 absent
    ]


def _make_slot(slot_idx: int, edge) -> SlotInfo:
    """Build a SlotInfo from a sumolib edge object (no movement data yet)."""
    return SlotInfo(
        slot_idx=slot_idx,
        edge_id=edge.getID(),
        bearing=_approach_bearing(edge),
        movements={
            'l': MovementInfo('l'),
            's': MovementInfo('s'),
            'r': MovementInfo('r'),
        },
    )

# ---------------------------------------------------------------------------
# Movement-to-lane mapping
# ---------------------------------------------------------------------------

def _populate_movements(slots: list[Optional[SlotInfo]], node) -> None:
    """Fill each SlotInfo's MovementInfo with the lanes that serve it.

    A lane is assigned to a direction if it has at least one SUMO connection
    in that direction.  If a lane serves multiple directions (e.g. shared
    through/right lane) it appears in both MovementInfo lists — consistent
    with reading each movement's detector independently.
    """
    edge_to_slot: dict[str, SlotInfo] = {}
    for s in slots:
        if s is not None:
            edge_to_slot[s.edge_id] = s

    # Collect which directions each lane serves.
    lane_dirs: dict[str, set[str]] = {}   # lane_id -> set of SUMO direction chars
    for in_edge in node.getIncoming():
        if in_edge.getID() not in edge_to_slot:
            continue
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                d = conn.getDirection()     # 'l', 's', 'r', 'L', 'R', 't', ...
                lane_id = conn.getFromLane().getID()
                # Normalise: SUMO uses lowercase for simple, uppercase for
                # partial-conflict.  We bucket L/l → 'l', R/r → 'r', rest → 's'.
                if d.lower() == 'l':
                    canon = 'l'
                elif d.lower() == 'r':
                    canon = 'r'
                else:
                    canon = 's'
                lane_dirs.setdefault(lane_id, set()).add(canon)

    # Now populate MovementInfo for each slot.
    for in_edge in node.getIncoming():
        slot = edge_to_slot.get(in_edge.getID())
        if slot is None:
            continue
        all_ids, all_dets = [], []
        for lane in in_edge.getLanes():
            lid = lane.getID()
            dl  = _det_len(lane)
            all_ids.append(lid)
            all_dets.append(dl)
            for canon_dir in lane_dirs.get(lid, set()):
                mv = slot.movements.get(canon_dir)
                if mv is not None:
                    mv.lane_ids.append(lid)
                    mv.det_lengths.append(dl)
        slot.all_lane_ids    = all_ids
        slot.all_det_lengths = all_dets

# ---------------------------------------------------------------------------
# Phase state string builder  (canonical 4-phase scheme, PLAN §4)
# ---------------------------------------------------------------------------

def _build_phase_strings(
    slots: list[Optional[SlotInfo]],
    node,
) -> tuple[list[str], list[str], str]:
    """Compute SUMO state strings for the 8 canonical phases.

    Returns
    -------
    green_states  : list[str] — 8 phase green state strings
    yellow_states : list[str] — 8 matching yellow state strings
    all_red       : str
    """
    edge_to_slot_idx: dict[str, int] = {
        s.edge_id: s.slot_idx for s in slots if s is not None
    }

    # Collect all TL link indices.
    link_rows: list[tuple[int, str, str]] = []  # (tl_idx, from_edge_id, direction)
    for in_edge in node.getIncoming():
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                tl_idx = conn.getTLLinkIndex()
                if tl_idx < 0:
                    continue
                d = conn.getDirection()
                if d.lower() == 'l':
                    canon = 'l'
                elif d.lower() == 'r':
                    canon = 'r'
                else:
                    canon = 's'
                link_rows.append((tl_idx, in_edge.getID(), canon))

    if not link_rows:
        return ([""] * NUM_PHASES, [""] * NUM_PHASES, "")

    n_links = max(r[0] for r in link_rows) + 1
    phase_states = [['r'] * n_links for _ in range(NUM_PHASES)]

    for tl_idx, from_edge_id, canon_dir in link_rows:
        slot_idx = edge_to_slot_idx.get(from_edge_id)
        if slot_idx is None:
            continue
        key = (slot_idx, canon_dir)
        for ph_idx, sig_char in SLOT_DIR_PHASES.get(key, []):
            existing = phase_states[ph_idx][tl_idx]
            # Priority: G > g > r
            if existing == 'r' or (existing == 'g' and sig_char == 'G'):
                phase_states[ph_idx][tl_idx] = sig_char

    green_states  = [''.join(s) for s in phase_states]
    yellow_states = [
        ''.join('y' if c in 'Gg' else 'r' for c in s)
        for s in green_states
    ]
    all_red = 'r' * n_links
    return green_states, yellow_states, all_red


def _build_phase_served_lanes(
    link_rows: list[tuple[int, str, str, str]],   # (tl_idx, from_edge, dir, from_lane)
    green_states: list[str],
) -> list[set]:
    """For each phase, collect incoming lane IDs that have protected green ('G')."""
    served: list[set] = [set() for _ in range(len(green_states))]
    for ph_idx, state in enumerate(green_states):
        for tl_idx, char in enumerate(state):
            if char == 'G':
                # Find which lane corresponds to this tl_idx
                for row in link_rows:
                    if row[0] == tl_idx:
                        served[ph_idx].add(row[3])   # from_lane
    return served

# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_junction_info(node) -> Optional[JunctionInfo]:
    """Build a JunctionInfo for *node*.

    Returns None if the node has no controlled TL links (skip it).
    """
    incoming = node.getIncoming()
    n_incoming = len(incoming)

    if n_incoming == 4:
        slots = _order_slots_4way(list(incoming))
    elif n_incoming == 3:
        slots = _order_slots_3way(list(incoming))
    else:
        # Unsupported junction type (2-way or 5+); caller should skip.
        return None

    n_slots = n_incoming   # 3 or 4

    _populate_movements(slots, node)

    # Build phase strings.
    green_states, yellow_states, all_red = _build_phase_strings(slots, node)
    if not all_red:
        return None   # no controlled links

    # Build full link_rows including from_lane for phase_served_lanes.
    edge_to_slot_idx: dict[str, int] = {
        s.edge_id: s.slot_idx for s in slots if s is not None
    }
    link_rows_with_lane: list[tuple[int, str, str, str]] = []
    for in_edge in node.getIncoming():
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                tl_idx = conn.getTLLinkIndex()
                if tl_idx < 0:
                    continue
                d = conn.getDirection()
                canon = 'l' if d.lower() == 'l' else ('r' if d.lower() == 'r' else 's')
                link_rows_with_lane.append((
                    tl_idx,
                    in_edge.getID(),
                    canon,
                    conn.getFromLane().getID(),
                ))

    phase_served = _build_phase_served_lanes(link_rows_with_lane, green_states)

    # Build conn_to_phases: all phase indices that give protected 'G' to each
    # connection. A movement can be served by a paired base phase and by a
    # single-slot clearance phase.
    conn_to_phases: dict[tuple[str, str], list[int]] = {}
    for in_edge in node.getIncoming():
        for out_edge in in_edge.getOutgoing():
            for conn in in_edge.getConnections(out_edge):
                tl_idx = conn.getTLLinkIndex()
                if tl_idx < 0:
                    continue
                key = (in_edge.getID(), out_edge.getID())
                phases = conn_to_phases.setdefault(key, [])
                for ph_idx, state in enumerate(green_states):
                    if tl_idx < len(state) and state[tl_idx] == 'G' and ph_idx not in phases:
                        phases.append(ph_idx)

    # All incoming lanes + detector lengths (for reward).
    all_lane_det: list[tuple[str, float]] = []
    for in_edge in node.getIncoming():
        for lane in in_edge.getLanes():
            all_lane_det.append((lane.getID(), _det_len(lane)))

    return JunctionInfo(
        junction_id=node.getID(),
        n_slots=n_slots,
        slots=slots,
        phase_states=green_states,
        yellow_states=yellow_states,
        all_red=all_red,
        all_lane_det=all_lane_det,
        phase_served_lanes=phase_served,
        conn_to_phases=conn_to_phases,
    )
