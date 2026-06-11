"""Generate rectangular SUMO grid networks for movement-policy experiments."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    row: int
    col: int
    degree: int
    x: float
    y: float
    node_type: str | None


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: str
    from_node: str
    to_node: str
    lanes: int
    speed: float


@dataclass(frozen=True)
class ConnectionSpec:
    node_id: str
    from_edge: str
    to_edge: str
    from_lane: int
    to_lane: int


@dataclass(frozen=True)
class FlowSpec:
    flow_id: str
    from_edge: str
    to_edge: str
    begin: int
    end: int
    probability: float


@dataclass(frozen=True)
class TLLinkSpec:
    tl_link_index: int
    approach: str
    direction: str
    outgoing_edge_id: str | None = None
    request_index: int | None = None

    @property
    def axis(self) -> str:
        return "vertical" if self.approach in {"north", "south"} else "horizontal"


def node_id(row: int, col: int) -> str:
    return f"N{row}_{col}"


def build_node_specs(rows: int, cols: int, spacing: float) -> list[NodeSpec]:
    _validate_dimensions(rows, cols)
    specs: list[NodeSpec] = []
    for row in range(rows):
        for col in range(cols):
            degree = _grid_degree(row, col, rows, cols)
            specs.append(
                NodeSpec(
                    node_id=node_id(row, col),
                    row=row,
                    col=col,
                    degree=degree,
                    x=col * spacing,
                    y=(rows - 1 - row) * spacing,
                    node_type="traffic_light" if degree >= 3 else None,
                )
            )
    return specs


def build_edge_specs(
    nodes: list[NodeSpec],
    speed: float = 13.89,
) -> list[EdgeSpec]:
    by_pos = {(node.row, node.col): node for node in nodes}
    specs: list[EdgeSpec] = []
    for node in nodes:
        for d_row, d_col in ((0, 1), (1, 0)):
            other = by_pos.get((node.row + d_row, node.col + d_col))
            if other is None:
                continue
            specs.append(_edge_between(node, other, speed))
            specs.append(_edge_between(other, node, speed))
    return specs


def build_connection_specs(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
) -> list[ConnectionSpec]:
    node_by_id = {node.node_id: node for node in nodes}
    edge_by_pair = {(edge.from_node, edge.to_node): edge for edge in edges}
    incoming_by_node: dict[str, list[EdgeSpec]] = {}
    outgoing_by_node: dict[str, list[EdgeSpec]] = {}
    for edge in edges:
        incoming_by_node.setdefault(edge.to_node, []).append(edge)
        outgoing_by_node.setdefault(edge.from_node, []).append(edge)

    specs: list[ConnectionSpec] = []
    for current in nodes:
        outgoing = outgoing_by_node.get(current.node_id, [])
        for incoming in sorted(incoming_by_node.get(current.node_id, []), key=lambda edge: edge.edge_id):
            candidates = [
                edge for edge in outgoing
                if edge.to_node != incoming.from_node
            ]
            if len(candidates) == 1:
                target = candidates[0]
                for lane_idx in range(incoming.lanes):
                    specs.append(
                        ConnectionSpec(
                            node_id=current.node_id,
                            from_edge=incoming.edge_id,
                            to_edge=target.edge_id,
                            from_lane=lane_idx,
                            to_lane=min(lane_idx, target.lanes - 1),
                        )
                    )
                continue
            ordered = _ordered_turn_edges(
                node_by_id[incoming.from_node],
                current,
                [node_by_id[edge.to_node] for edge in candidates],
            )
            for lane_idx, target_node in enumerate(ordered):
                to_edge = edge_by_pair[(current.node_id, target_node.node_id)]
                specs.append(
                    ConnectionSpec(
                        node_id=current.node_id,
                        from_edge=incoming.edge_id,
                        to_edge=to_edge.edge_id,
                        from_lane=lane_idx,
                        to_lane=min(lane_idx, to_edge.lanes - 1),
                    )
                )
                if current.degree == 3 and to_edge.lanes > incoming.lanes:
                    for extra_to_lane in range(incoming.lanes, to_edge.lanes):
                        specs.append(
                            ConnectionSpec(
                                node_id=current.node_id,
                                from_edge=incoming.edge_id,
                                to_edge=to_edge.edge_id,
                                from_lane=lane_idx,
                                to_lane=extra_to_lane,
                            )
                        )
    return specs


def build_route_flows(
    edges: list[EdgeSpec],
    begin: int = 0,
    end: int = 3600,
    probability: float = 0.03,
) -> list[FlowSpec]:
    max_row = max(max(_parse_node_id(edge.from_node)[0], _parse_node_id(edge.to_node)[0]) for edge in edges)
    max_col = max(max(_parse_node_id(edge.from_node)[1], _parse_node_id(edge.to_node)[1]) for edge in edges)
    source_edges = [
        edge for edge in edges
        if _is_boundary_node(edge.from_node, max_row, max_col)
    ]
    sink_edges = [
        edge for edge in edges
        if _is_boundary_node(edge.to_node, max_row, max_col)
    ]
    flows: list[FlowSpec] = []
    for idx, source in enumerate(sorted(source_edges, key=lambda edge: edge.edge_id)):
        opposite = _opposite_boundary_edge(source, sink_edges, max_row, max_col)
        if opposite is None or opposite.edge_id == source.edge_id:
            continue
        flows.append(
            FlowSpec(
                flow_id=f"flow_{idx}",
                from_edge=source.edge_id,
                to_edge=opposite.edge_id,
                begin=begin,
                end=end,
                probability=probability,
            )
        )
    return flows


def build_safe_phase_states(
    links: list[TLLinkSpec],
    n_links: int,
) -> list[str]:
    """Build conflict-valid protected movement phases for generated grids.

    Link fields are `(tl_link_index, approach, direction)`, where approach is
    `north/south/east/west` and direction is SUMO-style `r/s/l`.
    """
    phase_rules: list[list[tuple[set[str], set[str]]]] = [
        [({"north", "south"}, {"r", "s"})],
        [({"north", "south"}, {"l"}), ({"east", "west"}, {"r"})],
        [({"east", "west"}, {"r", "s"})],
        [({"east", "west"}, {"l"}), ({"north", "south"}, {"r"})],
        [({"north"}, {"r", "s", "l"})],
        [({"south"}, {"r", "s", "l"})],
        [({"east"}, {"r", "s", "l"})],
        [({"west"}, {"r", "s", "l"})],
    ]
    states: list[str] = []
    for rule in phase_rules:
        chars = ["r"] * n_links
        phase_links = [
            link for link in links
            if any(
                link.approach in approaches and link.direction.lower() in directions
                for approaches, directions in rule
            )
        ]
        if _has_movement_conflict(phase_links):
            continue
        for link in phase_links:
            chars[link.tl_link_index] = "G"
        state = "".join(chars)
        if "G" in state and state not in states:
            states.append(state)
    return states


def build_conflict_phase_states(
    links: list[TLLinkSpec],
    n_links: int,
    are_foes,
) -> list[str]:
    """Build maximal phases from SUMO foes plus same-outgoing-edge conflicts."""
    indexed_links = sorted(links, key=lambda link: link.tl_link_index)
    valid_sets: list[frozenset[int]] = []
    for size in range(1, len(indexed_links) + 1):
        for phase_links in itertools.combinations(indexed_links, size):
            if _has_sumo_or_outgoing_edge_conflict(list(phase_links), are_foes):
                continue
            valid_sets.append(frozenset(link.tl_link_index for link in phase_links))

    maximal_sets = [
        candidate for candidate in valid_sets
        if not any(candidate < other for other in valid_sets)
    ]
    states: list[str] = []
    for phase_set in sorted(maximal_sets, key=lambda item: (-len(item), sorted(item))):
        chars = ["r"] * n_links
        for tl_idx in phase_set:
            chars[tl_idx] = "G"
        state = "".join(chars)
        if state not in states:
            states.append(state)
    return states


def _has_sumo_or_outgoing_edge_conflict(links: list[TLLinkSpec], are_foes) -> bool:
    for idx, first in enumerate(links):
        for second in links[idx + 1:]:
            if _sumo_requests_are_foes(first, second, are_foes):
                return True
            if (
                first.outgoing_edge_id is not None
                and first.outgoing_edge_id == second.outgoing_edge_id
            ):
                return True
    return False


def _sumo_requests_are_foes(first: TLLinkSpec, second: TLLinkSpec, are_foes) -> bool:
    first_request = (
        first.request_index
        if first.request_index is not None
        else first.tl_link_index
    )
    second_request = (
        second.request_index
        if second.request_index is not None
        else second.tl_link_index
    )
    return bool(
        are_foes(first_request, second_request)
        or are_foes(second_request, first_request)
    )


def _has_movement_conflict(links: list[TLLinkSpec]) -> bool:
    for idx, first in enumerate(links):
        for second in links[idx + 1:]:
            if _movements_conflict(first, second):
                return True
    return False


def _movements_conflict(first: TLLinkSpec, second: TLLinkSpec) -> bool:
    if first.approach == second.approach:
        return False

    first_direction = first.direction.lower()
    second_direction = second.direction.lower()
    if _opposite_approaches(first.approach, second.approach):
        return not (
            first_direction in {"r", "s"} and second_direction in {"r", "s"}
            or first_direction == "l" and second_direction == "l"
        )

    return not (
        first_direction == "l" and second_direction == "r"
        or first_direction == "r" and second_direction == "l"
    )


def _opposite_approaches(first: str, second: str) -> bool:
    return {first, second} in (
        {"north", "south"},
        {"east", "west"},
    )


def generate_grid(
    rows: int,
    cols: int,
    out_dir: Path,
    spacing: float = 200.0,
    speed: float = 13.89,
    netconvert: bool = True,
    phase_mode: str = "conflict-edge",
) -> Path:
    nodes = build_node_specs(rows=rows, cols=cols, spacing=spacing)
    edges = build_edge_specs(nodes, speed=speed)
    connections = build_connection_specs(nodes, edges)

    out_dir.mkdir(parents=True, exist_ok=True)
    nod_path = out_dir / "grid.nod.xml"
    edg_path = out_dir / "grid.edg.xml"
    con_path = out_dir / "grid.con.xml"
    net_path = out_dir / "grid.net.xml"

    _write_nodes(nod_path, nodes)
    _write_edges(edg_path, edges)
    _write_connections(con_path, connections)
    _write_additional(out_dir / "grid.add.xml")
    _write_routes(out_dir / "grid.rou.xml", build_route_flows(edges))
    _write_sumocfg(out_dir / "grid.sumocfg")

    if netconvert:
        _run_netconvert(nod_path, edg_path, con_path, net_path)

    _write_tll_from_net(net_path, out_dir / "grid.tll.xml", phase_mode=phase_mode)
    _print_summary(nodes, edges, connections, net_path)
    return net_path


def _edge_between(from_node: NodeSpec, to_node: NodeSpec, speed: float) -> EdgeSpec:
    return EdgeSpec(
        edge_id=f"{from_node.node_id}_to_{to_node.node_id}",
        from_node=from_node.node_id,
        to_node=to_node.node_id,
        lanes=_incoming_lanes_for_target(to_node),
        speed=speed,
    )


def _incoming_lanes_for_target(node: NodeSpec) -> int:
    if node.node_type != "traffic_light":
        return 2
    return 3 if node.degree >= 4 else 2


def _ordered_turn_edges(
    incoming_from: NodeSpec,
    current: NodeSpec,
    outgoing_nodes: list[NodeSpec],
) -> list[NodeSpec]:
    in_vec = (current.col - incoming_from.col, current.row - incoming_from.row)

    def turn_rank(target: NodeSpec) -> tuple[int, str]:
        out_vec = (target.col - current.col, target.row - current.row)
        cross = in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0]
        dot = in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]
        if cross > 0:
            rank = 0  # right
        elif dot > 0:
            rank = 1  # straight
        else:
            rank = 2  # left
        return rank, target.node_id

    return sorted(outgoing_nodes, key=turn_rank)


def _grid_degree(row: int, col: int, rows: int, cols: int) -> int:
    degree = 0
    if row > 0:
        degree += 1
    if row < rows - 1:
        degree += 1
    if col > 0:
        degree += 1
    if col < cols - 1:
        degree += 1
    return degree


def _validate_dimensions(rows: int, cols: int) -> None:
    if rows < 3 or cols < 3:
        raise ValueError("Grid dimensions must be at least 3x3.")


def _write_nodes(path: Path, nodes: list[NodeSpec]) -> None:
    root = ET.Element("nodes")
    for node in nodes:
        attrs = {
            "id": node.node_id,
            "x": f"{node.x:.2f}",
            "y": f"{node.y:.2f}",
        }
        if node.node_type is not None:
            attrs["type"] = node.node_type
        ET.SubElement(root, "node", attrs)
    _write_xml(path, root)


def _write_edges(path: Path, edges: list[EdgeSpec]) -> None:
    root = ET.Element("edges")
    for edge in edges:
        ET.SubElement(
            root,
            "edge",
            {
                "id": edge.edge_id,
                "from": edge.from_node,
                "to": edge.to_node,
                "numLanes": str(edge.lanes),
                "speed": f"{edge.speed:.2f}",
            },
        )
    _write_xml(path, root)


def _write_connections(path: Path, connections: list[ConnectionSpec]) -> None:
    root = ET.Element("connections")
    for connection in connections:
        ET.SubElement(
            root,
            "connection",
            {
                "from": connection.from_edge,
                "to": connection.to_edge,
                "fromLane": str(connection.from_lane),
                "toLane": str(connection.to_lane),
            },
        )
    _write_xml(path, root)


def _write_additional(path: Path) -> None:
    _write_xml(path, ET.Element("additional"))


def _write_routes(path: Path, flows: list[FlowSpec]) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", {"id": "car", "accel": "2.6", "decel": "4.5", "length": "5.0", "maxSpeed": "13.89"})
    for flow in flows:
        ET.SubElement(
            root,
            "flow",
            {
                "id": flow.flow_id,
                "type": "car",
                "begin": str(flow.begin),
                "end": str(flow.end),
                "probability": f"{flow.probability:.4f}",
                "from": flow.from_edge,
                "to": flow.to_edge,
            },
        )
    _write_xml(path, root)


def _opposite_boundary_edge(
    source: EdgeSpec,
    sink_edges: list[EdgeSpec],
    max_row: int,
    max_col: int,
) -> EdgeSpec | None:
    source_from = _parse_node_id(source.from_node)
    mirror = (max_row - source_from[0], max_col - source_from[1])
    candidates = [
        edge for edge in sink_edges
        if _parse_node_id(edge.to_node) == mirror
    ]
    if candidates:
        return sorted(candidates, key=lambda edge: edge.edge_id)[0]
    far_side = [
        edge for edge in sink_edges
        if _boundary_side(_parse_node_id(edge.to_node), max_row, max_col)
        != _boundary_side(source_from, max_row, max_col)
    ]
    return sorted(far_side, key=lambda edge: edge.edge_id)[0] if far_side else None


def _is_boundary_node(node: str, max_row: int, max_col: int) -> bool:
    row, col = _parse_node_id(node)
    return row == 0 or row == max_row or col == 0 or col == max_col


def _boundary_side(pos: tuple[int, int], max_row: int, max_col: int) -> str:
    row, col = pos
    if row == 0:
        return "top"
    if row == max_row:
        return "bottom"
    if col == 0:
        return "left"
    if col == max_col:
        return "right"
    return "interior"


def _parse_node_id(value: str) -> tuple[int, int]:
    row_text, col_text = value.removeprefix("N").split("_", 1)
    return int(row_text), int(col_text)


def _write_sumocfg(path: Path) -> None:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", {"value": "grid.net.xml"})
    ET.SubElement(input_el, "route-files", {"value": "grid.rou.xml"})
    ET.SubElement(input_el, "additional-files", {"value": "grid.tll.xml,grid.add.xml"})
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", {"value": "0"})
    ET.SubElement(time_el, "end", {"value": "3600"})
    ET.SubElement(time_el, "step-length", {"value": "1.0"})
    report_el = ET.SubElement(root, "report")
    ET.SubElement(report_el, "no-step-log", {"value": "true"})
    _write_xml(path, root)


def _write_xml(path: Path, root: ET.Element) -> None:
    xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    path.write_text(xml, encoding="utf-8")


def _run_netconvert(nod_path: Path, edg_path: Path, con_path: Path, net_path: Path) -> None:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME environment variable is required to run netconvert.")
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import sumolib

    subprocess.run(
        [
            sumolib.checkBinary("netconvert"),
            "--node-files", str(nod_path),
            "--edge-files", str(edg_path),
            "--connection-files", str(con_path),
            "--output-file", str(net_path),
            "--no-turnarounds",
        ],
        check=True,
    )


def _write_tll_from_net(
    net_path: Path,
    tll_path: Path,
    phase_mode: str = "conflict-edge",
) -> None:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME environment variable is required to generate traffic lights.")
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import sumolib

    if phase_mode not in {"protected", "conflict-edge"}:
        raise ValueError(f"Unsupported phase mode: {phase_mode}")

    net = sumolib.net.readNet(
        str(net_path),
        withConnections=True,
        withFoes=phase_mode == "conflict-edge",
    )
    root = ET.Element("additional")
    for node in sorted(net.getNodes(), key=lambda item: item.getID()):
        if node.getType() != "traffic_light":
            continue
        link_specs: list[TLLinkSpec] = []
        for incoming in node.getIncoming():
            approach = _approach_name(incoming.getFromNode().getID(), node.getID())
            for outgoing in incoming.getOutgoing():
                for conn in incoming.getConnections(outgoing):
                    tl_idx = conn.getTLLinkIndex()
                    if tl_idx < 0:
                        continue
                    request_idx = conn.getJunctionIndex()
                    link_specs.append(
                        TLLinkSpec(
                            tl_link_index=tl_idx,
                            approach=approach,
                            direction=conn.getDirection().lower(),
                            outgoing_edge_id=conn.getTo().getID(),
                            request_index=request_idx if request_idx >= 0 else None,
                        )
                    )
        if not link_specs:
            continue
        n_links = max(link.tl_link_index for link in link_specs) + 1
        if phase_mode == "conflict-edge":
            green_states = build_conflict_phase_states(
                link_specs,
                n_links,
                are_foes=node.areFoes,
            )
        else:
            green_states = build_safe_phase_states(link_specs, n_links)
        all_red = "r" * n_links
        logic = ET.SubElement(
            root,
            "tlLogic",
            {
                "id": node.getID(),
                "type": "static",
                "programID": "movement_safe",
                "offset": "0",
            },
        )
        for green in green_states:
            ET.SubElement(logic, "phase", {"duration": "25", "state": green})
            ET.SubElement(logic, "phase", {"duration": "3", "state": _yellow_state(green)})
            ET.SubElement(logic, "phase", {"duration": "2", "state": all_red})
    _write_xml(tll_path, root)


def _approach_name(from_node: str, to_node: str) -> str:
    from_row, from_col = _parse_node_id(from_node)
    to_row, to_col = _parse_node_id(to_node)
    if from_row < to_row:
        return "north"
    if from_row > to_row:
        return "south"
    if from_col < to_col:
        return "west"
    if from_col > to_col:
        return "east"
    raise ValueError(f"Cannot infer approach for {from_node} -> {to_node}.")


def _yellow_state(green: str) -> str:
    return "".join("y" if char in {"G", "g"} else "r" for char in green)


def _print_summary(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
    connections: list[ConnectionSpec],
    net_path: Path,
) -> None:
    tls = [node.node_id for node in nodes if node.node_type == "traffic_light"]
    print(f"Wrote {net_path}")
    print(f"  nodes={len(nodes)} edges={len(edges)} traffic_lights={len(tls)} connections={len(connections)}")
    print(f"  traffic lights: {', '.join(tls)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rectangular SUMO grid network.")
    parser.add_argument("--rows", type=int, required=True, help="Number of grid rows")
    parser.add_argument("--cols", type=int, required=True, help="Number of grid columns")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--spacing", type=float, default=200.0, help="Distance between adjacent nodes in metres")
    parser.add_argument("--speed", type=float, default=13.89, help="Default speed in m/s")
    parser.add_argument("--no-netconvert", action="store_true", help="Only write plain XML files")
    parser.add_argument(
        "--phase-mode",
        choices=("conflict-edge", "protected"),
        default="conflict-edge",
        help="How to synthesize generated grid traffic-light phases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_grid(
        rows=args.rows,
        cols=args.cols,
        out_dir=args.out,
        spacing=args.spacing,
        speed=args.speed,
        netconvert=not args.no_netconvert,
        phase_mode=args.phase_mode,
    )


if __name__ == "__main__":
    main()
