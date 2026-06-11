"""Inspect SUMO movement conflicts and synthesized phase sets."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Point it to your SUMO installation directory."
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib  # noqa: E402

from scripts.generate_grid_network import _approach_name  # noqa: E402
from src.movement.phase_synthesis import (  # noqa: E402
    TrafficLightLinkSpec,
    build_conflict_phase_states,
)
from src.movement.schema import LaneId  # noqa: E402


class ConflictAnalysisMode(str, Enum):
    SUMO_FOES = "sumo-foes"
    CONFLICT_EDGE = "conflict-edge"


@dataclass(frozen=True)
class AnalyzedTrafficLightLink:
    traffic_light_link_index: int
    request_index: int
    approach: str
    direction: str
    incoming_lane_id: LaneId
    outgoing_lane_id: LaneId
    outgoing_edge_id: str

    def to_synthesis_spec(self, include_outgoing_edge: bool) -> TrafficLightLinkSpec:
        return TrafficLightLinkSpec(
            traffic_light_link_index=self.traffic_light_link_index,
            incoming_lane_id=self.incoming_lane_id,
            outgoing_lane_id=self.outgoing_lane_id,
            outgoing_edge_id=self.outgoing_edge_id if include_outgoing_edge else None,
            request_index=self.request_index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print movement conflicts and generated phase states for one SUMO TLS node.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--net", type=Path, required=True, help="Path to SUMO .net.xml")
    parser.add_argument("--tls", required=True, help="Traffic-light node id")
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ConflictAnalysisMode),
        default=ConflictAnalysisMode.CONFLICT_EDGE.value,
        help="Conflict rule used for generated phase states",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_mode = ConflictAnalysisMode(args.mode)
    net = sumolib.net.readNet(str(args.net), withConnections=True, withFoes=True)
    node = net.getNode(args.tls)
    links = _collect_tls_links(node, args.tls)
    number_of_links = max(link.traffic_light_link_index for link in links) + 1

    print(f"TLS {args.tls}: {len(links)} controlled movements")
    for link in sorted(links, key=lambda item: item.traffic_light_link_index):
        print(
            f"{link.traffic_light_link_index:2d} request={link.request_index:2d} "
            f"{link.approach:5s} {link.direction:1s} -> {link.outgoing_edge_id}"
        )

    if analysis_mode == ConflictAnalysisMode.SUMO_FOES:
        phase_links = [
            link.to_synthesis_spec(include_outgoing_edge=False)
            for link in links
        ]
    else:
        phase_links = [
            link.to_synthesis_spec(include_outgoing_edge=True)
            for link in links
        ]

    states = build_conflict_phase_states(
        phase_links,
        number_of_links=number_of_links,
        are_foes=node.areFoes,
    )
    print(f"\n{analysis_mode.value} maximal phase states: {len(states)}")
    for state in states:
        enabled = [str(idx) for idx, char in enumerate(state) if char == "G"]
        print(f"{state}  links={','.join(enabled)}")


def _collect_tls_links(node, tls_id: str) -> list[AnalyzedTrafficLightLink]:
    links: list[AnalyzedTrafficLightLink] = []
    for incoming in node.getIncoming():
        approach = _approach_name(incoming.getFromNode().getID(), node.getID())
        for outgoing in incoming.getOutgoing():
            for conn in incoming.getConnections(outgoing):
                if conn.getTLSID() != tls_id:
                    continue
                tl_idx = conn.getTLLinkIndex()
                request_idx = conn.getJunctionIndex()
                if tl_idx < 0 or request_idx < 0:
                    continue
                links.append(
                    AnalyzedTrafficLightLink(
                        traffic_light_link_index=tl_idx,
                        request_index=request_idx,
                        approach=approach,
                        direction=conn.getDirection().lower(),
                        incoming_lane_id=LaneId(conn.getFromLane().getID()),
                        outgoing_lane_id=LaneId(conn.getToLane().getID()),
                        outgoing_edge_id=conn.getTo().getID(),
                    )
                )
    return sorted(links, key=lambda link: link.traffic_light_link_index)


if __name__ == "__main__":
    main()
