"""Inspect SUMO movement conflicts and synthesized phase sets."""
from __future__ import annotations

import argparse
import os
import sys
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

from scripts.generate_grid_network import (  # noqa: E402
    TLLinkSpec,
    _approach_name,
    build_conflict_phase_states,
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
        choices=("sumo-foes", "conflict-edge"),
        default="conflict-edge",
        help="Conflict rule used for generated phase states",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = sumolib.net.readNet(str(args.net), withConnections=True, withFoes=True)
    node = net.getNode(args.tls)
    links = _collect_tls_links(node, args.tls)
    n_links = max(link.tl_link_index for link in links) + 1

    print(f"TLS {args.tls}: {len(links)} controlled movements")
    for link in sorted(links, key=lambda item: item.tl_link_index):
        print(
            f"{link.tl_link_index:2d} request={link.request_index:2d} "
            f"{link.approach:5s} {link.direction:1s} -> {link.outgoing_edge_id}"
        )

    if args.mode == "sumo-foes":
        phase_links = [
            TLLinkSpec(
                link.tl_link_index,
                link.approach,
                link.direction,
                None,
                link.request_index,
            )
            for link in links
        ]
    else:
        phase_links = links

    states = build_conflict_phase_states(
        phase_links,
        n_links=n_links,
        are_foes=node.areFoes,
    )
    print(f"\n{args.mode} maximal phase states: {len(states)}")
    for state in states:
        enabled = [str(idx) for idx, char in enumerate(state) if char == "G"]
        print(f"{state}  links={','.join(enabled)}")


def _collect_tls_links(node, tls_id: str) -> list[TLLinkSpec]:
    links: list[TLLinkSpec] = []
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
                    TLLinkSpec(
                        tl_link_index=tl_idx,
                        approach=approach,
                        direction=conn.getDirection().lower(),
                        outgoing_edge_id=conn.getTo().getID(),
                        request_index=request_idx,
                    )
                )
    return sorted(links, key=lambda link: link.tl_link_index)


if __name__ == "__main__":
    main()
