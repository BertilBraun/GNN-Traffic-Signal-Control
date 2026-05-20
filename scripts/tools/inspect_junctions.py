"""Print details for a list of junction IDs in a SUMO network.

Usage
-----
    python scripts/inspect_junctions.py configs/city/city.net.xml jid1 jid2 ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

import sumolib  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    net = sumolib.net.readNet(sys.argv[1], withConnections=True)
    node_map = {n.getID(): n for n in net.getNodes()}
    for jid in sys.argv[2:]:
        node = node_map.get(jid)
        if node is None:
            print(f'\n{jid}: NOT FOUND')
            continue
        ins = node.getIncoming()
        outs = node.getOutgoing()
        in_neighbors = {e.getFromNode().getID() for e in ins}
        out_neighbors = {e.getToNode().getID() for e in outs}
        unique = (in_neighbors | out_neighbors) - {jid}
        n_lanes = sum(len(e.getLanes()) for e in ins)
        print(f'\n{jid}')
        print(f'  type: {node.getType()}')
        print(f'  in={len(ins)} out={len(outs)} unique_neighbors={len(unique)} total_in_lanes={n_lanes}')
        for e in ins:
            lanes = e.getLanes()
            print(
                f'    IN  {e.getID()} from={e.getFromNode().getID():40s} len={e.getLength():6.1f}m nlanes={len(lanes)}'
            )
        for e in outs:
            lanes = e.getLanes()
            print(f'    OUT {e.getID()} to  ={e.getToNode().getID():40s} len={e.getLength():6.1f}m nlanes={len(lanes)}')


if __name__ == '__main__':
    main()
