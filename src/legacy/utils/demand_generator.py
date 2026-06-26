"""Random traffic demand generator for SUMO environments.

At construction, scans the network for "entry" edges (non-TL → TL) and
"exit" edges (TL → non-TL), then pre-computes the shortest path for every
valid (entry, exit) pair using sumolib.  At each episode, ``generate()``
draws per-pair flow rates from a Dirichlet-like distribution whose total
sums to a uniform sample from ``flow_range``, and writes a temporary SUMO
routes XML file.

Typical usage (handled automatically by TrafficEnv)
----------------------------------------------------
    randomizer = DemandRandomizer(net, flow_range=(300, 1000))
    print(f"{randomizer.n_od_pairs} O-D pairs discovered")

    rng  = np.random.default_rng(42)
    path = randomizer.generate(episode_length=1200, rng=rng)
    # pass path to SUMO via --route-files
    # ... after episode ...
    os.unlink(path)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


class DemandRandomizer:
    """Pre-computes O-D routes and writes randomised demand files per episode.

    Parameters
    ----------
    net :
        ``sumolib.net.Net`` object (already loaded by TrafficEnv).
    flow_range :
        ``(min, max)`` total vehicles-per-hour across the whole network.
        Each episode draws a target from ``Uniform(min, max)`` and
        distributes it across O-D pairs via Exponential weights.
    min_rate :
        O-D pairs assigned a rate below this threshold (veh/h) are omitted
        from the file to suppress near-zero flows.
    """

    def __init__(
        self,
        net,  # sumolib.net.Net
        flow_range: tuple[int, int] = (300, 1000),
        min_rate: float = 5.0,
    ) -> None:
        self._flow_range = flow_range
        self._min_rate = min_rate
        # Each entry: (from_edge_id, to_edge_id, space-separated edge-id string)
        self._od_routes: list[tuple[str, str, str]] = []
        self._build_od_table(net)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_od_table(self, net) -> None:
        """Identify entry/exit edges and pre-compute sumolib shortest paths."""
        tl_types = {
            'traffic_light',
            'traffic_light_unregulated',
            'traffic_light_right_on_red',
        }

        def is_tl(node) -> bool:
            return node.getType() in tl_types

        entry_edges = [e for e in net.getEdges() if not is_tl(e.getFromNode()) and is_tl(e.getToNode())]
        exit_edges = [e for e in net.getEdges() if is_tl(e.getFromNode()) and not is_tl(e.getToNode())]

        for src in entry_edges:
            for dst in exit_edges:
                if src is dst:
                    continue
                result = net.getShortestPath(src, dst)
                if result is None:
                    continue
                path, _cost = result
                if path is None or len(path) < 2:
                    continue
                edge_ids = ' '.join(e.getID() for e in path)
                self._od_routes.append((src.getID(), dst.getID(), edge_ids))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_od_pairs(self) -> int:
        """Number of valid O-D pairs found in the network."""
        return len(self._od_routes)

    def generate(
        self,
        episode_length: int,
        rng: np.random.Generator,
    ) -> str:
        """Write a temporary SUMO routes XML file and return its path.

        The caller is responsible for deleting the file after use.

        Parameters
        ----------
        episode_length :
            Simulation seconds for the episode — sets ``end`` on each
            ``<flow>`` element.
        rng :
            NumPy random Generator (seeded by TrafficEnv).

        Returns
        -------
        str
            Absolute path to the temporary routes XML file.
        """
        n = len(self._od_routes)
        if n == 0:
            raise RuntimeError(
                'DemandRandomizer found no valid O-D routes in the network. '
                'Check that the network has traffic-light junctions.'
            )

        # Dirichlet-like: Exp(1) weights normalised to a uniform total.
        target_total = float(rng.uniform(self._flow_range[0], self._flow_range[1]))
        weights = rng.exponential(1.0, size=n)
        rates = weights / weights.sum() * target_total  # sums to target_total

        fd, path = tempfile.mkstemp(suffix='.xml', prefix='demand_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('<routes>\n\n')

            # ---- Route definitions ----
            for i, (_from_e, _to_e, edges) in enumerate(self._od_routes):
                f.write(f'    <route id="r{i}" edges="{edges}"/>\n')

            f.write('\n')

            # ---- Flow elements (skip near-zero rates) ----
            for i, (_from_e, _to_e, _) in enumerate(self._od_routes):
                rate = float(rates[i])
                if rate < self._min_rate:
                    continue
                f.write(
                    f'    <flow id="f{i}" route="r{i}"'
                    f' begin="0" end="{episode_length}"'
                    f' vehsPerHour="{rate:.1f}"'
                    f' departLane="best" departSpeed="random"/>\n'
                )

            f.write('\n</routes>\n')

        return path
