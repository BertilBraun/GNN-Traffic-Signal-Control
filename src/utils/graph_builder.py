"""Graph construction utilities for GNN traffic signal control.

GraphBuilder
    Converts per-junction observation dicts into torch_geometric.data.Data
    objects.  Edge connectivity and edge features are computed once from the
    static SUMO network at construction time; only node features change each
    decision step.

RunningNormalizer
    Online Welford z-score normalizer.  Accumulated during IL training stages
    and frozen before RL (per PLAN §3).

Typical usage
-------------
    builder    = GraphBuilder(env._net, env.junction_infos)
    normalizer = builder.normalizer   # convenience reference

    # During IL training (update running stats):
    graph = builder.build(obs, update_normalizer=True)

    # During eval / RL (stats frozen):
    normalizer.freeze()
    graph = builder.build(obs)
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# RunningNormalizer
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Online Welford z-score normalizer for fixed-size 1-D feature vectors.

    Accumulates mean and variance from individual samples (or small batches)
    using Welford's online algorithm, which is numerically stable.

    Parameters
    ----------
    dim :
        Feature vector length.
    epsilon :
        Small constant added to std to avoid division by zero.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8) -> None:
        self.dim     = dim
        self.epsilon = epsilon
        self.n       = 0
        self._mean   = np.zeros(dim, dtype=np.float64)
        self._M2     = np.zeros(dim, dtype=np.float64)  # sum of squared deviations
        self.frozen  = False

    # ------------------------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a batch of samples.

        Parameters
        ----------
        x : np.ndarray
            Shape (dim,) for a single sample, or (N, dim) for a batch.
            Ignored when the normalizer is frozen.
        """
        if self.frozen:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis]   # add batch dim → (1, dim)
        for sample in x:
            self.n += 1
            delta        = sample - self._mean
            self._mean  += delta / self.n
            delta2       = sample - self._mean
            self._M2    += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return z-score normalized copy of *x* (float32).

        If fewer than 2 samples have been seen, returns *x* unchanged.
        """
        x = np.asarray(x, dtype=np.float32)
        if self.n < 2:
            return x
        std  = np.sqrt(self._M2 / self.n + self.epsilon).astype(np.float32)
        mean = self._mean.astype(np.float32)
        return (x - mean) / std

    def freeze(self) -> None:
        """Prevent further stat updates.  Call before RL training."""
        self.frozen = True

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "n":       self.n,
            "mean":    self._mean.copy(),
            "M2":      self._M2.copy(),
            "frozen":  self.frozen,
            "dim":     self.dim,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, d: dict) -> None:
        self.n       = d["n"]
        self._mean   = np.array(d["mean"], dtype=np.float64)
        self._M2     = np.array(d["M2"],   dtype=np.float64)
        self.frozen  = d.get("frozen", False)


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Builds torch_geometric.data.Data from per-junction observation dicts.

    Constructed once from the static SUMO network topology.  Each call to
    :meth:`build` produces a fresh Data object with updated node features
    but the same (pre-computed) edge index and edge attributes.

    Edge connectivity
        Directed edge A → B is included whenever there is a SUMO road segment
        whose from-node is junction A and whose to-node is junction B, and
        both A and B are managed TL junctions.  Parallel SUMO edges between
        the same junction pair are averaged.

    Edge features (PLAN §1): [road_length_m, n_lanes, speed_limit_m_s]
        Statically z-score normalised across all edges at construction time.

    Node features
        The 41-dim observation vector, optionally z-score normalised via the
        :attr:`normalizer` (RunningNormalizer).  Pass
        ``update_normalizer=True`` during IL data collection; call
        ``normalizer.freeze()`` before RL.
    """

    def __init__(self, net, junction_infos: dict) -> None:
        """
        Parameters
        ----------
        net :
            sumolib.net.Net object (from ``sumolib.net.readNet``).
        junction_infos :
            ``dict[junction_id → JunctionInfo]`` — only TL junctions.
        """
        # Canonical ordering: sorted junction IDs (must be stable across calls).
        self.junction_ids: list[str] = sorted(junction_infos.keys())
        self._jid_to_idx: dict[str, int] = {
            jid: i for i, jid in enumerate(self.junction_ids)
        }
        tl_set = set(self.junction_ids)

        # ------------------------------------------------------------------
        # Build directed edges between TL junctions.
        # Accumulate raw features for parallel edges, then average.
        # ------------------------------------------------------------------
        _raw: dict[tuple[int, int], list[tuple[float, int, float]]] = {}

        for edge in net.getEdges():
            from_id = edge.getFromNode().getID()
            to_id   = edge.getToNode().getID()
            if from_id not in tl_set or to_id not in tl_set:
                continue
            src = self._jid_to_idx[from_id]
            dst = self._jid_to_idx[to_id]
            if src == dst:
                continue    # skip self-loops (shouldn't happen, but be safe)
            length  = float(edge.getLength())
            n_lanes = int(edge.getLaneNumber())
            speed   = float(edge.getSpeed())
            _raw.setdefault((src, dst), []).append((length, n_lanes, speed))

        if not _raw:
            raise RuntimeError(
                "GraphBuilder found no direct SUMO edges between TL junctions. "
                "Ensure the network has road segments directly connecting "
                "signalized junctions."
            )

        # Sort for deterministic ordering.
        edge_keys = sorted(_raw.keys())
        src_list, dst_list, feat_list = [], [], []
        for src, dst in edge_keys:
            rows = _raw[(src, dst)]
            src_list.append(src)
            dst_list.append(dst)
            feat_list.append([
                float(np.mean([r[0] for r in rows])),   # avg length
                float(np.mean([r[1] for r in rows])),   # avg lanes
                float(np.mean([r[2] for r in rows])),   # avg speed
            ])

        # Z-score normalise edge features over all edges (static).
        ef = np.array(feat_list, dtype=np.float32)   # (E, 3)
        ef_mean = ef.mean(axis=0)
        ef_std  = ef.std(axis=0) + 1e-8
        ef_norm = (ef - ef_mean) / ef_std

        # Store as CPU tensors; caller moves to device if needed.
        self._edge_index: torch.Tensor = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )                                                # (2, E)
        self._edge_attr: torch.Tensor = torch.tensor(
            ef_norm, dtype=torch.float32
        )                                                # (E, 3)

        # Node-feature normalizer — lifecycle managed by the training loop.
        self.normalizer = RunningNormalizer(dim=41)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        obs: dict[str, np.ndarray],
        update_normalizer: bool = False,
    ) -> Data:
        """Produce a torch_geometric Data object for the current timestep.

        Parameters
        ----------
        obs :
            ``dict[junction_id → np.ndarray(41,)]`` — current observations.
        update_normalizer :
            If True, feed the stacked node features into the running
            normalizer before normalising.  Set to True during IL data
            collection; leave False during eval / RL rollouts.

        Returns
        -------
        torch_geometric.data.Data with:
            ``x``          — (N, 41) float32 node features (normalised)
            ``edge_index`` — (2, E) int64
            ``edge_attr``  — (E, 3) float32 edge features (pre-normalised)
        """
        x_raw = np.stack(
            [obs[jid] for jid in self.junction_ids],
            axis=0,
        )   # (N, 41), float32

        if update_normalizer:
            self.normalizer.update(x_raw)

        x_norm = self.normalizer.normalize(x_raw)   # (N, 41), float32

        return Data(
            x          = torch.tensor(x_norm, dtype=torch.float32),
            edge_index = self._edge_index,
            edge_attr  = self._edge_attr,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_junctions(self) -> int:
        return len(self.junction_ids)

    @property
    def n_edges(self) -> int:
        return self._edge_index.shape[1]
