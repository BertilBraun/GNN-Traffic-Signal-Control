"""GATv2 policy network for multi-junction traffic signal control.

Architecture (PLAN §7)
----------------------
    Encoder:          node_dim → hidden → hidden  (MLP, ReLU, LayerNorm)
    Message passing:  n_layers × GATv2Conv(hidden, heads, edge_dim)
                      + ReLU + residual + LayerNorm
    Classifier head:  hidden → 64 → n_classes  (per-node MLP)

A single forward pass on the whole junction graph returns per-node logits
of shape (N, n_classes), where N is the number of junctions.

Notes
-----
- ``add_self_loops=False`` in every GATv2Conv because the residual connection
  already provides each node with its own representation, and self-loops
  require matching edge-feature tensors that we don't have for them.
- With ``concat=True`` (default), GATv2Conv outputs ``heads × head_dim``
  per node.  We set ``head_dim = hidden // heads`` so the output stays at
  ``hidden`` dimensions throughout, making residual addition trivially shaped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from src.environment.phase_schema import NUM_PHASES, OBS_DIM


class GATPolicy(nn.Module):
    """GATv2-based traffic signal policy with shared parameters across junctions.

    Parameters
    ----------
    node_dim :
        Input feature dimension per junction (45).
    hidden :
        Hidden dimension throughout the network (128).
    heads :
        Number of attention heads per GATv2 layer (4).
        Must divide *hidden* evenly.
    n_layers :
        Number of GATv2 message-passing layers (3).
    edge_dim :
        Edge feature dimension (3: length, lanes, speed).
    n_classes :
        Number of output phase classes (8).
    """

    def __init__(
        self,
        node_dim:  int = OBS_DIM,
        hidden:    int = 128,
        heads:     int = 4,
        n_layers:  int = 3,
        edge_dim:  int = 3,
        n_classes: int = NUM_PHASES,
    ) -> None:
        super().__init__()

        if hidden % heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by heads ({heads})."
            )
        head_dim = hidden // heads

        # ------------------------------------------------------------------
        # Encoder MLP: node_dim → hidden → hidden
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # ------------------------------------------------------------------
        # GATv2 message-passing layers
        # ------------------------------------------------------------------
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels    = hidden,
                out_channels   = head_dim,   # per head; concat → hidden
                heads          = heads,
                edge_dim       = edge_dim,
                add_self_loops = False,
                concat         = True,
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])

        # ------------------------------------------------------------------
        # Classifier (actor) head: hidden → 64 → n_classes
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        # ------------------------------------------------------------------
        # Value (critic) head: hidden → 64 → 1  (per-node scalar V(s))
        # Shares the encoder + GAT backbone with the actor.
        # ------------------------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Shared backbone
    # ------------------------------------------------------------------

    def _encode(self, data: Data) -> torch.Tensor:
        """Run the encoder + GATv2 message-passing layers.

        Returns the per-node embedding of shape (N, hidden).
        """
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr

        x = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = norm(x + residual)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: Data) -> torch.Tensor:
        """Compute per-node phase logits.

        Parameters
        ----------
        data :
            torch_geometric.data.Data with fields ``x`` (N, node_dim),
            ``edge_index`` (2, E), ``edge_attr`` (E, edge_dim).

        Returns
        -------
        logits : torch.Tensor of shape (N, n_classes)
        """
        return self.classifier(self._encode(data))

    def forward_actor_critic(
        self, data: Data
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward pass and return both actor logits and critic values.

        Parameters
        ----------
        data :
            torch_geometric.data.Data (or a PyG Batch of multiple graphs).

        Returns
        -------
        logits : (N, n_classes) — unnormalised action log-probabilities
        values : (N,)           — per-node state-value estimates V(s)
        """
        h = self._encode(data)
        return self.classifier(h), self.value_head(h).squeeze(-1)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        """Greedy (argmax) phase prediction.  Returns (N,) LongTensor."""
        return self.forward(data).argmax(dim=-1)

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
