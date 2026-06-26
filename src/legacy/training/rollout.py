"""PPO rollout buffer for multi-junction traffic signal control.

Collects one or more complete episodes worth of (graph, action, log_prob,
reward, value, done) transitions, then computes GAE advantages so the PPO
update loop can sample random minibatches.

Design notes
------------
- Graphs are stored as CPU ``torch_geometric.data.Data`` objects.  Only the
  node-feature tensor ``x`` differs per step; ``edge_index`` and ``edge_attr``
  are references to the same pre-computed tensors throughout an episode.
- Per-step tensors (actions, log_probs, rewards, values) live on CPU.
- ``iterate_minibatches`` batches *timestep* graphs (not individual junctions)
  because the GNN must see the full graph topology to produce any output.
  With N junctions per graph, a minibatch of B timestep-graphs contains B·N
  junction-transitions, matching the PLAN's "minibatch_size=256" when B=16 and
  N=16.
- GAE is computed per-junction independently, respecting episode boundaries
  flagged by the ``done`` boolean.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data


class RolloutBuffer:
    """Stores multi-episode rollout data and computes GAE advantages.

    Parameters
    ----------
    n_junctions :
        Number of TL junctions (N).  Used to validate tensor shapes.
    gamma :
        PPO discount factor γ (default 0.99).
    lam :
        GAE smoothing parameter λ (default 0.95).
    """

    def __init__(
        self,
        n_junctions: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.n_junctions = n_junctions
        self.gamma = gamma
        self.lam = lam
        self.clear()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the buffer (call before each collection phase)."""
        self._graphs: list[Data] = []
        self._actions: list[torch.Tensor] = []  # (N,) LongTensor each
        self._log_probs: list[torch.Tensor] = []  # (N,) FloatTensor each
        self._rewards: list[torch.Tensor] = []  # (N,) FloatTensor each
        self._values: list[torch.Tensor] = []  # (N,) FloatTensor each
        self._dones: list[bool] = []  # True at episode end

        # Populated by compute_returns_and_advantages()
        self.advantages: torch.Tensor | None = None  # (T, N)
        self.returns: torch.Tensor | None = None  # (T, N)
        self.actions: torch.Tensor | None = None  # (T, N)
        self.old_log_probs: torch.Tensor | None = None  # (T, N)

    def add(
        self,
        graph: Data,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        done: bool,
    ) -> None:
        """Append one decision-step to the buffer.

        All tensors are moved to CPU before storing.

        Parameters
        ----------
        graph :
            torch_geometric.data.Data for this timestep (N nodes).
        actions :
            (N,) LongTensor — sampled phase for each junction.
        log_probs :
            (N,) FloatTensor — log-probability of the sampled action.
        rewards :
            (N,) FloatTensor — per-junction reward for this step.
        values :
            (N,) FloatTensor — critic estimate V(s) for this step.
        done :
            True if this is the final step of an episode.
        """
        self._graphs.append(graph.cpu())
        self._actions.append(actions.cpu())
        self._log_probs.append(log_probs.cpu())
        self._rewards.append(rewards.cpu())
        self._values.append(values.cpu())
        self._dones.append(done)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(self, use_mc_targets: bool = False) -> None:
        """Compute GAE advantages and critic targets.

        Must be called after all transitions have been added and before
        ``iterate_minibatches``.  Populates ``self.advantages``,
        ``self.returns``, ``self.actions``, and ``self.old_log_probs``.

        Parameters
        ----------
        use_mc_targets :
            When True, use pure Monte Carlo returns (Σ γ^k r_{t+k}) as the
            critic target instead of the standard GAE returns (A_t + V_t).
            MC returns are a stable, value-independent target — useful during
            value warmup where bootstrapped targets form a moving chasing loop.
            GAE advantages are always computed and stored for policy updates.
        """
        T = len(self._graphs)
        if T == 0:
            raise RuntimeError('Cannot compute advantages: buffer is empty.')

        rewards = torch.stack(self._rewards)  # (T, N)
        values = torch.stack(self._values)  # (T, N)
        dones = torch.tensor(self._dones, dtype=torch.float32)  # (T,)  1.0 = done

        advantages = torch.zeros(T, self.n_junctions)
        last_gae = torch.zeros(self.n_junctions)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(self.n_junctions)
                next_done = 1.0
            else:
                next_value = values[t + 1]
                next_done = dones[t].item()  # done at t → next is new ep

            delta = rewards[t] + self.gamma * next_value * (1.0 - next_done) - values[t]
            last_gae = delta + self.gamma * self.lam * (1.0 - next_done) * last_gae
            advantages[t] = last_gae

        if use_mc_targets:
            # Pure Monte Carlo returns: Σ_{k=0}^{T-t-1} γ^k r_{t+k}
            # No value bootstrapping → stable target regardless of V(s) accuracy.
            mc_returns = torch.zeros(T, self.n_junctions)
            running = torch.zeros(self.n_junctions)
            for t in reversed(range(T)):
                next_done = dones[t].item() if t < T - 1 else 1.0
                running = rewards[t] + self.gamma * running * (1.0 - next_done)
                mc_returns[t] = running
            critic_targets = mc_returns
        else:
            critic_targets = advantages + values  # standard GAE returns

        self.advantages = advantages  # (T, N)
        self.returns = critic_targets  # (T, N)
        self.actions = torch.stack(self._actions)  # (T, N)
        self.old_log_probs = torch.stack(self._log_probs)  # (T, N)

    # ------------------------------------------------------------------
    # Minibatch iteration
    # ------------------------------------------------------------------

    def iterate_minibatches(
        self,
        n_graphs_per_batch: int,
        device: torch.device | str = 'cpu',
    ):
        """Yield shuffled minibatches of timestep-graphs for PPO updates.

        Each yielded batch contains ``n_graphs_per_batch`` randomly selected
        timestep-graphs.  All tensors are moved to ``device``.

        Parameters
        ----------
        n_graphs_per_batch :
            Number of full graphs (timesteps) per minibatch.  With N=16
            junctions, ``n_graphs_per_batch=16`` gives 256 junction-transitions
            per minibatch, matching the PLAN's minibatch_size setting.
        device :
            Target device for the returned tensors.

        Yields
        ------
        batch_graph : PyG Batch  — combined graph of (n_graphs_per_batch × N) nodes
        acts        : (B·N,) LongTensor
        old_lps     : (B·N,) FloatTensor
        advs        : (B·N,) FloatTensor  (normalised within this minibatch)
        rets        : (B·N,) FloatTensor
        """
        if self.advantages is None:
            raise RuntimeError('Call compute_returns_and_advantages() before iterating minibatches.')
        dev = torch.device(device)
        T = len(self._graphs)
        idx = torch.randperm(T)

        for start in range(0, T, n_graphs_per_batch):
            batch_idx = idx[start : start + n_graphs_per_batch]

            # Batch the selected graphs into one PyG Batch object.
            batch_graph = Batch.from_data_list([self._graphs[i] for i in batch_idx]).to(dev)

            # Flatten (B, N) → (B·N,) to match GNN output ordering.
            acts = self.actions[batch_idx].to(dev).reshape(-1)
            old_lps = self.old_log_probs[batch_idx].to(dev).reshape(-1)
            advs = self.advantages[batch_idx].to(dev).reshape(-1)
            rets = self.returns[batch_idx].to(dev).reshape(-1)

            # Normalise advantages within the minibatch for stability.
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            yield batch_graph, acts, old_lps, advs, rets

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of timestep-graphs stored."""
        return len(self._graphs)

    @property
    def total_transitions(self) -> int:
        """Total number of (junction, timestep) transition pairs."""
        return len(self._graphs) * self.n_junctions
