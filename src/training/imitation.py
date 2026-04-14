"""Imitation Learning training loop (PLAN §8 — Stage 2).

Runs the GreedyExpert through TrafficEnv, collects (graph, label) pairs at
every 15-second decision step, and trains the GATPolicy with cross-entropy
loss.  Running normalizer statistics are accumulated throughout and frozen
at the end so they can be reused in RL stages.

TensorBoard logs
----------------
Per training step
    loss/cross_entropy          — cross-entropy vs expert labels
    accuracy/phase_match        — fraction of junctions where model == expert

Per training episode
    episode/avg_loss            — mean step loss
    episode/avg_match_rate      — mean per-step match rate
    episode/wait_density        — mean global wait density (s/m)
    policy/expert/switch_rate   — fraction of junctions that switched this episode
    policy/expert/phase_dist    — histogram of expert phase choices (phases 0-3)

Every eval_every episodes (model vs expert side-by-side)
    eval/{model,expert}/avg_waiting_time   — s / vehicle (from tripinfo)
    eval/{model,expert}/avg_travel_time    — s / vehicle (from tripinfo)
    eval/{model,expert}/throughput         — completed vehicles / hour
    eval/{model,expert}/max_queue_length   — peak halting vehicles on any lane
    eval/{model,expert}/phase_switch_freq  — switches / junction / minute
    eval/{model,expert}/wait_density       — mean global wait density (s/m)

Per-junction (every eval run)
    junctions/{id}/{model,expert}/wait_density   — avg wait density (s/m)
    junctions/{id}/{model,expert}/max_queue      — peak halting vehicles
    junctions/{id}/{model,expert}/phase_hist     — histogram of phase choices

Checkpoint
----------
    <checkpoint_dir>/il_policy.pt       — model state_dict
    <checkpoint_dir>/normalizer.npz     — RunningNormalizer state (n, mean, M2)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import TrafficEnv
from src.environment.expert import GreedyExpert
from src.utils.graph_builder import GraphBuilder
from src.model.gat_policy import GATPolicy
from src.training.eval_episode import (
    EvalMetrics,
    run_eval_episode,
    make_model_policy,
    make_expert_policy,
)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_il(
    cfg_path: str,
    n_episodes: int = 50,
    episode_length: int = 1200,
    lr: float = 3e-4,
    log_dir: str = 'runs/il',
    checkpoint_dir: str = 'checkpoints/il',
    device: Optional[str] = None,
    grad_clip: float = 0.5,
    eval_every: int = 5,
    print_every: int = 5,
) -> GATPolicy:
    """Run the IL training loop and return the trained model.

    Parameters
    ----------
    cfg_path :
        Path to the SUMO .sumocfg file.
    n_episodes :
        Number of training episodes.
    episode_length :
        Simulation seconds per episode (1200 s = Stage 2 default for speed).
    lr :
        Adam learning rate.
    log_dir :
        TensorBoard log directory.
    checkpoint_dir :
        Directory to save model and normalizer checkpoints.
    device :
        PyTorch device string.  Auto-detected if None.
    grad_clip :
        Max gradient norm.
    eval_every :
        Run model-vs-expert evaluation every N training episodes.
        Set to 0 to disable mid-training eval (a final eval still runs).
    print_every :
        Print console summary every N episodes.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    env = TrafficEnv(cfg_path, gui=False, episode_length=episode_length)
    expert = GreedyExpert(env.junction_infos)
    builder = GraphBuilder(env._net, env.junction_infos)
    model = GATPolicy().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    n_junctions = len(env.junction_ids)
    steps_per_ep = episode_length // 15

    print(
        f'\nImitation Learning'
        f'\n  Junctions:     {n_junctions}'
        f'\n  Episodes:      {n_episodes}'
        f'\n  Episode len:   {episode_length} s  ({steps_per_ep} decision steps)'
        f'\n  Device:        {dev}'
        f'\n  Parameters:    {model.n_parameters():,}'
        f'\n  Eval every:    {eval_every} episodes'
        f'\n  TensorBoard:   {log_dir}'
        f'\n  Checkpoints:   {checkpoint_dir}\n'
    )

    global_step = 0
    t0 = time.time()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(n_episodes):
        obs = env.reset()
        expert.reset()
        done = False

        ep_loss: float = 0.0
        ep_match: float = 0.0
        ep_wait: float = 0.0
        ep_steps: int = 0
        ep_switches: int = 0
        ep_phases: list[int] = []  # all expert phase choices (all junctions, all steps)

        model.train()

        while not done:
            # Expert labels for this step.
            actions = expert.act()

            # Build graph (update normalizer during training).
            graph = builder.build(obs, update_normalizer=True).to(dev)

            # Labels in canonical junction order.
            labels = torch.tensor(
                [actions[jid] for jid in builder.junction_ids],
                dtype=torch.long,
                device=dev,
            )

            # Collect expert phase choices for episode histogram.
            ep_phases.extend(labels.cpu().tolist())

            # Forward + loss.
            logits = model(graph)  # (N, 4)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # Per-step metrics.
            with torch.no_grad():
                match = (logits.argmax(dim=-1) == labels).float().mean().item()

            ep_loss += loss.item()
            ep_match += match
            ep_steps += 1
            global_step += 1

            writer.add_scalar('loss/cross_entropy', loss.item(), global_step)
            writer.add_scalar('accuracy/phase_match', match, global_step)

            # Step environment with expert actions.
            obs, _rewards, done, info = env.step(actions)
            ep_wait += info['global_wait']
            ep_switches += sum(1 for sw in info['switches'].values() if sw)

            # Keep expert hold-interval counts in sync.
            for jid in env.junction_ids:
                expert.notify_applied(jid, actions[jid])

        # ------------------------------------------------------------------
        # Episode-level logging
        # ------------------------------------------------------------------
        avg_loss = ep_loss / max(1, ep_steps)
        avg_match = ep_match / max(1, ep_steps)
        avg_wait = ep_wait / max(1, ep_steps)
        switch_rate = ep_switches / max(1, ep_steps * n_junctions)

        writer.add_scalar('episode/avg_loss', avg_loss, episode)
        writer.add_scalar('episode/avg_match_rate', avg_match, episode)
        writer.add_scalars('wait_density', {'training': avg_wait}, episode)
        writer.add_scalar('policy/expert/switch_rate', switch_rate, episode)

        if ep_phases:
            writer.add_histogram(
                'policy/expert/phase_dist',
                np.array(ep_phases, dtype=np.float32),
                episode,
            )

        if (episode + 1) % print_every == 0 or episode == 0:
            elapsed = time.time() - t0
            print(
                f'  ep {episode + 1:4d}/{n_episodes}'
                f'  loss={avg_loss:.4f}'
                f'  match={avg_match:.3f}'
                f'  wait={avg_wait:.4f} s/m'
                f'  sw={switch_rate:.2f}'
                f'  [{elapsed:5.0f}s]'
            )

        # ------------------------------------------------------------------
        # Mid-training evaluation: model vs expert on same demand
        # ------------------------------------------------------------------
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            model.eval()
            _run_and_log_eval(env, model, expert, builder, dev, writer, episode)
            model.train()

    # ------------------------------------------------------------------
    # Final evaluation.
    # ------------------------------------------------------------------
    model.eval()
    print('\nRunning final evaluation...')
    _run_and_log_eval(env, model, expert, builder, dev, writer, n_episodes - 1)

    # ------------------------------------------------------------------
    # Freeze normalizer and save checkpoints.
    # ------------------------------------------------------------------
    builder.normalizer.freeze()
    env.close()  # idempotent if already closed by last eval
    writer.close()

    _save_checkpoint(model, builder, checkpoint_dir)
    print(f'\nTraining complete.  Checkpoints saved to {checkpoint_dir}/')
    return model


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------


def _run_and_log_eval(
    env: TrafficEnv,
    model: GATPolicy,
    expert: GreedyExpert,
    builder: GraphBuilder,
    dev: torch.device,
    writer: SummaryWriter,
    episode: int,
) -> None:
    """Run model + expert eval episodes and log all metrics to TensorBoard.

    Both use the same config/demand file, so metrics are directly comparable.
    Each call to run_eval_episode starts and closes its own SUMO process.
    """
    model_policy = make_model_policy(model, builder, dev)
    model_metrics = run_eval_episode(env, model_policy)

    expert.reset()
    expert_policy = make_expert_policy(expert)
    expert_metrics = run_eval_episode(
        env,
        expert_policy,
        on_step=lambda actions: _notify_expert(expert, actions),
    )

    _log_eval_scalars(writer, model_metrics, expert_metrics, episode)
    _log_junction_metrics(writer, model_metrics, expert_metrics, episode)
    _print_eval_summary(model_metrics, expert_metrics, episode)


def _notify_expert(expert: GreedyExpert, actions: dict[str, int]) -> None:
    for jid, phase in actions.items():
        expert.notify_applied(jid, phase)


def _log_eval_scalars(
    writer: SummaryWriter,
    model_m: EvalMetrics,
    expert_m: EvalMetrics,
    episode: int,
) -> None:
    # add_scalars puts both series on the same chart in the Scalars tab.
    metrics = [
        ('eval/avg_waiting_time',  'avg_waiting_time'),
        ('eval/avg_travel_time',   'avg_travel_time'),
        ('eval/throughput',        'throughput_per_hour'),
        ('eval/max_queue_length',  'max_queue_length'),
        ('eval/phase_switch_freq', 'phase_switch_freq'),
    ]
    for tag, attr in metrics:
        writer.add_scalars(tag, {
            'model':  getattr(model_m,  attr),
            'expert': getattr(expert_m, attr),
        }, episode)

    # wait_density shares a chart with the training series logged each episode.
    writer.add_scalars('wait_density', {
        'eval_model':  model_m.avg_wait_density,
        'eval_expert': expert_m.avg_wait_density,
    }, episode)


def _log_junction_metrics(
    writer: SummaryWriter,
    model_m: EvalMetrics,
    expert_m: EvalMetrics,
    episode: int,
) -> None:
    for jid in model_m.per_junction_wait_density:
        writer.add_scalars(f'junctions/{jid}/wait_density', {
            'model':  model_m.per_junction_wait_density[jid],
            'expert': expert_m.per_junction_wait_density[jid],
        }, episode)

        writer.add_scalars(f'junctions/{jid}/max_queue', {
            'model':  float(model_m.per_junction_max_queue[jid]),
            'expert': float(expert_m.per_junction_max_queue[jid]),
        }, episode)

        # Histograms stay per-tag (no add_scalars equivalent for histograms).
        for label, m in (('model', model_m), ('expert', expert_m)):
            counts = m.per_junction_phase_counts[jid]
            phases_flat = [ph for ph, cnt in enumerate(counts) for _ in range(cnt)]
            if phases_flat:
                writer.add_histogram(
                    f'junctions/{jid}/{label}/phase_hist',
                    np.array(phases_flat, dtype=np.float32),
                    episode,
                )


def _print_eval_summary(
    model: EvalMetrics,
    expert: EvalMetrics,
    episode: int,
) -> None:
    print(f'\n  [eval @ ep {episode + 1}]')
    print(f'  {"Metric":<24}  {"Model":>10}  {"Expert":>10}')
    print(f'  {"-" * 48}')
    rows = [
        ('avg_waiting_time  (s)', model.avg_waiting_time, expert.avg_waiting_time),
        ('avg_travel_time   (s)', model.avg_travel_time, expert.avg_travel_time),
        ('throughput    (veh/h)', model.throughput_per_hour, expert.throughput_per_hour),
        ('max_queue      (vehs)', model.max_queue_length, expert.max_queue_length),
        ('switch_freq (/j/min)', model.phase_switch_freq, expert.phase_switch_freq),
        ('wait_density    (s/m)', model.avg_wait_density, expert.avg_wait_density),
    ]
    for name, mv, ev in rows:
        print(f'  {name:<24}  {mv:>10.3f}  {ev:>10.3f}')
    print()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: GATPolicy,
    builder: GraphBuilder,
    directory: str,
) -> None:
    model_path = os.path.join(directory, 'il_policy.pt')
    torch.save(model.state_dict(), model_path)

    nd = builder.normalizer.state_dict()
    norm_path = os.path.join(directory, 'normalizer.npz')
    np.savez(norm_path, n=np.array(nd['n']), mean=nd['mean'], M2=nd['M2'])

    print(f'  Saved model      → {model_path}')
    print(f'  Saved normalizer → {norm_path}')


def load_checkpoint(
    checkpoint_dir: str,
    device: Optional[str] = None,
) -> tuple[GATPolicy, dict]:
    """Load a saved IL checkpoint.

    Returns
    -------
    model :
        GATPolicy with loaded weights (eval mode).
    norm_state :
        dict with keys ``n``, ``mean``, ``M2`` — pass to
        ``RunningNormalizer.load_state_dict``.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GATPolicy()
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, 'il_policy.pt'),
            map_location=device,
            weights_only=True,
        )
    )
    model.to(torch.device(device)).eval()

    npz = np.load(os.path.join(checkpoint_dir, 'normalizer.npz'))
    return model, {'n': int(npz['n']), 'mean': npz['mean'], 'M2': npz['M2']}
