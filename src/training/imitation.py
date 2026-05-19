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
    average_eval_metrics,
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
    n_eval_seeds: int = 5,
    dagger_beta_end: float = 0.0,
    demand_seed: Optional[int] = None,
    debug: bool = False,
    flow_range: tuple[int, int] = (300, 1000),
    min_green_steps: int = 2,
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
    n_eval_seeds :
        Number of demand seeds averaged in the final evaluation.
    dagger_beta_end :
        Final expert-driving probability (DAgger annealing). beta decays
        linearly from 1.0 (episode 0, always expert drives) to this value.
        Labels are always from the expert; only the environment stepping is
        mixed. 0.0 = model drives entirely by end; 1.0 = pure BC (no DAgger).
    demand_seed :
        When set, every episode uses identical traffic demand (overfit probe).
        Set env._rng to this seed before each env.reset().
    debug :
        When True, log per-step gradient norm and logit entropy to TensorBoard,
        plus per-episode raw observation statistics (min/max/mean per feature).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    env = TrafficEnv(
        cfg_path, gui=False, episode_length=episode_length, flow_range=flow_range, min_green_steps=min_green_steps
    )
    expert = GreedyExpert(env.junction_infos)
    builder = GraphBuilder(env._net, env.junction_infos)
    model = GATPolicy().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    n_junctions = len(env.junction_ids)
    steps_per_ep = episode_length // 15

    demand_tag = f'fixed seed {demand_seed} (overfit mode)' if demand_seed is not None else 'randomised'
    print(
        f'\nImitation Learning'
        f'\n  Junctions:     {n_junctions}'
        f'\n  Episodes:      {n_episodes}'
        f'\n  Episode len:   {episode_length} s  ({steps_per_ep} decision steps)'
        f'\n  DAgger beta:   1.0 → {dagger_beta_end} ({"disabled" if dagger_beta_end == 1.0 else "annealing"})'
        f'\n  Demand:        {demand_tag}'
        f'\n  Debug:         {debug}'
        f'\n  Device:        {dev}'
        f'\n  Parameters:    {model.n_parameters():,}'
        f'\n  Eval every:    {eval_every} episodes'
        f'\n  TensorBoard:   {log_dir}'
        f'\n  Checkpoints:   {checkpoint_dir}\n'
    )

    global_step = 0
    t0 = time.time()
    best_wait_density = float('inf')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(n_episodes):
        if demand_seed is not None:
            env._rng = np.random.default_rng(demand_seed)
        obs = env.reset()
        expert.reset()
        done = False

        # DAgger: beta = probability of using expert to step env this episode.
        # Labels are always expert. beta decays linearly 1.0 → dagger_beta_end.
        beta = 1.0 - (1.0 - dagger_beta_end) * (episode / max(1, n_episodes - 1))

        ep_loss: float = 0.0
        ep_match: float = 0.0
        ep_wait: float = 0.0
        ep_steps: int = 0
        ep_switches: int = 0
        ep_phases: list[int] = []  # all expert phase choices (all junctions, all steps)
        ep_raw_obs: list[np.ndarray] = []  # (debug) raw obs vectors before normalisation

        model.train()

        while not done:
            # Expert labels for this step (always from expert, regardless of driver).
            actions = expert.act()

            if debug:
                for raw_vec in obs.values():
                    ep_raw_obs.append(raw_vec)

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

            # Forward + loss (always on expert labels).
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

            if debug:
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * probs.log().clamp(min=-20)).sum(dim=-1).mean().item()
                writer.add_scalar('debug/grad_norm', grad_norm, global_step)
                writer.add_scalar('debug/logit_entropy', entropy, global_step)

            # DAgger: step with expert (prob beta) or model (prob 1-beta).
            if np.random.random() < beta:
                step_actions = actions
            else:
                with torch.no_grad():
                    model_phases = logits.argmax(dim=-1).cpu().tolist()
                step_actions = {jid: model_phases[i] for i, jid in enumerate(builder.junction_ids)}

            obs, _rewards, done, info = env.step(step_actions)
            ep_wait += info['global_wait']
            ep_switches += sum(1 for sw in info['switches'].values() if sw)

            for jid in env.junction_ids:
                expert.notify_applied(jid, step_actions[jid])

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
        writer.add_scalar('dagger/beta', beta, episode)

        if ep_phases:
            writer.add_histogram(
                'policy/expert/phase_dist',
                np.array(ep_phases, dtype=np.float32),
                episode,
            )

        if debug and ep_raw_obs:
            obs_mat = np.stack(ep_raw_obs)  # (T*N, 41)
            writer.add_scalar('debug/obs_min', float(obs_mat.min()), episode)
            writer.add_scalar('debug/obs_max', float(obs_mat.max()), episode)
            writer.add_scalar('debug/obs_mean', float(obs_mat.mean()), episode)
            writer.add_scalar('debug/obs_abs_mean', float(np.abs(obs_mat).mean()), episode)

        if (episode + 1) % print_every == 0 or episode == 0:
            elapsed = time.time() - t0
            print(
                f'  ep {episode + 1:4d}/{n_episodes}'
                f'  loss={avg_loss:.4f}'
                f'  match={avg_match:.3f}'
                f'  wait={avg_wait:.4f} s/m'
                f'  sw={switch_rate:.2f}'
                f'  β={beta:.2f}'
                f'  [{elapsed:5.0f}s]'
            )

        # ------------------------------------------------------------------
        # Mid-training evaluation: model vs expert on same demand
        # ------------------------------------------------------------------
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            model.eval()
            model_metrics = _run_and_log_eval(env, model, expert, builder, dev, writer, episode, eval_seeds=(42,))
            if model_metrics.avg_wait_density < best_wait_density:
                best_wait_density = model_metrics.avg_wait_density
                _save_checkpoint(model, builder, checkpoint_dir, tag='best')
                print(f'  → new best  wait_density={best_wait_density:.4f} s/m  (checkpoint saved)')
            model.train()

    # ------------------------------------------------------------------
    # Final evaluation — averaged over n_eval_seeds for robustness.
    # ------------------------------------------------------------------
    eval_seeds = tuple(range(42, 42 + n_eval_seeds))
    model.eval()
    print(f'\nRunning final evaluation (avg of {n_eval_seeds} seeds: {eval_seeds})...')
    _run_and_log_eval(env, model, expert, builder, dev, writer, n_episodes - 1, eval_seeds=eval_seeds)

    # ------------------------------------------------------------------
    # Freeze normalizer and save checkpoints.
    # ------------------------------------------------------------------
    builder.normalizer.freeze()
    env.close()  # idempotent if already closed by last eval
    writer.close()

    _save_checkpoint(model, builder, checkpoint_dir, tag='final')
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
    eval_seeds: tuple[int, ...] = (42,),
) -> EvalMetrics:
    """Run model + expert eval episodes and log averaged metrics to TensorBoard.

    Each seed produces one model episode and one expert episode on identical
    demand.  Results are averaged across all seeds before logging.  The
    training RNG is saved and restored so subsequent episodes stay stochastic.
    Returns the averaged model metrics so callers can track the best checkpoint.
    """
    orig_rng = env._rng
    model_policy = make_model_policy(model, builder, dev)

    model_runs: list[EvalMetrics] = []
    expert_runs: list[EvalMetrics] = []

    for seed in eval_seeds:
        env._rng = np.random.default_rng(seed)
        model_runs.append(run_eval_episode(env, model_policy))

        expert.reset()
        expert_policy = make_expert_policy(expert)
        env._rng = np.random.default_rng(seed)  # same seed → same demand
        expert_runs.append(
            run_eval_episode(
                env,
                expert_policy,
                on_step=lambda actions: _notify_expert(expert, actions),
            )
        )

    env._rng = orig_rng  # restore stochastic training RNG

    model_metrics = average_eval_metrics(model_runs)
    expert_metrics = average_eval_metrics(expert_runs)

    _log_eval_scalars(writer, model_metrics, expert_metrics, episode)
    _log_junction_metrics(writer, model_metrics, expert_metrics, episode)
    _print_eval_summary(model_metrics, expert_metrics, episode, n_seeds=len(eval_seeds))
    return model_metrics


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
        ('eval/avg_waiting_time', 'avg_waiting_time'),
        ('eval/avg_travel_time', 'avg_travel_time'),
        ('eval/throughput', 'throughput_per_hour'),
        ('eval/max_queue_length', 'max_queue_length'),
        ('eval/phase_switch_freq', 'phase_switch_freq'),
    ]
    for tag, attr in metrics:
        writer.add_scalars(
            tag,
            {
                'model': getattr(model_m, attr),
                'expert': getattr(expert_m, attr),
            },
            episode,
        )

    # wait_density shares a chart with the training series logged each episode.
    writer.add_scalars(
        'wait_density',
        {
            'eval_model': model_m.avg_wait_density,
            'eval_expert': expert_m.avg_wait_density,
        },
        episode,
    )


def _log_junction_metrics(
    writer: SummaryWriter,
    model_m: EvalMetrics,
    expert_m: EvalMetrics,
    episode: int,
) -> None:
    for jid in model_m.per_junction_wait_density:
        writer.add_scalars(
            f'junctions/{jid}/wait_density',
            {
                'model': model_m.per_junction_wait_density[jid],
                'expert': expert_m.per_junction_wait_density[jid],
            },
            episode,
        )

        writer.add_scalars(
            f'junctions/{jid}/max_queue',
            {
                'model': float(model_m.per_junction_max_queue[jid]),
                'expert': float(expert_m.per_junction_max_queue[jid]),
            },
            episode,
        )

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
    n_seeds: int = 1,
) -> None:
    seed_tag = f'avg of {n_seeds} seeds' if n_seeds > 1 else '1 seed'
    print(f'\n  [eval @ ep {episode + 1}  ({seed_tag})]')
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
    tag: str = 'final',
) -> None:
    model_path = os.path.join(directory, f'il_policy_{tag}.pt')
    torch.save(model.state_dict(), model_path)

    nd = builder.normalizer.state_dict()
    norm_path = os.path.join(directory, f'normalizer_{tag}.npz')
    np.savez(norm_path, n=np.array(nd['n']), mean=nd['mean'], M2=nd['M2'])

    print(f'  Saved model      → {model_path}')
    print(f'  Saved normalizer → {norm_path}')


def load_checkpoint(
    checkpoint_dir: str,
    device: Optional[str] = None,
    tag: str = 'best',
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

    # Resolve file paths: prefer tagged names, fall back to legacy bare names
    # so checkpoints saved before the tag scheme still load cleanly.
    def _resolve(stem: str, ext: str) -> str:
        tagged = os.path.join(checkpoint_dir, f'{stem}_{tag}{ext}')
        legacy = os.path.join(checkpoint_dir, f'{stem}{ext}')
        if os.path.exists(tagged):
            return tagged
        if os.path.exists(legacy):
            print(f'  [load_checkpoint] "{stem}_{tag}{ext}" not found, loading legacy "{stem}{ext}"')
            return legacy
        raise FileNotFoundError(f'No checkpoint found at {tagged} or {legacy}')

    model = GATPolicy()
    state = torch.load(_resolve('il_policy', '.pt'), map_location=device, weights_only=True)
    # strict=False allows loading IL checkpoints that predate the value head:
    # missing keys (value_head.*) keep their random initialisation; unexpected
    # keys are silently ignored so RL checkpoints load cleanly too.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [load_checkpoint] New keys (random init): {missing}')
    if unexpected:
        print(f'  [load_checkpoint] Ignored keys: {unexpected}')
    model.to(torch.device(device)).eval()

    npz = np.load(_resolve('normalizer', '.npz'))
    return model, {'n': int(npz['n']), 'mean': npz['mean'], 'M2': npz['M2']}
