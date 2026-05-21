"""PPO reinforcement learning training loop (PLAN §8 — Stages 3/4).

Loads from an IL checkpoint, adds a value head (randomly initialised), then
fine-tunes with Proximal Policy Optimisation on the 4×4 irregular grid.

Algorithm outline (one iteration)
----------------------------------
1. Collect N_ep episodes.  The first BURN_IN_STEPS decision-steps of each
   episode use a fixed-time phase-cycling controller to bring the network to
   near-steady-state before the GNN takes over.
2. Compute GAE advantages (γ=0.99, λ=0.95) across the full buffer.
3. Run K=10 PPO epochs, each shuffling the buffer into minibatches of
   n_graphs_per_batch timestep-graphs.
4. Optionally run evaluation episodes (model vs expert) every eval_every
   iterations.

TensorBoard logs
----------------
Per update iteration:
    loss/policy            — PPO clipped surrogate loss
    loss/value             — critic MSE loss
    loss/entropy           — mean entropy of the policy
    loss/total             — combined weighted loss

Per episode (collected during rollout):
    episode/mean_reward    — mean total reward per junction per episode
    episode/mean_return    — mean bootstrapped return (after GAE)

Per eval run (every eval_every iterations):
    eval/{model,expert}/avg_waiting_time
    eval/{model,expert}/avg_travel_time
    eval/{model,expert}/throughput
    eval/{model,expert}/max_queue_length
    eval/{model,expert}/phase_switch_freq
    eval/{model,expert}/wait_density
    junctions/{id}/{model,expert}/wait_density
    junctions/{id}/{model,expert}/max_queue

Checkpoints
-----------
    <checkpoint_dir>/rl_policy_iter<N>.pt   — model state_dict every save_every
    <checkpoint_dir>/rl_policy_latest.pt    — always the most recent save
    <checkpoint_dir>/normalizer.npz         — normalizer state (copied from IL)
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import TrafficEnv
from src.environment.expert import GreedyExpert
from src.environment.phase_schema import NUM_PHASES
from src.model.gat_policy import GATPolicy
from src.training.imitation import load_checkpoint
from src.training.rollout import RolloutBuffer
from src.training.eval_episode import (
    EvalMetrics,
    average_eval_metrics,
    run_eval_episode,
    make_model_policy,
    make_expert_policy,
)
from src.utils.graph_builder import GraphBuilder


# ---------------------------------------------------------------------------
# Fixed-time burn-in controller
# ---------------------------------------------------------------------------

BURN_IN_CYCLE_LENGTH = 2  # decision intervals per phase during burn-in


def _fixed_time_actions(
    junction_ids: list[str],
    step_in_episode: int,
) -> dict[str, int]:
    """Round-robin phase assignment for burn-in.

    Cycles through phases 0→1→...→7→0 with each phase held for
    BURN_IN_CYCLE_LENGTH decision intervals (30 s per phase → 240 s cycle).
    All junctions share the same cycle offset so they are synchronised, which
    avoids any single approach being starved during burn-in.
    """
    phase = (step_in_episode // BURN_IN_CYCLE_LENGTH) % NUM_PHASES
    return {jid: phase for jid in junction_ids}


# ---------------------------------------------------------------------------
# Backbone freeze helper
# ---------------------------------------------------------------------------


def _set_backbone_grad(model: GATPolicy, requires_grad: bool) -> None:
    """Enable or disable gradients for the shared backbone + classifier.

    Called at the start of each PPO update step.  During value warmup
    (requires_grad=False) only the value_head trains; the backbone and
    classifier are frozen so random value-head gradients cannot corrupt
    the IL-trained policy weights.
    """
    for module in (model.encoder, model.convs, model.norms, model.classifier):
        for p in module.parameters():
            p.requires_grad = requires_grad


# ---------------------------------------------------------------------------
# In-process (sequential) collection helper
# ---------------------------------------------------------------------------


def _collect_episode_inline(
    env: TrafficEnv,
    model: GATPolicy,
    builder: GraphBuilder,
    junction_ids: list[str],
    burn_in_steps: int,
) -> dict:
    """Collect one episode using the already-initialised env and builder.

    Called in the main process — zero reinitialisation overhead.
    Returns the same dict schema as _collect_episode_worker so the caller
    can merge both paths identically.
    """
    obs = env.reset()
    done = False

    for bi in range(burn_in_steps):
        if done:
            break
        obs, _, done, _ = env.step(_fixed_time_actions(junction_ids, bi))

    graphs, actions_list, log_probs_list, rewards_list, values_list, dones_list = [], [], [], [], [], []
    ep_rewards = []

    if not done:
        while not done:
            graph = builder.build(obs, update_normalizer=False).to(next(model.parameters()).device)
            with torch.no_grad():
                logits, values = model.forward_actor_critic(graph)
                dist = Categorical(logits=logits)
                acts_t = dist.sample()
                lps = dist.log_prob(acts_t)

            step_actions = {jid: int(acts_t[i].item()) for i, jid in enumerate(junction_ids)}
            obs, rewards_dict, done, _ = env.step(step_actions)

            rewards_t = torch.tensor(
                [rewards_dict[jid] for jid in junction_ids],
                dtype=torch.float32,
            )
            ep_rewards.append(rewards_t)
            graphs.append(graph.cpu())
            actions_list.append(acts_t.cpu())
            log_probs_list.append(lps.cpu())
            rewards_list.append(rewards_t)
            values_list.append(values.cpu())
            dones_list.append(done)

    return {
        'graphs': graphs,
        'actions': actions_list,
        'log_probs': log_probs_list,
        'rewards': rewards_list,
        'values': values_list,
        'dones': dones_list,
        'mean_reward': torch.stack(ep_rewards).mean().item() if ep_rewards else 0.0,
    }


# ---------------------------------------------------------------------------
# Parallel collection worker
# ---------------------------------------------------------------------------


def _collect_episode_worker(args: dict) -> dict:
    """Collect one episode in a subprocess and return serialisable buffer data.

    This function must live at module level so Python's multiprocessing
    (spawn mode on Windows) can pickle it.  All heavy imports are done at the
    top of the module, so subprocess startup cost is paid once per worker
    process when using a persistent ProcessPoolExecutor.

    Parameters in *args* dict
    -------------------------
    cfg_path, model_state, norm_state, junction_ids, n_junctions,
    episode_length, burn_in_steps, min_green_steps
    """
    import sys as _sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).parent.parent.parent
    _sys.path.insert(0, str(_ROOT))

    import torch as _torch
    from torch.distributions import Categorical as _Categorical
    from src.environment import TrafficEnv as _TrafficEnv
    from src.model.gat_policy import GATPolicy as _GATPolicy
    from src.utils.graph_builder import GraphBuilder as _GraphBuilder

    cfg_path = args['cfg_path']
    model_state = args['model_state']
    norm_state = args['norm_state']
    episode_length = args['episode_length']
    burn_in_steps = args['burn_in_steps']
    min_green_steps = args.get('min_green_steps', 2)
    demand_seed = args.get('demand_seed', None)
    flow_range = args.get('flow_range', (300, 1000))

    env = _TrafficEnv(
        cfg_path,
        gui=False,
        episode_length=episode_length,
        min_green_steps=min_green_steps,
        flow_range=flow_range,
    )
    model = _GATPolicy()
    model.load_state_dict(model_state, strict=False)
    model.eval()

    builder = _GraphBuilder(env._net, env.junction_infos)
    builder.normalizer.load_state_dict(norm_state)
    builder.normalizer.freeze()

    junction_ids = builder.junction_ids

    import numpy as _np

    if demand_seed is not None:
        env._rng = _np.random.default_rng(demand_seed)
    obs = env.reset()
    done = False

    # Burn-in with fixed-time controller.
    for bi in range(burn_in_steps):
        if done:
            break
        actions = _fixed_time_actions(junction_ids, bi)
        obs, _, done, _ = env.step(actions)

    graphs, actions_list, log_probs_list, rewards_list, values_list, dones_list = [], [], [], [], [], []
    ep_rewards = []

    if not done:
        while not done:
            graph = builder.build(obs, update_normalizer=False)
            with _torch.no_grad():
                logits, values = model.forward_actor_critic(graph)
                dist = _Categorical(logits=logits)
                acts_t = dist.sample()
                lps = dist.log_prob(acts_t)

            step_actions = {jid: acts_t[i].item() for i, jid in enumerate(junction_ids)}
            obs, rewards_dict, done, _ = env.step(step_actions)

            rewards_t = _torch.tensor(
                [rewards_dict[jid] for jid in junction_ids],
                dtype=_torch.float32,
            )
            ep_rewards.append(rewards_t)
            graphs.append(graph.cpu())
            actions_list.append(acts_t.cpu())
            log_probs_list.append(lps.cpu())
            rewards_list.append(rewards_t)
            values_list.append(values.cpu())
            dones_list.append(done)

    env.close()

    mean_reward = _torch.stack(ep_rewards).mean().item() if ep_rewards else 0.0
    return {
        'graphs': graphs,
        'actions': actions_list,
        'log_probs': log_probs_list,
        'rewards': rewards_list,
        'values': values_list,
        'dones': dones_list,
        'mean_reward': mean_reward,
    }


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_rl(
    cfg_path: str,
    il_checkpoint_dir: str,
    n_iterations: int = 500,
    episodes_per_update: int = 1,
    n_workers: int = 1,  # parallel SUMO processes for collection
    episode_length: int = 3600,
    burn_in_steps: int = 20,  # 20 × 15 s = 300 s
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    value_warmup_iters: int = 20,  # iters to train only value head, backbone frozen
    warmup_epochs: int = 10,  # PPO epochs used *only* during value warmup
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.05,
    max_grad_norm: float = 0.5,
    min_green_steps: int = 2,
    minibatch_junctions: int = 256,  # junction-transitions per minibatch
    eval_every: int = 50,
    save_every: int = 50,
    print_every: int = 10,
    n_eval_seeds: int = 5,
    log_dir: str = 'runs/rl',
    checkpoint_dir: str = 'checkpoints/rl',
    device: Optional[str] = None,
    demand_seed: Optional[int] = None,
    flow_range: tuple[int, int] = (300, 1000),
) -> GATPolicy:
    """Run PPO fine-tuning from an IL checkpoint.

    Parameters
    ----------
    cfg_path :
        Path to the SUMO .sumocfg file.
    il_checkpoint_dir :
        Directory produced by ``train_il`` containing ``il_policy.pt`` and
        ``normalizer.npz``.
    n_iterations :
        Total number of collect → update cycles.
    episodes_per_update :
        Episodes collected per iteration (Stage 3: 4, Stage 4: 1).
    episode_length :
        Simulation seconds per episode.
    burn_in_steps :
        Number of decision steps (× 15 s) to run with the fixed-time
        controller before handing control to the GNN.  300 s → 20 steps.
    lr :
        Adam learning rate.
    gamma / lam :
        PPO discount and GAE smoothing factors.
    clip_eps :
        PPO surrogate clip bound ε.
    n_epochs :
        PPO update passes over the rollout buffer per iteration.
    value_coeff :
        Relative weight of the critic MSE loss.
    entropy_coeff :
        Entropy bonus coefficient (exploration incentive).
    max_grad_norm :
        Gradient clipping threshold.
    minibatch_junctions :
        Approximate number of junction-transitions per minibatch.  Rounded
        down to the nearest whole graph (÷ n_junctions).
    eval_every :
        Run model-vs-expert evaluation every N iterations (0 = never).
    save_every :
        Save a numbered checkpoint every N iterations.
    print_every :
        Print console summary every N iterations.
    log_dir / checkpoint_dir :
        TensorBoard log and checkpoint output directories.
    device :
        PyTorch device string.  Auto-detected if None.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    env = TrafficEnv(
        cfg_path, gui=False, episode_length=episode_length, min_green_steps=min_green_steps, flow_range=flow_range
    )
    expert = GreedyExpert(env.junction_infos)

    # Load IL weights + frozen normalizer.
    model, norm_state = load_checkpoint(il_checkpoint_dir, device=str(dev))

    # Zero-init the value head output layer.
    # Random init produces large arbitrary outputs (often -2 to +2), which
    # biases bootstrapped returns (= GAE + values) far from actual rewards
    # (~-0.5 to -0.7), preventing the critic from converging during warmup.
    # With zero init, initial value predictions are 0, so returns ≈ raw GAE
    # rewards and the critic has a stable, unbiased target from step 1.
    with torch.no_grad():
        last_linear = model.value_head[-1]
        assert isinstance(last_linear, torch.nn.Linear)
        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    model.train()

    builder = GraphBuilder(env._net, env.junction_infos)
    builder.normalizer.load_state_dict(norm_state)
    builder.normalizer.freeze()

    junction_ids = builder.junction_ids
    n_junctions = len(junction_ids)

    # Minibatch size in graphs (round down to ≥ 1).
    n_graphs_per_batch = max(1, minibatch_junctions // n_junctions)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    effective_workers = min(n_workers, episodes_per_update)
    parallel = effective_workers > 1

    print(
        f'\nPPO Reinforcement Learning'
        f'\n  IL checkpoint:    {il_checkpoint_dir}'
        f'\n  Junctions:        {n_junctions}'
        f'\n  Iterations:       {n_iterations}'
        f'\n  Episodes/update:  {episodes_per_update}'
        f'\n  Workers:          {effective_workers} ({"parallel" if parallel else "sequential"})'
        f'\n  Episode length:   {episode_length} s'
        f'\n  Burn-in steps:    {burn_in_steps} ({burn_in_steps * 15} s)'
        f'\n  Min green steps:  {min_green_steps} ({min_green_steps * 15} s)'
        f'\n  Demand:           {"fixed seed " + str(demand_seed) + " (deterministic)" if demand_seed is not None else "randomised"}  flow={flow_range}'
        f'\n  Entropy coeff:    {entropy_coeff}'
        f'\n  value_warmup:     {value_warmup_iters} iters (backbone frozen, {warmup_epochs} epochs/iter)'
        f'\n  n_epochs:         {n_epochs}'
        f'\n  Graphs/minibatch: {n_graphs_per_batch}  '
        f'(~{n_graphs_per_batch * n_junctions} junction-transitions)'
        f'\n  Device:           {dev}'
        f'\n  Parameters:       {model.n_parameters():,}'
        f'\n  TensorBoard:      {log_dir}'
        f'\n  Checkpoints:      {checkpoint_dir}\n'
    )

    # Create the process pool once so worker processes survive across iterations.
    # Model weights travel per-iteration via worker_args['model_state'], so
    # a persistent pool always uses the current weights without respawning.
    pool = ProcessPoolExecutor(max_workers=effective_workers) if parallel else None

    t0 = time.time()
    last_completed_iteration = -1

    # ------------------------------------------------------------------
    # Main training loop
    # Ctrl-C is caught below; the finally block always saves a checkpoint.
    # ------------------------------------------------------------------
    try:
        for iteration in range(n_iterations):
            # ---- 1. Collect rollout -----------------------------------------
            buffer = RolloutBuffer(n_junctions, gamma=gamma, lam=lam)
            model.eval()

            ep_reward_sums = []

            if parallel:
                worker_args = {
                    'cfg_path': cfg_path,
                    'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                    'norm_state': builder.normalizer.state_dict(),
                    'episode_length': episode_length,
                    'burn_in_steps': burn_in_steps,
                    'min_green_steps': min_green_steps,
                    'flow_range': flow_range,
                    'demand_seed': demand_seed,
                }
                futures = [pool.submit(_collect_episode_worker, worker_args) for _ in range(episodes_per_update)]
                try:
                    results = [f.result() for f in as_completed(futures)]
                except KeyboardInterrupt:
                    for f in futures:
                        f.cancel()
                    raise

                for result in results:
                    for g, a, lp, r, v, d in zip(
                        result['graphs'],
                        result['actions'],
                        result['log_probs'],
                        result['rewards'],
                        result['values'],
                        result['dones'],
                    ):
                        buffer.add(g, a, lp, r, v, d)
                    ep_reward_sums.append(result['mean_reward'])

            else:
                # Sequential in-process collection — reuses the already-initialised
                # env and builder; zero reinitialisation overhead per iteration.
                for _ep in range(episodes_per_update):
                    if demand_seed is not None:
                        env._rng = np.random.default_rng(demand_seed)
                    result = _collect_episode_inline(env, model, builder, junction_ids, burn_in_steps)
                    for g, a, lp, r, v, d in zip(
                        result['graphs'],
                        result['actions'],
                        result['log_probs'],
                        result['rewards'],
                        result['values'],
                        result['dones'],
                    ):
                        buffer.add(g, a, lp, r, v, d)
                    ep_reward_sums.append(result['mean_reward'])

            # Skip update if no data (shouldn't happen in normal operation).
            if len(buffer) == 0:
                continue

            writer.add_scalar(
                'episode/mean_reward',
                float(np.mean(ep_reward_sums)) if ep_reward_sums else 0.0,
                iteration,
            )

            # ---- 2. GAE + returns -------------------------------------------
            # During warmup: MC returns give a stable, value-independent
            # critic target so the value head doesn't chase a moving target.
            # After warmup: standard GAE returns (advantages + V) for PPO.
            warming_up = iteration < value_warmup_iters
            buffer.compute_returns_and_advantages(use_mc_targets=warming_up)

            mean_return = buffer.returns.mean().item()
            writer.add_scalar('episode/mean_return', mean_return, iteration)

            # Explained variance: how much of return variance does V(s) explain?
            # EV > 0 → V(s) is state-conditional; EV ≈ 0 → global mean predictor.
            with torch.no_grad():
                flat_values = torch.stack(buffer._values).reshape(-1)
                flat_returns = buffer.returns.reshape(-1)
                residual_var = (flat_returns - flat_values).var().item()
                return_var = flat_returns.var().item()
                explained_var = 1.0 - residual_var / (return_var + 1e-8)
            writer.add_scalar('diagnostics/explained_variance', explained_var, iteration)

            # ---- 3. PPO update ----------------------------------------------
            _set_backbone_grad(model, requires_grad=not warming_up)
            model.train()

            policy_losses = []
            value_losses = []
            entropy_losses = []
            total_losses = []

            epochs_this_iter = warmup_epochs if warming_up else n_epochs
            for epoch in range(epochs_this_iter):
                for batch_graph, acts, old_lps, advs, rets in buffer.iterate_minibatches(
                    n_graphs_per_batch, device=dev
                ):
                    logits, values = model.forward_actor_critic(batch_graph)

                    dist = Categorical(logits=logits)
                    entropy = dist.entropy().mean()
                    v_loss = F.mse_loss(values, rets)

                    if warming_up:
                        # Only value loss — policy is frozen.
                        loss = value_coeff * v_loss
                        pg_loss = torch.zeros(1)
                    else:
                        new_lps = dist.log_prob(acts)
                        ratio = (new_lps - old_lps).exp()
                        pg_loss = -torch.min(
                            ratio * advs,
                            ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advs,
                        ).mean()
                        loss = pg_loss + value_coeff * v_loss - entropy_coeff * entropy

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()

                    policy_losses.append(pg_loss.item())
                    value_losses.append(v_loss.item())
                    entropy_losses.append(entropy.item())
                    total_losses.append(loss.item())

            writer.add_scalar('loss/policy', float(np.mean(policy_losses)), iteration)
            writer.add_scalar('loss/value', float(np.mean(value_losses)), iteration)
            writer.add_scalar('loss/entropy', float(np.mean(entropy_losses)), iteration)
            writer.add_scalar('loss/total', float(np.mean(total_losses)), iteration)

            # ---- 4. Periodic checkpointing ----------------------------------
            if save_every > 0 and (iteration + 1) % save_every == 0:
                _save_checkpoint(model, builder, checkpoint_dir, iteration + 1)

            # ---- 5. Periodic evaluation -------------------------------------
            if eval_every > 0 and (iteration + 1) % eval_every == 0:
                model.eval()
                periodic_seeds = tuple(range(42, 42 + n_eval_seeds))
                _run_and_log_eval(env, model, expert, builder, dev, writer, iteration, eval_seeds=periodic_seeds)
                model.train()

            # ---- 6. Console summary -----------------------------------------
            if (iteration + 1) % print_every == 0 or iteration == 0:
                elapsed = time.time() - t0
                phase = 'warmup' if warming_up else 'rl'
                print(
                    f'  [{phase}] iter {iteration + 1:5d}/{n_iterations}'
                    f'  rew={float(np.mean(ep_reward_sums)) if ep_reward_sums else 0.0:+.4f}'
                    f'  ret={mean_return:.4f}'
                    f'  EV={explained_var:+.3f}'
                    f'  pi_loss={float(np.mean(policy_losses)):.4f}'
                    f'  v_loss={float(np.mean(value_losses)):.4f}'
                    f'  H={float(np.mean(entropy_losses)):.3f}'
                    f'  [{elapsed:5.0f}s]'
                )

            last_completed_iteration = iteration

    except KeyboardInterrupt:
        print(f'\nTraining interrupted after iteration {last_completed_iteration + 1}.')

    finally:
        # ------------------------------------------------------------------
        # Always runs — on normal completion, interrupt, or any exception.
        # ------------------------------------------------------------------
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        model.eval()
        interrupted = last_completed_iteration < n_iterations - 1

        if not interrupted:
            eval_seeds = tuple(range(42, 42 + n_eval_seeds))
            print(f'\nRunning final evaluation (avg of {n_eval_seeds} seeds: {eval_seeds})...')
            _run_and_log_eval(
                env,
                model,
                expert,
                builder,
                dev,
                writer,
                last_completed_iteration,
                eval_seeds=eval_seeds,
            )

        _save_checkpoint(
            model,
            builder,
            checkpoint_dir,
            last_completed_iteration + 1,
            final=True,
        )
        env.close()
        writer.close()

        status = 'interrupted' if interrupted else 'complete'
        print(f'\nRL training {status}.  Checkpoints: {checkpoint_dir}/')

    return model


# ---------------------------------------------------------------------------
# Eval helper (mirrors imitation.py pattern)
# ---------------------------------------------------------------------------


def _run_and_log_eval(
    env: TrafficEnv,
    model: GATPolicy,
    expert: GreedyExpert,
    builder: GraphBuilder,
    dev: torch.device,
    writer: SummaryWriter,
    iteration: int,
    eval_seeds: tuple[int, ...] = (42,),
) -> None:
    """Run model + expert eval episodes and log averaged metrics to TensorBoard.

    Each seed produces one model episode and one expert episode on identical
    demand.  Results are averaged across all seeds before logging.
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
        env._rng = np.random.default_rng(seed)
        expert_runs.append(
            run_eval_episode(
                env,
                expert_policy,
                on_step=lambda actions: _notify_expert(expert, actions),
            )
        )

    env._rng = orig_rng

    model_metrics = average_eval_metrics(model_runs)
    expert_metrics = average_eval_metrics(expert_runs)

    _log_eval_scalars(writer, model_metrics, expert_metrics, iteration)
    _log_junction_metrics(writer, model_metrics, expert_metrics, iteration)
    _print_eval_summary(model_metrics, expert_metrics, iteration, n_seeds=len(eval_seeds))


def _notify_expert(expert: GreedyExpert, actions: dict[str, int]) -> None:
    for jid, phase in actions.items():
        expert.notify_applied(jid, phase)


def _log_eval_scalars(
    writer: SummaryWriter,
    model_m: EvalMetrics,
    expert_m: EvalMetrics,
    step: int,
) -> None:
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
            step,
        )

    writer.add_scalars(
        'eval/wait_density',
        {
            'model': model_m.avg_wait_density,
            'expert': expert_m.avg_wait_density,
        },
        step,
    )


def _log_junction_metrics(
    writer: SummaryWriter,
    model_m: EvalMetrics,
    expert_m: EvalMetrics,
    step: int,
) -> None:
    for jid in model_m.per_junction_wait_density:
        writer.add_scalars(
            f'junctions/{jid}/wait_density',
            {
                'model': model_m.per_junction_wait_density[jid],
                'expert': expert_m.per_junction_wait_density[jid],
            },
            step,
        )
        writer.add_scalars(
            f'junctions/{jid}/max_queue',
            {
                'model': float(model_m.per_junction_max_queue[jid]),
                'expert': float(expert_m.per_junction_max_queue[jid]),
            },
            step,
        )
        for label, m in (('model', model_m), ('expert', expert_m)):
            counts = m.per_junction_phase_counts[jid]
            phases_flat = [ph for ph, cnt in enumerate(counts) for _ in range(cnt)]
            if phases_flat:
                writer.add_histogram(
                    f'junctions/{jid}/{label}/phase_hist',
                    np.array(phases_flat, dtype=np.float32),
                    step,
                )


def _print_eval_summary(
    model: EvalMetrics,
    expert: EvalMetrics,
    step: int,
    n_seeds: int = 1,
) -> None:
    seed_tag = f'avg of {n_seeds} seeds' if n_seeds > 1 else '1 seed'
    print(f'\n  [eval @ iter {step + 1}  ({seed_tag})]')
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
        delta = mv - ev
        marker = '  <' if mv < ev else ''
        print(f'  {name:<24}  {mv:>10.3f}  {ev:>10.3f}  ({delta:+.3f}){marker}')
    print()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: GATPolicy,
    builder: GraphBuilder,
    directory: str,
    iteration: int,
    final: bool = False,
) -> None:
    tag = 'final' if final else f'iter{iteration:04d}'
    model_path = os.path.join(directory, f'rl_policy_{tag}.pt')
    latest_path = os.path.join(directory, 'rl_policy_latest.pt')

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), latest_path)

    # Save normalizer alongside (same as IL checkpoint format for compatibility).
    nd = builder.normalizer.state_dict()
    norm_path = os.path.join(directory, 'normalizer.npz')
    np.savez(norm_path, n=np.array(nd['n']), mean=nd['mean'], M2=nd['M2'])

    print(f'  Saved checkpoint → {model_path}')


def load_rl_checkpoint(
    checkpoint_dir: str,
    device: Optional[str] = None,
) -> tuple[GATPolicy, dict]:
    """Load the latest RL checkpoint (mirrors IL load_checkpoint interface).

    Returns
    -------
    model : GATPolicy with loaded weights (eval mode)
    norm_state : dict with keys ``n``, ``mean``, ``M2``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GATPolicy()
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, 'rl_policy_latest.pt'),
            map_location=device,
            weights_only=True,
        )
    )
    model.to(torch.device(device)).eval()

    npz = np.load(os.path.join(checkpoint_dir, 'normalizer.npz'))
    return model, {'n': int(npz['n']), 'mean': npz['mean'], 'M2': npz['M2']}
