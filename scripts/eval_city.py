"""Zero-shot evaluation of the trained RL model on a city network.

Runs N seeds each of the RL model and the greedy expert, then prints a
side-by-side metric comparison table.

Usage
-----
    python scripts/eval_city.py \\
        --ckpt checkpoints/rl/2026-05-19_20-02-40 \\
        --cfg  configs/city/city.sumocfg \\
        --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.environment import TrafficEnv
from src.environment.expert import GreedyExpert
from src.training.eval_episode import (
    EvalMetrics,
    average_eval_metrics,
    make_expert_policy,
    make_model_policy,
    run_eval_episode,
)
from src.training.ppo import load_rl_checkpoint
from src.utils.graph_builder import GraphBuilder

DEFAULT_CFG = str(ROOT / 'configs' / 'city' / 'city.sumocfg')
DEFAULT_CKPT = str(ROOT / 'checkpoints' / 'rl' / '2026-05-19_20-02-40')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Zero-shot city eval: model vs expert')
    p.add_argument('--ckpt', default=DEFAULT_CKPT, help='RL checkpoint directory')
    p.add_argument('--cfg', default=DEFAULT_CFG, help='Path to .sumocfg')
    p.add_argument('--seeds', type=int, default=5, help='Number of eval seeds (42..)')
    p.add_argument('--ep-len', type=int, default=3600, help='Episode length (s)')
    p.add_argument(
        '--flow-range',
        nargs=2,
        type=int,
        default=[700, 1200],
        metavar=('MIN', 'MAX'),
        help='Traffic demand range (veh/h)',
    )
    p.add_argument(
        '--demand-min-rate',
        type=float,
        default=5.0,
        metavar='R',
        help='Min veh/h per O-D pair to include in demand file (lower = more active flows at light total demand)',
    )
    p.add_argument('--device', default=None)
    return p.parse_args()


def _print_table(expert: EvalMetrics, model: EvalMetrics, seeds: int, ep_len: int) -> None:
    rows = [
        ('avg_waiting_time (s)', expert.avg_waiting_time, model.avg_waiting_time),
        ('avg_travel_time (s)', expert.avg_travel_time, model.avg_travel_time),
        ('throughput (veh/h)', expert.throughput_per_hour, model.throughput_per_hour),
        ('max_queue (vehs)', expert.max_queue_length, model.max_queue_length),
        ('switch_freq (/j/min)', expert.phase_switch_freq, model.phase_switch_freq),
        ('wait_density (s/m)', expert.avg_wait_density, model.avg_wait_density),
        ('tls_passes (/veh)', expert.avg_tls_passes_per_vehicle, model.avg_tls_passes_per_vehicle),
        ('tls_stops (/veh)', expert.avg_stops_before_tls_per_vehicle, model.avg_stops_before_tls_per_vehicle),
        ('nonstop_tls_rate', expert.nonstop_tls_pass_rate, model.nonstop_tls_pass_rate),
        ('best_nonstop_streak', expert.avg_best_nonstop_tls_streak, model.avg_best_nonstop_tls_streak),
    ]

    header = f'{"Metric":<28}  {"Expert":>10}  {"Model":>10}  {"Delta":>10}'
    sep = '-' * len(header)
    print(f'\n{sep}')
    print(f'  Zero-shot city eval  ({seeds} seeds, {seeds * ep_len}s total sim time)')
    print(sep)
    print(f'  {header}')
    print(f'  {sep}')

    for name, ev, mv in rows:
        if ev != 0:
            delta_pct = 100.0 * (mv - ev) / abs(ev)
            delta_str = f'{delta_pct:+.1f}%'
        else:
            delta_str = 'N/A'
        print(f'  {name:<28}  {ev:>10.1f}  {mv:>10.1f}  {delta_str:>10}')

    print(sep)
    print()


def main() -> None:
    args = _parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = list(range(42, 42 + args.seeds))

    env = TrafficEnv(
        args.cfg,
        gui=False,
        episode_length=args.ep_len,
        flow_range=tuple(args.flow_range),
        demand_min_rate=args.demand_min_rate,
    )
    junction_ids = env.junction_ids
    junction_infos = env.junction_infos

    print(f'\n{"=" * 62}')
    print(f'  Config    : {args.cfg}')
    print(f'  Checkpoint: {args.ckpt}')
    print(f'  Junctions : {len(junction_ids)}')
    print(f'  Seeds     : {seeds}')
    print(f'  Ep length : {args.ep_len} s  |  Flow: {args.flow_range}')
    print(f'{"=" * 62}\n')

    # ------------------------------------------------------------------ expert
    print('Running expert ...')
    expert = GreedyExpert(junction_infos)
    expert_policy = make_expert_policy(expert)

    expert_results: list[EvalMetrics] = []
    for seed in seeds:
        env._rng = np.random.default_rng(seed)
        expert.reset()
        m = run_eval_episode(
            env,
            expert_policy,
            on_step=lambda a: [expert.notify_applied(j, p) for j, p in a.items()],
        )
        expert_results.append(m)
        print(f'  seed {seed}  wait={m.avg_waiting_time:.1f}s  wd={m.avg_wait_density:.4f}')

    # ------------------------------------------------------------------ model
    print('\nLoading model checkpoint ...')
    model, norm_state = load_rl_checkpoint(args.ckpt, device=device)
    builder = GraphBuilder(env._net, junction_infos)
    builder.normalizer.load_state_dict(norm_state)
    builder.normalizer.freeze()
    model_policy = make_model_policy(model, builder, device)

    print('Running model (zero-shot) ...')
    model_results: list[EvalMetrics] = []
    for seed in seeds:
        env._rng = np.random.default_rng(seed)
        m = run_eval_episode(env, model_policy)
        model_results.append(m)
        print(f'  seed {seed}  wait={m.avg_waiting_time:.1f}s  wd={m.avg_wait_density:.4f}')

    env.close()

    expert_avg = average_eval_metrics(expert_results)
    model_avg = average_eval_metrics(model_results)

    _print_table(expert_avg, model_avg, args.seeds, args.ep_len)


if __name__ == '__main__':
    main()
