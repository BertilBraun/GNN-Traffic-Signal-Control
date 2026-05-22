"""Re-accumulate RunningNormalizer statistics from any SUMO network.

The normalizer statistics baked into a checkpoint reflect the feature
distributions of the network it was trained on.  When transferring to a
different network (different lane lengths, block spacings, junction counts)
those statistics are wrong.  This script drives the target network with the
greedy expert, feeds every observation into a fresh RunningNormalizer, then
writes a new checkpoint directory with the same model weights but network-
calibrated normalizer stats.

The output directory is compatible with eval_city.py (--ckpt) and
train_rl.py (--il-ckpt via warmstart adapter).

Usage
-----
    python scripts/renormalize.py ^
        --ckpt checkpoints/rl/2026-05-19_23-35-19 ^
        --cfg  configs/city/city.sumocfg ^
        --episodes 30 ^
        --flow-range 3000 6000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.environment import TrafficEnv
from src.environment.phase_schema import NUM_PHASES
from src.utils.graph_builder import GraphBuilder

_FIXED_TIME_CYCLE = 2  # decision intervals per phase (matches PPO burn-in)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Re-accumulate normalizer stats for a target network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt', required=True, help='Source RL checkpoint directory (model weights to copy)')
    p.add_argument('--cfg', required=True, help='Path to .sumocfg for the target network')
    p.add_argument('--episodes', type=int, default=30, help='Warmup episodes to collect stats from')
    p.add_argument('--ep-len', type=int, default=1800, help='Episode length in seconds')
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
        help='Min veh/h per O-D pair to include in demand file',
    )
    p.add_argument(
        '--out',
        default=None,
        help='Output checkpoint directory (default: auto-timestamped under checkpoints/renorm/<network>)',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    network_name = Path(args.cfg).parent.name
    out_dir = (
        Path(args.out)
        if args.out
        else (ROOT / 'checkpoints' / 'renorm' / network_name / datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 62}')
    print(f'  Network           : {network_name}')
    print(f'  Source checkpoint : {args.ckpt}')
    print(f'  Config            : {args.cfg}')
    print(f'  Warmup episodes   : {args.episodes}  x  {args.ep_len}s')
    print(f'  Flow range        : {args.flow_range}')
    print(f'  Output            : {out_dir}')
    print(f'{"=" * 62}\n')

    t_init = time.perf_counter()
    env = TrafficEnv(
        args.cfg,
        gui=False,
        episode_length=args.ep_len,
        flow_range=tuple(args.flow_range),
        demand_min_rate=args.demand_min_rate,
    )
    builder = GraphBuilder(env._net, env.junction_infos)
    print(f'  Init (env + demand table): {time.perf_counter() - t_init:.1f}s\n')

    rng = np.random.default_rng(0)
    t_total = time.perf_counter()

    for episode in range(args.episodes):
        env._rng = np.random.default_rng(int(rng.integers(0, 2**31)))

        t_reset = time.perf_counter()
        obs = env.reset()
        reset_s = time.perf_counter() - t_reset

        done = False
        step_count = 0
        t_sumo = t_build = 0.0
        while not done:
            t0 = time.perf_counter()
            builder.build(obs, update_normalizer=True)
            t_build += time.perf_counter() - t0

            phase = (step_count // _FIXED_TIME_CYCLE) % NUM_PHASES
            actions = {jid: phase for jid in env.junction_ids}

            t0 = time.perf_counter()
            obs, _, done, _ = env.step(actions)
            t_sumo += time.perf_counter() - t0

            step_count += 1

        elapsed = time.perf_counter() - t_total
        print(
            f'  Episode {episode + 1:3d}/{args.episodes}'
            f'  steps={step_count}'
            f'  reset={reset_s:.1f}s'
            f'  sumo={t_sumo:.1f}s  build={t_build:.2f}s'
            f'  normalizer_n={builder.normalizer.n}'
            f'  elapsed={elapsed:.0f}s'
        )

    env.close()
    builder.normalizer.freeze()

    nd = builder.normalizer.state_dict()
    std = np.sqrt(nd['M2'] / nd['n'] + builder.normalizer.epsilon)
    print(f'\nNormalizer stats after {nd["n"]} samples:')
    print(f'  mean[0:6] = {nd["mean"][:6].round(4)}')
    print(f'  std [0:6] = {std[:6].round(4)}')

    shutil.copy2(Path(args.ckpt) / 'rl_policy_latest.pt', out_dir / 'rl_policy_latest.pt')
    np.savez(out_dir / 'normalizer.npz', n=np.array(nd['n']), mean=nd['mean'], M2=nd['M2'])

    print(f'\nCheckpoint written to: {out_dir}')
    print('\nNext step:')
    print('  python scripts/eval_city.py ^')
    print(f'      --ckpt {out_dir} ^')
    print(f'      --flow-range {args.flow_range[0]} {args.flow_range[1]}')
    print()


if __name__ == '__main__':
    main()
