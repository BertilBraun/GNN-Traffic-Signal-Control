"""Run expert or RL model on the irregular 4×4 grid, optionally with SUMO-GUI.

Usage
-----
    # Expert controller with GUI
    python scripts/run_grid.py --mode expert --gui

    # RL model with GUI
    python scripts/run_grid.py --mode model --ckpt checkpoints/rl/<stamp> --gui

    # Headless model run with fixed demand seed
    python scripts/run_grid.py --mode model --ckpt checkpoints/rl/<stamp> --demand-seed 42

    # Enable POI labels in SUMO-GUI if not visible:
    #   View > Visualization Settings > POIs > tick 'Show name'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import traci

from src.environment import TrafficEnv
from src.environment.expert import GreedyExpert
from src.environment.junction_info import JunctionInfo
from src.training.ppo import load_rl_checkpoint
from src.utils.graph_builder import GraphBuilder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRID_CFG = str(ROOT / 'configs' / 'grid_4x4' / 'grid.sumocfg')

_POI_Y_OFFSET = 30.0
_POI_LAYER = 200.0

_COL_CLEAR = (30, 210, 60, 230)
_COL_MODERATE = (210, 170, 0, 230)
_COL_HEAVY = (220, 40, 40, 230)
_THR_CLEAR = 0.05
_THR_HEAVY = 0.30

# ---------------------------------------------------------------------------
# Wait-density overlay
# ---------------------------------------------------------------------------


class WaitDensityOverlay:
    """Floating POI labels above each junction showing live congestion.

    green  — clear      (w < 0.05 s/m)
    amber  — forming    (0.05 ≤ w < 0.30 s/m)
    red    — heavy      (w ≥ 0.30 s/m)

    Label: "<jid> P<phase>[>] w=<wait_density:.3f>"
    """

    def __init__(self, junction_infos: dict[str, JunctionInfo], net) -> None:
        self._net = net
        self._junctions = junction_infos
        self._active_ids: dict[str, str] = {}

    def update(self, switches: dict[str, bool], phases: dict[str, int]) -> None:
        for jid, ji in self._junctions.items():
            w = self._local_wait(ji)
            sw_marker = '>' if switches.get(jid, False) else ' '
            ph = phases.get(jid, 0)
            new_id = f'{jid} P{ph}{sw_marker} w={w:.3f}'

            prev_id = self._active_ids.get(jid)
            if prev_id is not None:
                try:
                    traci.poi.remove(prev_id)
                except traci.exceptions.TraCIException:
                    pass

            if w < _THR_CLEAR:
                color = _COL_CLEAR
            elif w < _THR_HEAVY:
                color = _COL_MODERATE
            else:
                color = _COL_HEAVY

            node = self._net.getNode(jid)
            x, y = node.getCoord()
            traci.poi.add(
                new_id,
                x,
                y + _POI_Y_OFFSET,
                color,
                layer=_POI_LAYER,
                imgFile='',
                width=3.0,
                height=3.0,
                angle=0.0,
            )
            self._active_ids[jid] = new_id

    def clear(self) -> None:
        for poi_id in self._active_ids.values():
            try:
                traci.poi.remove(poi_id)
            except traci.exceptions.TraCIException:
                pass
        self._active_ids.clear()

    @staticmethod
    def _local_wait(ji: JunctionInfo) -> float:
        total_wait = sum(traci.lane.getWaitingTime(lid) for lid, _ in ji.all_lane_det)
        total_len = sum(dl for _, dl in ji.all_lane_det) or 1.0
        return total_wait / total_len


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _expert_actions(expert: GreedyExpert) -> dict[str, int]:
    return expert.act()


def _model_actions(
    obs: dict[str, np.ndarray],
    model: torch.nn.Module,
    builder: GraphBuilder,
    junction_ids: list[str],
    device: torch.device,
) -> dict[str, int]:
    graph = builder.build(obs, update_normalizer=False).to(device)
    with torch.no_grad():
        logits, _ = model.forward_actor_critic(graph)
    phases = logits.argmax(dim=-1)
    return {jid: int(phases[i].item()) for i, jid in enumerate(junction_ids)}


# ---------------------------------------------------------------------------
# Console reward display
# ---------------------------------------------------------------------------


def _format_step(rewards: dict[str, float], switches: dict[str, bool]) -> str:
    parts = []
    for jid in sorted(rewards):
        sw = '→' if switches.get(jid, False) else ' '
        r = rewards[jid]
        bar = '▓' * min(20, max(0, int((r + 5) * 2)))
        parts.append(f'  {jid:6s}{sw} r={r:+6.3f}  {bar}')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(
    mode: str,
    cfg: str,
    gui: bool,
    episode_length: int,
    ckpt_dir: str | None,
    device: str | None,
    demand_seed: int | None,
    flow_range: tuple[int, int],
    min_green_steps: int,
) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    env = TrafficEnv(
        cfg,
        gui=gui,
        episode_length=episode_length,
        flow_range=flow_range,
        min_green_steps=min_green_steps,
    )
    junction_ids = env.junction_ids

    if mode == 'expert':
        expert = GreedyExpert(env.junction_infos)
        model = builder = None
        policy_label = 'GreedyExpert'
    else:
        if ckpt_dir is None:
            raise ValueError('--ckpt is required for --mode model')
        model, norm_state = load_rl_checkpoint(ckpt_dir, device=device)
        builder = GraphBuilder(env._net, env.junction_infos)
        builder.normalizer.load_state_dict(norm_state)
        builder.normalizer.freeze()
        expert = None
        policy_label = f'RL model  ({ckpt_dir})'

    overlay = WaitDensityOverlay(env.junction_infos, env._net) if gui else None

    print(f'\n{"=" * 62}')
    print(f'  {policy_label}')
    print(f'  Junctions: {len(junction_ids)}   Episode: {episode_length} s   GUI: {gui}')
    print(f'  Min green: {min_green_steps} steps ({min_green_steps * 15} s)')
    print(f'  Demand seed: {demand_seed}   Flow range: {flow_range}')
    if gui:
        print('\n  POI labels: View > Visualization Settings > POIs > Show name')
    print(f'{"=" * 62}\n')

    if demand_seed is not None:
        env._rng = np.random.default_rng(demand_seed)
    obs = env.reset()
    done = False
    step = 0
    ep_reward: dict[str, float] = {jid: 0.0 for jid in junction_ids}

    while not done:
        step += 1

        if mode == 'expert':
            actions = _expert_actions(expert)
        else:
            actions = _model_actions(obs, model, builder, junction_ids, dev)

        obs, rewards, done, info = env.step(actions)

        if mode == 'expert':
            for jid in junction_ids:
                expert.notify_applied(jid, env._current_phases[jid])

        for jid, r in rewards.items():
            ep_reward[jid] += r

        if overlay is not None:
            overlay.update(info['switches'], env._current_phases)

        t = info['sim_time']
        gw = info['global_wait']
        print(f'[t={t:5.0f}s  step={step:3d}  {mode}]  global_wait={gw:.4f}')
        print(_format_step(rewards, info['switches']))
        print()

    if overlay is not None:
        overlay.clear()

    print(f'\n{"=" * 62}  EPISODE DONE')
    print('  Cumulative reward per junction:')
    for jid in sorted(ep_reward):
        print(f'    {jid:6s}  {ep_reward[jid]:+.2f}')
    total = sum(ep_reward.values())
    mean = total / max(1, len(ep_reward))
    print(f'  Total: {total:+.1f}   Mean/junction: {mean:+.3f}')
    print(f'{"=" * 62}\n')

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Run expert or RL model on the 4×4 grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--mode',
        choices=['expert', 'model'],
        default='expert',
        help='Controller to run',
    )
    p.add_argument(
        '--ckpt',
        default=None,
        metavar='DIR',
        help='RL checkpoint directory (required for --mode model)',
    )
    p.add_argument('--gui', action='store_true', help='Open SUMO-GUI')
    p.add_argument('--cfg', default=GRID_CFG, help='Path to .sumocfg')
    p.add_argument('--ep-len', type=int, default=3600, help='Episode length in seconds')
    p.add_argument('--demand-seed', type=int, default=None, help='Fix demand RNG seed')
    p.add_argument(
        '--flow-range',
        nargs=2,
        type=int,
        default=[700, 1200],
        metavar=('MIN', 'MAX'),
        help='Traffic demand range (veh/h)',
    )
    p.add_argument(
        '--min-green-steps',
        type=int,
        default=2,
        help='Minimum decision steps before a phase switch is allowed',
    )
    p.add_argument('--device', default=None, help='PyTorch device (model mode only)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        mode=args.mode,
        cfg=args.cfg,
        gui=args.gui,
        episode_length=args.ep_len,
        ckpt_dir=args.ckpt,
        device=args.device,
        demand_seed=args.demand_seed,
        flow_range=tuple(args.flow_range),
        min_green_steps=args.min_green_steps,
    )
