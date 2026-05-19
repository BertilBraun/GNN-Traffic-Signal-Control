"""Environment correctness verification.

Runs six numbered checks bottom-up, without any model or training.
Each check prints concrete values so failures are immediately actionable.

Usage
-----
    python scripts/verify_env.py
    python scripts/verify_env.py --cfg configs/grid_4x4/grid.sumocfg
    python scripts/verify_env.py --episodes 3   # more data per check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import TrafficEnv
from src.environment.expert import GreedyExpert
from src.utils.graph_builder import GraphBuilder

DEFAULT_CFG = str(ROOT / 'configs' / 'grid_4x4' / 'grid.sumocfg')
EP_LEN = 300  # short episodes — enough data, fast to run

# 41-dim feature names matching sumo_env.py layout
_FEAT_NAMES: list[str] = []
for _s in range(4):
    for _d in ('left', 'thru', 'rght'):
        for _f in ('q_dens', 'app_dns', 'wait_dns'):
            _FEAT_NAMES.append(f's{_s}_{_d}_{_f}')
for _p in range(4):
    _FEAT_NAMES.append(f'phase{_p}_oh')
_FEAT_NAMES.append('elapsed')
assert len(_FEAT_NAMES) == 41


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _sep(char: str = '-', width: int = 62) -> None:
    print(char * width)


def _ok(msg: str) -> None:
    print(f'  ✓  {msg}')


def _warn(msg: str) -> None:
    print(f'  ⚠  {msg}')


def _fail(msg: str) -> None:
    print(f'  ✗  {msg}')


def _header(n: int, total: int, title: str) -> None:
    print(f'\n[{n}/{total}] {title}')
    _sep()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_startup(cfg_path: str) -> tuple[bool, TrafficEnv, GreedyExpert, GraphBuilder]:
    env = TrafficEnv(cfg_path, gui=False, episode_length=EP_LEN, demand_seed=0)
    expert = GreedyExpert(env.junction_infos)
    builder = GraphBuilder(env._net, env.junction_infos)

    obs = env.reset()
    ok = True

    n_junc = len(env.junction_ids)
    print(f'  Junctions found : {n_junc}')
    print(f'  Junction IDs    : {env.junction_ids[:5]}' + (' ...' if n_junc > 5 else ''))

    if n_junc == 0:
        _fail('No junctions found — check network / sumocfg path')
        ok = False
    else:
        _ok(f'{n_junc} junctions')

    if set(obs.keys()) != set(env.junction_ids):
        _fail('obs keys do not match junction_ids')
        ok = False
    else:
        _ok('obs keys match junction_ids')

    env.close()
    return ok, env, expert, builder


def check_obs_shapes(cfg_path: str, env: TrafficEnv) -> bool:
    env._rng = np.random.default_rng(0)
    obs = env.reset()
    passed = True

    shapes_ok = all(v.shape == (41,) for v in obs.values())
    dtypes_ok = all(v.dtype == np.float32 for v in obs.values())
    nan_found = any(np.any(np.isnan(v)) for v in obs.values())
    inf_found = any(np.any(np.isinf(v)) for v in obs.values())

    if shapes_ok:
        _ok('all obs shapes (41,)')
    else:
        _fail(f'unexpected shapes: { {k: v.shape for k, v in obs.items() if v.shape != (41,)} }')
        passed = False

    if dtypes_ok:
        _ok('all dtypes float32')
    else:
        _warn(f'non-float32 dtypes: { {k: v.dtype for k, v in obs.items() if v.dtype != np.float32} }')

    if nan_found:
        _fail('NaN values found in obs')
        passed = False
    else:
        _ok('no NaN')

    if inf_found:
        _fail('Inf values found in obs')
        passed = False
    else:
        _ok('no Inf')

    env.close()
    return passed


def check_obs_bounds(cfg_path: str, env: TrafficEnv, expert: GreedyExpert, n_episodes: int) -> bool:
    all_obs: list[np.ndarray] = []  # each row: one junction at one step

    for ep in range(n_episodes):
        env._rng = np.random.default_rng(ep)
        obs = env.reset()
        expert.reset()
        done = False
        while not done:
            for v in obs.values():
                all_obs.append(v)
            actions = expert.act()
            obs, _, done, _ = env.step(actions)
            for jid in env.junction_ids:
                expert.notify_applied(jid, actions[jid])
        env.close()

    mat = np.stack(all_obs)  # (T*N, 41)
    passed = True

    # Table header
    print(f'\n  {"Feature":<22}  {"Min":>8}  {"Max":>8}  {"Mean":>8}  {"Zeros%":>7}  Notes')
    print(f'  {"-" * 22}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 7}  -----')

    for i, name in enumerate(_FEAT_NAMES):
        col = mat[:, i]
        mn = col.min()
        mx = col.max()
        mean = col.mean()
        zeros = 100.0 * (col == 0.0).mean()
        notes = ''

        # Checks
        if i < 36:  # movement features — must be non-negative
            if mn < -1e-4:
                notes += 'NEG! '
                passed = False
        if 36 <= i < 40:  # phase one-hot
            bad_vals = col[(col != 0.0) & (col != 1.0)]
            if len(bad_vals):
                notes += 'NOT_BINARY! '
                passed = False
            ph_sum_ok = True
            for row_idx in range(0, len(all_obs)):
                ph_slice = mat[row_idx, 36:40]
                if abs(ph_slice.sum() - 1.0) > 1e-4:
                    ph_sum_ok = False
                    break
            if not ph_sum_ok:
                notes += 'ONEHOT_SUM≠1! '
                passed = False
        if i == 40:  # elapsed
            if mn < -1e-4 or mx > 1.0 + 1e-4:
                notes += f'RANGE_ERR({mn:.3f},{mx:.3f})! '
                passed = False
        if mx > 500:
            notes += 'LARGE '

        print(f'  {name:<22}  {mn:>8.4f}  {mx:>8.4f}  {mean:>8.4f}  {zeros:>6.1f}%  {notes}')

    # Phase one-hot sum check summary
    ph_sums = mat[:, 36:40].sum(axis=1)
    if np.allclose(ph_sums, 1.0, atol=1e-4):
        _ok('phase one-hot sums to 1.0 for all observations')
    else:
        _fail(f'phase one-hot sums not all 1.0: min={ph_sums.min():.4f} max={ph_sums.max():.4f}')
        passed = False

    return passed


def check_ordering(cfg_path: str, env: TrafficEnv, builder: GraphBuilder) -> bool:
    passed = True
    env._rng = np.random.default_rng(0)
    obs = env.reset()
    graph = builder.build(obs, update_normalizer=False)
    env.close()

    env_ids = env.junction_ids
    bld_ids = builder.junction_ids

    print(f'  env.junction_ids[:4]     : {env_ids[:4]}')
    print(f'  builder.junction_ids[:4] : {bld_ids[:4]}')

    if env_ids == bld_ids:
        _ok('ordering matches exactly')
    else:
        _fail('ordering MISMATCH — labels will be misaligned')
        passed = False

    n = len(env_ids)
    if graph.x.shape == (n, 41):
        _ok(f'graph.x shape ({n}, 41)')
    else:
        _fail(f'graph.x shape {tuple(graph.x.shape)}, expected ({n}, 41)')
        passed = False

    if graph.edge_index.shape[0] == 2:
        _ok(f'edge_index shape (2, {graph.edge_index.shape[1]})')
    else:
        _fail(f'edge_index shape {tuple(graph.edge_index.shape)}')
        passed = False

    if graph.edge_attr is not None and graph.edge_attr.shape[1] == 3:
        _ok(f'edge_attr shape ({graph.edge_attr.shape[0]}, 3)')
    else:
        _warn(f'edge_attr unexpected: {graph.edge_attr.shape if graph.edge_attr is not None else None}')

    return passed


def check_expert_coverage(cfg_path: str, env: TrafficEnv, expert: GreedyExpert, n_episodes: int) -> bool:
    phase_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    total_switches = 0
    total_steps = 0
    n_junctions = len(env.junction_ids)
    passed = True

    for ep in range(n_episodes):
        env._rng = np.random.default_rng(ep)
        obs = env.reset()
        expert.reset()
        done = False
        while not done:
            actions = expert.act()
            for phase in actions.values():
                phase_counts[phase] += 1
            obs, _, done, info = env.step(actions)
            total_switches += sum(1 for sw in info['switches'].values() if sw)
            total_steps += 1
            for jid in env.junction_ids:
                expert.notify_applied(jid, actions[jid])
        env.close()

    total_actions = sum(phase_counts.values())
    print(f'  Phase distribution ({total_actions} total junction-steps):')
    for ph, cnt in phase_counts.items():
        bar = '#' * int(50 * cnt / total_actions)
        print(f'    Phase {ph}: {cnt:5d}  ({100 * cnt / total_actions:5.1f}%)  {bar}')

    if all(cnt > 0 for cnt in phase_counts.values()):
        _ok('all 4 phases selected')
    else:
        unused = [p for p, c in phase_counts.items() if c == 0]
        _fail(f'phases never selected: {unused} — phase definition or junction geometry issue')
        passed = False

    # Switch frequency: switches / junction / minute
    sim_minutes = n_episodes * EP_LEN / 60.0
    switch_freq = total_switches / n_junctions / sim_minutes
    print(f'  Switch frequency: {switch_freq:.3f} /junction/min')
    if 0.3 <= switch_freq <= 4.0:
        _ok(f'switch freq {switch_freq:.3f} in [0.3, 4.0]')
    else:
        _warn(f'switch freq {switch_freq:.3f} outside [0.3, 4.0] — check phase hold logic')

    return passed


def check_reward_sanity(cfg_path: str, env: TrafficEnv, expert: GreedyExpert) -> bool:
    passed = True

    def run_episode(policy_fn) -> tuple[float, list[float]]:
        env._rng = np.random.default_rng(99)
        obs = env.reset()
        expert.reset()
        done = False
        wait_sum = 0.0
        steps = 0
        all_rewards: list[float] = []
        while not done:
            actions = policy_fn(obs, env.junction_ids)
            obs, rewards, done, info = env.step(actions)
            wait_sum += info['global_wait']
            steps += 1
            all_rewards.extend(rewards.values())
            if hasattr(policy_fn, '_is_expert'):
                for jid in env.junction_ids:
                    expert.notify_applied(jid, actions[jid])
        env.close()
        return wait_sum / max(1, steps), all_rewards

    def expert_policy(obs, jids):
        return expert.act()

    expert_policy._is_expert = True  # type: ignore[attr-defined]

    def donothing_policy(obs, jids):
        return {jid: 0 for jid in jids}

    expert_wait, expert_rewards = run_episode(expert_policy)
    donothing_wait, _ = run_episode(donothing_policy)

    print(f'  Expert avg wait density    : {expert_wait:.4f} s/m')
    print(f'  Always-phase-0 wait density: {donothing_wait:.4f} s/m')

    if donothing_wait > 0 and expert_wait < donothing_wait:
        ratio = donothing_wait / expert_wait
        _ok(f'expert is {ratio:.2f}× better than do-nothing')
    elif expert_wait == 0 and donothing_wait == 0:
        _warn('both policies have 0 wait — episode too short or no demand')
    else:
        _warn(f'expert ({expert_wait:.4f}) NOT better than do-nothing ({donothing_wait:.4f})')
        passed = False

    if expert_rewards:
        r_arr = np.array(expert_rewards)
        print(
            f'  Expert rewards — min: {r_arr.min():.4f}  max: {r_arr.max():.4f}  '
            f'mean: {r_arr.mean():.4f}  std: {r_arr.std():.4f}'
        )
        if r_arr.min() < -50:
            _warn(f'very large negative rewards ({r_arr.min():.2f}) — check reward scale')
        if r_arr.max() > 50:
            _warn(f'very large positive rewards ({r_arr.max():.2f}) — check reward scale')
        if np.all(r_arr <= 0):
            _warn('all rewards non-positive — switch penalty may dominate')
        if np.all(r_arr >= 0):
            _warn('all rewards non-negative — check reward sign')

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Environment correctness verification')
    p.add_argument('--cfg', default=DEFAULT_CFG, help='Path to .sumocfg')
    p.add_argument('--episodes', type=int, default=3, help='Episodes per check')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_ep = args.episodes
    total = 6
    results: list[bool] = []

    _sep('=')
    print(' ENVIRONMENT VERIFICATION')
    print(f' Config  : {args.cfg}')
    print(f' Episodes: {n_ep} per check  (ep_len={EP_LEN}s)')
    _sep('=')

    # [1] Startup
    _header(1, total, 'Environment startup')
    ok1, env, expert, builder = check_startup(args.cfg)
    results.append(ok1)

    # [2] Obs shapes
    _header(2, total, 'Observation shapes and types')
    results.append(check_obs_shapes(args.cfg, env))

    # [3] Obs bounds
    _header(3, total, f'Feature bounds ({n_ep} episodes, raw pre-normalisation)')
    results.append(check_obs_bounds(args.cfg, env, expert, n_ep))

    # [4] Ordering consistency
    _header(4, total, 'Junction/graph ordering consistency')
    results.append(check_ordering(args.cfg, env, builder))

    # [5] Expert coverage
    _header(5, total, f'Expert phase coverage ({n_ep} episodes)')
    results.append(check_expert_coverage(args.cfg, env, expert, n_ep))

    # [6] Reward sanity
    _header(6, total, 'Reward sanity (expert vs always-phase-0)')
    results.append(check_reward_sanity(args.cfg, env, expert))

    # Summary
    n_pass = sum(results)
    print()
    _sep('=')
    status = 'ALL PASS' if n_pass == total else f'{n_pass}/{total} PASS — FIX FAILURES BEFORE TRAINING'
    print(f'  {status}')
    _sep('=')
    print()

    sys.exit(0 if n_pass == total else 1)


if __name__ == '__main__':
    main()
