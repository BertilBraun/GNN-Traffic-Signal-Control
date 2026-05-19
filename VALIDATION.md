# Validation Log

Bottom-up verification of each layer before building on top of it.
The rule: if you can't solve a trivial version, nothing else matters.

---

## Layer 1 — Environment correctness

**Script:** `python scripts/verify_env.py`

Six numbered checks, all automated:

| # | Check | Result |
|---|-------|--------|
| 1 | Startup: junction count, obs keys match junction IDs | ✓ 6 junctions |
| 2 | Obs shapes `(41,)`, dtype `float32`, no NaN/Inf | ✓ |
| 3 | Feature bounds: all 41 dims flagged for negatives, bad one-hots, out-of-range elapsed | ✓ all clean |
| 4 | Junction ordering identical between `TrafficEnv` and `GraphBuilder` | ✓ exact match |
| 5 | Expert uses all 4 phases; switch frequency in `[0.3, 4.0]` /junction/min | ✓ 1.667 /j/min |
| 6 | Expert wait density < always-phase-0 baseline | ✓ 12.4× better |

**Notable findings from check 3:**
- Slots 13–15 (`s1_thru_*`) and 27–35 (`s3_*`) are permanently zero — those approach arms don't exist in this network. This is correct; the model must learn to ignore them.
- Phase 0 dominates the expert distribution (44.7%) — the network is not symmetric.
- All features are non-negative; elapsed is in `[0, 1]`.

**Status: PASS** — environment is trustworthy.

---

## Layer 2 — Imitation learning can overfit a single episode

**Command:**
```
python scripts/train_il.py --episodes 30 --ep-len 600 --demand-seed 42 \
    --eval-every 5 --dagger-beta-end 1.0 --debug --n-eval-seeds 1
```

`--demand-seed 42` resets `env._rng` to the same seed before every `env.reset()`,
so every training episode sees identical traffic demand.
`--dagger-beta-end 1.0` keeps pure behavioural cloning (no DAgger).
Eval also uses seed 42 (`--n-eval-seeds 1`) so training and eval see the same demand.

**Expected:** loss → ~0, match → ~100%, eval wait density ≈ expert within 30 episodes.

**Observed:** loss reached 0.0009, match above 99% by episode 15. Eval confirmed near-perfect approximation of the expert.

**What this proves:**
- The model has sufficient capacity.
- The training loop (forward pass, loss, backward, optimiser step) is correct.
- The `GraphBuilder` → `GATPolicy` pipeline produces correct gradients.
- The `RunningNormalizer` does not corrupt features in a way that breaks learning.

**Status: PASS**

---

## Layer 3 — Imitation learning generalises across random demand

**Command:**
```
python scripts/train_il.py --episodes 100 --ep-len 3600 \
    --dagger-beta-end 0.0 --n-eval-seeds 5
```

DAgger anneals the expert-driving probability from β=1.0 (episode 0) to β=0.0
(episode 99). Expert labels are always used; only the environment is stepped with
model actions as β falls.

**Results (selected eval checkpoints):**

| Episode | wait density model | wait density expert | ratio |
|---------|--------------------|---------------------|-------|
| 10      | 2.381 s/m          | 0.152 s/m           | 15.7× worse |
| 20      | 1.819 s/m          | 0.152 s/m           | 12.0× worse |
| 80      | 0.213 s/m          | 0.152 s/m           | 1.40× worse |
| 90      | 0.161 s/m          | 0.152 s/m           | 1.06× worse |
| 100     | 0.189 s/m          | 0.152 s/m           | 1.24× worse |

Final multi-seed evaluation (seeds 42–46):

| Metric | Model | Expert |
|--------|-------|--------|
| avg_waiting_time (s) | 102.7 | 85.9 |
| throughput (veh/h)   | 647.8 | 652.2 |
| wait_density (s/m)   | 0.125 | 0.101 |

**Notable:** at ep 90 (β=0.10), the single-seed eval showed the model within 6% of the expert. The slight regression at ep 100 (β=0.00) is expected — pure self-driving reintroduces compounding errors that DAgger couldn't fully eliminate.

**Notable:** during mid-training evals, the model occasionally *beat* the expert on throughput/wait time on individual seeds. This is the RL signal: the greedy expert is myopic and the model sometimes finds better phase coordination by chance.

**Status: PASS** — IL learns and generalises. The best checkpoint (≈ep 90) is a sound RL starting point.

---

## Layer 4 — PPO value warmup

**Command:**

```sh
python scripts/train_rl.py --il-ckpt checkpoints/il/<stamp> \
    --iterations 30 --warmup 30 --warmup-epochs 10 --ep-len 1200 --burn-in 10
```

**Observed:**

- v_loss dropped from 1.6 → ~0.2 during warmup (first 3–4 iterations saw largest drop)
- Explained variance (EV) grew from 0 → 0.3–0.45 once policy updates began, confirming the value head is state-conditional rather than a global mean predictor
- Residual v_loss floor (~0.2–0.4) is irreducible: policy stochasticity means the same state produces different future trajectories even with fixed demand and SUMO seeds

**Key fixes required:**

- GAE bootstrapped targets caused a moving-target chasing loop → switched to **MC returns** during warmup (stable, value-independent targets)
- SUMO internal RNG (driver speed jitter) caused return variance even with fixed demand seed → fixed by deriving SUMO `--seed` from `env._rng`

**Status: PASS** — value head converges to a useful state-conditional estimate

---

## Layer 5 & 6 — PPO policy improvement (randomised demand)

**Command:**

```sh
python scripts/train_rl.py --il-ckpt checkpoints/il/<stamp> \
    --iterations 200 --warmup 20 --episodes 2 --ep-len 1200 --burn-in 10 \
    --switch-penalty 0.0 --entropy-coeff 0.01 --clip 0.1
```

**Failed first attempt (iterations 1–100):**

- `--switch-penalty 0.1` caused reward hacking: policy learned to hold phases (switch_freq 1.77→1.05) to avoid the penalty, causing queue buildup (wait_density 0.067→0.456)
- `--entropy-coeff 0.05` pushed entropy from 0.14→0.43, randomising a near-optimal IL policy
- Root cause: switch penalty and entropy bonus are both adversarial to an IL-initialised policy

**Fixed run results (5-seed final eval, 200 iterations):**

| Metric | Model | Expert | Delta |
| --- | --- | --- | --- |
| avg_waiting_time (s) | 33.97 | 67.29 | **−50%** |
| avg_travel_time (s) | 120.3 | 153.9 | **−22%** |
| throughput (veh/h) | 421.8 | 412.2 | +2.3% |
| max_queue (vehs) | 5.0 | 6.0 | −17% |
| wait_density (s/m) | 0.014 | 0.044 | **−68%** |
| switch_freq (/j/min) | 3.633 | 1.718 | +112% |

**Notable:** RL discovered that aggressive phase cycling (3.6 switches/j/min vs expert's 1.7) prevents queue buildup more effectively than the greedy expert's conservative hold strategy. With 5s yellow+all-red per 15s interval, 33% of time is in transition — still net positive.

**Status: PASS** — PPO significantly outperforms the greedy expert baseline

---

## Decision log

| Date       | Decision |
|------------|----------|
| 2026-05-14 | DAgger adopted after pure BC showed 3.5× eval gap despite 85% train match rate |
| 2026-05-14 | Multi-seed final eval adopted (5 seeds) after single-seed showed high variance |
| 2026-05-14 | Zero-init on value head adopted to eliminate bootstrap bias at PPO start |
| 2026-05-14 | Best-checkpoint saving added so PPO starts from peak IL policy, not final |
