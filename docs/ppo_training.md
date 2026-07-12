# Movement PPO Training

The current controller trains a shared movement-level graph policy with proximal policy optimization (PPO). It can start from an imitation-learning checkpoint or, as in the current reference result, from random weights. The model is independent of city graph size and of the number of legal phases at each junction.

## From movement scores to legal phases

For every observation, the GNN produces a scalar score for each `Movement` node. Each traffic light has a phase-incidence matrix describing which movements receive green in each synthesized phase. Summing movement scores over each phase's incidence set produces local phase logits.

The runtime builds the same Boolean legal-action mask used during rollout collection and sampled evaluation. Phases blocked by minimum-green or transition constraints receive effectively negative-infinite logits. A categorical distribution is formed only over the remaining legal phases. SUMO signal programs, yellow transitions, and phase synthesis stay deterministic, so the learned model cannot invent an incompatible signal state.

The policy acts once per junction. A batch may therefore contain cities with different numbers of junctions and different local phase counts without padding the model to one global city-specific action space.

## PPO objective

PPO stores the sampled action, its log probability under the behavior policy, the legal mask, reward, value prediction, and bootstrapping state. Generalized advantage estimation produces an advantage for each junction decision. The actor minimizes the clipped surrogate objective:

```text
ratio = exp(new_log_probability - old_log_probability)
policy_objective = min(ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)
```

Clipping prevents one update from moving the policy too far from the distribution that collected the rollout. Approximate KL divergence and ratio-clipping frequency are logged as additional update diagnostics.

A value head pools movement embeddings by traffic light and predicts one value per junction. Its squared value error supplies the critic term. An entropy bonus rewards a broader legal phase distribution and discourages premature collapse; forced one-action decisions are excluded from the actor and entropy losses but still train the critic. The combined update contains policy, value, and entropy terms.

Rollouts are fixed-length segments rather than necessarily terminal episodes. If SUMO is still running at the segment boundary, the critic evaluates the next state and bootstraps both GAE and return targets. Only a genuinely terminated simulation uses a zero final value.

## Why actions are sampled

Rollout collection samples each junction's legal categorical phase distribution. This exploration is part of the policy optimized by PPO. The reference evaluation also uses `sample` mode at temperature `1.0`, so it measures the same stochastic controller semantics instead of replacing the trained policy with a greedy argmax controller after training.

Sampling means evaluation includes action randomness as well as demand and route randomness. Results must therefore average multiple seeds and report enough scenario metrics to reveal gridlock or incomplete trips. A deterministic learned-policy evaluation can be useful as a separate ablation, but it is a different policy and must be labeled accordingly.

Max-pressure and queue baselines are deterministic for a fixed configuration and seed. They are evaluated once and reused during periodic evaluation; later evaluation intervals need only run the changing learned policy.

## Multi-city rollouts and generalization

Each update may concatenate independent rollout segments from multiple cities. Every worker computes returns and advantages for its own segment before buffers are combined. A persistent process pool avoids restarting SUMO for each iteration, and libsumo provides the headless backend used by the reference run.

The intended generalization mechanism is parameter sharing over a common abstraction:

- directed lane-group nodes encode demand, queues, speed, occupancy, storage, and arrivals;
- movement nodes encode legal turns and turn-local state;
- typed message passing combines upstream demand with downstream supply;
- the same movement scorer is reused at every junction in every city;
- phase incidence converts variable-size movement sets into each junction's variable-size legal action space.

No city identity or fixed intersection index is required by the policy. Generalization is nevertheless an empirical question: topology-held-out validation does not substitute for independent training seeds and a final unseen-city evaluation.

## Reward and scenario sampling

The current multi-city reference uses the throughput reward mode. It combines local discharge throughput with a weighted global throughput signal, vehicle progress, a gridlock penalty, and a small speed-change penalty. The exact selected-run weights are documented in the [results report](results/city_first_pass_throughput_scratch_32_worker.md) and its committed experiment YAML.

Every rollout samples demand scale and target initial occupancy from configured ranges. Valid routes are generated from the shaped city network, vehicles are inserted at safe positions at simulation time zero, and one SUMO insertion step occurs before the first action. Periodic evaluation uses fixed seeds, fixed demand scale, and the same occupancy-generation procedure for all policies.

Training diagnostics include reward and return distributions, value scale, critic explained variance, normalized entropy, top-action probability, approximate KL, clipping frequency, policy/value loss, gradient norms, teleport counts, city-specific rollout congestion, and detailed timing.

Completion rate, vehicles remaining, teleports, throughput, and wait density should be interpreted together. Average waiting, travel, and time-loss values include only completed vehicles and can look artificially favorable when a poor controller leaves hard trips unfinished.

## Scratch training

The current experiment configuration supplies city splits, rollout allocation, reward weights, evaluation seeds, and worker counts. Random-scratch training additionally requires explicit graph feature and model dimensions:

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
uv run python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --scratch-random `
  --scratch-lane-feature-dim 29 `
  --scratch-movement-feature-dim 4 `
  --scratch-hidden-dim 64 `
  --scratch-num-hops 1
```

On Linux, set `export SUMO_HOME=/usr/share/sumo` and replace backslashes and PowerShell continuations with forward slashes and `\`.

## Evaluation

Evaluate the five-city experiment with the selected learned checkpoint and the two configured baselines:

```powershell
uv run python scripts\eval_multi_city.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --checkpoint artifacts\ppo_runs\city_first_pass_throughput_progress_025_sample_eval_v3\selected_iteration_0085\movement_policy_iter_0085_best.pt `
  --output-dir reports\city_first_pass_iteration_0085
```

The experiment config selects six seeds, 1,200 decision steps, sampled learned actions, and demand scale `1.0`.

## Resuming PPO

PPO checkpoints contain actor and critic parameters, optimizer state, completed iteration, best selection score, random-number-generator state, normalizers, and architecture metadata. Resume into the same run with:

```powershell
uv run python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --resume-checkpoint checkpoints\rl\<run>\movement_ppo_latest.pt `
  --iterations 500
```

`--iterations` is the final target iteration. Scenario and PPO parameters must remain consistent with the saved run. Do not interpret a latest checkpoint as the best checkpoint: the reference run regressed after iteration 85, and both the selected checkpoint and full trajectory are retained.
