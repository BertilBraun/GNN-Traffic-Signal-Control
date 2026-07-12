# Training and Evaluation

The movement GNN supports three initialization paths:

1. random-scratch PPO;
2. imitation learning followed by PPO;
3. PPO resume from a complete actor/critic/optimizer checkpoint.

All paths use the same movement graph, phase-incidence aggregation, and legal-action mask described in [Architecture and constraints](architecture.md).

## Imitation learning

Imitation learning trains the GNN movement scorer to reproduce a deterministic teacher such as max pressure. The teacher supplies movement scores and selected phases; training combines movement-score regression with phase-ranking cross-entropy.

Collect balanced samples from the rollout cities in an experiment config:

```powershell
uv run python scripts\collect_multi_city_il.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --output-dir datasets\city_first_pass_il
```

Train a shared checkpoint from the combined dataset:

```powershell
uv run python scripts\train_il.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --data datasets\city_first_pass_il\combined.jsonl `
  --ckpt-dir checkpoints\il\city_first_pass
```

The experiment YAML supplies sample counts, collection workers, epochs, batch size, and phase-loss coefficient unless an explicit CLI override is provided.

## PPO from imitation learning

Start PPO with the trained actor and matching normalizers:

```powershell
uv run python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass_4_worker.yaml `
  --il-checkpoint checkpoints\il\city_first_pass\movement_policy_best.pt
```

The critic is new even though the actor and shared GNN are initialized from imitation learning. `ppo.value_warmup_iterations` in the experiment YAML can train the value head before normal joint PPO updates. During warm-up:

- the actor and shared policy representation are frozen;
- the final value layer starts at zero;
- bootstrapped discounted returns train the critic;
- normal clipped PPO begins after the configured warm-up iterations.

Warm-up is useful when a competent imitation actor would otherwise be disturbed by an initially uncalibrated critic. The scratch iteration-85 run used no value warm-up because both actor and critic began together from random weights.

## Random-scratch PPO

Reproduce the architecture and experiment settings used by the iteration-85 run:

```powershell
uv run python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --scratch-random `
  --scratch-lane-feature-dim 29 `
  --scratch-movement-feature-dim 4 `
  --scratch-hidden-dim 64 `
  --scratch-num-hops 1
```

The YAML assigns 10 rollout jobs to each of four training cities, for 40 independent 350-decision-step segments per update, collected by 32 persistent libsumo workers.

## PPO objective

For each legal junction decision, PPO records the sampled action, behavior log probability, legal mask, reward, value, and bootstrap state. Generalized advantage estimation produces the policy advantage. The actor uses the clipped surrogate:

```text
ratio = exp(new_log_probability - old_log_probability)
objective = min(ratio * advantage,
                clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)
```

The critic minimizes value error. An entropy term encourages exploration over legal phases. Decisions forced to one legal phase train the critic but are excluded from actor and entropy loss. Approximate KL divergence, PPO clipping frequency, entropy, policy/value loss, gradients, returns, and explained variance are logged.

Rollouts are fixed-length segments. If SUMO is still running at a segment boundary, the critic estimates the next state and bootstraps GAE and return targets. Only genuine termination uses a zero final value.

## Reward

The scratch reference run used throughput reward mode with:

| component | weight |
| --- | ---: |
| local throughput | 1.0 |
| global reward | 0.2 |
| vehicle progress | 0.25 |
| gridlock penalty | 0.08 |
| speed change | 0.005 |

Training demand was sampled from configured city ranges, and target initial occupancy from 5–8%. SUMO gridlock teleporting was disabled. The reward weights, demand ranges, rollout allocation, and evaluation settings are committed in the experiment YAML.

## Stochastic rollout and evaluation

PPO rollouts sample one categorical action from each junction's currently legal phase logits. The reference learned-policy evaluation uses the same sampled legal-action semantics at temperature `1.0`. This evaluates the policy distribution optimized by PPO.

Max-pressure and queue baselines are deterministic for a fixed city, demand, and seed. They score the same legal phases using hand-designed pressure or queue values and select the maximum. The GUI runner uses greedy learned phase scores for visual stability; it is not the sampled evaluation path.

## Checkpoints

Training writes latest, periodic, and best checkpoints. PPO checkpoint files contain actor and critic parameters, optimizer state, completed iteration, random-number-generator state, feature normalizers, and architecture metadata. Policy-only files are convenient for inference.

Best-checkpoint selection is an internal training mechanism based on the configured periodic learned-policy evaluation score. It does not mean that the selected checkpoint beat a baseline on every city, and baseline performance is not encoded into the checkpoint. The iteration-85 report evaluates that frozen checkpoint city by city.

Resume a complete PPO state with:

```powershell
uv run python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --resume-checkpoint checkpoints\rl\<run>\movement_ppo_latest.pt `
  --iterations 500
```

`--iterations` is the final target iteration, not the number of additional iterations.

## Multi-city evaluation

Evaluate the frozen iteration-85 checkpoint across the configured cities and baselines:

```powershell
uv run python scripts\eval_multi_city.py `
  --experiment-config configs\training\city_first_pass_throughput_scratch_32_worker.yaml `
  --checkpoint artifacts\ppo_runs\city_first_pass_throughput_progress_025_sample_eval_v3\selected_iteration_0085\movement_policy_iter_0085_best.pt `
  --output-dir reports\city_first_pass_iteration_0085
```

The experiment configuration uses six fixed seeds, demand scale `1.0`, and 1,200 decision steps. Evaluation reports throughput, completion, teleports, wait density, queue metrics, and completed-trip waiting/travel metrics.

Completion and congestion must accompany completed-trip averages. Average waiting, travel, and time loss exclude unfinished vehicles and can look artificially favorable when a controller leaves difficult trips in the network.

## Final generalization protocol

Freiburg generated no PPO rollouts but was used during development, so it is validation rather than an untouched final test. A strong final claim requires multiple independent training seeds, fresh evaluation seeds, confidence intervals, and another unseen OSM city that was not used for checkpoint selection.
