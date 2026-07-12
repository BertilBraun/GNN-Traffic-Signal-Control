# PPO libsumo 200-step eval v3 run notes

This note documents the long PPO continuation run in:

- remote log dir: `/root/GNN-Traffic-Signal-Control/runs/rl/city_first_pass_4_worker_libsumo_200step_eval_v3`
- remote checkpoint dir: `/root/GNN-Traffic-Signal-Control/checkpoints/rl/city_first_pass_4_worker_libsumo_200step_eval_v3`
- local TensorBoard mirror: `runs/remote_tensorboard/city_first_pass_4_worker_libsumo_200step_eval_v3`
- local throughput plots: `runs/remote_tensorboard/plots`

The run is not a clean single-configuration experiment. Several reward, rollout, routing, city, and evaluation-cache changes were introduced while continuing from existing checkpoints. Treat trend plots as a run diary, not as a controlled ablation.

## Current interpretation

The optimizer looks healthy, but the learned policy is not clearly improving into a robust multi-city controller.

Observed from TensorBoard through evaluation iteration 720:

- PPO update diagnostics are stable: KL stays around `0.0007-0.0009`, clip fraction around `3%`, no KL early stops, entropy remains high, and explained variance is usually `0.97-0.99`.
- The critic fits the shaped rollout reward well, but that does not imply the policy is good under evaluation metrics.
- Shaped rollout reward and return improved after reward changes, especially after the flow/global reward work, but held-out evaluation did not improve consistently.
- Throughput gains are city-specific. Karlsruhe is the clean success case; Heidelberg is partially positive. Mannheim, Stuttgart, and Freiburg are mixed or weak.
- Wait density is a diagnostic/guardrail, not necessarily the primary objective, but the learned policy often has materially worse wait density than max-pressure/queue. This can be caused by real starvation, OSM network defects, or deliberate platoon holding. It needs per-junction inspection before using it as a headline metric.
- Training rollout teleports increased after roughly iteration 500. Eval teleports remained zero in the parsed data, but rollout teleports suggest sampled training states are becoming more problematic.

The most concerning signal is that PPO continues to optimize the shaped reward while per-city throughput/generalization is inconsistent. That points more to reward/data/evaluation alignment than to PPO numerical instability.

## Timeline of explicit changes

| Iteration | Change | Interpretation impact |
|---:|---|---|
| 0-20 | Initial PPO trial from the IL checkpoint | Short initial run with value warmup. |
| 20 | `c1392f0`: evaluation backend moved to libsumo | Evaluation runtime improvement; policy objective unchanged. |
| ~185 | timestamps and rollout scheduling work | Logging/runtime changes. |
| 186 | 16 rollout workers and optimized PPO batch collation | Runtime/sample throughput change. |
| 215 | evaluation every 15 iterations | More regular eval tracking. |
| 294 | `9abe014`: flow reward added and Mannheim excluded from rollouts | Major data/reward change. Mannheim is no longer a trained rollout city after this point. |
| 295 | reward weights set to `global_reward_weight=0.2`, `flow_reward_weight=0.2` | Stronger flow/global objective. |
| ~440 | `d5a0489`: fastest route generation and arrived-flow PPO reward | Major routing/reward-semantics change. Reward plots before/after this are not directly comparable. |
| 490 | `bd3da3a`: 20 workers, 30 rollout jobs, 500 steps/rollout | Big rollout distribution and throughput change. |
| 490 | `42802a6`: per-city rollout demand cap, Stuttgart `train_scale_max=1.05` | Reduces extreme Stuttgart training congestion. |
| 490 | `dc1c0cb`: save every iteration, retain latest 5 | Checkpointing only. |
| 517 | `4d0b27c`: pruned/rebuilt Stuttgart network | Stuttgart becomes a different environment; pre/post Stuttgart evals are not directly comparable. |
| 640 | resumed from iteration 640 to target 800 | No intended reward/config change. |
| 645 | deleted `.cache/evaluation/*.json` | Max-pressure/queue baselines recomputed. Baseline curves before/after this have a comparison break. |

Cleanest interpretation windows:

- `295-439`: flow/global reward, Mannheim excluded, before fastest routes/arrived-flow.
- `440-489`: fastest routes/arrived-flow, old rollout schedule.
- `490-516`: 20 workers / 30 jobs / 500 steps, old Stuttgart network.
- `518-644`: pruned Stuttgart network, current rollout setup, but stale baseline cache possible.
- `645+`: current setup plus recomputed baselines.

## Evaluation snapshot at iteration 720

Latest parsed evaluation snapshot from TensorBoard:

| City | Policy | Throughput / h | Completion | Wait density | Time loss |
|---|---|---:|---:|---:|---:|
| Karlsruhe | learned | 4140 | 0.828 | 0.040 | 132.8 |
| Karlsruhe | max-pressure | 3702 | 0.741 | 0.091 | 134.0 |
| Karlsruhe | queue | 3780 | 0.756 | 0.079 | 138.9 |
| Mannheim | learned | 3726 | 0.472 | 0.311 | 131.5 |
| Mannheim | max-pressure | 3960 | 0.500 | 0.158 | 146.5 |
| Mannheim | queue | 3930 | 0.497 | 0.156 | 145.4 |
| Stuttgart | learned | 4548 | 0.442 | 0.326 | 135.8 |
| Stuttgart | max-pressure | 4608 | 0.448 | 0.155 | 150.3 |
| Stuttgart | queue | 4710 | 0.457 | 0.124 | 151.2 |
| Heidelberg | learned | 3678 | 0.686 | 0.138 | 101.9 |
| Heidelberg | max-pressure | 3678 | 0.686 | 0.074 | 126.2 |
| Heidelberg | queue | 3594 | 0.671 | 0.099 | 123.5 |
| Freiburg | learned | 3048 | 0.428 | 0.402 | 119.8 |
| Freiburg | max-pressure | 3042 | 0.427 | 0.148 | 146.6 |
| Freiburg | queue | 3042 | 0.427 | 0.164 | 140.1 |

City-level read:

- Karlsruhe: learned is clearly better on throughput, completion, and wait density.
- Heidelberg: learned matches or beats throughput and has lower time loss, but wait density is worse than baselines.
- Mannheim: learned is worse on throughput/completion/wait density, despite lower time loss. Since Mannheim is excluded from rollouts, this should be interpreted as a generalization case, not a trained-city success/failure.
- Stuttgart: after pruning and baseline recomputation, learned is not better than max-pressure/queue on throughput or wait density.
- Freiburg: held-out throughput is roughly tied, but wait density is much worse.

## Final 5-seed evaluation at iteration 800

After PPO completed at iteration 800, a fresh evaluation suite was run with seeds `200-204`, `800` steps, demand scale `1.0`, libsumo backend, and policies `learned`, `max-pressure`, and `queue`.

Remote outputs:

- summary JSON: `/root/GNN-Traffic-Signal-Control/checkpoints/rl/city_first_pass_4_worker_libsumo_200step_eval_v3/eval/iter_0800_fresh_5seed/summary.json`
- summary CSV: `/root/GNN-Traffic-Signal-Control/checkpoints/rl/city_first_pass_4_worker_libsumo_200step_eval_v3/eval/iter_0800_fresh_5seed/summary.csv`
- TensorBoard log: `/root/GNN-Traffic-Signal-Control/runs/rl/city_first_pass_4_worker_libsumo_200step_eval_v3/eval_iter_0800_fresh_5seed`

Local copies:

- summary files: `reports/ppo_eval/iter_0800_fresh_5seed/`
- final eval TensorBoard: `runs/remote_tensorboard/city_first_pass_4_worker_libsumo_200step_eval_v3_eval_iter_0800_fresh_5seed/`
- training TensorBoard: `runs/remote_tensorboard/city_first_pass_4_worker_libsumo_200step_eval_v3/`
- checkpoints: `checkpoints/remote_rl/city_first_pass_4_worker_libsumo_200step_eval_v3/`

Final 5-seed mean results:

| City | Policy | Throughput / h | Completion | Wait density | Time loss | Nonstop pass rate |
|---|---|---:|---:|---:|---:|---:|
| Karlsruhe | learned | 3785.4 | 0.886 | 0.0709 | 144.9 | 0.518 |
| Karlsruhe | max-pressure | 3652.2 | 0.853 | 0.0943 | 151.4 | 0.372 |
| Karlsruhe | queue | 3457.8 | 0.808 | 0.1796 | 152.8 | 0.362 |
| Mannheim | learned | 3575.7 | 0.532 | 0.4999 | 157.4 | 0.561 |
| Mannheim | max-pressure | 3673.8 | 0.541 | 0.3969 | 171.1 | 0.493 |
| Mannheim | queue | 3723.3 | 0.550 | 0.3961 | 163.1 | 0.484 |
| Stuttgart | learned | 4691.7 | 0.517 | 0.5225 | 156.5 | 0.546 |
| Stuttgart | max-pressure | 4615.2 | 0.509 | 0.3421 | 178.5 | 0.392 |
| Stuttgart | queue | 4546.8 | 0.502 | 0.3956 | 171.3 | 0.397 |
| Heidelberg | learned | 3202.2 | 0.691 | 0.3822 | 117.5 | 0.600 |
| Heidelberg | max-pressure | 3422.7 | 0.741 | 0.1747 | 141.5 | 0.426 |
| Heidelberg | queue | 3461.4 | 0.749 | 0.1757 | 142.1 | 0.425 |
| Freiburg | learned | 2552.4 | 0.414 | 0.9392 | 128.2 | 0.570 |
| Freiburg | max-pressure | 2779.2 | 0.450 | 0.4671 | 167.1 | 0.377 |
| Freiburg | queue | 2737.8 | 0.444 | 0.4818 | 163.4 | 0.371 |

Final read:

- Karlsruhe is a real success case: learned wins throughput, completion, wait density, time loss, and smoothness.
- Stuttgart is mixed: learned has the best throughput/completion and lower time loss, but much worse wait density than baselines. This looks like a flow/smoothness tradeoff rather than a clean win.
- Mannheim is not a success: learned loses throughput and completion, and wait density is worse, although time loss and nonstop pass rate are better.
- Heidelberg is not a success on throughput/completion: learned has better time loss and smoothness, but baselines move more vehicles through.
- Freiburg held-out is poor: learned loses throughput/completion and has roughly double the wait density of the baselines, although time loss and nonstop pass rate are better.

The final 5-seed evaluation confirms the TensorBoard impression: the learned policy is not a robust all-city improvement. It has learned a smoother/less stopping behavior in several places, but it often trades that for lower completion/throughput or higher wait density.

This argues against continuing the same training run further. More iterations are unlikely to fix the mismatch without changing the objective, data split, or evaluation target.

## What the throughput plots show

Generated local plots:

- `runs/remote_tensorboard/plots/eval_throughput_all_cities.svg`
- `runs/remote_tensorboard/plots/eval_throughput_karlsruhe_oststadt.svg`
- `runs/remote_tensorboard/plots/eval_throughput_mannheim_innenstadt.svg`
- `runs/remote_tensorboard/plots/eval_throughput_stuttgart_mitte.svg`
- `runs/remote_tensorboard/plots/eval_throughput_heidelberg_bergheim.svg`
- `runs/remote_tensorboard/plots/eval_throughput_freiburg_altstadt.svg`

The plots include a dotted line at iteration 645 for the baseline cache reset.

High-level plot interpretation:

- Karlsruhe has a credible upward learned-policy trend.
- Heidelberg has some positive movement, but not enough to declare a robust win.
- Mannheim does not show a useful learned-policy trend.
- Stuttgart is hard to reason about because pruning changed the network at iteration 517 and stale baseline cache was cleared at 645. Post-645 learned does not obviously beat baselines.
- Freiburg is not convincingly improving; held-out behavior is roughly flat/noisy and not clearly superior.

## Why five minutes per iteration is a problem

At roughly 4-5 minutes per iteration, rerunning long exploratory variants is expensive and slow to interpret. The current setup is good enough for a final continuation but too expensive for broad reward search.

The expensive loop should not be used as the main tool for discovering reward design. Use it only after narrower probes indicate that the reward/eval alignment is plausible.

## Recommended next steps

1. Let the current run finish to 800 unless it clearly degrades.

2. At 800, run a focused multi-seed evaluation on current networks:
   - learned checkpoint at 800
   - best historical checkpoint if still available
   - max-pressure
   - queue
   - 3-5 eval seeds
   - same current pruned/rebuilt networks

3. Add per-junction outlier reporting before making reward decisions:
   - worst 10 junctions by wait density
   - worst 10 junctions by max queue
   - phase usage counts for those junctions
   This separates policy failure from impossible OSM artifacts.

4. Use throughput/completion/time-loss/stops/nonstop-pass metrics as primary evaluation metrics. Keep wait density as a starvation/network-defect guardrail.

5. Do not do another full 500+ iteration reward experiment immediately. First run short controlled probes:
   - one fixed checkpoint
   - deterministic eval only
   - compare reward components and per-junction pathologies
   - optionally 50-100 iteration mini-runs only after reward/eval metrics are aligned

6. Consider simplifying the training set for the next controlled experiment:
   - either explicitly train on Karlsruhe/Heidelberg/Stuttgart and evaluate Freiburg/Mannheim as held-out/generalization,
   - or reintroduce Mannheim if it is meant to count as a train-city success metric.

## Planned outlier stress pass

Run a one-off stress evaluation separate from the main PPO training loop:

- cities: Karlsruhe, Mannheim, Stuttgart, Heidelberg, Freiburg
- seeds: five per city if runtime allows, at least three if not
- demand: maximum stress demand scale
- horizon: about 800 evaluation steps
- policies: at minimum the learned checkpoint; include max-pressure/queue if runtime allows
- outputs: top 10 junctions per run by wait density and max queue
- summary: flag junctions that recur in the top 10 across seeds for a city

This pass should run on the remote node and write artifacts under a scratch/results folder, for example `.tmp/outlier_eval/`. Its purpose is to identify whether high wait-density metrics are caused by a small number of pathological OSM junctions rather than general policy behavior.

The stress pass completed on the remote node:

- checkpoint: `checkpoints/rl/city_first_pass_4_worker_libsumo_200step_eval_v3/movement_policy_latest.pt`
- policy: learned
- backend: libsumo
- seeds: `100-104`
- demand scale: `1.2`
- steps: `800`
- completed runs: `25/25`
- artifact directory: `/root/GNN-Traffic-Signal-Control/.tmp/outlier_eval/stress_20260710_800_steps_scale_1p2/`
- summary JSON: `/root/GNN-Traffic-Signal-Control/.tmp/outlier_eval/stress_20260710_800_steps_scale_1p2/recurring_outliers_summary.json`
- summary CSV: `/root/GNN-Traffic-Signal-Control/.tmp/outlier_eval/stress_20260710_800_steps_scale_1p2/recurring_outliers_summary.csv`

The stress pass did not reveal a single obvious broken junction that explains the evaluation behavior. Every city has recurring top-10 wait/queue junctions under high demand, but the recurrence pattern looks like normal bottleneck sets in dense OSM networks rather than one isolated impossible junction dominating the metric.

Most recurring learned-policy stress outliers:

- Karlsruhe: multiple recurring `5/5` wait and queue clusters, including `cluster_11362973189_11362973190_1247533339_1725564649_#9more`, `cluster_1476083280_1476083281_1476083282_1476083283_#10more`, and `cluster_14892151_1725103321_21618145_5864209624`.
- Mannheim: several recurring wait/queue clusters, including `cluster_1126992321_1126992336_1126992394_132331226_#13more` and `cluster_1757469961_1757469965_1757473743_1757473744_#1more`; also some wait-only recurrent clusters.
- Stuttgart: recurrent queue/wait clusters include `cluster_1332657885_80296454`, `cluster_1566983600_1566983654_1566983657_78689710_#5more`, and `cluster_31128107_6297802686`; several queue-only outliers recur across all five seeds.
- Heidelberg: recurring clusters include `cluster_10704229380_10704229382_10704229385_10704229386_#9more`, `cluster_11247600_13318863087_2290584079_269603382_#7more`, and `cluster_269603383_270237900_270237901_2980068472_#3more`.
- Freiburg: recurring clusters include `cluster_1245631036_12463900665_12463993027_12463993037_#14more`, `cluster_2229658082_2229658085_2406471843_26997402_#16more`, and `cluster_600507808_662882072_662882080_662882084_#1more`.

Interpretation: high wait density should still be treated as a guardrail, but the stress pass does not support dismissing the final evaluation as a single-junction artifact.

## Open questions

- Is the headline objective throughput/completion, or smoother green-wave behavior? The reward and checkpoint selection should encode this explicitly.
- Should Mannheim remain excluded from rollouts? If yes, it should be reported as generalization, not as a trained-city metric.
- Should checkpoint selection optimize held-out score, train aggregate score, or a constrained objective such as throughput with a wait-density/starvation cap?
- Are Stuttgart/Freiburg wait-density spikes caused by policy behavior or by a small number of broken junctions?
