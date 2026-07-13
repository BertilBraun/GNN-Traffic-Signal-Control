# Iteration-85 frozen-checkpoint sensitivity protocol

This analysis is deliberately separate from the iteration-85 evidence bundle. It does not retrain or alter the learned policy.

## Planned protocol

- checkpoint: `selected_iteration_0085/movement_policy_iter_0085_best.pt`
- cities: all four rollout cities and held-out-from-rollouts Freiburg
- policies: sampled learned policy at temperature 1.0, max pressure, and longest queue
- standard horizon: 1,200 simulated seconds
- fresh scenario seed: 200
- demand sensitivity: 0.8 and 1.2 (the archived evaluation already covers demand 1.0 with seeds 100--105)
- longer-horizon check: Freiburg at demand 1.0, seed 200, 2,400 simulated seconds

The initially proposed Cartesian product of seeds 200--202 and demand scales 0.8, 1.0, and 1.2 would require 135 episodes. A local TraCI timing probe did not complete even its first 1,200-second learned episode within approximately 90 seconds, implying a multi-hour serial run. The bounded protocol therefore uses one genuinely fresh seed at the two sensitivity scales and reserves the longer horizon for Freiburg. It is supportive analysis only: no confidence intervals, significance tests, or broad robustness claim will be based on one fresh seed.

## Execution and recovery

The first standard-protocol launch was interrupted when the host crashed. The evaluator writes its combined CSV and JSON only after every requested episode finishes, so no partial summary could be recovered. During recovery, inspection found that `eval_multi_city.py` did not forward the configured `ppo.eval_workers` value to the already implemented parallel evaluator. The CLI was updated to forward that setting and to accept an explicit `--workers` override.

A three-worker Freiburg long-horizon restart completed successfully. An initial 30-worker standard restart caused severe CPU contention and was deliberately stopped after verifying and terminating its full process tree. The final standard run used eight workers and completed all 30 episodes. The host suspended during this run, so elapsed wall-clock time is not a useful performance measurement. The uninterrupted Freiburg three-episode batch took approximately 19 minutes. These operational restarts did not change the checkpoint, scenario seeds, demand, policies, simulator settings, or metrics.

## Exact successful commands

```powershell
uv run python scripts/eval_multi_city.py `
  --experiment-config reports/freiburg_iteration_0085_sensitivity.yaml `
  --policies learned max-pressure queue `
  --checkpoint artifacts/ppo_runs/city_first_pass_throughput_progress_025_sample_eval_v3/selected_iteration_0085/movement_policy_iter_0085_best.pt `
  --output-dir reports/iteration_0085_sensitivity_freiburg_long `
  --device cpu `
  --seeds 200 `
  --steps 2400 `
  --demand-scales 1.0 `
  --log-dir reports/iteration_0085_sensitivity_freiburg_long/tensorboard `
  --cache-dir reports/iteration_0085_sensitivity_cache
```

```powershell
uv run python scripts/eval_multi_city.py `
  --experiment-config configs/training/city_first_pass_throughput_scratch_32_worker.yaml `
  --policies learned max-pressure queue `
  --checkpoint artifacts/ppo_runs/city_first_pass_throughput_progress_025_sample_eval_v3/selected_iteration_0085/movement_policy_iter_0085_best.pt `
  --output-dir reports/iteration_0085_sensitivity_standard `
  --device cpu `
  --workers 8 `
  --seeds 200 `
  --steps 1200 `
  --demand-scales 0.8 1.2 `
  --log-dir reports/iteration_0085_sensitivity_standard/tensorboard `
  --cache-dir reports/iteration_0085_sensitivity_cache
```

## Fresh-seed demand sensitivity results

All values below are single episodes at seed 200. Throughput is vehicles per hour, completion is the fraction of departed vehicles completing within the horizon, and wait density is seconds per meter.

| City | Demand | Policy | Throughput | Completion | Wait density | Teleports |
|---|---:|---|---:|---:|---:|---:|
| Karlsruhe | 0.8 | Learned | 2388 | 85.1% | 0.209 | 0 |
|  |  | Max pressure | 2361 | 84.2% | 0.230 | 0 |
|  |  | Queue | 2310 | 82.4% | 0.323 | 0 |
| Karlsruhe | 1.2 | Learned | 2460 | 77.9% | 0.387 | 0 |
|  |  | Max pressure | 2550 | 80.9% | 0.258 | 0 |
|  |  | Queue | 2505 | 79.4% | 0.314 | 0 |
| Mannheim | 0.8 | Learned | 2013 | 44.3% | 1.082 | 0 |
|  |  | Max pressure | 2967 | 64.9% | 0.524 | 0 |
|  |  | Queue | 3063 | 67.0% | 0.437 | 0 |
| Mannheim | 1.2 | Learned | 2358 | 43.6% | 0.956 | 0 |
|  |  | Max pressure | 3231 | 59.3% | 0.677 | 0 |
|  |  | Queue | 3363 | 61.6% | 0.585 | 0 |
| Stuttgart | 0.8 | Learned | 2919 | 46.0% | 1.193 | 0 |
|  |  | Max pressure | 3108 | 49.1% | 0.951 | 0 |
|  |  | Queue | 3534 | 55.5% | 0.638 | 0 |
| Stuttgart | 1.2 | Learned | 3582 | 42.8% | 1.363 | 0 |
|  |  | Max pressure | 3954 | 47.4% | 0.890 | 0 |
|  |  | Queue | 3837 | 46.2% | 1.174 | 0 |
| Heidelberg | 0.8 | Learned | 2025 | 64.2% | 0.668 | 0 |
|  |  | Max pressure | 2721 | 86.3% | 0.032 | 1 |
|  |  | Queue | 2640 | 83.7% | 0.129 | 0 |
| Heidelberg | 1.2 | Learned | 1866 | 49.2% | 1.128 | 0 |
|  |  | Max pressure | 3024 | 79.3% | 0.125 | 0 |
|  |  | Queue | 3006 | 78.8% | 0.157 | 0 |
| Freiburg | 0.8 | Learned | 2463 | 57.9% | 0.579 | 0 |
|  |  | Max pressure | 2439 | 57.5% | 0.348 | 0 |
|  |  | Queue | 2091 | 49.3% | 0.697 | 0 |
| Freiburg | 1.2 | Learned | 2727 | 51.9% | 0.576 | 0 |
|  |  | Max pressure | 2502 | 47.7% | 0.658 | 0 |
|  |  | Queue | 2292 | 43.7% | 0.754 | 0 |

At seed 200, learned control led throughput at low demand in Karlsruhe and Freiburg, and at high demand in Freiburg. It trailed both heuristics in Mannheim, Stuttgart, and Heidelberg at both sensitivity scales, and was slightly behind both at high-demand Karlsruhe. Thus, the fresh episode preserves the promising Freiburg result but also shows that the favorable archived Heidelberg result is not stable across this seed-and-demand change.

## Freiburg longer-horizon check

| Policy | Throughput | Completion | Wait density | Teleports |
|---|---:|---:|---:|---:|
| Learned | 1762.5 | 48.9% | 2.220 | 0 |
| Max pressure | 1522.5 | 42.3% | 2.618 | 0 |
| Queue | 1257.0 | 34.8% | 4.234 | 0 |

At seed 200 and a 2,400-second horizon, learned control remained ahead of both heuristics on throughput, completion, and wait density. Absolute throughput is lower and congestion metrics are worse than in the archived 1,200-second Freiburg evaluation, but this run simultaneously changes the seed and horizon. It therefore supports only a within-run policy comparison, not a causal estimate of horizon sensitivity.

## Comparison with the archived evaluation

The archived result used demand 1.0, seeds 100--105, and a 1,200-second horizon. It reported the learned policy above both baselines in Heidelberg and Freiburg, close to the best baseline in Karlsruhe and Stuttgart, and behind both in Mannheim. The new seed-200 sensitivity episodes show:

- the Freiburg ordering is preserved at demand 0.8 and 1.2, and in the longer demand-1.0 run;
- the Mannheim weakness is preserved and becomes larger in these episodes;
- Karlsruhe remains competitive, leading at demand 0.8 but trailing slightly at demand 1.2;
- Stuttgart trails the heuristics at both added demand scales;
- Heidelberg reverses the archived ordering at both added demand scales.

These results strengthen the narrow claim that the shared controller can execute successfully and sometimes lead strong baselines under changed demand on a held-out-from-rollouts topology. They also reinforce that one run and one checkpoint do not establish consistent cross-city superiority.

## Output files

- `reports/iteration_0085_sensitivity_standard/summary.csv`
- `reports/iteration_0085_sensitivity_standard/summary.json`
- `reports/iteration_0085_sensitivity_freiburg_long/summary.csv`
- `reports/iteration_0085_sensitivity_freiburg_long/summary.json`

TensorBoard event files and the evaluation cache are operational by-products and are not required to reproduce the reported tables from the summary exports.

## Limitations

- Each added condition has only one seed, so there are no confidence intervals or significance tests.
- Learned actions remain sampled at temperature 1.0; a single episode includes action-sampling variance as well as traffic-scenario variance.
- Demand-scale comparisons also regenerate demand and are not paired vehicle-for-vehicle counterfactuals.
- The longer-horizon check changes both seed and horizon relative to the archive.
- Freiburg was excluded from PPO rollout generation but was periodically monitored during development.
- One Heidelberg max-pressure episode at demand 0.8 recorded one teleport; all other added episodes recorded zero.
- Host crashes and suspensions affected runtime measurement but not the completed result files.
