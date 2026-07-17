# Current-Method City PPO Preflight

## Decision

The city PPO gate should retain demand around `1.0`. The previous iteration-85 experiment trained
Karlsruhe and Heidelberg at demand scales `0.8-1.2`, capped Mannheim and Stuttgart at `0.8-1.05`,
and evaluated at `1.0`. Scale `1.0` was already congested: max-pressure completion in the previous
report ranged from approximately 47% to 82% across the five cities.

The new gate configuration is
[`city_current_local_reward_2hop_gate_30.yaml`](../../configs/training/city_current_local_reward_2hop_gate_30.yaml).
It retains those demand ranges while changing the intended method:

- five-second decisions with immediate three-second yellow;
- one-decision minimum green;
- 6-8% initial occupancy and a 15-second warm-up;
- two graph hops, supplied explicitly at scratch launch;
- local progress/discharge/braking/gridlock reward `1/10/10/0.02`;
- zero global, flow, throughput, and direct switch reward;
- four PPO epochs, entropy coefficient `0.001`, and 200 decisions per rollout.

## Action-sample balance

Equal city rollout counts substantially overweight larger controlled graphs. The old allocation used
ten 350-decision rollouts per city:

| city | controllers | old action samples/update |
| --- | ---: | ---: |
| Karlsruhe | 41 | 143,500 |
| Mannheim | 84 | 294,000 |
| Stuttgart | 69 | 241,500 |
| Heidelberg | 56 | 196,000 |

The new 200-decision allocation is inverse to controller count:

| city | rollouts/update | action samples/update |
| --- | ---: | ---: |
| Karlsruhe | 26 | 213,200 |
| Mannheim | 13 | 218,400 |
| Stuttgart | 16 | 220,800 |
| Heidelberg | 20 | 224,000 |

The total is 876,400 junction/action samples per update, within 0.2% of the previous 875,000-sample
budget. PPO packs minibatches by actual junction/action samples with a target of 16,384.

## Local occupancy and warm-up check

At fixed 7% target occupancy, seed `9301`, demand `1.0`, and a 15-second native warm-up, every city
generated every requested initial vehicle:

| city | generated/requested |
| --- | ---: |
| Karlsruhe | 878 / 878 |
| Mannheim | 1,172 / 1,172 |
| Stuttgart | 1,151 / 1,151 |
| Heidelberg | 758 / 758 |
| Freiburg | 958 / 958 |

All five movement graphs loaded with their expected controller counts. No warm-up or evaluation
teleport was observed.

## Demand and saturation check

One max-pressure episode per city and demand used seed `9301`, 1,800 simulated seconds, the new
initialization and warm-up, and pinned libsumo:

| city | demand | throughput/h | completion | wait density | teleports |
| --- | ---: | ---: | ---: | ---: | ---: |
| Karlsruhe | 0.8 / 1.0 / 1.2 | 1,706 / 1,726 / 1,924 | 69.2% / 64.4% / 67.0% | 1.220 / 1.684 / 1.438 | 0 |
| Mannheim | 0.8 / 1.0 / 1.2 | 2,178 / 2,472 / 2,276 | 51.9% / 53.0% / 44.8% | 2.317 / 2.206 / 2.880 | 0 |
| Stuttgart | 0.8 / 1.0 / 1.2 | 2,812 / 2,722 / 2,910 | 46.5% / 39.8% / 37.1% | 2.212 / 3.327 / 3.091 | 0 |
| Heidelberg | 0.8 / 1.0 / 1.2 | 2,518 / 2,790 / 2,976 | 87.1% / 85.6% / 82.3% | 0.303 / 0.343 / 0.418 | 0 |
| Freiburg | 0.8 / 1.0 / 1.2 | 1,332 / 1,334 / 1,382 | 33.9% / 30.2% / 28.1% | 3.146 / 4.056 / 4.581 | 0 |

This confirms that `1.0` is a meaningful, difficult city demand. It also supports retaining the old
Mannheim and Stuttgart `1.05` training caps. Freiburg becomes heavily saturated over a 1,800-second
horizon and should remain visible validation rather than a checkpoint claim based on completed-trip
waiting alone.

A same-seed `1.0`, 1,200-second check was also completed for direct comparison with the previous
report. Under the new initialization and timing, max-pressure completion was 56.5% Karlsruhe, 58.8%
Mannheim, 44.9% Stuttgart, 66.2% Heidelberg, and 50.8% Freiburg, with zero teleports. Periodic gate
evaluation therefore remains at 1,200 seconds for comparability and cost; the 1,800-second condition
is retained as a final stress check.

Raw local outputs are under:

- `reports/city_current_method_preflight_max_pressure_seed9301`;
- `reports/city_current_method_preflight_max_pressure_seed100_1200`.

## Thirty-iteration gate

Launch from random weights with `--scratch-num-hops 2`. Stop on non-finite training, worker failure,
evaluation teleports, or collapse below fixed-time and random control. At iteration 30:

- throughput and completion should improve from iteration 0 without widespread wait-density
  regression;
- at least three of four rollout cities should be within 10% throughput and five completion
  percentage points of the better of max-pressure and queue;
- competitive cities should keep wait density within 1.25 times that baseline;
- both sampled and greedy policies should remain viable;
- Freiburg should improve from initialization, but it remains visible validation.

Extend only when iterations 20-30 still improve materially. If the gate passes, repeat the frozen
design with independent training seeds and use fresh evaluation seeds plus a city excluded from
rollouts, monitoring, and checkpoint decisions.
