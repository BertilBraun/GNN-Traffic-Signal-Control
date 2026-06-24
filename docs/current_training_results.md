# Current Training Results

Status as of June 19, 2026: movement-based PPO is now producing useful
policies on the generated grid networks.

## Setup

The current successful policy was trained from the ETA-to-queue-tail IL schema
and fine-tuned with PPO on the regenerated 3x3 dedicated-lane grid.

Key PPO settings:

```text
decision interval: 10 s
rollout length: 360 decisions per rollout
rollouts per update: 3
parallel workers: 3
value warmup: 20 iterations
learning rate: 2e-4
PPO clip: 0.1
target KL: 0.03
entropy coefficient: 0.01
speed-change reward weight: 0.00 (not used in this run)
SUMO gridlock teleporting: disabled with --time-to-teleport -1
```

The best checkpoint from run `2026-06-16_15-35-33` occurred near iteration
270. Iteration 300 remained strong but was slightly worse than the best
checkpoint, so downstream checks should use `movement_policy_best.pt`.

## 3x3 Fixed-Scenario Overfit

The fixed-seed 3x3 test demonstrated that PPO can improve a deterministic
scenario instead of merely preserving the imitation policy.

At iteration 270:

| policy | completion | throughput | wait | travel | time loss | stops / vehicle | nonstop pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max-pressure | 85.1% | 1404/h | 37.87 s | 118.12 s | 58.27 s | 1.74 | 24.9% |
| queue | 86.5% | 1428/h | 38.30 s | 119.55 s | 59.44 s | 1.77 | 24.1% |
| learned | 87.3% | 1440/h | 21.33 s | 97.41 s | 37.80 s | 1.08 | 55.3% |

## 4x4 Transfer Check

### Demand Scale 0.65

The same best checkpoint was evaluated on the generated 4x4 dedicated-lane grid
with unseen seeds `100`, `101`, and `102`, 600 simulation seconds, 10 s
decisions, demand scale `0.65`, and initial occupancy sampled from 5% to 8%.

| policy | completion | throughput | wait | travel | time loss | queue | wait density | stops / vehicle | nonstop pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max-pressure | 73.6% | 2776/h | 81.41 s | 193.47 s | 116.33 s | 0.44 | 0.0618 | 3.34 | 18.4% |
| queue | 74.3% | 2804/h | 76.18 s | 188.89 s | 111.57 s | 0.40 | 0.0507 | 3.34 | 19.2% |
| learned | 77.7% | 2930/h | 55.72 s | 162.11 s | 84.09 s | 0.31 | 0.0602 | 2.31 | 45.1% |

This is evidence of useful transfer from the 3x3 grid to a larger generated
grid. The learned policy improves completed-trip travel metrics and stop
behavior substantially. Detector-local wait density is mixed: learned beats
max-pressure but not queue on this short 3-seed evaluation.

### Demand Scale 0.85

The same checkpoint was also evaluated on the 4x4 grid with unseen seeds `100`
through `104`, 1200 simulation seconds, 10 s decisions, demand scale `0.85`,
and initial occupancy sampled from 5% to 8%.

| policy | completion | throughput | wait | travel | time loss | queue | wait density | stops / vehicle | nonstop pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max-pressure | 81.6% | 3318.0/h | 112.83 s | 245.76 s | 161.53 s | 0.68 | 0.1070 | 3.84 | 14.3% |
| queue | 83.3% | 3390.6/h | 95.22 s | 225.95 s | 141.63 s | 0.57 | 0.0791 | 3.78 | 16.1% |
| learned | 85.9% | 3493.8/h | 75.20 s | 197.87 s | 113.32 s | 0.46 | 0.0904 | 2.79 | 38.6% |

This higher-demand check is stronger evidence that the policy did not only
learn an under-saturated scenario. Learned remains clearly better on waiting
time, travel time, time loss, stops per vehicle, and nonstop pass rate. Queue
still has lower detector-local wait density, so wait density should be treated
as a secondary diagnostic rather than the primary policy-selection metric.

## Interpretation

The current result is strong enough to stop treating PPO as broken. The next
risk is generalization: generated-grid transfer is encouraging, but city
networks will introduce irregular topology, uneven approach lengths, different
phase structures, and demand calibration problems.

For the next generated-grid PPO run, train with a rollout demand distribution
instead of a fixed scale. A practical first range is `0.4` to `0.85`, with
evaluation held fixed at a representative high-demand value such as `0.85`.
