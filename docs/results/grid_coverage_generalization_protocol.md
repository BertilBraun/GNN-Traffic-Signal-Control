# 4×4 Signal-Coverage Generalization Protocol

## Question

The earlier 6×6 ablation evaluated a policy trained only on fully signalized grids. This follow-up
asks the fairer question:

> If PPO is trained across several partially signalized layouts, does it generalize to new signal
> masks and to lower or higher signal coverage?

## Training design

- geometry: fixed 4×4 grid;
- controller-eligible junctions: 12;
- training coverage: 6 of 12 signals, or 50%;
- five distinct training masks;
- every eligible junction is signalized in either two or three training masks;
- one separate 50% mask is used for visible validation and checkpoint selection;
- 18 rollouts per training mask, giving
  `5 masks × 18 rollouts × 200 decisions × 6 actions = 108,000` junction/action samples per iteration;
- one scratch training seed, `6101`;
- 30 PPO iterations, with periodic evaluation every 10 iterations.

The control timing, demand, initialization, reward, entropy, and PPO settings remain those of the
grid-shape study: two hops, 5-second decisions, immediate 3-second yellow, one-decision minimum green,
6–8% initial occupancy, 15-second warm-up, training demand `0.6–0.8`, local progress/discharge/braking/
gridlock reward weights `1/10/10/0.02`, no global/flow/throughput/switch reward, entropy `0.001`, four
PPO epochs, and 200 decisions per rollout.

## Final evaluation

The final evaluation uses mask seeds not seen in training or checkpoint selection:

| coverage | signals | fresh masks |
| ---: | ---: | ---: |
| 25% | 3 of 12 | 5 |
| 50% | 6 of 12 | 5 |
| 75% | 9 of 12 | 5 |
| 100% | 12 of 12 | 1 |

Traffic seeds `8401–8403` and demand `0.7` are used for the compact final evaluation. Learned sampled,
learned greedy, max-pressure, queue, fixed-time, and uniform-random policies are evaluated. Confidence
intervals pair learned and baseline episodes by signal mask and traffic seed, then pool masks at the
same coverage level.

The generated `03_of_12_mask_03` diagnostic is preserved but excluded because it duplicated
`03_of_12_mask_01`; the distinct `mask_06` scenario replaces it.

## Interpretation

This is an exploratory one-training-seed run. It can distinguish a pure full-coverage distribution
shift from learned robustness to unsignalized gaps, but it cannot establish training-seed stability.
If it succeeds, independent training replicas are the appropriate next step before making a strong
coverage-generalization claim.
