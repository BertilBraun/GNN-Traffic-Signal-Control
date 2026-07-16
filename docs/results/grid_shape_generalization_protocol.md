# Grid Shape and Size Generalization Protocol

This study tests whether the successful two-hop local-reward PPO controller transfers across synthetic
network size and aspect ratio without changing its control timing, reward, or PPO settings.

## Frozen scientific settings

The study preserves:

- two GNN message-passing hops;
- 5-second decisions;
- immediate yellow onset and 3-second yellow duration;
- minimum green of one decision;
- initial occupancy sampled uniformly from 6–8%;
- 15-second warm-up;
- training demand sampled uniformly from 0.6–0.8;
- local progress weight `1`, discharge weight `10`, braking-only weight `10`, and local gridlock weight
  `0.02`;
- zero global, flow, throughput, and direct switch reward;
- entropy coefficient `0.001`;
- four PPO epochs;
- 200 decisions per rollout;
- sampled and greedy learned evaluation;
- max-pressure, queue, fixed-time, and uniform-random baselines.

## Matched topology suite

The generator now accepts a short axis of two junctions. A 2×2 grid remains invalid because it has no
degree-three or degree-four controlled junction.

| scenario | role | controlled junctions |
| --- | --- | ---: |
| 5×2 | train | 6 |
| 3×3 | train | 5 |
| 4×4 | train | 12 |
| 5×3 | train | 11 |
| 5×5 | train | 21 |
| 6×6 | visible validation and checkpoint selection | 32 |
| 2×3, 3×2 | evaluation only | 2 |
| 2×5 | transposed evaluation of trained 5×2 | 6 |
| 3×5 | transposed evaluation of trained 5×3 | 11 |

The maximum training dimension is five junctions, and the largest evaluation topology is 6×6. The
6×6 topology contributes no PPO rollouts and is the only split used for checkpoint selection. Its
final result uses fresh seeds that were not used in periodic development evaluation.

## PPO sample balance

One PPO transition contains one action, reward, advantage, and value per controlled junction. Equal
rollout counts therefore overweight larger graphs.

The gate allocates rollouts inversely to controller count:

| training shape | controllers | rollouts | action samples per iteration |
| --- | ---: | ---: | ---: |
| 5×2 | 6 | 18 | 21,600 |
| 3×3 | 5 | 21 | 21,000 |
| 4×4 | 12 | 9 | 21,600 |
| 5×3 | 11 | 10 | 22,000 |
| 5×5 | 21 | 5 | 21,000 |

PPO minibatches are also packed by actual junction/action samples (`16,384` target samples per batch)
instead of a fixed number of variable-size graph transitions. TensorBoard records realized action
samples and non-forced policy samples per training shape.

## Experimental gates

The first mixed-shape run trains for 30 iterations from random weights. Periodic development
evaluation uses seeds `4101–4103` at demand `0.7` on every listed topology. The learned checkpoint
score is computed only from the 6×6 held-out split.

The run extends to 60 iterations only if:

1. learned sampled or greedy control improves within at least three training shapes;
2. 6×6 transfer improves relative to iteration 0 without teleport or completion failure;
3. the transposed 2×5 and 3×5 cases do not show an orientation-specific failure;
4. PPO diagnostics show no persistent KL stopping, entropy collapse, or value divergence.

If the gate succeeds, the complete train-shape × evaluation-shape matrix will compare single-shape
3×3, 4×4, and 5×5 training, a rectangular 5×2+5×3 mixture, and the full mixed-small-grid design under
an approximately equal total junction/action-sample budget. The final mixed design will use at least
three independent training seeds.

The final size/shape matrix uses fresh evaluation seeds `7101–7106` at demand scales `0.6`, `0.7`, and
`0.8`. These seeds are not used for periodic checkpoint selection. Matrix comparisons use the common
training seed `5101`; the full mixed design is additionally repeated with training seeds `5102` and
`5103`.

## Signal coverage ablation

The separate 6×6 coverage suite keeps node coordinates, roads, lane counts, routes, base demand, and
evaluation demand fixed while retaining 32, 24, 16, or 8 signals. Signal sets are nested and spatially
distributed. Removing a signal does not reduce incoming lane capacity.

The coverage study runs only after the size/shape gate. It reports the same traffic outcomes against
signal coverage `88.9%`, `66.7%`, `44.4%`, and `22.2%`. It uses fresh evaluation seeds `8101–8106`
at fixed demand scale `0.7`.

## Validation and outputs

`scripts/validate_grid_generalization_suite.py` produces:

- route reachability and boundary-source coverage;
- controller count and signal coverage;
- lane storage and base/scaled demand;
- requested, generated, and realized initial occupancy;
- post-warm-up population and warm-up teleports;
- max-pressure saturation probes at demand `0.6`, `0.7`, and `0.8`;
- topology plots for every matched and coverage scenario.

Final reporting includes throughput, completion, completed-trip waiting time, switches per junction
per minute, wait density, teleports, and paired confidence intervals.
