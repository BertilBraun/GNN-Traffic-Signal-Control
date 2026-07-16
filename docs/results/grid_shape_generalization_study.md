# Controlled Grid-Shape Generalization Study

## Executive conclusion

A two-hop local-reward PPO controller trained only on grids whose longest axis is at most five
junctions transfers zero-shot to the larger 6×6 grid. Across three independent mixed-grid training
seeds and six fresh evaluation seeds, both sampled and greedy learned control improve 6×6 throughput,
completion, and completed-trip waiting time relative to max-pressure at demand scales `0.6`, `0.7`,
and `0.8`. No learned-policy evaluation teleported a vehicle.

The result is not an unconditional dominance claim:

- learned greedy control produces higher wait density than max-pressure at demand `0.7` and `0.8`;
- the single-shape 5×5 policy is slightly better than the mixed-grid policy on 6×6 and on average
  across the evaluated geometries;
- performance does not degrade gracefully when signal coverage is reduced. At 75% coverage,
  throughput is approximately tied with max-pressure but wait density rises sharply. At 50% and 25%
  coverage, learned throughput and completion collapse relative to max-pressure;
- the 6×6 grid was visible during periodic development evaluation and checkpoint selection. Its final
  episodes use fresh seeds, but it is a held-out rollout topology rather than a completely hidden
  model-selection topology.

The supported scientific conclusion is therefore:

> The local two-hop policy has genuine synthetic size and aspect-ratio transfer under full signal
> coverage, but mixed-shape training is not required for that transfer and the learned controller is
> not robust to substantial signal-coverage mismatch.

## Repository and provenance

- starting commit: `42495c2eebdf43a1cb313a223488ca02049dd03d`
- study branch: `codex/grid-size-generalization-study`
- remote experiment repository:
  `/workspace/GNN-Traffic-Signal-Control-2hop-repeat-01`
- local preserved result root:
  `C:\Projects\GNN-Traffic-Light-Optimization-Results`

All pre-existing runs, grids, checkpoints, logs, and the unrelated untracked
`docs/assets/training-eval-visualization.png` were preserved.

## Frozen scientific protocol

The study retained the paper settings without silently retuning them:

- two GNN message-passing hops;
- 5-second decisions;
- yellow begins immediately and lasts 3 seconds;
- minimum green of one decision;
- initial occupancy sampled from 6–8%;
- 15-second warm-up;
- training demand sampled from `0.6–0.8`;
- local reward weights: progress `1`, discharge `10`, braking-only `10`, local gridlock `0.02`;
- no global, throughput, flow, or direct switching reward;
- entropy coefficient `0.001`;
- four PPO epochs;
- 200 decisions per rollout;
- sampled and greedy learned evaluation;
- max-pressure, queue, fixed-time, and uniform-random baselines.

Training, periodic development evaluation, and reported final evaluation use the pinned `libsumo`
backend. A controlled check found that TraCI and libsumo can produce different numerical results for
an otherwise identical seeded episode, so the two backends were not mixed in reported comparisons.

## Grid suite

The current generator was extended to rectangular grids with a short axis of two. A 2×2 case remains
invalid because it contains no degree-three or degree-four controller-eligible junction.

| scenario | role | controller-eligible junctions |
| --- | --- | ---: |
| 5×2 | mixed and rectangular training | 6 |
| 3×3 | mixed and single-shape training | 5 |
| 4×4 | mixed and single-shape training | 12 |
| 5×3 | mixed and rectangular training | 11 |
| 5×5 | mixed and single-shape training | 21 |
| 6×6 | visible held-out validation and checkpoint selection | 32 |
| 2×3, 3×2 | evaluation only | 2 |
| 2×5 | transposed evaluation of trained 5×2 | 6 |
| 3×5 | transposed evaluation of trained 5×3 | 11 |

The final matrix contains ten geometries. The maximum training dimension is five and the maximum
evaluation dimension is six. The originally proposed 7×7 final test was removed after the requested
scope reduction.

### Static generator validation

For every geometry, the validator confirmed:

- every generated route is reachable;
- all expected boundary source and destination flows exist;
- edge, lane, and lane-length totals scale consistently;
- base demand is paired exactly between transposed rectangles;
- controller counts match degree-three and degree-four eligibility;
- topology images show the intended routing and signal placement.

Examples of matched quantities:

| paired or square geometry | routes | lane length (m) | base demand (veh/h) | controllers |
| --- | ---: | ---: | ---: | ---: |
| 2×3 and 3×2 | 10 | 8,601.6 | 2,221.2 | 2 |
| 2×5 and 5×2 | 14 | 14,336.0 | 2,746.8 | 6 |
| 3×5 and 5×3 | 16 | 23,347.2 | 3,824.64 | 11 |
| 4×4 | 16 | 25,420.8 | 3,997.44 | 12 |
| 5×5 | 20 | 41,484.8 | 5,313.6 | 21 |
| 6×6 | 24 | 61,696.0 | 6,523.2 | 32 |

The 5×3 training-demand visualization was also inspected interactively. The first visual run was
judged cleaner than the repeated run, but neither showed a demand-scaling defect.

### Dynamic generator validation

`scripts/validate_grid_generalization_suite.py` probes each geometry at a fixed 7% occupancy and
15-second warm-up, then runs max-pressure saturation episodes at demand `0.6`, `0.7`, and `0.8`.
It records requested/generated initial vehicles, realized occupancy, post-warm-up population,
warm-up teleports, throughput, completion, waiting time, switching, wait density, and evaluation
teleports. The resulting CSV/JSON and topology images are preserved under the report artifacts
listed below.

This validator uses TraCI as an independent diagnostic harness. Its traffic outcomes are used only
to detect routing, occupancy, warm-up, saturation, or teleport failures; all comparative policy
numbers reported below come from the pinned-libsumo evaluation matrix.

The completed dynamic audit contains 42 rows: 14 scenarios × three demand levels. It found:

- every route flow reachable in every scenario;
- requested and generated initial vehicle counts identical;
- realized initial occupancy between 6.920% and 7.002%;
- positive, stable post-warm-up populations in every scenario;
- zero warm-up and evaluation teleports;
- the expected saturation response as demand rises. For example, diagnostic max-pressure completion
  on 6×6 falls from 84.2% at demand `0.6` to 43.3% at `0.8`, while wait density rises from `0.054`
  to `2.752` seconds/meter;
- identical diagnostic outcomes for the matched 6×6 grid and corrected 32/32 coverage grid,
  confirming that the coverage suite preserves the full-coverage geometry and demand.

## PPO sample balancing

Equal rollout counts are not equal training weights because each graph transition contributes one
action, reward, advantage, and value per controlled junction. The mixed-grid schedule therefore
allocates rollouts inversely to controller count:

| training shape | controllers | rollouts per iteration | action samples per iteration |
| --- | ---: | ---: | ---: |
| 5×2 | 6 | 18 | 21,600 |
| 3×3 | 5 | 21 | 21,000 |
| 4×4 | 12 | 9 | 21,600 |
| 5×3 | 11 | 10 | 22,000 |
| 5×5 | 21 | 5 | 21,000 |

PPO minibatches are packed by actual junction/action samples with a target of `16,384` samples per
batch. TensorBoard records realized total and non-forced policy samples by training geometry. The
single-shape and rectangular comparison designs use approximately matched total action-sample
budgets rather than equal graph-rollout counts.

## Training runs and gate

Seven final training runs were used:

| design | training seed | selected checkpoint |
| --- | ---: | ---: |
| 3×3 only | 5101 | iteration 60 |
| 4×4 only | 5101 | iteration 60 |
| 5×5 only | 5101 | iteration 60 |
| rectangles, 5×2 + 5×3 | 5101 | iteration 60 |
| mixed five-shape design | 5101 | iteration 60 |
| mixed five-shape design | 5102 | iteration 60 |
| mixed five-shape design | 5103 | iteration 60 |

Development evaluation used seeds `4101–4103` at demand `0.7`. The 30-iteration gate passed:

- within-training-shape performance improved strongly;
- all designs transferred to 6×6, although sampled 3×3-only control learned more slowly;
- transposed rectangles did not show an orientation failure;
- no development evaluation teleported a vehicle;
- PPO logs showed no persistent KL stopping, entropy collapse, or value divergence.

For the three mixed training replicas, the mean development trajectory on 6×6 was:

| iteration | sampled throughput | sampled completion | greedy throughput | greedy completion |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,814 | 33.8% | 43 | 1.6% |
| 5 | 2,163 | 39.0% | 3,987 | 71.8% |
| 10 | 2,833 | 49.8% | 4,814 | 82.6% |
| 15 | 3,635 | 63.4% | 4,912 | 84.2% |
| 30 | 4,698 | 80.6% | 4,947 | 84.8% |
| 60 | 4,969 | 85.2% | 4,980 | 85.4% |
| max-pressure reference | 4,458 | 76.4% | 4,458 | 76.4% |

Iteration 60 was selected because all designs retained or improved 6×6 transfer without instability.
Immutable `movement_policy_iter_0030.pt` and `movement_policy_iter_0060.pt` files were preserved.
For all three mixed runs, the selected best policy is state-identical to the explicit iteration-60
evaluation policy.

## Final evaluation design

The final matrix uses six fresh evaluation seeds, `7101–7106`, at demand scales `0.6`, `0.7`, and
`0.8`. These seeds were not used for checkpoint selection.

- common baselines: 10 geometries × 4 policies × 6 seeds × 3 demands = 720 seed rows;
- each learned checkpoint: 10 geometries × 2 action modes × 6 seeds × 3 demands = 360 seed rows;
- seven learned checkpoints were evaluated;
- all learned rows have zero teleports;
- one baseline row teleported one vehicle: uniform-random on 5×3, seed `7104`, demand `0.7`;
- max-pressure and queue are exactly equal on these symmetric synthetic grids.

Common baselines were evaluated once and reused analytically for each checkpoint. This avoids
rerunning deterministic episodes while retaining seed-paired comparisons.

At demand `0.7` on 6×6, the complete baseline set is:

| baseline | throughput | completion | waiting (s) | wait density (s/m) | switches | teleports |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max-pressure | 4,429 | 78.2% | 156.8 | 0.119 | 4.97 | 0 |
| queue | 4,429 | 78.2% | 156.8 | 0.119 | 4.97 | 0 |
| fixed-time | 2,584 | 46.7% | 449.1 | 0.533 | 5.97 | 0 |
| uniform-random | 1,934 | 36.8% | 478.1 | 0.495 | 10.43 | 0 |

## Held-out 6×6 result

The following values aggregate three mixed training seeds and six fresh evaluation seeds per demand.
Throughput is vehicles/hour, waiting is completed-trip waiting time, and switching is per junction
per minute.

| demand | policy | throughput | completion | waiting (s) | wait density (s/m) | switches | teleports |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | learned sampled | 4,344 | 87.2% | 101.3 | 0.077 | 5.76 | 0 |
| 0.6 | learned greedy | 4,375 | 87.8% | 92.9 | 0.095 | 3.24 | 0 |
| 0.6 | max-pressure | 4,168 | 83.6% | 122.7 | 0.082 | 4.74 | 0 |
| 0.7 | learned sampled | 4,795 | 84.8% | 120.5 | 0.114 | 5.47 | 0 |
| 0.7 | learned greedy | 4,875 | 86.1% | 111.3 | 0.131 | 3.21 | 0 |
| 0.7 | max-pressure | 4,429 | 78.2% | 156.8 | 0.119 | 4.97 | 0 |
| 0.8 | learned sampled | 5,174 | 82.2% | 144.0 | 0.175 | 5.12 | 0 |
| 0.8 | learned greedy | 5,267 | 83.5% | 130.1 | 0.193 | 3.09 | 0 |
| 0.8 | max-pressure | 4,521 | 71.5% | 202.7 | 0.131 | 5.07 | 0 |

### Paired 95% confidence intervals versus max-pressure

Each row contains 18 pairs: three independent training replicas × six fresh evaluation seeds.
Intervals are mean paired difference ± 95% half-width.

| demand | action mode | throughput | completion | waiting (s) | wait density (s/m) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.6 | sampled | +176.1 ± 44.8 | +3.54 ± 0.89 pp | −21.4 ± 2.6 | −0.005 ± 0.031 |
| 0.6 | greedy | +206.8 ± 39.6 | +4.15 ± 0.79 pp | −29.8 ± 2.9 | +0.014 ± 0.032 |
| 0.7 | sampled | +366.3 ± 43.2 | +6.55 ± 0.80 pp | −36.3 ± 4.0 | −0.004 ± 0.060 |
| 0.7 | greedy | +446.0 ± 43.5 | +7.90 ± 0.75 pp | −45.5 ± 4.5 | +0.012 ± 0.060 |
| 0.8 | sampled | +653.3 ± 88.4 | +10.64 ± 1.38 pp | −58.6 ± 6.9 | +0.044 ± 0.016 |
| 0.8 | greedy | +746.3 ± 67.6 | +11.96 ± 1.06 pp | −72.6 ± 7.8 | +0.061 ± 0.022 |

Throughput, completion, and completed-trip waiting gains are clear at all three demands. Wait-density
differences are uncertain at `0.6` and `0.7`, but significantly worse for both learned action modes
at `0.8`. Completed-trip waiting must therefore not be interpreted alone: it excludes vehicles that
remain trapped at the end of an episode, whereas wait density exposes accumulated network-wide delay.

The confidence intervals pair on `(training replica, evaluation seed)`. They describe the realized
three-replica experiment, but with only three independent training seeds they should not be treated
as a high-powered population estimate over all possible PPO initializations.

## Training-design comparison

At demand `0.7`, greedy control on held-out 6×6 gives:

| training design | throughput | completion | waiting (s) | wait density (s/m) | switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 4,783 | 84.5% | 124.6 | 0.280 | 3.13 |
| 4×4 | 4,852 | 85.7% | 114.3 | 0.133 | 3.31 |
| 5×5 | **4,915** | **86.8%** | **98.2** | **0.091** | 3.59 |
| rectangles | 4,860 | 85.9% | 110.5 | 0.152 | 3.18 |
| mixed | 4,863 | 85.9% | 115.2 | 0.142 | 3.05 |
| max-pressure | 4,429 | 78.2% | 156.8 | 0.119 | 4.97 |

The 5×5-only policy is the strongest 6×6 policy in this comparison. It also has the largest mean
throughput gain across all ten geometries:

| demand | policy | 3×3 | 4×4 | 5×5 | rectangles | mixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.7 | sampled | +56 | +78 | **+101** | +89 | +84 |
| 0.7 | greedy | +72 | +100 | **+119** | +105 | +103 |
| 0.8 | sampled | +148 | +194 | **+237** | +195 | +201 |
| 0.8 | greedy | +182 | +215 | **+253** | +235 | +233 |

These are mean throughput differences from max-pressure across the ten evaluation geometries.
Training on diverse shapes is therefore not required for zero-shot size transfer under this local
policy architecture. Exposure to the largest permitted training graph appears more useful than
shape diversity under an equal action-sample budget.

### Aspect-ratio and orientation transfer

No meaningful transposition failure appears at demand `0.7`. For mixed sampled control:

- 5×2: 1,993 vehicles/hour, 91.9% completion, wait density 0.108;
- 2×5: 2,005 vehicles/hour, 92.5% completion, wait density 0.123;
- 5×3: 2,796 vehicles/hour, 91.2% completion, wait density 0.100;
- 3×5: 2,791 vehicles/hour, 91.0% completion, wait density 0.103.

The small paired differences support aspect-ratio transfer within the matched generator.

## Signal-coverage ablation

Coverage is defined over the 32 degree-three or degree-four controller-eligible junctions in the 6×6
grid, not all 36 lattice nodes. The four corners are never controller candidates. The corrected,
nested coverage suite retains 32, 24, 16, or 8 signals while preserving node coordinates, roads,
lanes, routes, base demand, and incoming lane capacity.

The ablation uses demand `0.7`, fresh evaluation seeds `8101–8106`, and all three mixed training
replicas. Values below aggregate 18 learned episodes per action mode and six max-pressure episodes.

| coverage | policy | throughput | completion | waiting (s) | wait density (s/m) | switches | teleports |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | learned sampled | 4,866 | 85.5% | 121.9 | 0.115 | 5.43 | 0 |
| 100% | learned greedy | 4,909 | 86.2% | 110.4 | 0.138 | 3.17 | 0 |
| 100% | max-pressure | 4,296 | 75.5% | 169.5 | 0.185 | 4.90 | 0 |
| 75% | learned sampled | 4,427 | 80.9% | 129.2 | 1.227 | 4.99 | 0 |
| 75% | learned greedy | 4,426 | 81.4% | 115.4 | 1.499 | 2.82 | 0 |
| 75% | max-pressure | 4,555 | 80.1% | 151.6 | 0.083 | 4.89 | 0 |
| 50% | learned sampled | 3,493 | 66.8% | 152.6 | 3.186 | 3.57 | 0 |
| 50% | learned greedy | 3,429 | 66.2% | 134.3 | 3.864 | 1.92 | 0 |
| 50% | max-pressure | 4,411 | 77.4% | 165.9 | 0.270 | 4.31 | 0 |
| 25% | learned sampled | 3,803 | 70.3% | 131.9 | 2.974 | 2.29 | 0 |
| 25% | learned greedy | 3,776 | 70.5% | 122.3 | 3.295 | 1.37 | 0 |
| 25% | max-pressure | 4,679 | 82.1% | 132.4 | 0.120 | 4.33 | 0 |

Paired 95% intervals versus max-pressure:

| coverage | action mode | throughput | completion | wait density |
| ---: | --- | ---: | ---: | ---: |
| 100% | sampled | +570.0 ± 134.8 | +10.02 ± 2.27 pp | −0.070 ± 0.111 |
| 100% | greedy | +612.6 ± 140.5 | +10.73 ± 2.38 pp | −0.047 ± 0.111 |
| 75% | sampled | −128.4 ± 120.2 | +0.79 ± 1.69 pp | +1.144 ± 0.497 |
| 75% | greedy | −129.2 ± 134.4 | +1.31 ± 2.06 pp | +1.415 ± 0.567 |
| 50% | sampled | −918.0 ± 425.5 | −10.63 ± 8.13 pp | +2.916 ± 1.135 |
| 50% | greedy | −982.3 ± 361.8 | −11.28 ± 6.98 pp | +3.593 ± 1.354 |
| 25% | sampled | −875.8 ± 223.5 | −11.85 ± 4.50 pp | +2.855 ± 0.910 |
| 25% | greedy | −902.6 ± 198.8 | −11.68 ± 4.00 pp | +3.175 ± 0.838 |

The 75% result is already unsafe to describe as graceful degradation: aggregate throughput is near
max-pressure, but the network carries roughly one additional second of accumulated wait per meter.
At 50% and 25%, both throughput and completion are substantially worse.

Max-pressure throughput is non-monotonic in coverage and is highest at 25%. Removing a signal changes
the junction to SUMO priority and safety semantics, which can remove signal delay. Consequently this
ablation isolates signal availability while preserving geometry and demand, but it does not preserve
identical junction service semantics.

Training-seed sensitivity also rises sharply below full coverage. Seed 5101 remains relatively
strong, while seed 5103 fails badly at 50% and 25%. This supports the conclusion that the full-coverage
policy has not learned a stable missing-controller strategy.

## Protocol corrections and excluded diagnostics

Several audit findings were corrected before the authoritative final analyses:

1. Standalone evaluation constructed learned-policy configuration with deterministic action selection
   by default, ignoring the sampled action mode from the experiment configuration. This was fixed and
   covered by tests.
2. The first coverage configuration omitted sampled action selection, causing `learned` and
   `learned-greedy` to be identical. Those run directories were preserved, but only the
   `*_sampled_fixed` reruns are used above.
3. Early coverage labels used all 36 lattice nodes as the denominator. Corrected directories use the
   32 controller-eligible nodes and are named `eligible_signals_*_of_32`.
4. TraCI and libsumo were found not to be numerically interchangeable. Final grid evaluations were
   explicitly pinned to libsumo.
5. Common deterministic baselines were factored out and analytically paired with each learned
   checkpoint, eliminating redundant simulations without changing comparisons.

No result directory or diagnostic run was deleted.

## Plots

The analysis produces:

- learning curves through iteration 60;
- train-design × evaluation-shape throughput-difference heatmaps for demands `0.6`, `0.7`, and `0.8`;
- paired confidence-interval CSVs;
- performance versus signal coverage for sampled learned, greedy learned, and max-pressure control.

Primary local plot locations:

- `C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_generalization_analysis\demand_07\learning_curves.png`
- `C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_generalization_analysis\demand_07\train_shape_by_evaluation_shape.png`
- `C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_signal_coverage_analysis\all_training_seeds_sampled_fixed\performance_vs_signal_coverage.png`

All three were visually inspected after generation. Legends, labels, color scales, and plotted policy
sets are visible and consistent with the underlying CSVs.

## Preserved artifacts

### Final matrix

`C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_generalization_final`

- `common_baselines_fresh7101_7106`
- `train_3x3_seed5101_iter0060`
- `train_4x4_seed5101_iter0060`
- `train_5x5_seed5101_iter0060`
- `train_rectangles_seed5101_iter0060`
- `train_mixed_seed5101_best0060`
- `train_mixed_seed5102_best0060`
- `train_mixed_seed5103_best0060`

### Analysis

`C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_generalization_analysis`

### Signal coverage

`C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_signal_coverage_final`

Authoritative learned coverage directories end in `_sampled_fixed`.

### Signal-coverage analysis

`C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_signal_coverage_analysis\all_training_seeds_sampled_fixed`

### Training checkpoints and development evaluations

`C:\Projects\GNN-Traffic-Light-Optimization-Results\comparison_training_runs`

The three mixed training replicas are preserved in separate result directories under the same local
result root. The complete corresponding artifacts also remain in the remote experiment repository.

### Grid validation

- corrected static CSV/JSON and topology titles:
  `C:\Projects\GNN-Traffic-Light-Optimization\reports\grid_generalization_validation_static_eligible_titles`
- local complete dynamic occupancy, warm-up, demand, saturation, and teleport audit:
  `C:\Projects\GNN-Traffic-Light-Optimization-Results\grid_generalization_validation_full_dynamic_parallel_9101`
- corresponding remote dynamic audit:
  `/workspace/GNN-Traffic-Signal-Control-2hop-repeat-01/reports/grid_generalization_validation_full_dynamic_parallel_9101`

## Limitations

- All networks are generated by one matched synthetic generator. The experiment tests scale and
  aspect ratio, not irregular real-city topology.
- The maximum zero-shot size increment is from a training maximum of 5×5 to 6×6.
- The 6×6 topology was visible for development evaluation and checkpoint selection.
- Only three independent PPO training seeds were run.
- Paired intervals combine training and evaluation replicas rather than fitting a hierarchical model.
- Completed-trip waiting is survivor-biased when completion is low; wait density and completion must
  be interpreted alongside it.
- Partial signal coverage changes local right-of-way semantics as well as which junctions are
  controllable.

## Repository validation

The final worktree passed:

- `uv run ruff format`;
- `uv run ruff check --fix`;
- `uv run pytest`: 276 tests passed.

## Reproducibility commands

The study launchers and exact experiment configurations are versioned in the repository. Core
analysis is reproduced with:

```powershell
uv run python scripts\analyze_grid_generalization_results.py `
  --matrix-summary ... `
  --matrix-baseline-summary ... `
  --paired-policy learned `
  --paired-policy learned-greedy `
  --output-directory ...
```

The full structural and dynamic validation is reproduced with:

```powershell
uv run python scripts\validate_grid_generalization_suite.py `
  --suite all `
  --simulation-seed 9101 `
  --simulation-steps 1800 `
  --workers 6 `
  --output-directory reports\grid_generalization_validation_full_dynamic_parallel_9101
```

Repository validation is completed with:

```powershell
uv run ruff format
uv run ruff check --fix
uv run pytest
```
