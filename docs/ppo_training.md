# Movement PPO Training

PPO starts from an imitation-learning movement checkpoint. The actor continues to output movement scores, which are aggregated into a local categorical distribution over each traffic light's valid phases.

The default PPO topology is the regenerated 3x3 grid. Use an IL checkpoint
trained with the same current feature schema. Old checkpoints contain numeric
lane-group IDs and zero-valued control/flow columns and are intentionally
incompatible.

The current schema includes ETA-to-queue-tail lane features. The policy can see
how many moving vehicles in the detector are likely to catch the stopped queue
within 5, 10, or 15 seconds, plus minimum/mean ETA. Any IL checkpoint created
before those features must be regenerated before PPO can load it.

The critic pools movement embeddings by traffic light and predicts one state value per traffic light. Training begins with value-only warmup: the actor and shared backbone are frozen, the final value layer is zero-initialized, and bootstrapped discounted returns are used as critic targets. Normal clipped PPO updates begin after the configured warmup iterations.

Rollouts are fixed-size data segments rather than terminal episodes. When a
rollout ends while SUMO is still running, the critic evaluates the resulting
next state and that value bootstraps both GAE and the warmup return targets.
Only a genuinely terminated simulation uses a zero final value.

Multiple independent rollout segments can be collected per PPO update. Each
worker computes returns and advantages for its own segment before the parent
process concatenates the ready-to-train buffers. A persistent process pool is
used when more than one worker is requested, so SUMO worker startup cost is not
paid again on every iteration.

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --rollouts-per-update 4 `
  --num-workers 4
```

`--steps-per-rollout` remains the number of decision steps per independent
environment, so total PPO data per update is approximately
`rollouts-per-update * steps-per-rollout` timestep graphs.

## Randomized Initial Traffic

Each rollout samples a target initial occupancy and generates valid vehicle routes starting from random network edges. Vehicles are inserted at simulation time zero with safe random positions. PPO collection begins immediately so the policy observes vehicles moving toward intersections rather than an artificially stabilized queued state.
SUMO advances one insertion step before the first action so those vehicles are
actually present in the initial observation.

Defaults:

```text
initial occupancy: 5% to 8%
training demand scale: 100%
evaluation demand scale: 100%
decision interval: 10 s
SUMO gridlock teleporting: disabled
```

Both can be configured:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --initial-occupancy-min 0.05 `
  --initial-occupancy-max 0.08
```

PPO can sample background demand per rollout. `--demand-scale` remains the
fixed-value shorthand; when min/max are supplied, every rollout samples a
deterministic scale from that range using its rollout seed.

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --demand-scale-min 0.4 `
  --demand-scale-max 0.85 `
  --eval-demand-scale 0.85
```

Passing only `--demand-scale 0.65` is equivalent to training with
`--demand-scale-min 0.65 --demand-scale-max 0.65`.

Set both occupancy bounds to zero to disable initial population generation.
Periodic evaluation uses the same deterministic occupancy range. For a fixed
evaluation seed, every policy receives the same generated initial vehicles and
background demand.

## Reward And Diagnostics

The reward is the negative interval-average speed-deficit density on incoming
lanes plus a weighted global speed-deficit density. Each vehicle contributes
according to `1 - mean_speed / speed_limit`: stopped vehicles contribute fully,
slow vehicles still contribute, and free-flow vehicles contribute zero. The
cost is normalized by incoming lane length and integrated across every
simulation second in the decision interval. This avoids the legacy endpoint
wait-delta reward's ability to improve when a stopped queue merely begins
moving and its consecutive waiting counters reset.

An auxiliary smoothness penalty is enabled with a small default weight of
`0.02`. It tracks each vehicle's absolute speed change between consecutive
simulation seconds while the vehicle is on incoming lanes, normalizes the
change by the lane speed limit, and applies the same lane-length density
normalization as the delay term. The intent is to mildly discourage stop-go
behavior without overpowering the main speed-deficit objective. Configure it
with `--speed-change-weight`; use `0.0` to disable it.

The reward is clipped to `[-1, 1]` by default. TensorBoard logs the local and
global delay-density components and the speed-change density separately.
Teleport events are logged but are not penalized by default. They can include
route/lane-feasibility failures that are not attributable to one signal action;
penalizing every junction for them can overwhelm the wait-density reward.
SUMO gridlock teleporting is disabled by default with `--time-to-teleport -1`.
This makes gridlock visible through degraded queue, wait, and completion
metrics rather than having SUMO silently remove stuck vehicles. Rollouts with
more than 999 teleports are excluded from optimization by default; this is
mostly a guard against route/lane feasibility failures when teleporting has
not been disabled. PPO update epochs stop when approximate KL exceeds `0.03`.

Training logs include mean reward and return, critic explained variance,
rollout and update duration, teleport count, and all PPO losses. Diagnostic
scalars also include reward range and clipping frequency, return/value/
advantage scale, normalized policy entropy, top-action probability, legal
policy-decision fraction, approximate KL divergence, PPO ratio clipping
frequency, and backbone/value-head gradient norms.

Min-green-forced decisions are excluded from policy advantage normalization,
policy loss, and entropy. They still train the critic. The default entropy
coefficient is `0.01` and applies only when more than one phase is legal.

Periodic evaluation metrics are also written to TensorBoard. Completion rate,
vehicles remaining, and teleports must be considered alongside waiting and
travel time: those two averages include only completed vehicles and can improve
artificially when a poor policy leaves difficult trips unfinished.

Max-pressure and queue evaluations are deterministic for a fixed configuration
and seed, so training evaluates them once and reuses those records. Later
evaluation intervals run only the learned policy.

Best-checkpoint selection uses completion-adjusted average time loss:

```text
score =
    average_time_loss / completion_rate
    + evaluation_steps * teleports / departed_vehicles
```

Lower is better. Dividing by completion rate prevents a policy from appearing
better by leaving difficult trips unfinished. The teleport term prevents SUMO
teleportation from acting like successful discharge. The score is logged as
`eval/<policy>/checkpoint_selection_score`.

The best learned checkpoint is saved as `movement_policy_best.pt` and
`movement_ppo_best.pt`; the final iteration remains available through the
`latest` checkpoints.

## Current Reference Result

The current reference checkpoint is
`checkpoints\rl\2026-06-16_15-35-33\movement_policy_best.pt`. It was trained
from the ETA-to-queue-tail IL schema on the generated 3x3 dedicated-lane grid
and reached its best fixed-seed validation near PPO iteration 270.

A preliminary 4x4 transfer evaluation at demand scale `0.65`, seeds
`100 101 102`, 600 simulation seconds, and 10 s decisions showed:

```text
metric                 max-pressure      queue      learned
completion rate              73.6%      74.3%        77.7%
throughput                  2776/h     2804/h       2930/h
average waiting time        81.41s     76.18s       55.72s
average travel time        193.47s    188.89s      162.11s
average time loss          116.33s    111.57s       84.09s
TLS stops / vehicle           3.34       3.34         2.31
nonstop TLS pass rate        18.4%      19.2%        45.1%
```

The result is encouraging but still preliminary. The next validation should use
more seeds, longer episodes, and a demand sweep to measure where the learned
policy degrades.

A harder 4x4 evaluation at demand scale `0.85`, seeds `100` through `104`, and
1200 simulation seconds still favored the learned policy:

```text
metric                 max-pressure      queue      learned
completion rate              81.6%      83.3%        85.9%
throughput                  3318/h     3391/h       3494/h
average waiting time       112.83s     95.22s       75.20s
average travel time        245.76s    225.95s      197.87s
average time loss          161.53s    141.63s      113.32s
TLS stops / vehicle           3.84       3.78         2.79
nonstop TLS pass rate        14.3%      16.1%        38.6%
```

For further generated-grid PPO training, sample rollout demand over roughly
`0.4..0.85` and keep evaluation fixed at a representative target demand.

## Resuming PPO

PPO checkpoints now contain:

- actor and critic parameters;
- Adam optimizer state and moments;
- completed iteration;
- best checkpoint-selection score;
- CPU and CUDA random-number-generator state;
- normalizers and IL architecture metadata.

Resume into the same run directory with:

```powershell
python scripts\train_rl.py `
  --resume-checkpoint checkpoints\rl\<run>\movement_ppo_latest.pt `
  --iterations 500
```

`--iterations` is the final target iteration. Resuming an iteration-300
checkpoint with `--iterations 500` runs iterations 301 through 500. When
`--ckpt-dir` and `--log-dir` are omitted, the directories are inferred from the
resumed checkpoint's run name.

Scenario and PPO hyperparameters must still be supplied consistently when they
differ from defaults. The saved Adam learning rate is restored from the
optimizer state.

## Visual Training

Use SUMO-GUI for rollout collection:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --gui
```

Periodic evaluation remains headless. The GUI closes and reopens for each rollout iteration because every iteration samples a new initial population and SUMO seed.

## Fixed-Scenario Overfit Test

Use one fixed rollout seed to test whether PPO can improve a single deterministic
3x3 scenario. The SUMO demand and generated initial population remain identical
across iterations. Policy actions are still sampled stochastically.

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --fixed-rollout-seed 100 `
  --eval-seeds 100 `
  --time-to-teleport -1
```

Improvement on this test demonstrates that the policy parameterization and PPO
update can fit at least one scenario. Failure indicates an optimization,
credit-assignment, or action-parameterization problem rather than insufficient
scenario diversity.
