# Movement PPO Training

PPO starts from an imitation-learning movement checkpoint. The actor continues to output movement scores, which are aggregated into a local categorical distribution over each traffic light's valid phases.

The default PPO topology is the regenerated 3x3 grid. Use an IL checkpoint
trained with the same current feature schema. Old checkpoints contain numeric
lane-group IDs and zero-valued control/flow columns and are intentionally
incompatible.

The critic pools movement embeddings by traffic light and predicts one state value per traffic light. Training begins with value-only warmup: the actor and shared backbone are frozen, the final value layer is zero-initialized, and Monte Carlo returns are used as critic targets. Normal clipped PPO updates begin after the configured warmup iterations.

## Randomized Initial Traffic

Each rollout samples a target initial occupancy and generates valid vehicle routes starting from random network edges. Vehicles are inserted at simulation time zero with safe random positions. PPO collection begins immediately so the policy observes vehicles moving toward intersections rather than an artificially stabilized queued state.

Defaults:

```text
initial occupancy: 5% to 8%
training demand scale: 65%
evaluation demand scale: 100%
```

Both can be configured:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --initial-occupancy-min 0.05 `
  --initial-occupancy-max 0.08
```

Set both occupancy bounds to zero to disable initial population generation.

The training load is intentionally below the full evaluation load. A stochastic
policy near the start of PPO is weaker than max-pressure; running it at the
maximum sustainable demand can push SUMO into gridlock, make simulation much
slower, and produce teleport events before useful learning begins.

## Reward And Diagnostics

The reward is the reduction in local wait density plus a weighted global
wait-density reduction over one decision interval. It is clipped to `[-1, 1]`
by default. This matches the legacy delta-wait formulation and avoids critic
targets based on ever-growing absolute accumulated waiting time.
Teleport events are logged but are not penalized by default. They can include
route/lane-feasibility failures that are not attributable to one signal action;
penalizing every junction for them can overwhelm the wait-density reward.

Training logs include mean reward and return, critic explained variance,
rollout and update duration, teleport count, and all PPO losses.

Min-green-forced decisions are excluded from policy advantage normalization,
policy loss, and entropy. They still train the critic. The default entropy
coefficient is `0.01` and applies only when more than one phase is legal.

Periodic evaluation metrics are also written to TensorBoard. Completion rate,
vehicles remaining, and teleports must be considered alongside waiting and
travel time: those two averages include only completed vehicles and can improve
artificially when a poor policy leaves difficult trips unfinished.

Max-pressure and queue evaluations are deterministic for a fixed configuration
and seed, so training evaluates them once and reuses those records. Later
evaluation intervals run only the learned policy. The learned checkpoint with
the highest evaluation throughput is saved as `movement_policy_best.pt` and
`movement_ppo_best.pt`; the final iteration remains available through the
`latest` checkpoints.

## Visual Training

Use SUMO-GUI for rollout collection:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --gui
```

Periodic evaluation remains headless. The GUI closes and reopens for each rollout iteration because every iteration samples a new initial population and SUMO seed.
