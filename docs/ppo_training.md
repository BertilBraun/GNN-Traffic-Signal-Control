# Movement PPO Training

PPO starts from an imitation-learning movement checkpoint. The actor continues to output movement scores, which are aggregated into a local categorical distribution over each traffic light's valid phases.

The critic pools movement embeddings by traffic light and predicts one state value per traffic light. Training begins with value-only warmup: the actor and shared backbone are frozen, the final value layer is zero-initialized, and Monte Carlo returns are used as critic targets. Normal clipped PPO updates begin after the configured warmup iterations.

## Randomized Initial Traffic

Each rollout samples a target initial occupancy and generates valid vehicle routes starting from random network edges. Vehicles are inserted at simulation time zero with safe random positions. PPO collection begins immediately so the policy observes vehicles moving toward intersections rather than an artificially stabilized queued state.

Defaults:

```text
initial occupancy: 8% to 16%
```

Both can be configured:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --initial-occupancy-min 0.08 `
  --initial-occupancy-max 0.16
```

Set both occupancy bounds to zero to disable initial population generation.

## Visual Training

Use SUMO-GUI for rollout collection:

```powershell
python scripts\train_rl.py `
  --il-checkpoint checkpoints\il\<run>\movement_policy_best.pt `
  --gui
```

Periodic evaluation remains headless. The GUI closes and reopens for each rollout iteration because every iteration samples a new initial population and SUMO seed.
