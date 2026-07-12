# City First Pass Remote Runbook

> **Operational reference.** This runbook records remote execution and recovery
> commands. Some examples use earlier worker configurations. It is not the
> scientific results report; use the
> [iteration-85 scratch PPO report](results/city_first_pass_throughput_scratch_32_worker.md)
> for the current result and claims.

This runbook is for the first rented-node city training pass. It assumes a
Linux machine with SUMO installed, 32 CPU cores, one CUDA GPU, at least 32 GB
RAM, and the generated city configs already present under `configs/`.

Use `configs/training/city_first_pass_16_worker.yaml` for this run. It schedules
four rollout workers per training city, sixteen rollout workers total, and keeps
Freiburg as held-out evaluation only.

Deterministic baseline evaluations are cached under `.cache/evaluation` by
default. Re-running the same max-pressure or queue evaluation parameters reuses
the cached metrics; learned-policy evaluations always run again.

## 1. Machine Sanity Check

```bash
python -m pytest -q
ruff check scripts src tests
sumo --version
nvidia-smi
```

## 2. Baseline Evaluation Smoke

```bash
python scripts/eval_multi_city.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --policies max-pressure queue \
  --seeds 100 \
  --demand-scales 1.0 \
  --steps 300 \
  --output-dir reports/city_first_pass_16_worker_baseline_smoke
```

## 3. Multi-City IL Collection

```bash
python scripts/collect_multi_city_il.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --output-dir data/il/city_first_pass_16_worker \
  --workers 16
```

The expected combined dataset is:

```text
data/il/city_first_pass_16_worker/combined.jsonl
```

The config keeps `samples_per_city: 9600`, `samples_per_simulation: 80`, and the
collector default `sample_stride: 3`. Each collection simulation therefore keeps
80 samples, advances through 240 raw decisions, and stops after about 2400 SUMO
seconds with the 10-second decision interval. That requires 120 simulations per
training city, or 480 collection jobs total across the four training cities.

## 4. IL Training

Disable automatic final multi-city evaluation during the training command. Run
evaluation explicitly afterward so failures are easier to attribute.

```bash
python scripts/train_il.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --data data/il/city_first_pass_16_worker/combined.jsonl \
  --device cuda \
  --ckpt-dir checkpoints/il/city_first_pass_16_worker \
  --log-dir runs/il/city_first_pass_16_worker \
  --eval-every-epochs 0
```

## 5. IL Policy Evaluation

```bash
python scripts/eval_multi_city.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --checkpoint checkpoints/il/city_first_pass_16_worker/movement_policy_best.pt \
  --policies learned max-pressure queue \
  --seeds 100 101 102 \
  --demand-scales 0.8 1.0 1.2 \
  --steps 1800 \
  --device cuda \
  --output-dir reports/city_first_pass_16_worker_il_eval \
  --log-dir runs/eval/city_first_pass_16_worker_il_eval
```

Proceed to PPO if the learned policy is at least directionally close to
max-pressure on completion rate, wait density, and throughput, and does not
show city-specific collapse.

## 6. PPO Pilot: 100 Iterations

```bash
python scripts/train_rl.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --il-checkpoint checkpoints/il/city_first_pass_16_worker/movement_policy_best.pt \
  --iterations 100 \
  --num-workers 16 \
  --device cuda \
  --ckpt-dir checkpoints/rl/city_first_pass_16_worker \
  --log-dir runs/rl/city_first_pass_16_worker
```

Watch TensorBoard during the pilot:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Primary checks:

```text
rollout/<city>/completion_rate
rollout/<city>/teleport_count
rollout/<city>/reward_mean
eval/aggregate/held_out/learned/average_wait_density_s_per_m
eval/aggregate/held_out/learned/throughput_per_hour
train/approx_kl
train/entropy
train/value_loss
```

Value loss can rise while explained variance improves if returns have larger
scale. Treat explained variance, reward, completion, teleports, KL, and
evaluation metrics as the main sanity signals.

## 7. Continue To 1000 Iterations

Resume only if the 100-iteration pilot is stable: no worker failure loop, no
held-out collapse, no persistent teleport spike, and no KL explosions.

```bash
python scripts/train_rl.py \
  --experiment-config configs/training/city_first_pass_16_worker.yaml \
  --resume-checkpoint checkpoints/rl/city_first_pass_16_worker/movement_ppo_latest.pt \
  --iterations 1000 \
  --num-workers 16 \
  --device cuda
```

`--iterations` is the final target iteration. Resuming from iteration 100 with
`--iterations 1000` runs iterations 101 through 1000.

## 8. Stop Conditions

Stop and inspect before continuing if any of these hold for more than one eval
cycle:

```text
held-out Freiburg completion rate drops sharply while train cities improve
teleport count becomes nonzero or spikes
approximate KL repeatedly hits the early-stop threshold
policy entropy collapses early
all cities show worsening wait density and throughput
workers spend time in repeated failed or skipped rollouts
```
