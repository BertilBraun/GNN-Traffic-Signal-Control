# Remote/Linux Training Operations

This is operational guidance for reproducing or extending the multi-city experiment. Scientific claims and checkpoint metrics belong in the [iteration-85 result report](results/city_first_pass_throughput_scratch_32_worker.md).

## Launch

The maintained launcher is [`train.sh`](../train.sh):

```bash
./train.sh
```

Its default experiment is `configs/training/city_first_pass_throughput_scratch_32_worker.yaml`, with random scratch initialization, CUDA, and a final target of iteration 85. It installs dependencies on apt-based hosts, rebuilds the saved city recipes, validates the repository, starts TensorBoard, and writes checkpoints and logs under run-specific directories. The validation and training console is appended to `runs/rl/<run>/train_stdout.log` automatically.

Common overrides:

```bash
SKIP_BUILD=1 ./train.sh
START_TENSORBOARD=0 ./train.sh
RUN_NAME=city_first_pass_repeat FINAL_ITERATIONS=85 ./train.sh
IL_CHECKPOINT=checkpoints/il/city_first_pass/movement_policy_best.pt ./train.sh
RESUME_CHECKPOINT=checkpoints/rl/city_first_pass_repeat/movement_ppo_latest.pt FINAL_ITERATIONS=200 ./train.sh
```

The experiment configuration supplies worker counts, rollout jobs and horizon, reward weights, PPO epochs, evaluation cadence, seeds, and learned-action sampling mode. Avoid duplicating those values as launcher overrides.

## Machine checks

Before a long run, confirm:

```bash
sumo --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run ruff check scripts src tests
uv run pytest -q
```

The 32-worker configuration assumes enough CPU and memory for 32 persistent libsumo rollout workers. CUDA accelerates model updates; SUMO rollout remains CPU-heavy.

## Monitoring and stop conditions

The launcher starts:

```bash
uv run tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Watch per-city reward, throughput, completion, wait density, and teleports together with explained variance, entropy, approximate KL, policy loss, and value loss. Stop and inspect when worker failures repeat, teleports spike, KL repeatedly triggers early stopping, entropy collapses, or all cities regress across multiple evaluations.

For rented instances, run training in a persistent terminal or service. The launcher already records the complete console stream, so no additional `tee` pipeline is required.

```bash
./train.sh
```

## Resume and recovery

`movement_ppo_latest.pt` contains model, critic, optimizer, random-number-generator state, normalizers, architecture metadata, and completed iteration. Resume it with `RESUME_CHECKPOINT`; `FINAL_ITERATIONS` is the final target, not an additional-iteration count.

Before terminating an ephemeral machine, retrieve at least:

- the selected policy-only checkpoint;
- the selected full PPO checkpoint;
- the selected evaluation CSV and JSON;
- the TensorBoard event data through the selected iteration;
- `train_stdout.log` and run metadata.

Verify local file sizes and hashes before destroying the remote instance.
