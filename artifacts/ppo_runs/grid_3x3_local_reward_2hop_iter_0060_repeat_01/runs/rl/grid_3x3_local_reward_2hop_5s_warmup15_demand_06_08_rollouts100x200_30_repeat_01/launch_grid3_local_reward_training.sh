#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/GNN-Traffic-Signal-Control-2hop-repeat-01
ulimit -n 65536
export SUMO_HOME=/usr/share/sumo
unset PYTHONPATH
export PYTHONUNBUFFERED=1

exec > >(tee -a runs/rl/grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01/train_stdout.log) 2>&1
exec uv run python scripts/train_rl.py \
    --experiment-config configs/training/grid_3x3_local_reward_2hop_30.yaml \
    --scratch-random \
    --scratch-num-hops 2 \
    --iterations 30 \
    --device cuda \
    --ckpt-dir checkpoints/rl/grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01 \
    --log-dir runs/rl/grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01
