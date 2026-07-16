#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/GNN-Traffic-Signal-Control-2hop-repeat-01
unset PYTHONPATH
export PYTHONUNBUFFERED=1

exec > >(tee -a runs/rl/grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01/tensorboard_stdout.log) 2>&1
exec uv run tensorboard \
    --logdir runs/rl/grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01 \
    --host 127.0.0.1 \
    --port 6011
