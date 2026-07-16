#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-grid_shape_generalization_mixed_2hop_gate_seed_5101}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/training/grid_shape_generalization_mixed_2hop_gate_30.yaml}"
FINAL_ITERATIONS="${FINAL_ITERATIONS:-30}"
TRAINING_SEED="${TRAINING_SEED:-5101}"
DEVICE="${DEVICE:-cuda}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65535}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

cd "$REPOSITORY_ROOT"
ulimit -n "$OPEN_FILE_LIMIT"
export SUMO_HOME
export PYTHONPATH="$REPOSITORY_ROOT:$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "runs/rl/$RUN_NAME" "checkpoints/rl/$RUN_NAME"
exec > >(tee -a "runs/rl/$RUN_NAME/train_stdout.log") 2>&1

printf 'repository=%s\n' "$REPOSITORY_ROOT"
printf 'run_name=%s\n' "$RUN_NAME"
printf 'experiment_config=%s\n' "$EXPERIMENT_CONFIG"
printf 'training_seed=%s\n' "$TRAINING_SEED"
printf 'final_iterations=%s\n' "$FINAL_ITERATIONS"
printf 'device=%s\n' "$DEVICE"
printf 'open_file_limit=%s\n' "$(ulimit -n)"
printf 'pythonpath=%s\n' "$PYTHONPATH"

uv sync --group dev
uv run ruff format --check scripts src tests
uv run ruff check scripts src tests
uv run pytest -q
uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA unavailable')"

if ! pgrep -f "tensorboard --logdir runs --host 0.0.0.0 --port $TENSORBOARD_PORT" >/dev/null 2>&1; then
    nohup uv run tensorboard --logdir runs --host 0.0.0.0 --port "$TENSORBOARD_PORT" \
        > "runs/rl/$RUN_NAME/tensorboard.log" 2>&1 &
    printf '%s\n' "$!" > "runs/rl/$RUN_NAME/tensorboard.pid"
fi

initialization=(
    --scratch-random
    --scratch-lane-feature-dim 29
    --scratch-movement-feature-dim 4
    --scratch-hidden-dim 64
    --scratch-num-hops 2
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    initialization=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

uv run python scripts/train_rl.py \
    --experiment-config "$EXPERIMENT_CONFIG" \
    "${initialization[@]}" \
    --iterations "$FINAL_ITERATIONS" \
    --device "$DEVICE" \
    --seed "$TRAINING_SEED" \
    --ckpt-dir "checkpoints/rl/$RUN_NAME" \
    --log-dir "runs/rl/$RUN_NAME"
