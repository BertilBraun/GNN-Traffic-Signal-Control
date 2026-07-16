#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-grid_shape_generalization_mixed_2hop_gate_seed_5101}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/training/grid_shape_generalization_mixed_2hop_gate_30.yaml}"
FINAL_ITERATIONS="${FINAL_ITERATIONS:-30}"
TRAINING_SEED="${TRAINING_SEED:-5101}"
DEVICE="${DEVICE:-cuda}"
NUM_HOPS="${NUM_HOPS:-2}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65535}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

cd "$REPOSITORY_ROOT"
ulimit -n "$OPEN_FILE_LIMIT"
export SUMO_HOME
# The uv environment supplies pinned libsumo/sumolib packages. Adding SUMO's
# system tools here can mask them with an incompatible system Python package.
export PYTHONPATH="$REPOSITORY_ROOT"
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
printf 'num_hops=%s\n' "$NUM_HOPS"
printf 'open_file_limit=%s\n' "$(ulimit -n)"
printf 'pythonpath=%s\n' "$PYTHONPATH"

uv sync --group dev
uv run ruff format --check scripts src tests
uv run ruff check scripts src tests
uv run pytest -q
uv run python -c "import libsumo, torch; print(f'libsumo={libsumo.__file__}'); raise SystemExit(0 if torch.cuda.is_available() else 'CUDA unavailable')"

tensorboard_session="tb_${RUN_NAME}"
if ! tmux has-session -t "$tensorboard_session" 2>/dev/null; then
    tmux new-session -d -s "$tensorboard_session" \
        "cd '$REPOSITORY_ROOT' && exec uv run tensorboard \
        --logdir 'runs/rl/$RUN_NAME' --host 127.0.0.1 --port '$TENSORBOARD_PORT' \
        > 'runs/rl/$RUN_NAME/tensorboard.log' 2>&1"
fi

initialization=(
    --scratch-random
    --scratch-lane-feature-dim 29
    --scratch-movement-feature-dim 4
    --scratch-hidden-dim 64
    --scratch-num-hops "$NUM_HOPS"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    initialization=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

training_overrides=()
if [[ -n "${EVAL_EVERY_OVERRIDE:-}" ]]; then
    training_overrides+=(--eval-every "$EVAL_EVERY_OVERRIDE")
fi

uv run python scripts/train_rl.py \
    --experiment-config "$EXPERIMENT_CONFIG" \
    "${initialization[@]}" \
    "${training_overrides[@]}" \
    --iterations "$FINAL_ITERATIONS" \
    --device "$DEVICE" \
    --seed "$TRAINING_SEED" \
    --ckpt-dir "checkpoints/rl/$RUN_NAME" \
    --log-dir "runs/rl/$RUN_NAME"
