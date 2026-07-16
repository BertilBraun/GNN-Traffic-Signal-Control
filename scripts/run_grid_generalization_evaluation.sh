#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-reports/grid_generalization_final/$RUN_LABEL}"
TENSORBOARD_DIRECTORY="${TENSORBOARD_DIRECTORY:-runs/rl/grid_generalization_final/$RUN_LABEL}"
WORKERS="${WORKERS:-32}"
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65535}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
SEEDS="${SEEDS:-7101 7102 7103 7104 7105 7106}"
DEMAND_SCALES="${DEMAND_SCALES:-0.6 0.7 0.8}"
POLICIES="${POLICIES:-}"
CITIES="${CITIES:-}"
BACKEND="${BACKEND:-libsumo}"

cd "$REPOSITORY_ROOT"
ulimit -n "$OPEN_FILE_LIMIT"
export SUMO_HOME
export PYTHONPATH="$REPOSITORY_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_DIRECTORY" "$TENSORBOARD_DIRECTORY"
exec > >(tee -a "$OUTPUT_DIRECTORY/evaluation_stdout.log") 2>&1

printf 'run_label=%s\n' "$RUN_LABEL"
printf 'experiment_config=%s\n' "$EXPERIMENT_CONFIG"
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'output_directory=%s\n' "$OUTPUT_DIRECTORY"
printf 'workers=%s\n' "$WORKERS"
printf 'seeds=%s\n' "$SEEDS"
printf 'demand_scales=%s\n' "$DEMAND_SCALES"
printf 'policies=%s\n' "${POLICIES:-experiment-config-default}"
printf 'cities=%s\n' "${CITIES:-all}"
printf 'backend=%s\n' "$BACKEND"
printf 'open_file_limit=%s\n' "$(ulimit -n)"

read -r -a seed_arguments <<< "$SEEDS"
read -r -a demand_scale_arguments <<< "$DEMAND_SCALES"
checkpoint_arguments=()
if [[ -n "$CHECKPOINT" ]]; then
    checkpoint_arguments=(--checkpoint "$CHECKPOINT")
fi
policy_arguments=()
if [[ -n "$POLICIES" ]]; then
    read -r -a policy_values <<< "$POLICIES"
    policy_arguments=(--policies "${policy_values[@]}")
fi
city_arguments=()
if [[ -n "$CITIES" ]]; then
    read -r -a city_values <<< "$CITIES"
    city_arguments=(--cities "${city_values[@]}")
fi

uv run python -c "import libsumo; print(f'libsumo={libsumo.__file__}')"
uv run python scripts/eval_multi_city.py \
    --experiment-config "$EXPERIMENT_CONFIG" \
    "${checkpoint_arguments[@]}" \
    "${policy_arguments[@]}" \
    "${city_arguments[@]}" \
    --backend "$BACKEND" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --log-dir "$TENSORBOARD_DIRECTORY" \
    --workers "$WORKERS" \
    --seeds "${seed_arguments[@]}" \
    --demand-scales "${demand_scale_arguments[@]}"
