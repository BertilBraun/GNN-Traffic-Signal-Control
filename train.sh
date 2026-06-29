#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/GNN-Traffic-Signal-Control.git}"
REPO_DIR="${REPO_DIR:-$HOME/GNN-Traffic-Signal-Control}"
BRANCH="${BRANCH:-master}"
RUN_NAME="${RUN_NAME:-city_first_pass_16_worker}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/training/city_first_pass_16_worker.yaml}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-16}"
IL_EPOCHS="${IL_EPOCHS:-20}"
IL_SAMPLES_PER_BATCH="${IL_SAMPLES_PER_BATCH:-128}"
PPO_PILOT_ITERATIONS="${PPO_PILOT_ITERATIONS:-100}"
PPO_TOTAL_ITERATIONS="${PPO_TOTAL_ITERATIONS:-1000}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_command() {
    log "$*"
    "$@"
}

sudo_command() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

bootstrap_repository() {
    if [[ -d "$SCRIPT_DIR/.git" ]]; then
        cd "$SCRIPT_DIR"
        if [[ "${SKIP_GIT_UPDATE:-0}" != "1" ]]; then
            run_command git fetch origin "$BRANCH"
            run_command git checkout "$BRANCH"
            run_command git pull --ff-only origin "$BRANCH"
        fi
        return
    fi
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        run_command git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        run_command git -C "$REPO_DIR" fetch origin "$BRANCH"
        run_command git -C "$REPO_DIR" checkout "$BRANCH"
        run_command git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
    fi
    exec "$REPO_DIR/train.sh" "$@"
}

install_system_dependencies() {
    if [[ "${INSTALL_SYSTEM_DEPS:-1}" != "1" ]]; then
        log "Skipping system dependency install because INSTALL_SYSTEM_DEPS=${INSTALL_SYSTEM_DEPS:-unset}"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; install git, curl, build-essential, SUMO, and SUMO tools manually."
        return
    fi
    run_command sudo_command apt-get update
    run_command sudo_command env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-venv \
        sumo \
        sumo-tools
}

install_uv_if_missing() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    log "uv not found; installing uv into the current user account."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        printf 'uv installation did not put uv on PATH. Add ~/.local/bin to PATH and rerun.\n' >&2
        exit 1
    fi
}

configure_sumo_environment() {
    if [[ -z "${SUMO_HOME:-}" ]]; then
        if [[ -d /usr/share/sumo ]]; then
            export SUMO_HOME=/usr/share/sumo
        else
            printf 'SUMO_HOME is not set and /usr/share/sumo does not exist.\n' >&2
            exit 1
        fi
    fi
    export PYTHONPATH="$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
    run_command sumo --version
}

sync_python_environment() {
    run_command uv sync --group dev
}

check_cuda() {
    run_command uv run python -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
    if [[ "$DEVICE" == cuda* ]]; then
        run_command uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA requested but torch.cuda.is_available() is false')"
    fi
}

build_city_networks() {
    if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
        log "Skipping city network rebuild because SKIP_BUILD=1"
        return
    fi
    local build_file
    for build_file in \
        configs/karlsruhe_oststadt/karlsruhe_oststadt.build.yaml \
        configs/stuttgart_mitte/stuttgart_mitte.build.yaml \
        configs/heidelberg_bergheim/heidelberg_bergheim.build.yaml \
        configs/freiburg_altstadt/freiburg_altstadt.build.yaml \
        configs/mannheim_innenstadt/mannheim_innenstadt.build.yaml
    do
        run_command uv run python scripts/network_workbench.py --build-file "$build_file" rebuild
    done
}

run_validation() {
    run_command uv run ruff check scripts src tests
    if [[ "${RUN_TESTS:-1}" == "1" ]]; then
        run_command uv run pytest -q
    else
        log "Skipping pytest because RUN_TESTS=$RUN_TESTS"
    fi
}

start_tensorboard() {
    if [[ "${START_TENSORBOARD:-1}" != "1" ]]; then
        log "Skipping TensorBoard because START_TENSORBOARD=$START_TENSORBOARD"
        return
    fi
    mkdir -p runs
    if pgrep -f "tensorboard --logdir runs --host 0.0.0.0 --port $TENSORBOARD_PORT" >/dev/null 2>&1; then
        log "TensorBoard already appears to be running on port $TENSORBOARD_PORT"
        return
    fi
    log "Starting TensorBoard on 0.0.0.0:$TENSORBOARD_PORT"
    nohup uv run tensorboard --logdir runs --host 0.0.0.0 --port "$TENSORBOARD_PORT" > runs/tensorboard.log 2>&1 &
    echo "$!" > runs/tensorboard.pid
}

run_baseline_smoke() {
    local output_dir="reports/${RUN_NAME}_baseline_smoke"
    if [[ -f "$output_dir/summary.json" && "${FORCE_BASELINE_SMOKE:-0}" != "1" ]]; then
        log "Skipping baseline smoke; $output_dir/summary.json already exists."
        return
    fi
    run_command uv run python scripts/eval_multi_city.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --policies max-pressure queue \
        --seeds 100 \
        --demand-scales 1.0 \
        --steps 300 \
        --output-dir "$output_dir"
}

collect_imitation_data() {
    local output_dir="data/il/$RUN_NAME"
    local combined_dataset="$output_dir/combined.jsonl"
    if [[ -f "$combined_dataset" && "${FORCE_IL_COLLECTION:-0}" != "1" ]]; then
        log "Skipping IL collection; $combined_dataset already exists."
        return
    fi
    run_command uv run python scripts/collect_multi_city_il.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --output-dir "$output_dir" \
        --workers "$WORKERS"
}

train_imitation_policy() {
    local combined_dataset="data/il/$RUN_NAME/combined.jsonl"
    local checkpoint_path="checkpoints/il/$RUN_NAME/movement_policy_best.pt"
    if [[ -f "$checkpoint_path" && "${FORCE_IL_TRAIN:-0}" != "1" ]]; then
        log "Skipping IL training; $checkpoint_path already exists."
        return
    fi
    run_command uv run python scripts/train_il.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --data "$combined_dataset" \
        --device "$DEVICE" \
        --epochs "$IL_EPOCHS" \
        --samples-per-batch "$IL_SAMPLES_PER_BATCH" \
        --ckpt-dir "checkpoints/il/$RUN_NAME" \
        --log-dir "runs/il/$RUN_NAME" \
        --eval-every-epochs 0
}

evaluate_imitation_policy() {
    local output_dir="reports/${RUN_NAME}_il_eval"
    if [[ -f "$output_dir/summary.json" && "${FORCE_IL_EVAL:-0}" != "1" ]]; then
        log "Skipping IL evaluation; $output_dir/summary.json already exists."
        return
    fi
    run_command uv run python scripts/eval_multi_city.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --checkpoint "checkpoints/il/$RUN_NAME/movement_policy_best.pt" \
        --policies learned max-pressure queue \
        --seeds 100 101 102 \
        --demand-scales 0.8 1.0 1.2 \
        --steps 1800 \
        --device "$DEVICE" \
        --output-dir "$output_dir" \
        --log-dir "runs/eval/${RUN_NAME}_il_eval"
}

run_ppo_pilot() {
    local checkpoint_path="checkpoints/rl/$RUN_NAME/movement_ppo_latest.pt"
    if [[ -f "$checkpoint_path" && "${FORCE_PPO_PILOT:-0}" != "1" ]]; then
        log "Skipping PPO pilot; $checkpoint_path already exists."
        return
    fi
    run_command uv run python scripts/train_rl.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --il-checkpoint "checkpoints/il/$RUN_NAME/movement_policy_best.pt" \
        --iterations "$PPO_PILOT_ITERATIONS" \
        --steps-per-rollout 1800 \
        --rollouts-per-update 16 \
        --num-workers "$WORKERS" \
        --eval-every 25 \
        --eval-steps 1800 \
        --eval-seeds 100 101 102 \
        --eval-policies learned max-pressure queue \
        --save-every 25 \
        --device "$DEVICE" \
        --ckpt-dir "checkpoints/rl/$RUN_NAME" \
        --log-dir "runs/rl/$RUN_NAME"
}

continue_ppo_to_total() {
    if [[ "${RUN_FULL_PPO:-0}" != "1" ]]; then
        log "PPO pilot complete. Set RUN_FULL_PPO=1 to continue to $PPO_TOTAL_ITERATIONS iterations."
        return
    fi
    run_command uv run python scripts/train_rl.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        --resume-checkpoint "checkpoints/rl/$RUN_NAME/movement_ppo_latest.pt" \
        --iterations "$PPO_TOTAL_ITERATIONS" \
        --steps-per-rollout 1800 \
        --rollouts-per-update 16 \
        --num-workers "$WORKERS" \
        --eval-every 25 \
        --eval-steps 1800 \
        --eval-seeds 100 101 102 \
        --eval-policies learned max-pressure queue \
        --save-every 25 \
        --device "$DEVICE" \
        --ckpt-dir "checkpoints/rl/$RUN_NAME" \
        --log-dir "runs/rl/$RUN_NAME"
}

main() {
    install_system_dependencies
    install_uv_if_missing
    bootstrap_repository "$@"
    configure_sumo_environment
    sync_python_environment
    check_cuda
    build_city_networks
    run_validation
    start_tensorboard
    run_baseline_smoke
    collect_imitation_data
    train_imitation_policy
    evaluate_imitation_policy
    run_ppo_pilot
    continue_ppo_to_total
    log "Training script finished."
}

main "$@"
