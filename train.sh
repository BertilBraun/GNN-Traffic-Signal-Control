#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/GNN-Traffic-Signal-Control.git}"
REPO_DIR="${REPO_DIR:-$HOME/GNN-Traffic-Signal-Control}"
BRANCH="${BRANCH:-master}"
RUN_NAME="${RUN_NAME:-city_first_pass_throughput_scratch_32_worker}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/training/city_first_pass_throughput_scratch_32_worker.yaml}"
DEVICE="${DEVICE:-cuda}"
FINAL_ITERATIONS="${FINAL_ITERATIONS:-85}"
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
            run_command git pull --ff-only origin "$BRANCH"
        fi
        return
    fi
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        run_command git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    elif [[ "${SKIP_GIT_UPDATE:-0}" != "1" ]]; then
        run_command git -C "$REPO_DIR" fetch origin "$BRANCH"
        run_command git -C "$REPO_DIR" checkout "$BRANCH"
        run_command git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
    fi
    exec "$REPO_DIR/train.sh" "$@"
}

install_system_dependencies() {
    if [[ "${INSTALL_SYSTEM_DEPS:-1}" != "1" ]]; then
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; install Git, build tools, SUMO, and SUMO tools manually."
        return
    fi
    run_command sudo_command apt-get update
    run_command sudo_command env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential ca-certificates curl git python3 python3-venv sumo sumo-tools
}

install_uv_if_missing() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    run_command curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
    run_command sh /tmp/uv-install.sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { log "uv was installed but is not on PATH."; exit 1; }
}

configure_environment() {
    export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
    [[ -d "$SUMO_HOME" ]] || { log "SUMO_HOME does not exist: $SUMO_HOME"; exit 1; }
    export PYTHONPATH="$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
    run_command uv sync --group dev
    run_command sumo --version
    if [[ "$DEVICE" == cuda* ]]; then
        run_command uv run python -c \
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA requested but unavailable')"
    fi
}

configure_logging() {
    mkdir -p "runs/rl/$RUN_NAME"
    exec > >(tee -a "runs/rl/$RUN_NAME/train_stdout.log") 2>&1
}

build_city_networks() {
    if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
        return
    fi
    local build_file
    for build_file in \
        configs/karlsruhe_oststadt/karlsruhe_oststadt.build.yaml \
        configs/mannheim_innenstadt/mannheim_innenstadt.build.yaml \
        configs/stuttgart_mitte/stuttgart_mitte.build.yaml \
        configs/heidelberg_bergheim/heidelberg_bergheim.build.yaml \
        configs/freiburg_altstadt/freiburg_altstadt.build.yaml
    do
        run_command uv run python scripts/network_workbench.py --build-file "$build_file" rebuild
    done
}

validate_repository() {
    run_command uv run ruff check scripts src tests
    if [[ "${RUN_TESTS:-1}" == "1" ]]; then
        run_command uv run pytest -q
    fi
}

start_tensorboard() {
    if [[ "${START_TENSORBOARD:-1}" != "1" ]]; then
        return
    fi
    mkdir -p runs
    if pgrep -f "tensorboard --logdir runs --host 0.0.0.0 --port $TENSORBOARD_PORT" >/dev/null 2>&1; then
        return
    fi
    nohup uv run tensorboard --logdir runs --host 0.0.0.0 --port "$TENSORBOARD_PORT" \
        > runs/tensorboard.log 2>&1 &
    printf '%s\n' "$!" > runs/tensorboard.pid
}

initialization_arguments() {
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
        printf '%s\0%s\0' --resume-checkpoint "$RESUME_CHECKPOINT"
        return
    fi
    if [[ -n "${IL_CHECKPOINT:-}" ]]; then
        printf '%s\0%s\0' --il-checkpoint "$IL_CHECKPOINT"
        return
    fi
    printf '%s\0' \
        --scratch-random \
        --scratch-lane-feature-dim 29 \
        --scratch-movement-feature-dim 4 \
        --scratch-hidden-dim 64 \
        --scratch-num-hops 1
}

train_policy() {
    local -a initialization
    mapfile -d '' -t initialization < <(initialization_arguments)
    run_command uv run python scripts/train_rl.py \
        --experiment-config "$EXPERIMENT_CONFIG" \
        "${initialization[@]}" \
        --iterations "$FINAL_ITERATIONS" \
        --device "$DEVICE" \
        --ckpt-dir "checkpoints/rl/$RUN_NAME" \
        --log-dir "runs/rl/$RUN_NAME"
}

main() {
    install_system_dependencies
    install_uv_if_missing
    bootstrap_repository "$@"
    configure_logging
    configure_environment
    build_city_networks
    validate_repository
    start_tensorboard
    train_policy
}

main "$@"
