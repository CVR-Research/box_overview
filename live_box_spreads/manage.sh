#!/usr/bin/env bash
# Helper script to ensure the Conda env exists, install deps, verify env vars,
# and launch either the snapshot collector or the dashboard (or both).

set -euo pipefail

ENV_NAME="box_live"
PY_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIVE_DIR="${PROJECT_ROOT}/live_box_spreads"
REQ_FILE="${LIVE_DIR}/requirements.txt"
INGEST="${LIVE_DIR}/ingest.py"
DASH="${LIVE_DIR}/dashboard/app.py"

log() { printf "\033[1m%s\033[0m\n" "$*"; }

require_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda command not found. Install Anaconda/Miniconda first." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
}

ensure_env() {
    if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "Conda env ${ENV_NAME} already exists."
    else
        log "Creating conda env ${ENV_NAME} (python ${PY_VERSION})…"
        conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
    fi
}

install_requirements() {
    log "Installing/upgrading project requirements inside ${ENV_NAME}…"
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip >/dev/null
    conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"
}

require_env_vars() {
    if [[ -z "${TT_USERNAME:-}" || -z "${TT_PASSWORD:-}" ]]; then
        cat >&2 <<EOF
Missing TT_USERNAME or TT_PASSWORD environment variables.
Export them in your shell before running this script, e.g.:
    export TT_USERNAME="your_username"
    export TT_PASSWORD="your_password"
EOF
        exit 1
    fi
}

run_choice() {
    printf "\nChoose an action:\n"
    printf "  1) Collect a single snapshot\n"
    printf "  2) Start continuous collector (--loop)\n"
    printf "  3) Launch dashboard (expects snapshots already)\n"
    printf "  4) Run collector (--loop) + dashboard together\n"
    printf "Select [1-4]: "
    read -r choice
    case "${choice}" in
        1) conda run -n "${ENV_NAME}" python "${INGEST}" ;;
        2) conda run -n "${ENV_NAME}" python "${INGEST}" --loop ;;
        3) conda run -n "${ENV_NAME}" python "${DASH}" ;;
        4)
            log "Starting collector in background…"
            conda run -n "${ENV_NAME}" python "${INGEST}" --loop &
            COL_PID=$!
            trap 'kill ${COL_PID} 2>/dev/null || true' EXIT INT TERM
            sleep 2
            log "Launching dashboard…"
            conda run -n "${ENV_NAME}" python "${DASH}"
            ;;
        *)
            echo "Invalid choice." >&2
            exit 1
            ;;
    esac
}

require_conda
ensure_env
install_requirements
require_env_vars
run_choice
