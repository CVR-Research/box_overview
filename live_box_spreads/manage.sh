#!/usr/bin/env bash
# Helper script to set up the environment and launch the box spread monitor.

set -euo pipefail

ENV_NAME="box_live"
PY_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

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
        log "Creating conda env ${ENV_NAME} (python ${PY_VERSION})..."
        conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
    fi
}

install_package() {
    log "Installing package (editable) inside ${ENV_NAME}..."
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip >/dev/null
    conda run -n "${ENV_NAME}" python -m pip install -e "${SCRIPT_DIR}[dev]"
}

require_env_vars() {
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        set +a
    fi
    if [[ -z "${ALPACA_API_KEY:-}" || -z "${ALPACA_API_SECRET:-}" ]]; then
        cat >&2 <<EOF
Missing ALPACA_API_KEY or ALPACA_API_SECRET environment variables.
Export them in your shell or add them to .env, e.g.:
    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"
EOF
        exit 1
    fi
}

run_choice() {
    printf "\nChoose an action:\n"
    printf "  1) Collect a single snapshot\n"
    printf "  2) Start continuous collector (--loop)\n"
    printf "  3) Launch dashboard (snapshot mode)\n"
    printf "  4) Run collector (--loop) + dashboard together\n"
    printf "  5) Launch dashboard (stream mode)\n"
    printf "Select [1-5]: "
    read -r choice
    case "${choice}" in
        1) conda run -n "${ENV_NAME}" boxmon ingest --config "${SCRIPT_DIR}/config.yaml" ;;
        2) conda run -n "${ENV_NAME}" boxmon ingest --loop --config "${SCRIPT_DIR}/config.yaml" ;;
        3) conda run -n "${ENV_NAME}" boxmon dashboard --config "${SCRIPT_DIR}/config.yaml" ;;
        4)
            log "Starting collector in background..."
            conda run -n "${ENV_NAME}" boxmon ingest --loop --config "${SCRIPT_DIR}/config.yaml" &
            COL_PID=$!
            trap 'kill ${COL_PID} 2>/dev/null || true' EXIT INT TERM
            sleep 2
            log "Launching dashboard..."
            conda run -n "${ENV_NAME}" boxmon dashboard --config "${SCRIPT_DIR}/config.yaml"
            ;;
        5)
            log "Launching dashboard (stream mode)..."
            conda run -n "${ENV_NAME}" boxmon dashboard --stream --config "${SCRIPT_DIR}/config.yaml"
            ;;
        *)
            echo "Invalid choice." >&2
            exit 1
            ;;
    esac
}

require_conda
ensure_env
install_package
require_env_vars
run_choice
