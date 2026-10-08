#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export DOWNSTREAM_GPU_IDS="${DOWNSTREAM_GPU_IDS:-0,1,2,3}"
export ENCODER_GAIN="${ENCODER_GAIN:-32}"

exec bash \
  "$SCRIPT_DIR/run_persistent_ddruos_lpld_ipc100_2gpu_lambda1e5_targets.sh" \
  "$@"
