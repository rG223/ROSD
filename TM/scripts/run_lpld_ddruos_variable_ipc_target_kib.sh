#!/usr/bin/env bash
set -euo pipefail

# Stable public entry point for variable-IPC, target-rate LPLD-ROSD runs.
# Configuration is passed through environment variables. The implementation
# remains in the historical launcher so existing experiment commands keep working.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

exec bash \
  "$ROOT/TM/scripts/run_persistent_ddruos_lpld_ipc100_2gpu_lambda1e5_targets.sh" \
  "$@"
