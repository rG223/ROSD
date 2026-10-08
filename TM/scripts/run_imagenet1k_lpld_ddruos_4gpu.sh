#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_imagenet1k_lpld_ddruos_ipc50_multirate_4gpu.sh" "$@"
