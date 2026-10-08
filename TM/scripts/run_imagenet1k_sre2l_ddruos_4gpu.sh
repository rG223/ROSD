#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_imagenet1k_sre2l_ddruos_ipc50_g20_dual610_4gpu.sh" "$@"
