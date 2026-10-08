#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
exec bash "$ROOT/TM/scripts/run_persistent_ddruos_sre2l_ipc100_bn1_2gpu_multix.sh" "$@"
