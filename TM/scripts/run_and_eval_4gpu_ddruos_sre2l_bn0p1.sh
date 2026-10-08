#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="$1"
ITERATIONS="${ITERATIONS:-4000}"

OUT="$RUN_DIR" BN_WEIGHT="${BN_WEIGHT:-0.1}" ITERATIONS="$ITERATIONS" \
  bash "$ROOT/TM/scripts/run_4gpu_ddruos_sre2l_joint_random.sh"

bash "$ROOT/TM/scripts/postprocess_eval_after_joint_generic.sh" \
  "$RUN_DIR" "$ITERATIONS" "${LABEL_GROUPS:-30}"
