#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA3_ROOT="${DATA3_ROOT:-/data3/ypliu/DD-RUO-1}"
ASSET_DIR="${ASSET_DIR:-$DATA3_ROOT/assets/tiny_imagenet}"
SOURCE_TEACHER="${SOURCE_TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
TEACHER="${TEACHER:-$ASSET_DIR/ddruo_teacher.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${1:-$DATA3_ROOT/results/tiny_imagenet_ipc200/SRe2L_DDRUOS_IPC200_G20_combinedDual610_dualLR2e-3_wImg0p25_wLabel1_dualMax1_worldNorm_fastBN_TF32_gain16_GPU0123_seed0_${STAMP}}"

mkdir -p "$ASSET_DIR" "$RUN_DIR"
if [[ ! -f "$TEACHER" ]]; then
  cp --reflink=auto "$SOURCE_TEACHER" "$TEACHER.tmp"
  mv "$TEACHER.tmp" "$TEACHER"
fi

export ROOT TEACHER
export UTILITY_PROFILE=sre2l
export IPC=200
export ITERATIONS=4000
export LABEL_GROUPS=20
export ENCODER_GAIN=16
export GPU_IDS=0,1,2,3
export DOWNSTREAM_GPU_IDS=0,1,2,3
export TARGETS_CSV=610
export LAMBDA_IMAGE=0.00001
export LAMBDA_LABEL=0.0001
export COMBINED_DUAL=1
export COMBINED_TARGET_KIB=610
export DUAL_INIT=0.0001
export DUAL_LR=0.002
export DUAL_RHO=0.00005
export DUAL_EMA_DECAY=0.95
export DUAL_UPDATE_EVERY=10
export DUAL_DEADBAND=0.02
export DUAL_MIN=-0.00005
export DUAL_MAX=1
export IMAGE_RATE_GRADIENT_WEIGHT=0.25
export LABEL_RATE_GRADIENT_WEIGHT=1.0
export LABEL_FEATURE_CHUNK=500
export LABEL_ENTROPY_BATCH=500
export UTILITY_BATCH_SIZE=400
export ALLOW_TF32=1
export CHECKPOINT_EVERY=1000

printf '%s\n' "$RUN_DIR" > "$DATA3_ROOT/latest_sre2l_ddruos_ipc200_run.txt"
exec bash "$ROOT/TM/scripts/run_persistent_ddruos_lpld_ipc100_2gpu_lambda1e5_targets.sh" "$RUN_DIR"
