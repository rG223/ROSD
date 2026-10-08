#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the ImageNet root or parent used by core.utils.get_dataset}"
TEACHER_PATH="${TEACHER_PATH:?Set TEACHER_PATH to a ResNet18BN teacher checkpoint}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SUBSET="${SUBSET:-imagefruit}"
IPC="${IPC:-102}"
TARGET_KIB="${TARGET_KIB:-500}"
JOINT_ITERS="${JOINT_ITERS:-200}"
SEED="${SEED:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/results/cim_ddruo/ImageNet/$SUBSET/$IPC/CIM_DDRUO_factor2_target${TARGET_KIB}_joint${JOINT_ITERS}_seed${SEED}_${STAMP}}"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$(dirname "$RUN_DIR")/latest.txt"

cd "$ROOT"
printf 'run_dir=%s\n' "$RUN_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
"$PYTHON" -u TM/cim_ddruo_tensorpool.py \
  --data_path "$DATA_PATH" \
  --subset "$SUBSET" \
  --ipc "$IPC" \
  --teacher_path "$TEACHER_PATH" \
  --save_path "$RUN_DIR" \
  --cim_factor 2 \
  --cim_feature_chunk 16 \
  --cim_augmentation crop_cutout_flip \
  --ldb 0.1 \
  --codec_lr 0.001 \
  --lr_it 1000 \
  --warmup_rate_control dual \
  --warmup_target_kib "$TARGET_KIB" \
  --warmup_rate_margin 1.0 \
  --warmup_dual_lr 0.001 \
  --warmup_min_teacher_acc 0 \
  --rate_control dual \
  --latent_target_kib "$TARGET_KIB" \
  --dual_lr 0.0001 \
  --stage1_iterations "$JOINT_ITERS" \
  --stage2_iterations 0 \
  --layers_v v5 \
  --arm 32 \
  --dim 4 \
  --log_every 5 \
  --checkpoint_every 100 \
  --network_mse_threshold 5e-7 \
  --seed "$SEED" \
  >"$RUN_DIR/stdout.log" 2>&1

printf 'image-only initialization complete: %s\n' "$RUN_DIR"
