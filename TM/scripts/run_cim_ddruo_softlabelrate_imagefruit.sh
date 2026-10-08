#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
FKD_PYTHON="${FKD_PYTHON:-$PYTHON}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the ImageNet root or parent used by core.utils.get_dataset}"
TEACHER_PATH="${TEACHER_PATH:?Set TEACHER_PATH to a ResNet18BN teacher checkpoint}"
SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to the completed image-only CIM-DD-RUO run}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SUBSET="${SUBSET:-imagefruit}"
IPC="${IPC:-102}"
TOTAL_TARGET_KIB="${TOTAL_TARGET_KIB:-1200}"
JOINT_ITERS="${JOINT_ITERS:-200}"
LABEL_STEP="${LABEL_STEP:-0.5}"
LABEL_GROUPS="${LABEL_GROUPS:-300}"
SEED="${SEED:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/results/cim_ddruo/ImageNet/$SUBSET/$IPC/CIM_DDRUO_softLabelRate_step${LABEL_STEP}_total${TOTAL_TARGET_KIB}_joint${JOINT_ITERS}_seed${SEED}_${STAMP}}"

test -f "$SOURCE_RUN/pool_init.pt" || {
  printf 'missing source pool: %s\n' "$SOURCE_RUN/pool_init.pt" >&2
  exit 1
}
test -f "$SOURCE_RUN/cim_references.pt" || {
  printf 'missing CIM references: %s\n' "$SOURCE_RUN/cim_references.pt" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
cp --reflink=auto "$SOURCE_RUN/cim_references.pt" "$RUN_DIR/cim_references.pt"
printf '%s\n' "$RUN_DIR" > "$(dirname "$RUN_DIR")/latest_softlabel_rate.txt"

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
  --resume_pool "$SOURCE_RUN/pool_init.pt" \
  --cim_factor 2 \
  --cim_feature_chunk 16 \
  --cim_augmentation crop_cutout_flip \
  --optimize_label_rate \
  --label_step "$LABEL_STEP" \
  --label_groups "$LABEL_GROUPS" \
  --label_entropy_lr 0.001 \
  --label_entropy_warmup 20 \
  --label_feature_chunk 128 \
  --label_model_bits 16 \
  --total_target_kib "$TOTAL_TARGET_KIB" \
  --ldb 0.1 \
  --codec_lr 0.001 \
  --lr_it 1000 \
  --rate_control dual \
  --latent_target_kib 500 \
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

OUTPUT="$RUN_DIR/downstream_fkd_quantized_seed${SEED}"
LABELS="$OUTPUT/fkd_labels"
CODEC="$RUN_DIR/label_codec_${JOINT_ITERS}.pt"
mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$FKD_PYTHON" -u TM/sre2l_fkd.py \
  --mode relabel \
  --synthetic_path "$RUN_DIR/synthetic.pt" \
  --teacher_path "$TEACHER_PATH" \
  --codec_checkpoint "$CODEC" \
  --fkd_path "$LABELS" \
  --output_path "$OUTPUT" \
  --epochs 300 \
  --crop_size 128 \
  --min_crop_scale 0.08 \
  --cutmix_alpha 1.0 \
  --temperature 20 \
  --loader_batch 128 \
  --workers 8 \
  --fkd_seed 42 \
  --seed "$SEED" \
  --device_ids 0 1 2 3 \
  >"$OUTPUT/relabel.stdout.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$FKD_PYTHON" -u TM/sre2l_fkd.py \
  --mode train \
  --synthetic_path "$RUN_DIR/synthetic.pt" \
  --teacher_path "$TEACHER_PATH" \
  --fkd_path "$LABELS" \
  --output_path "$OUTPUT" \
  --data_path "$DATA_PATH" \
  --dataset ImageNet \
  --subset "$SUBSET" \
  --epochs 300 \
  --crop_size 128 \
  --temperature 20 \
  --train_batch 256 \
  --workers 8 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --eval_every 10 \
  --seed "$SEED" \
  --device_ids 0 1 2 3 \
  >"$OUTPUT/train.stdout.log" 2>&1

printf 'complete: %s\n' "$RUN_DIR"
