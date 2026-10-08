#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?usage: $0 RUN_DIR}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-python}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?set TEACHER to the ResNet18-BN checkpoint}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_GPU="${TRAIN_GPU:-0}"
BASE="$RUN/downstream_10x_30groups_cutmix_lb64_tb64_dkr_step_ca_seed0"
FKD="$BASE/fkd_pool"
OUT="$BASE/seed0"

test -f "$RUN/synthetic.pt"
test -f "$RUN/label_codec_4000.pt"
mkdir -p "$BASE" "$FKD" "$OUT"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exactly 30 retained augmentation groups: 31,200 full batches / 9,360.
# CutMix is intentionally enabled: the no-CutMix ablation memorizes these
# synthetic images while its Tiny-ImageNet validation accuracy stays low.
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
  --mode relabel_pool --synthetic_path "$RUN/synthetic.pt" \
  --teacher_path "$TEACHER" --codec_checkpoint "$RUN/label_codec_4000.pt" \
  --fkd_path "$FKD" --output_path "$BASE" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
  --pool_compression 3.333333333333334 --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 64 --workers 12 \
  --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 \
  > "$BASE/relabel_pool.stdout.log" 2>&1
touch "$BASE/relabel_complete"

# One seed on one GPU preserves a true train-mode BN batch of 64.
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON" -u TM/sre2l_fkd.py \
  --mode train_pool --synthetic_path "$RUN/synthetic.pt" \
  --teacher_path "$TEACHER" --fkd_path "$FKD" --output_path "$OUT" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
  --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
  --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
  --warmup_epochs 5 --warmup_start_factor 0.01 \
  --scale_loss_by_temperature_squared --eval_every 10 \
  --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
  --dkr_min_temperature 2 --ca_dynamic \
  --seed 0 --device_ids 0 \
  > "$OUT/train_pool.stdout.log" 2>&1

touch "$BASE/complete"
