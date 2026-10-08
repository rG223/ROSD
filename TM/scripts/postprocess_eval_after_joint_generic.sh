#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_DIR [ITERATIONS] [LABEL_GROUPS]" >&2
  exit 2
fi

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-python}"
RUN="$1"
ITERATIONS="${2:-4000}"
LABEL_GROUPS="${3:-30}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?set TEACHER to the ResNet18-BN checkpoint}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_GPU="${TRAIN_GPU:-0}"
# CutMix is part of the validated FKD replay recipe. On this IPC-100
# TensorPool payload, disabling it makes the student memorize synthetic
# classes (>95% train accuracy) while validation remains near chance.
DISABLE_CUTMIX="${DISABLE_CUTMIX:-0}"
USE_DKR_CA="${USE_DKR_CA:-1}"
BASE="$RUN/downstream_${LABEL_GROUPS}groups_seed0_bs64"
FKD="$BASE/fkd_pool"
OUT="$BASE/seed0"
POOL_COMPRESSION=$(awk "BEGIN {printf \"%.10g\", 100/$LABEL_GROUPS}")
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
  exit 2
fi

test -f "$RUN/joint_complete"
test -f "$RUN/label_codec_${ITERATIONS}.pt"
mkdir -p "$RUN/postquant" "$BASE" "$FKD" "$OUT"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pids=()
for rank in 0 1 2 3; do
  class_start=$((rank * 50))
  class_end=$((class_start + 50))
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$rank]}" "$PYTHON" -u TM/quantize_decode_tensorpool_shard.py \
    --input "$RUN/rank${rank}_${class_start}_${class_end}/pool_final_global_keys.pt" \
    --output "$RUN/postquant/shard${rank}.pt" \
    --quantized_output "$RUN/postquant/pool_quantized_shard${rank}.pt" \
    --class_start "$class_start" --workers 12 --mse_threshold 5e-7 \
    --teacher_path "$TEACHER" \
    --label_codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
    --postquant_label_rate_samples 1 \
    > "$RUN/postquant/shard${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
  --inputs "$RUN/postquant/shard0.pt" "$RUN/postquant/shard1.pt" \
           "$RUN/postquant/shard2.pt" "$RUN/postquant/shard3.pt" \
  --output "$RUN/synthetic.pt" \
  > "$RUN/postquant/merge.log" 2>&1
touch "$RUN/postquant_complete"

# Relabel inference uses all four GPUs; labels correspond to 30 augmented
# views per synthetic image when LABEL_GROUPS=30.
RELABEL_EXTRA=()
if [[ "$DISABLE_CUTMIX" == 1 ]]; then
  RELABEL_EXTRA+=(--disable_cutmix)
fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
  --mode relabel_pool --synthetic_path "$RUN/synthetic.pt" \
  --teacher_path "$TEACHER" --codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
  --fkd_path "$FKD" --output_path "$BASE" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
  --pool_compression "$POOL_COMPRESSION" --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 128 --workers 12 \
  --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 "${RELABEL_EXTRA[@]}" \
  > "$BASE/relabel_pool.stdout.log" 2>&1

# Match the established Tiny-ImageNet SRe2L evaluation recipe: one GPU and
# a real BN batch of 64, rather than DataParallel shards of 16 samples.
TRAIN_EXTRA=()
if [[ "$USE_DKR_CA" == 1 ]]; then
  TRAIN_EXTRA+=(--dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7)
  TRAIN_EXTRA+=(--dkr_min_temperature 2 --ca_dynamic)
fi
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON" -u TM/sre2l_fkd.py \
  --mode train_pool --synthetic_path "$RUN/synthetic.pt" \
  --teacher_path "$TEACHER" --fkd_path "$FKD" --output_path "$OUT" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
  --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
  --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
  --warmup_epochs 5 --warmup_start_factor 0.01 \
  --scale_loss_by_temperature_squared --eval_every 10 \
  --seed 0 --device_ids 0 "${TRAIN_EXTRA[@]}" \
  > "$OUT/train_pool.stdout.log" 2>&1

touch "$BASE/complete"
touch "$RUN/downstream_complete"
