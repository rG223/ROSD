#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:?Set RUN_DIR to a completed ImageNet-1K ROSD run}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ITERATIONS="${ITERATIONS:-4000}"
IPC="${IPC:-50}"
TRAIN_LABEL_GROUPS="${LABEL_GROUPS:-20}"
TARGETS_CSV="${TARGETS_CSV:-623,721,954,1500}"
SOURCE_POOL_COMPRESSION="${SOURCE_POOL_COMPRESSION:-3}"
UTILITY_MODE="${UTILITY_MODE:-sre2l}"
WORKERS="${WORKERS:-12}"

POSTQUANT="$RUN/postquant_imagenet1k_iter${ITERATIONS}"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter${ITERATIONS}.pt"
CODEC="$RUN/label_codec_${ITERATIONS}.pt"
OUT="$RUN/downstream_multirate_${TARGETS_CSV//,/_}_seed0"
SOURCE_POOL="$OUT/fkd_pool_source"

mkdir -p "$POSTQUANT" "$OUT"
exec >>"$OUT/pipeline.log" 2>&1
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

for required in "$TEACHER" "$CODEC"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 1
  fi
done

echo "[$(date --iso-8601=seconds)] post-quantization decode start"
for wave in "0 1" "2 3"; do
  pids=()
  for rank in $wave; do
    class_start=$((rank * 250))
    class_end=$((class_start + 250))
    input="$RUN/rank${rank}_${class_start}_${class_end}/pool_${ITERATIONS}_global_keys.pt"
    shard="$POSTQUANT/shard${rank}.pt"
    if [[ -f "$shard" ]]; then
      continue
    fi
    if [[ ! -f "$input" ]]; then
      echo "Missing rank checkpoint: $input" >&2
      exit 1
    fi
    gpu=$(echo "$GPU_IDS" | cut -d, -f$((rank + 1)))
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/quantize_decode_tensorpool_shard.py \
      --input "$input" --output "$shard" \
      --quantized_output "$POSTQUANT/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --classes_per_shard 250 --ipc "$IPC" \
      --image_size 224 --max_iter "$ITERATIONS" --workers "$WORKERS" \
      --mse_threshold 5e-7 --encoder_gain 16 \
      --teacher_path "$TEACHER" --label_codec_checkpoint "$CODEC" \
      --postquant_label_rate_samples 1 \
      > "$POSTQUANT/shard${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

if [[ ! -f "$SYNTHETIC" ]]; then
  read -r LABEL_KIB LABEL_MODEL_KIB < <(
    "$PYTHON" - "$CODEC" <<'PY'
import sys
import torch
config = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
print(config["estimated_label_kib_per_class"], config["label_model_kib_per_class"])
PY
  )
  "$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "$POSTQUANT/shard0.pt" "$POSTQUANT/shard1.pt" \
             "$POSTQUANT/shard2.pt" "$POSTQUANT/shard3.pt" \
    --output "$SYNTHETIC" --ipc "$IPC" --num_classes 1000 \
    --image_size 224 --label_kib "$LABEL_KIB" \
    --label_model_kib "$LABEL_MODEL_KIB" --encoder_gain 16 \
    --label_groups "$TRAIN_LABEL_GROUPS" --utility_mode "$UTILITY_MODE" \
    > "$POSTQUANT/merge.log" 2>&1
fi
touch "$POSTQUANT/complete"

if [[ ! -f "$SOURCE_POOL/pool_summary.pt" ]]; then
  mkdir -p "$SOURCE_POOL"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --codec_checkpoint "$CODEC" \
    --fkd_path "$SOURCE_POOL" --output_path "$OUT/relabel_source" \
    --data_path "$DATA_PATH" --dataset ImageNet1K --epochs 300 \
    --pool_compression "$SOURCE_POOL_COMPRESSION" --crop_size 224 \
    --min_crop_scale 0.08 --temperature 20 --loader_batch 128 \
    --workers "$WORKERS" --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 \
    --fast_downstream --log_every 100 \
    > "$OUT/relabel_source.stdout.log" 2>&1
fi

IFS=',' read -r -a targets <<< "$TARGETS_CSV"
for target in "${targets[@]}"; do
  pool="$OUT/fkd_pool_target${target}"
  train_out="$OUT/train_target${target}"
  mkdir -p "$pool" "$train_out"
  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    "$PYTHON" TM/build_fkd_pool_target.py \
      --sources "$SOURCE_POOL" --output "$pool" \
      --target_total_kib "$target" --num_classes 1000 \
      > "$OUT/build_target${target}.log" 2>&1
  fi
  echo "[$(date --iso-8601=seconds)] target=${target} KiB/class train start"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --fkd_path "$pool" \
    --output_path "$train_out" --data_path "$DATA_PATH" \
    --dataset ImageNet1K --epochs 300 --crop_size 224 --temperature 20 \
    --train_batch 128 --workers "$WORKERS" --optimizer adamw \
    --learning_rate 0.001 --weight_decay 0.01 --eval_every 10 --seed 0 \
    --device_ids 0 1 2 3 --dkr_schedule step --dkr_step_size 30 \
    --dkr_step_gamma 0.7 --dkr_min_temperature 2 --ca_dynamic \
    --fast_downstream > "$train_out/train_pool.stdout.log" 2>&1
  touch "$train_out/complete"
  echo "[$(date --iso-8601=seconds)] target=${target} complete"
done

touch "$OUT/complete"
echo "[$(date --iso-8601=seconds)] multi-rate evaluation complete"
