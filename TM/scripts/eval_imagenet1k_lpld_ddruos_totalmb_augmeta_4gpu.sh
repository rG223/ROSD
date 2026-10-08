#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/LPLD_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260830_233358}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
ITERATIONS="${ITERATIONS:-4000}"
IPC="${IPC:-50}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS="${WORKERS:-8}"
SOURCE_POOL_COMPRESSION="${SOURCE_POOL_COMPRESSION:-7.5}"
TARGETS_MB_CSV="${TARGETS_MB_CSV:-1200,600,700,300}"

POSTQUANT="$RUN/postquant_imagenet1k_iter${ITERATIONS}"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter${ITERATIONS}.pt"
OUT="$RUN/downstream_totalMB_augmeta_${TARGETS_MB_CSV//,/_}_seed0"
SOURCE_POOL="$OUT/fkd_pool_source_c${SOURCE_POOL_COMPRESSION//./p}"
JOBS="$OUT/jobs.json"

mkdir -p "$POSTQUANT" "$OUT"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

exec >>"$OUT/pipeline.log" 2>&1
trap 'status=$?; echo "[$(date --iso-8601=seconds)] pipeline failed status=$status"; exit "$status"' ERR

test -f "$RUN/joint_complete"
test -f "$RUN/label_codec_${ITERATIONS}.pt"
test -f "$TEACHER"

echo "[$(date --iso-8601=seconds)] post-quantization decode start"
for wave in "0 1 2 3"; do
  pids=()
  for rank in $wave; do
    class_start=$((rank * 250))
    class_end=$((class_start + 250))
    shard="$POSTQUANT/shard${rank}.pt"
    if [[ -f "$shard" ]]; then continue; fi
    gpu="$(echo "$GPU_IDS" | cut -d, -f$((rank + 1)))"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/quantize_decode_tensorpool_shard.py \
      --input "$RUN/rank${rank}_${class_start}_${class_end}/pool_${ITERATIONS}_global_keys.pt" \
      --output "$shard" --quantized_output "$POSTQUANT/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --classes_per_shard 250 --ipc "$IPC" \
      --image_size 224 --max_iter "$ITERATIONS" --workers 12 \
      --mse_threshold 5e-7 --encoder_gain 16 \
      --teacher_path "$TEACHER" \
      --label_codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
      --postquant_label_rate_samples 1 \
      >"$POSTQUANT/shard${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

if [[ ! -f "$SYNTHETIC" ]]; then
  read -r LABEL_KIB LABEL_MODEL_KIB < <(
    "$PYTHON" - "$RUN/label_codec_${ITERATIONS}.pt" <<'PY'
import sys
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
print(c["estimated_label_kib_per_class"], c["label_model_kib_per_class"])
PY
  )
  "$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "$POSTQUANT/shard0.pt" "$POSTQUANT/shard1.pt" \
             "$POSTQUANT/shard2.pt" "$POSTQUANT/shard3.pt" \
    --output "$SYNTHETIC" --ipc "$IPC" --num_classes 1000 \
    --image_size 224 --label_kib "$LABEL_KIB" --label_model_kib "$LABEL_MODEL_KIB" \
    --encoder_gain 16 --label_groups 20 --utility_mode lpld_class_bn \
    >"$POSTQUANT/merge.log" 2>&1
fi
touch "$POSTQUANT/complete"

if [[ ! -f "$SOURCE_POOL/pool_summary.pt" ]]; then
  mkdir -p "$SOURCE_POOL"
  echo "[$(date --iso-8601=seconds)] source FKD pool generation start"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" --teacher_path "$TEACHER" \
    --codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$SOURCE_POOL" --output_path "$OUT/relabel_source" \
    --data_path "$DATA_PATH" --dataset ImageNet1K --epochs 300 \
    --pool_compression "$SOURCE_POOL_COMPRESSION" --crop_size 224 \
    --min_crop_scale 0.08 --temperature 20 --loader_batch 128 --workers 12 \
    --relabel_forward_accum 1 \
    --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 --fast_downstream --log_every 100 \
    >"$OUT/relabel_source.stdout.log" 2>&1
fi

IFS=',' read -r -a targets_mb <<< "$TARGETS_MB_CSV"
if [[ "${#targets_mb[@]}" -ne 4 ]]; then
  echo "TARGETS_MB_CSV must contain exactly four values" >&2
  exit 2
fi

for target_mb in "${targets_mb[@]}"; do
  pool="$OUT/fkd_pool_total${target_mb}MB"
  mkdir -p "$pool"
  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    "$PYTHON" TM/build_fkd_pool_target.py \
      --sources "$SOURCE_POOL" --output "$pool" --target_total_mb "$target_mb" \
      --num_classes 1000 --include_augmentation_metadata \
      >"$OUT/build_total${target_mb}MB.log" 2>&1
  fi
done

"$PYTHON" - "$JOBS" "$OUT" "$SYNTHETIC" "$TEACHER" "${targets_mb[@]}" <<'PY'
import json
import sys
from pathlib import Path

jobs_path, out, synthetic, teacher, *targets = sys.argv[1:]
common = [
    "--mode", "train_pool", "--synthetic_path", synthetic,
    "--teacher_path", teacher, "--data_path", "/data2/home/ypliu",
    "--dataset", "ImageNet1K", "--epochs", "300", "--crop_size", "224",
    "--temperature", "20", "--train_batch", "128", "--workers", "8",
    "--optimizer", "adamw", "--learning_rate", "0.001",
    "--weight_decay", "0.01", "--eval_every", "10", "--seed", "0",
    "--device_ids", "0", "--fast_downstream",
]
jobs = []
for gpu, target in enumerate(targets):
    pool = Path(out) / f"fkd_pool_total{target}MB"
    output = Path(out) / f"train_total{target}MB"
    output.mkdir(parents=True, exist_ok=True)
    jobs.append({
        "name": f"total{target}MB",
        "cuda_visible_devices": str(gpu),
        "argv": common + ["--fkd_path", str(pool), "--output_path", str(output)],
        "stdout_path": str(output / "train_pool.stdout.log"),
    })
Path(jobs_path).write_text(json.dumps(jobs, indent=2))
PY

echo "[$(date --iso-8601=seconds)] four single-GPU LPLD-protocol jobs start"
"$PYTHON" -u TM/launch_shared_fkd_train.py \
  --synthetic_path "$SYNTHETIC" --jobs_json "$JOBS" \
  >"$OUT/shared_four_train.stdout.log" 2>&1
touch "$OUT/complete"
echo "[$(date --iso-8601=seconds)] all downstream jobs complete"
