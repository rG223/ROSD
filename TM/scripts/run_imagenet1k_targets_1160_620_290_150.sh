#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/SRe2L_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260826_124447}"
BASE="$RUN/downstream_official_lpqld_300e_seed0"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter4000.pt"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DEVICE_IDS="${DEVICE_IDS:-0 1 2 3}"
CODEC="$RUN/label_codec_4000.pt"
POOL20="$BASE/fkd_pool_g20"
EXTRA13="$BASE/fkd_pool_extra13_seed142"
POOL1160="$BASE/fkd_pool_target1160"
POOL620="$BASE/fkd_pool_target620"
POOL290="$BASE/fkd_pool_target290"
POOL150="$BASE/fkd_pool_target150"
JOBS="$BASE/targets_1160_620_290_150.json"

cd "$ROOT"
mkdir -p "$BASE/extra13_relabel"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

for required in "$SYNTHETIC" "$TEACHER" "$CODEC" "$POOL20/pool_summary.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$EXTRA13/pool_summary.pt" ]]; then
  mkdir -p "$EXTRA13"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --codec_checkpoint "$CODEC" \
    --fkd_path "$EXTRA13" --output_path "$BASE/extra13_relabel" \
    --data_path "$DATA_PATH" --dataset ImageNet1K --epochs 300 \
    --pool_compression 23.076923076923077 --crop_size 224 \
    --min_crop_scale 0.08 --temperature 20 --loader_batch 128 --workers 12 \
    --fkd_seed 142 --seed 0 --device_ids $DEVICE_IDS \
    --fast_downstream --log_every 100 \
    > "$BASE/extra13_relabel.stdout.log" 2>&1
fi

"$PYTHON" TM/build_fkd_pool_target.py \
  --sources "$POOL20" "$EXTRA13" --output "$POOL1160" \
  --target_total_kib 1160 --num_classes 1000 --pool_batches 12870 \
  > "$BASE/build_target1160.log" 2>&1

"$PYTHON" TM/build_fkd_pool_target.py \
  --sources "$POOL20" --output "$POOL620" \
  --target_total_kib 620 --num_classes 1000 --pool_batches 6299 \
  > "$BASE/build_target620.log" 2>&1

"$PYTHON" TM/build_fkd_pool_target.py \
  --sources "$POOL20" --output "$POOL290" \
  --target_total_kib 290 --num_classes 1000 --pool_batches 2283 \
  > "$BASE/build_target290.log" 2>&1

"$PYTHON" TM/build_fkd_pool_target.py \
  --sources "$POOL20" --output "$POOL150" \
  --target_total_kib 150 --num_classes 1000 --pool_batches 579 \
  > "$BASE/build_target150.log" 2>&1

"$PYTHON" - "$JOBS" "$BASE" "$SYNTHETIC" "$TEACHER" \
  "$DATA_PATH" "$DEVICE_IDS" <<'PY'
import json
import sys
from pathlib import Path

jobs_path, base, synthetic, teacher, data_path, device_ids = sys.argv[1:]
base = Path(base)
common = [
    "--mode", "train_pool", "--synthetic_path", synthetic,
    "--teacher_path", teacher, "--data_path", data_path,
    "--dataset", "ImageNet1K", "--epochs", "300", "--crop_size", "224",
    "--temperature", "20", "--train_batch", "128", "--workers", "8",
    "--optimizer", "adamw", "--learning_rate", "0.001",
    "--weight_decay", "0.01", "--eval_every", "10", "--seed", "0",
    "--device_ids", *device_ids.split(), "--dkr_schedule", "step",
    "--dkr_step_size", "30", "--dkr_step_gamma", "0.7",
    "--dkr_min_temperature", "2", "--ca_dynamic", "--fast_downstream",
]
specs = [
    ("rate1160", base / "fkd_pool_target1160", base / "train_target1160"),
    ("rate620", base / "fkd_pool_target620", base / "train_target620"),
    ("rate290", base / "fkd_pool_target290", base / "train_target290_v2"),
    ("rate150", base / "fkd_pool_target150", base / "train_target150_v2"),
]
jobs = []
for name, pool, output in specs:
    output.mkdir(parents=True, exist_ok=True)
    jobs.append({
        "name": name,
        "argv": common + ["--fkd_path", str(pool), "--output_path", str(output)],
        "stdout_path": str(output / "train_pool.stdout.log"),
    })
Path(jobs_path).write_text(json.dumps(jobs, indent=2))
PY

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/launch_shared_fkd_train.py \
  --synthetic_path "$SYNTHETIC" --jobs_json "$JOBS" \
  > "$BASE/shared_four_targets.stdout.log" 2>&1
touch "$BASE/targets_1160_620_290_150_complete"
