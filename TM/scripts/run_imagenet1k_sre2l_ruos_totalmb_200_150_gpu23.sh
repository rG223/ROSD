#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/SRe2L_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260826_124447}"
BASE="$RUN/downstream_official_lpqld_300e_seed0"
SOURCE_POOL="$BASE/fkd_pool_target1160"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter4000.pt"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
OUT="$BASE/totalMB_augmeta_200_150_seed0"
JOBS="$OUT/jobs.json"

cd "$ROOT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

for required in "$SOURCE_POOL/pool_summary.pt" "$SYNTHETIC" "$TEACHER"; do
  test -e "$required" || { echo "Missing required artifact: $required" >&2; exit 1; }
done

for target_mb in 200 150; do
  pool="$OUT/fkd_pool_total${target_mb}MB"
  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    mkdir -p "$pool"
    "$PYTHON" TM/build_fkd_pool_target.py \
      --sources "$SOURCE_POOL" --output "$pool" \
      --target_total_mb "$target_mb" --num_classes 1000 \
      --include_augmentation_metadata \
      >"$OUT/build_total${target_mb}MB.log" 2>&1
  fi
done

"$PYTHON" - "$JOBS" "$OUT" "$SYNTHETIC" "$TEACHER" <<'PY'
import json
import sys
from pathlib import Path

jobs_path, out, synthetic, teacher = sys.argv[1:]
out = Path(out)
common = [
    "--mode", "train_pool", "--synthetic_path", synthetic,
    "--teacher_path", teacher, "--data_path", "/data2/home/ypliu",
    "--dataset", "ImageNet1K", "--epochs", "300", "--crop_size", "224",
    "--temperature", "20", "--train_batch", "128", "--workers", "8",
    "--optimizer", "adamw", "--learning_rate", "0.001",
    "--weight_decay", "0.01", "--eval_every", "10", "--seed", "0",
    "--device_ids", "0", "--dkr_schedule", "step",
    "--dkr_step_size", "30", "--dkr_step_gamma", "0.7",
    "--dkr_min_temperature", "2", "--ca_dynamic", "--fast_downstream",
]
jobs = []
for gpu, target in ((2, 200), (3, 150)):
    pool = out / f"fkd_pool_total{target}MB"
    output = out / f"train_total{target}MB"
    output.mkdir(parents=True, exist_ok=True)
    jobs.append({
        "name": f"total{target}MB",
        "cuda_visible_devices": str(gpu),
        "argv": common + ["--fkd_path", str(pool), "--output_path", str(output)],
        "stdout_path": str(output / "train_pool.stdout.log"),
    })
Path(jobs_path).write_text(json.dumps(jobs, indent=2))
PY

"$PYTHON" -u TM/launch_shared_fkd_train.py \
  --synthetic_path "$SYNTHETIC" --jobs_json "$JOBS" \
  >"$OUT/shared_train.stdout.log" 2>&1

touch "$OUT/complete"
