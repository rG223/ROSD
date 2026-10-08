#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/SRe2L_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260826_124447}"
BASE="$RUN/downstream_official_lpqld_300e_seed0"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter4000.pt"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
CODEC="$RUN/label_codec_4000.pt"
POOL20="$BASE/fkd_pool_g20"
POOL290="$BASE/fkd_pool_target290"
POOL150="$BASE/fkd_pool_target150"
EXTRA13="$BASE/fkd_pool_extra13_seed142"
POOL1160="$BASE/fkd_pool_target1160"
JOBS="$BASE/four_rate_jobs.json"

mkdir -p "$BASE"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

make_train_argv() {
  local pool="$1"
  local output="$2"
  printf '%s\n' \
    --mode train_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path /data2/home/ypliu --dataset ImageNet1K --epochs 300 \
    --crop_size 224 --temperature 20 --train_batch 128 --workers 12 \
    --optimizer adamw --learning_rate 0.001 --weight_decay 0.01 \
    --eval_every 10 --seed 0 --device_ids 0 1 2 3 \
    --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
    --dkr_min_temperature 2 --ca_dynamic --fast_downstream
}

# Materialize the launcher JSON without duplicating the 29 GiB image tensor.
"$PYTHON" - "$JOBS" "$BASE" "$POOL20" "$POOL290" "$POOL150" "$SYNTHETIC" "$TEACHER" <<'PY'
import json
import sys
from pathlib import Path

jobs_path, base, pool20, pool290, pool150, synthetic, teacher = sys.argv[1:]
common = [
    "--mode", "train_pool", "--synthetic_path", synthetic,
    "--teacher_path", teacher, "--data_path", "/data2/home/ypliu",
    "--dataset", "ImageNet1K", "--epochs", "300", "--crop_size", "224",
    "--temperature", "20", "--train_batch", "128", "--workers", "12",
    "--optimizer", "adamw", "--learning_rate", "0.001",
    "--weight_decay", "0.01", "--eval_every", "10", "--seed", "0",
    "--device_ids", "0", "1", "2", "3", "--dkr_schedule", "step",
    "--dkr_step_size", "30", "--dkr_step_gamma", "0.7",
    "--dkr_min_temperature", "2", "--ca_dynamic", "--fast_downstream",
]
specs = [
    ("rate743", pool20, Path(base) / "train_target743_batchReplay"),
    ("rate290", pool290, Path(base) / "train_target290"),
    ("rate150", pool150, Path(base) / "train_target150"),
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

"$PYTHON" -u TM/launch_shared_fkd_train.py \
  --synthetic_path "$SYNTHETIC" --jobs_json "$JOBS" \
  > "$BASE/shared_three_train.stdout.log" 2>&1 &
SHARED_PID=$!

# Wait until the parent has loaded the shared tensor and forked all jobs before
# loading a second tensor copy for the extra-label relabel process.
for _ in $(seq 1 120); do
  if [[ $(grep -c '^started name=' "$BASE/shared_three_train.stdout.log" 2>/dev/null || true) -eq 3 ]]; then
    break
  fi
  sleep 5
done
if [[ $(grep -c '^started name=' "$BASE/shared_three_train.stdout.log" 2>/dev/null || true) -ne 3 ]]; then
  echo "shared downstream jobs did not start" >&2
  exit 1
fi

# Existing 20 groups are reused. Only 13 independent additional groups are
# generated, reducing relabel work from 33 to 13 groups.
if [[ ! -f "$EXTRA13/pool_summary.pt" ]]; then
  mkdir -p "$EXTRA13"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --codec_checkpoint "$CODEC" \
    --fkd_path "$EXTRA13" --output_path "$BASE/extra13_relabel" \
    --data_path /data2/home/ypliu --dataset ImageNet1K --epochs 300 \
    --pool_compression 23.076923076923077 --crop_size 224 \
    --min_crop_scale 0.08 --temperature 20 --loader_batch 128 --workers 12 \
    --fkd_seed 142 --seed 0 --device_ids 0 1 2 3 \
    --fast_downstream --log_every 100 \
    > "$BASE/extra13_relabel.stdout.log" 2>&1
fi

"$PYTHON" TM/build_fkd_pool_target.py \
  --sources "$POOL20" "$EXTRA13" --output "$POOL1160" \
  --target_total_kib 1160 --num_classes 1000 \
  > "$BASE/build_target1160.log" 2>&1

OUT1160="$BASE/train_target1160"
mkdir -p "$OUT1160"
mapfile -t ARGV1160 < <(make_train_argv "$POOL1160" "$OUT1160")
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u TM/sre2l_fkd.py "${ARGV1160[@]}" \
  > "$OUT1160/train_pool.stdout.log" 2>&1 &
PID1160=$!

wait "$SHARED_PID"
wait "$PID1160"
touch "$BASE/four_rate_complete"
