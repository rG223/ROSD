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
WAIT_PIDS="${WAIT_PIDS:-2188876 2188882}"
QUEUE_LOG="$BASE/resume_620_1160_serial.stdout.log"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
exec >>"$QUEUE_LOG" 2>&1

echo "[$(date --iso-8601=seconds)] waiting for active targets: $WAIT_PIDS"
while true; do
  active=0
  for pid in $WAIT_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      active=1
    fi
  done
  [[ "$active" -eq 0 ]] && break
  sleep 60
done
echo "[$(date --iso-8601=seconds)] active targets finished; starting serial queue"
sleep 15

run_target() {
  local target="$1"
  local pool="$BASE/fkd_pool_target${target}"
  local output="$BASE/train_target${target}"
  local checkpoint="$output/checkpoint.pt"

  for required in "$SYNTHETIC" "$TEACHER" "$pool/pool_summary.pt" "$checkpoint"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required artifact: $required" >&2
      exit 1
    fi
  done

  echo "[$(date --iso-8601=seconds)] resuming target=${target}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --data_path "$DATA_PATH" \
    --dataset ImageNet1K --epochs 300 --crop_size 224 \
    --temperature 20 --train_batch 128 --workers 8 \
    --optimizer adamw --learning_rate 0.001 --weight_decay 0.01 \
    --eval_every 10 --seed 0 --device_ids $DEVICE_IDS \
    --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
    --dkr_min_temperature 2 --ca_dynamic --fast_downstream \
    --fkd_path "$pool" --output_path "$output" \
    --resume_checkpoint "$checkpoint" \
    >"$output/resume_pool.stdout.log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed target=${target}"
}

run_target 620
run_target 1160
touch "$BASE/targets_620_1160_serial_complete"
echo "[$(date --iso-8601=seconds)] serial queue complete"
