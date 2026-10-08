#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
SOURCE_RUN="${1:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_ipc100/LPLD_DDRUOS_officialVectorized_IPC100_BN0p05_4GPU_10x20x30x40x_seed0_20260819_140653}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"

if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain four comma-separated GPU IDs" >&2
  exit 2
fi

test -x "$PYTHON"
test -f "$SOURCE_RUN/synthetic.pt"
test -f "$TEACHER"
test -d "$DATA_PATH/train"
test -d "$DATA_PATH/val"

for compression in 10 20 30 40; do
  test -f "$SOURCE_RUN/downstream_${compression}x_lpld_seed0/fkd_pool/pool_summary.pt"
done

LOG="$SOURCE_RUN/lpqld_downstream_10x20x30x40x.log"
exec >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] LPQLD downstream evaluation start pid=$$"
echo "root=$ROOT source_run=$SOURCE_RUN gpu_ids=$GPU_IDS"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] failed status=$status"; exit "$status"' ERR

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

train_one() {
  local compression="$1" gpu="$2"
  local pool="$SOURCE_RUN/downstream_${compression}x_lpld_seed0/fkd_pool"
  local output="$SOURCE_RUN/downstream_${compression}x_lpqld_dkr_seed0"
  mkdir -p "$output"

  if [[ -f "$output/complete" ]]; then
    echo "[$(date --iso-8601=seconds)] ${compression}x already complete; skip"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] ${compression}x LPQLD train start gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SOURCE_RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
    --dkr_min_temperature 2 --eval_every 10 --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$output/complete"
  echo "[$(date --iso-8601=seconds)] ${compression}x LPQLD complete gpu=$gpu"
}

pids=()
compressions=(10 20 30 40)
for index in 0 1 2 3; do
  train_one "${compressions[$index]}" "${gpu_array[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more LPQLD downstream runs failed" >&2
  exit 1
fi

touch "$SOURCE_RUN/lpqld_downstream_complete"
echo "[$(date --iso-8601=seconds)] LPQLD downstream evaluation complete"
