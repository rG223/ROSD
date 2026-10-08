#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
SOURCE_RUN="${1:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_ipc100/LPLD_DDRUOS_officialVectorized_IPC100_BN0p05_4GPU_10x20x30x40x_seed0_20260819_140653}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"

[[ "${#gpu_array[@]}" -eq 4 ]]
test -x "$PYTHON"
test -f "$SOURCE_RUN/synthetic.pt"
test -f "$TEACHER"

LOG="$SOURCE_RUN/resume_lpld_lpqld_target_kib.log"
exec >> "$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] target-KiB resume start pid=$$"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] failed status=$status"; exit "$status"' ERR

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

resume_one() {
  local target="$1" method="$2" gpu="$3" pool output
  pool="$SOURCE_RUN/downstream_target${target}kib_lpqld_dkr_seed0/fkd_pool"
  if [[ "$method" == "lpqld" ]]; then
    output="$SOURCE_RUN/downstream_target${target}kib_lpqld_dkr_seed0/seed0"
  else
    output="$SOURCE_RUN/downstream_target${target}kib_lpld_seed0"
  fi
  test -f "$pool/pool_summary.pt"
  test -f "$output/checkpoint.pt"

  echo "[$(date --iso-8601=seconds)] resume target=$target method=$method gpu=$gpu"
  extra_args=()
  if [[ "$method" == "lpqld" ]]; then
    extra_args+=(--dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 --dkr_min_temperature 2)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SOURCE_RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --eval_every 10 --seed 0 --device_ids 0 \
    --resume_checkpoint "$output/checkpoint.pt" "${extra_args[@]}" \
    >> "$output/train_pool.stdout.log" 2>&1
  touch "$output/complete"
  echo "[$(date --iso-8601=seconds)] complete target=$target method=$method gpu=$gpu"
}

# These two checkpoints already reached epoch 100 before the external stop.
touch "$SOURCE_RUN/downstream_target295kib_lpld_seed0/complete"
touch "$SOURCE_RUN/downstream_target330kib_lpld_seed0/complete"

(resume_one 295 lpqld "${gpu_array[0]}") & p0=$!
(resume_one 330 lpqld "${gpu_array[1]}") & p1=$!
(resume_one 445 lpqld "${gpu_array[2]}"; resume_one 445 lpld "${gpu_array[2]}") & p2=$!
(resume_one 610 lpqld "${gpu_array[3]}"; resume_one 610 lpld "${gpu_array[3]}") & p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]]
touch "$SOURCE_RUN/resume_lpld_lpqld_target_kib_complete"
echo "[$(date --iso-8601=seconds)] target-KiB resume complete"
