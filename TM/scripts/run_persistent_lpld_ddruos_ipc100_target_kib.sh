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

LOG="$SOURCE_RUN/lpld_downstream_target295_330_445_610.log"
exec >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] LPLD target-KiB downstream start pid=$$"
echo "root=$ROOT source_run=$SOURCE_RUN gpu_ids=$GPU_IDS"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] failed status=$status"; exit "$status"' ERR

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

compression_for() {
  case "$1" in
    295) echo 7.240659 ;;
    330) echo 6.052376 ;;
    445) echo 3.929471 ;;
    610) echo 2.614160 ;;
    *) return 2 ;;
  esac
}

train_one() {
  local target="$1" gpu="$2" compression pool output
  compression="$(compression_for "$target")"
  pool="$SOURCE_RUN/downstream_target${target}kib_lpqld_dkr_seed0/fkd_pool"
  output="$SOURCE_RUN/downstream_target${target}kib_lpld_seed0"
  mkdir -p "$output"

  if [[ -f "$output/complete" ]]; then
    echo "[$(date --iso-8601=seconds)] target=${target} already complete; skip"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] target=${target} waiting for shared ${compression}x FKD pool"
  for _ in $(seq 1 240); do
    [[ -f "$pool/pool_summary.pt" ]] && break
    sleep 30
  done
  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    echo "target=${target} FKD pool was not ready within two hours" >&2
    return 1
  fi

  echo "[$(date --iso-8601=seconds)] target=${target} LPLD train start gpu=$gpu compression=${compression}x"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SOURCE_RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --eval_every 10 --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$output/complete"
  echo "[$(date --iso-8601=seconds)] target=${target} complete gpu=$gpu"
}

pids=()
targets=(295 330 445 610)
for index in 0 1 2 3; do
  train_one "${targets[$index]}" "${gpu_array[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more LPLD target-KiB runs failed" >&2
  exit 1
fi

touch "$SOURCE_RUN/lpld_downstream_target295_330_445_610_complete"
echo "[$(date --iso-8601=seconds)] LPLD target-KiB downstream complete"
