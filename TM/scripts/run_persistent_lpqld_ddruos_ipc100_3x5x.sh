#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
SOURCE_RUN="${1:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_ipc100/LPLD_DDRUOS_officialVectorized_IPC100_BN0p05_4GPU_10x20x30x40x_seed0_20260819_140653}"
GPU_IDS="${GPU_IDS:-2,3}"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"

if [[ "${#gpu_array[@]}" -ne 2 ]]; then
  echo "GPU_IDS must contain two comma-separated GPU IDs" >&2
  exit 2
fi

test -x "$PYTHON"
test -f "$SOURCE_RUN/synthetic.pt"
test -f "$SOURCE_RUN/label_codec_4000.pt"
test -f "$TEACHER"
test -d "$DATA_PATH/train"
test -d "$DATA_PATH/val"

LOG="$SOURCE_RUN/lpqld_downstream_3x5x.log"
exec >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] LPQLD 3x/5x pipeline start pid=$$"
echo "root=$ROOT source_run=$SOURCE_RUN gpu_ids=$GPU_IDS"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] failed status=$status"; exit "$status"' ERR

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

prune_ratio_for() {
  case "$1" in
    3) echo 0.6666666666666666 ;;
    # Avoid float-flooring 6240 expected batches to 6239.
    5) echo 0.7999999999999999 ;;
    *) return 2 ;;
  esac
}

run_one() {
  local compression="$1" gpu="$2" prune_ratio
  prune_ratio="$(prune_ratio_for "$compression")"
  local base="$SOURCE_RUN/downstream_${compression}x_lpqld_dkr_seed0"
  local pool="$base/fkd_pool" output="$base/seed0"
  mkdir -p "$pool" "$output"

  if [[ -f "$base/complete" ]]; then
    echo "[$(date --iso-8601=seconds)] ${compression}x already complete; skip"
    return 0
  fi

  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    echo "[$(date --iso-8601=seconds)] ${compression}x relabel start gpu=$gpu prune_ratio=$prune_ratio"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
      --mode relabel_pool --synthetic_path "$SOURCE_RUN/synthetic.pt" \
      --teacher_path "$TEACHER" --codec_checkpoint "$SOURCE_RUN/label_codec_4000.pt" \
      --fkd_path "$pool" --output_path "$base" --data_path "$DATA_PATH" \
      --dataset Tiny --epochs 100 --prune_ratio "$prune_ratio" \
      --crop_size 64 --min_crop_scale 0.08 --temperature 20 --disable_cutmix \
      --loader_batch 64 --workers 8 --fkd_seed 42 --seed 0 --device_ids 0 \
      > "$base/relabel_pool.stdout.log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] ${compression}x pool exists; skip relabel"
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
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] ${compression}x complete gpu=$gpu"
}

run_one 3 "${gpu_array[0]}" & pid3=$!
run_one 5 "${gpu_array[1]}" & pid5=$!

status=0
wait "$pid3" || status=1
wait "$pid5" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "3x or 5x pipeline failed" >&2
  exit 1
fi

touch "$SOURCE_RUN/lpqld_downstream_3x5x_complete"
echo "[$(date --iso-8601=seconds)] LPQLD 3x/5x pipeline complete"
