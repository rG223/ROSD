#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
RUN="${1:?usage: $0 RUN TARGETS_CSV GPU_IDS_CSV}"
TARGETS_CSV="${2:?usage: $0 RUN TARGETS_CSV GPU_IDS_CSV}"
GPU_IDS_CSV="${3:?usage: $0 RUN TARGETS_CSV GPU_IDS_CSV}"

IFS=',' read -r -a targets <<< "$TARGETS_CSV"
IFS=',' read -r -a gpus <<< "$GPU_IDS_CSV"
[[ "${#targets[@]}" -eq "${#gpus[@]}" ]] || {
  echo "TARGETS_CSV and GPU_IDS_CSV must have the same length" >&2
  exit 2
}

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

resume_target() {
  local target="$1" gpu="$2"
  local base="$RUN/downstream_target${target}kib_lpld_seed0"
  local pool="$base/fkd_pool" output="$base/seed0"
  local checkpoint="$output/checkpoint.pt"
  test -f "$pool/pool_summary.pt"
  test -f "$checkpoint"
  rm -f "$base/complete"
  echo "[$(date --iso-8601=seconds)] target=${target} cached resume start gpu=${gpu}" \
    >> "$RUN/pipeline.log"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --fast_downstream --cache_replay_batches --eval_every 10 --seed 0 \
    --device_ids 0 --resume_checkpoint "$checkpoint" \
    >> "$output/train_pool.stdout.log" 2>&1
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] target=${target} cached resume complete gpu=${gpu}" \
    >> "$RUN/pipeline.log"
}

pids=()
for index in "${!targets[@]}"; do
  resume_target "${targets[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]]
echo "[$(date --iso-8601=seconds)] cached target evaluations complete targets=$TARGETS_CSV" \
  >> "$RUN/pipeline.log"
