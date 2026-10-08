#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON")/torchrun}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/SRe2L_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260826_124447}"
BASE="$RUN/downstream_official_lpqld_300e_seed0"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter4000.pt"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-0}"
OUTPUT="$BASE/train_target1160"
POOL="$BASE/fkd_pool_target1160"
CHECKPOINT="${RESUME_CHECKPOINT:-$OUTPUT/checkpoint.pt}"
STDOUT_LOG="${STDOUT_LOG:-$OUTPUT/resume_pool_ddp.stdout.log}"

for required in \
  "$TORCHRUN" "$SYNTHETIC" "$TEACHER" \
  "$POOL/pool_summary.pt" "$CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 1
  fi
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if [[ "${WARM_SYNTHETIC_CACHE:-1}" == "1" ]]; then
  # The 29 GiB tensor stays mmap-backed in every rank. A single sequential
  # pass is substantially cheaper than four ranks faulting random image pages
  # during the first epoch and does not create another tensor copy.
  dd if="$SYNTHETIC" of=/dev/null bs=64M status=none
fi

"$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
  TM/sre2l_fkd_ddp.py \
  --mode train_pool --synthetic_path "$SYNTHETIC" \
  --teacher_path "$TEACHER" --data_path "$DATA_PATH" \
  --dataset ImageNet1K --epochs 300 --crop_size 224 \
  --temperature 20 --train_batch 128 --workers "$WORKERS_PER_RANK" \
  --optimizer adamw --learning_rate 0.001 --weight_decay 0.01 \
  --eval_every 10 --seed 0 --device_ids 0 1 2 3 \
  --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
  --dkr_min_temperature 2 --ca_dynamic --fast_downstream \
  --fkd_path "$POOL" --output_path "$OUTPUT" \
  --resume_checkpoint "$CHECKPOINT" \
  >"$STDOUT_LOG" 2>&1
