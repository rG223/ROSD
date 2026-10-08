#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_DDRUO="${PYTHON_DDRUO:-python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-python}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?Set TEACHER to the ResNet18-BN checkpoint}"
GPU="${GPU:-1}"
SEED="${SEED:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_ddif/Tiny_IPC50_DDiF_SRe2L_seed${SEED}_${STAMP}}"
OUTPUT="${RUN_DIR}/downstream_fkd_10x_seed${SEED}"
LABELS="${OUTPUT}/fkd_pool"

test -d "${DATA_PATH}/train"
test -f "${TEACHER}"
mkdir -p "${RUN_DIR}" "${OUTPUT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

"${PYTHON_DDRUO}" -u "${ROOT}/baselines/ddif_sre2l_tiny.py" \
  --dataset Tiny --data_path "${DATA_PATH}" --teacher_path "${TEACHER}" \
  --save_path "${RUN_DIR}" --ipc 50 --field_width 18 --field_bits 32 \
  --image_budget_kib 88.54 --expected_label_kib 221.46 \
  --init_iterations 2000 --init_pixels 4096 --init_field_chunk 5000 \
  --init_lr 5e-4 --iterations 1000 --field_lr 1e-4 --bn_weight 0.05 \
  --aug_mild_min_crop 0.5 --aug_full_min_crop 0.08 \
  --aug_mild_fraction 0.5 --aug_flip_probability 0.5 --aug_eval_trials 5 \
  --utility_slots_per_class 20 --field_chunk 2000 --export_chunk 512 \
  --log_every 10 --checkpoint_every 100 --seed "${SEED}" \
  > "${RUN_DIR}/stdout.log" 2>&1

"${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode relabel_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${LABELS}" \
  --output_path "${OUTPUT}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs 100 --pool_compression 10 --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 256 --workers 12 --fkd_seed 42 \
  --seed "${SEED}" --device_ids 0 > "${OUTPUT}/relabel.stdout.log" 2>&1

# These are the exact downstream settings used by the reported 33.11% run.
"${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode train_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${LABELS}" \
  --output_path "${OUTPUT}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs 100 --crop_size 64 --temperature 20 --train_batch 512 \
  --workers 12 --optimizer adamw --learning_rate 0.001 \
  --weight_decay 0.01 --eval_every 10 --seed "${SEED}" --device_ids 0 \
  > "${OUTPUT}/train.stdout.log" 2>&1

touch "${RUN_DIR}/.complete"
