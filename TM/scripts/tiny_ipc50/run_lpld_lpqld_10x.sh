#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_SRE2L="${PYTHON_SRE2L:-python}"
LPLD_ROOT="${LPLD_ROOT:?Set LPLD_ROOT to the LPLD repository}"
LPQLD_ROOT="${LPQLD_ROOT:?Set LPQLD_ROOT to the LPQLD repository}"
IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR to the recovered LPLD images}"
TEACHER="${TEACHER:?Set TEACHER to the Tiny-ImageNet teacher checkpoint}"
GPU_LPLD="${GPU_LPLD:-2}"
GPU_LPQLD="${GPU_LPQLD:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_baselines/lpld_lpqld_ipc50_10x_${STAMP}}"
CONFIG="${ROOT}/TM/configs/tiny_1x_fkd.yaml"

mkdir -p "${RUN_DIR}/lpld" "${RUN_DIR}/lpqld"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

run_lpld() {
  local out="${RUN_DIR}/lpld"
  cd "${LPLD_ROOT}/relabel_and_validate"
  CUDA_VISIBLE_DEVICES="${GPU_LPLD}" "${PYTHON_SRE2L}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGE_DIR}" \
    --gpus 0 --prune_ratio 0.9 > "${out}/relabel.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_LPLD}" "${PYTHON_SRE2L}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio 0.9 --granularity batch_to_epoch \
    --prune_metric random --gpus 0 --cfg_yaml "${CONFIG}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGE_DIR}" \
    --output_dir "${out}/downstream" --run_name lpld_tiny_ipc50_10x \
    --exp_name tiny_10x > "${out}/train.log" 2>&1
}

run_lpqld() {
  local out="${RUN_DIR}/lpqld"
  cd "${LPQLD_ROOT}/relabel_and_validate"
  CUDA_VISIBLE_DEVICES="${GPU_LPQLD}" "${PYTHON_SRE2L}" -u generate_soft_label_pruning_batch.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGE_DIR}" \
    --gpus 0 --prune_ratio 0.9 > "${out}/relabel.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_LPQLD}" "${PYTHON_SRE2L}" -u train_FKD_LPQLD.py \
    --model resnet18 --prune_ratio 0.9 --sample_metric random \
    --gpus 0 --cfg_yaml "${CONFIG}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGE_DIR}" \
    --output_dir "${out}/downstream" --run_name lpqld_tiny_ipc50_10x \
    --exp_name tiny_10x --temperature 20 --temp_scheduler step \
    --temp_step_size 30 --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
}

run_lpld & p0=$!
run_lpqld & p1=$!
status=0
wait "${p0}" || status=1; wait "${p1}" || status=1
exit "${status}"
