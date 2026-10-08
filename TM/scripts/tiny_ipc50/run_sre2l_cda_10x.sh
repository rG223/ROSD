#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 sre2l|cda}"
case "${METHOD}" in sre2l|cda) ;; *) echo "METHOD must be sre2l or cda" >&2; exit 2 ;; esac

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_SRE2L="${PYTHON_SRE2L:-python}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?Set TEACHER to the upstream teacher checkpoint}"
LPLD_ROOT="${LPLD_ROOT:?Set LPLD_ROOT to he-y/soft-label-pruning-for-dataset-distillation}"
SRE2L_ROOT="${SRE2L_ROOT:-}"
CDA_ROOT="${CDA_ROOT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_baselines/${METHOD}_ipc50_10x_${STAMP}}"
CONFIG="${ROOT}/TM/configs/tiny_1x_fkd.yaml"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ${#GPUS[@]} -ne 4 ]]; then echo "GPU_IDS must contain four IDs" >&2; exit 2; fi
if [[ "${METHOD}" == sre2l ]]; then
  RECOVERY_ROOT="${SRE2L_ROOT:?Set SRE2L_ROOT for SRe2L recovery}"
else
  RECOVERY_ROOT="${CDA_ROOT:?Set CDA_ROOT for CDA recovery}"
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/shards" "${RUN_DIR}/images"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

recover_shard() {
  local local_index=$1 start=$2 end=$3 gpu=${GPUS[$1]}
  local shard="${RUN_DIR}/shards/gpu${local_index}"
  mkdir -p "${shard}"
  if [[ "${METHOD}" == sre2l ]]; then
    cd "${RECOVERY_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_SRE2L}" -u recover_tiny.py \
      --arch-name resnet18 --arch-path "${TEACHER}" --exp-name images \
      --syn-data-path "${shard}" --batch-size 50 --lr 0.1 --r-bn 0.05 \
      --iteration 4000 --store-last-images --ipc-start "${start}" --ipc-end "${end}" \
      > "${RUN_DIR}/logs/recover_gpu${local_index}.log" 2>&1
  else
    cd "${RECOVERY_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_SRE2L}" -u recover_cda_tiny.py \
      --arch-name resnet18 --arch-path "${TEACHER}" --exp-name images \
      --syn-data-path "${shard}" --batch-size 100 --lr 0.4 --r-bn 0.05 \
      --iteration 4000 --store-best-images --easy2hard-mode cosine --milestone 1 \
      --ipc-start "${start}" --ipc-end "${end}" \
      > "${RUN_DIR}/logs/recover_gpu${local_index}.log" 2>&1
  fi
}

recover_shard 0 0 13 & p0=$!
recover_shard 1 13 25 & p1=$!
recover_shard 2 25 38 & p2=$!
recover_shard 3 38 50 & p3=$!
status=0
wait "${p0}" || status=1; wait "${p1}" || status=1
wait "${p2}" || status=1; wait "${p3}" || status=1
if [[ ${status} -ne 0 ]]; then exit ${status}; fi

for index in 0 1 2 3; do
  cp -a "${RUN_DIR}/shards/gpu${index}/images/." "${RUN_DIR}/images/"
done

cd "${LPLD_ROOT}/relabel_and_validate"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_SRE2L}" -u generate_soft_label_pruning.py \
  --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
  --fkd_path "${RUN_DIR}/labels" --train_dir "${RUN_DIR}/images" \
  --gpus 0 --prune_ratio 0.0 > "${RUN_DIR}/logs/relabel.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_SRE2L}" -u train_FKD_label_pruning_batch.py \
  --model resnet18 --prune_ratio 0.9 --granularity batch_to_epoch \
  --prune_metric random --gpus 0 --cfg_yaml "${CONFIG}" \
  --fkd_path "${RUN_DIR}/labels" --train_dir "${RUN_DIR}/images" \
  --output_dir "${RUN_DIR}/downstream" --run_name "${METHOD}_tiny_ipc50_10x" \
  --exp_name tiny_10x > "${RUN_DIR}/logs/train.log" 2>&1

touch "${RUN_DIR}/.complete"
