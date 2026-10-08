#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
SRE2L_ROOT="${SRE2L_ROOT:-/data2/home/ypliu/SRe2L/SRe2L/*small_dataset}"
CDA_ROOT="${CDA_ROOT:-/data2/home/ypliu/SRe2L/CDA}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE="${SOURCE:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
TEACHER="${TEACHER:-${SOURCE}/teacher/checkpoint_best.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_ipc100/Tiny_IPC100_SRe2L_then_CDA_10x_gpu2_${STAMP}}"
IPC=100
PRUNE_RATIO=0.9
SHARDS=4

mkdir -p "${RUN_ROOT}"/{sre2l,cda}
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=100
gpu=2
queue=SRe2L_then_CDA
recovery_iterations=4000
recovery_slot_shards=${SHARDS}
label_compression=10x
prune_ratio=${PRUNE_RATIO}
downstream_optimizer=SGD
downstream_learning_rate=0.2
temperature=20
EOF

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

recover_method() {
  local method=$1 out="${RUN_ROOT}/$1"
  mkdir -p "${out}/shards" "${out}/images" "${out}/logs"
  local pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    local start=$((shard * IPC / SHARDS))
    local end=$(((shard + 1) * IPC / SHARDS))
    local shard_root="${out}/shards/shard${shard}"
    mkdir -p "${shard_root}"
    if [[ "${method}" == "sre2l" ]]; then
      (
        cd "${SRE2L_ROOT}"
        CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -u recover_tiny.py \
          --arch-name resnet18 --arch-path "${TEACHER}" \
          --exp-name images --syn-data-path "${shard_root}" \
          --batch-size 50 --lr 0.1 --r-bn 0.05 --iteration 4000 \
          --store-last-images --ipc-start "${start}" --ipc-end "${end}" \
          > "${out}/logs/recover_shard${shard}.log" 2>&1
      ) &
    else
      (
        cd "${CDA_ROOT}"
        CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -u recover_cda_tiny.py \
          --arch-name resnet18 --arch-path "${TEACHER}" \
          --exp-name images --syn-data-path "${shard_root}" \
          --batch-size 100 --lr 0.4 --r-bn 0.05 --iteration 4000 \
          --store-best-images --easy2hard-mode cosine --milestone 1 \
          --ipc-start "${start}" --ipc-end "${end}" \
          > "${out}/logs/recover_shard${shard}.log" 2>&1
      ) &
    fi
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [[ ${status} -eq 0 ]] || return "${status}"
  for shard in $(seq 0 $((SHARDS - 1))); do
    cp -a "${out}/shards/shard${shard}/images/." "${out}/images/"
  done
  local count
  count=$(find "${out}/images" -type f -name '*.jpg' | wc -l)
  [[ "${count}" -eq 20000 ]] || {
    echo "${method}: expected 20000 images, found ${count}" >&2
    return 1
  }
  touch "${out}/recovery_complete"
}

relabel_and_train() {
  local method=$1 out="${RUN_ROOT}/$1"
  cd "${LPLD_ROOT}/relabel_and_validate"
  CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${out}/images" \
    --gpus 2 --prune_ratio 0.0 > "${out}/logs/relabel.log" 2>&1
  CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 2 \
    --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${out}/images" --output_dir "${out}/downstream_10x" \
    --run_name "${method}_tiny_ipc100_10x" --exp_name tiny_ipc100_10x \
    > "${out}/logs/train_10x.log" 2>&1
  touch "${out}/complete"
}

recover_method sre2l
relabel_and_train sre2l
recover_method cda
relabel_and_train cda
touch "${RUN_ROOT}/complete"
