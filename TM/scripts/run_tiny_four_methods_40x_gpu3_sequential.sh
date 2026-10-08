#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
LPQLD_ROOT="${LPQLD_ROOT:-/data2/home/ypliu/SRe2L/LPQLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
SRE2L_ROOT="${SRE2L_ROOT:-${ROOT}/results/tiny_imagenet_sre2l/Tiny_IPC50_SRe2L_bs50_bn0p05_parallel4_10x_20260809_111944}"
TEACHER="${TEACHER:-${SOURCE_ROOT}/teacher/checkpoint_best.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_40x/Tiny_IPC50_four_methods_40x_gpu3_${STAMP}}"
PRUNE_RATIO="0.975"

mkdir -p "${RUN_ROOT}"/{sre2l,cda,lpld,lpqld}
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=50
label_compression=40x
prune_ratio=${PRUNE_RATIO}
gpu3_queue=SRe2L_then_CDA_then_LPLD_then_LPQLD
downstream_epochs=100
batch_size=64
optimizer=SGD
learning_rate=0.2
momentum=0.9
weight_decay=0.0001
temperature=20
EOF

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

train_pruned_pool() {
  local method=$1 images=$2 labels=$3
  local out="${RUN_ROOT}/${method}"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 3 \
    --cfg_yaml "${CONFIG}" --fkd_path "${labels}" --train_dir "${images}" \
    --output_dir "${out}/downstream" \
    --run_name "${method}_tiny_ipc50_40x" --exp_name tiny_40x \
    > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpld() {
  local out="${RUN_ROOT}/lpld"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${SOURCE_ROOT}/lpld/images" \
    --gpus 3 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  train_pruned_pool lpld "${SOURCE_ROOT}/lpld/images" "${out}/labels"
}

run_lpqld() {
  local out="${RUN_ROOT}/lpqld"
  cd "${LPQLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning_batch.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${SOURCE_ROOT}/lpld/images" \
    --gpus 3 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  "${PYTHON}" -u train_FKD_LPQLD.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" --sample_metric random \
    --gpus 3 --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${SOURCE_ROOT}/lpld/images" --output_dir "${out}/downstream" \
    --run_name lpqld_tiny_ipc50_40x --exp_name tiny_40x \
    --temperature 20 --temp_scheduler step --temp_step_size 30 \
    --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

train_pruned_pool sre2l "${SRE2L_ROOT}/images" "${SRE2L_ROOT}/labels"
train_pruned_pool cda "${SOURCE_ROOT}/cda/images" "${SOURCE_ROOT}/cda/labels"
run_lpld
run_lpqld
touch "${RUN_ROOT}/complete"
