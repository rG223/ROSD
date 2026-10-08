#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
SRE2L_ROOT="${SRE2L_ROOT:-${ROOT}/results/tiny_imagenet_sre2l/Tiny_IPC50_SRe2L_bs50_bn0p05_parallel4_10x_20260809_111944}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_30x/Tiny_IPC50_CDA_then_SRe2L_30x_gpu3_${STAMP}}"
PRUNE_RATIO="0.9666666667"

mkdir -p "${RUN_ROOT}"/{cda,sre2l}
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=50
label_compression=30x
prune_ratio=${PRUNE_RATIO}
gpu3_queue=CDA_then_SRe2L
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

run_method() {
  local method=$1 images=$2 labels=$3
  local out="${RUN_ROOT}/${method}"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 3 \
    --cfg_yaml "${CONFIG}" --fkd_path "${labels}" --train_dir "${images}" \
    --output_dir "${out}/downstream" \
    --run_name "${method}_tiny_ipc50_30x" --exp_name tiny_30x \
    > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_method cda "${SOURCE_ROOT}/cda/images" "${SOURCE_ROOT}/cda/labels"
run_method sre2l "${SRE2L_ROOT}/images" "${SRE2L_ROOT}/labels"
touch "${RUN_ROOT}/complete"
