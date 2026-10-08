#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
LPQLD_ROOT="${LPQLD_ROOT:-/data2/home/ypliu/SRe2L/LPQLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
TEACHER="${TEACHER:-${SOURCE_ROOT}/teacher/checkpoint_best.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_30x/Tiny_IPC50_CDA_LPLD_LPQLD_30x_${STAMP}}"
PRUNE_RATIO="0.9666666667"

mkdir -p "${RUN_ROOT}"/{cda,lpld,lpqld}
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=50
label_compression=30x
prune_ratio=${PRUNE_RATIO}
gpu2_queue=LPLD_then_LPQLD
gpu3=CDA
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

run_lpld() {
  local out="${RUN_ROOT}/lpld"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${SOURCE_ROOT}/lpld/images" \
    --gpus 2 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 2 \
    --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${SOURCE_ROOT}/lpld/images" --output_dir "${out}/downstream" \
    --run_name lpld_tiny_ipc50_30x --exp_name tiny_30x \
    > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpqld() {
  local out="${RUN_ROOT}/lpqld"
  cd "${LPQLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning_batch.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${SOURCE_ROOT}/lpld/images" \
    --gpus 2 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  "${PYTHON}" -u train_FKD_LPQLD.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" --sample_metric random \
    --gpus 2 --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${SOURCE_ROOT}/lpld/images" --output_dir "${out}/downstream" \
    --run_name lpqld_tiny_ipc50_30x --exp_name tiny_30x \
    --temperature 20 --temp_scheduler step --temp_step_size 30 \
    --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_gpu2_queue() {
  run_lpld
  run_lpqld
}

run_cda() {
  local out="${RUN_ROOT}/cda"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 3 \
    --cfg_yaml "${CONFIG}" --fkd_path "${SOURCE_ROOT}/cda/labels" \
    --train_dir "${SOURCE_ROOT}/cda/images" --output_dir "${out}/downstream" \
    --run_name cda_tiny_ipc50_30x --exp_name tiny_30x \
    > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_gpu2_queue & p2=$!
run_cda & p3=$!
printf 'run_root=%s gpu2_queue_pid=%s cda_pid=%s\n' "${RUN_ROOT}" "${p2}" "${p3}" \
  | tee "${RUN_ROOT}/pids.txt"

status=0
wait "${p2}" || status=1
wait "${p3}" || status=1
printf 'exit_status=%s\n' "${status}" | tee "${RUN_ROOT}/finished.txt"
exit "${status}"
