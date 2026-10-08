#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
LPQLD_ROOT="${LPQLD_ROOT:-/data2/home/ypliu/SRe2L/LPQLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE="${SOURCE:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
TEACHER="${TEACHER:-${SOURCE}/teacher/checkpoint_best.pth}"
IMAGES="${IMAGES:-${ROOT}/results/tiny_imagenet_ipc100/official_lpld_ipc100/extracted/LPLD_tiny_rn18_4k_ipc100}"
GPU="${GPU:?Set GPU to the physical GPU index}"
COMPRESSION="${COMPRESSION:?Set COMPRESSION to 20 or 30}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_ipc100/Tiny_IPC100_LPLD_then_LPQLD_${COMPRESSION}x_gpu${GPU}_${STAMP}}"

case "${COMPRESSION}" in
  20) PRUNE_RATIO="0.95" ;;
  30) PRUNE_RATIO="0.9666666667" ;;
  *) echo "Unsupported COMPRESSION=${COMPRESSION}; expected 20 or 30" >&2; exit 2 ;;
esac

test -f "${TEACHER}"
test -d "${IMAGES}"
classes=$(find "${IMAGES}" -mindepth 1 -maxdepth 1 -type d | wc -l)
images=$(find "${IMAGES}" -type f -name '*.jpg' | wc -l)
[[ "${classes}" -eq 200 && "${images}" -eq 20000 ]] || {
  echo "Invalid IPC100 image set: classes=${classes}, images=${images}" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}"/{lpld,lpqld}
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=100
gpu=${GPU}
queue=LPLD_then_LPQLD
label_compression=${COMPRESSION}x
prune_ratio=${PRUNE_RATIO}
downstream_epochs=100
optimizer=SGD
learning_rate=0.2
momentum=0.9
weight_decay=0.0001
temperature=20
images=${IMAGES}
EOF

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

run_lpld() {
  local out="${RUN_ROOT}/lpld"
  cd "${LPLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
    --gpus "${GPU}" --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus "${GPU}" \
    --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
    --run_name "lpld_tiny_ipc100_${COMPRESSION}x" \
    --exp_name "tiny_ipc100_${COMPRESSION}x" > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpqld() {
  local out="${RUN_ROOT}/lpqld"
  cd "${LPQLD_ROOT}/relabel_and_validate"
  "${PYTHON}" -u generate_soft_label_pruning_batch.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
    --gpus "${GPU}" --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  "${PYTHON}" -u train_FKD_LPQLD.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" --sample_metric random \
    --gpus "${GPU}" --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
    --run_name "lpqld_tiny_ipc100_${COMPRESSION}x" \
    --exp_name "tiny_ipc100_${COMPRESSION}x" \
    --temperature 20 --temp_scheduler step --temp_step_size 30 \
    --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpld
run_lpqld
touch "${RUN_ROOT}/complete"
printf 'complete run_root=%s\n' "${RUN_ROOT}"
