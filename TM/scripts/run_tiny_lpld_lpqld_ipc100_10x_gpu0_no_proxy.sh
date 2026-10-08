#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
LPLD_ROOT="${LPLD_ROOT:-/data2/home/ypliu/SRe2L/LPLD}"
LPQLD_ROOT="${LPQLD_ROOT:-/data2/home/ypliu/SRe2L/LPQLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE="${SOURCE:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
TEACHER="${TEACHER:-${SOURCE}/teacher/checkpoint_best.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_ipc100/Tiny_IPC100_LPLD_then_LPQLD_10x_gpu0_${STAMP}}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${ROOT}/results/tiny_imagenet_ipc100/official_lpld_ipc100}"
ARCHIVE="${DOWNLOAD_ROOT}/LPLD_tiny_rn18_4k_ipc100.tar.gz"
IMAGES="${DOWNLOAD_ROOT}/extracted/LPLD_tiny_rn18_4k_ipc100"
URL='https://drive.google.com/uc?export=download&id=1cQDD8OfMfoshsDIaiWOQb95pn2q9veuk&confirm=t'
PRUNE_RATIO=0.9

mkdir -p "${RUN_ROOT}"/{lpld,lpqld} "${DOWNLOAD_ROOT}/extracted"
cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=100
gpu=0
queue=LPLD_then_LPQLD
label_compression=10x
prune_ratio=${PRUNE_RATIO}
download_proxy=disabled
download_url=${URL}
EOF

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

until [[ -f "${ARCHIVE}" ]] && tar -tzf "${ARCHIVE}" >/dev/null 2>&1; do
  echo "$(date -Is) direct no-proxy IPC100 download attempt" \
    | tee -a "${RUN_ROOT}/download.log"
  rm -f "${ARCHIVE}.part"
  if curl --noproxy '*' -L --fail --connect-timeout 20 --max-time 1800 \
      --retry 2 -o "${ARCHIVE}.part" "${URL}" \
      >> "${RUN_ROOT}/download.log" 2>&1; then
    mv "${ARCHIVE}.part" "${ARCHIVE}"
  else
    rm -f "${ARCHIVE}.part"
    echo "$(date -Is) direct route unavailable; retrying in 300s" \
      | tee -a "${RUN_ROOT}/download.log"
    sleep 300
  fi
done

if [[ ! -d "${IMAGES}" ]]; then
  tar -xzf "${ARCHIVE}" -C "${DOWNLOAD_ROOT}/extracted"
fi
classes=$(find "${IMAGES}" -mindepth 1 -maxdepth 1 -type d | wc -l)
images=$(find "${IMAGES}" -type f -name '*.jpg' | wc -l)
[[ "${classes}" -eq 200 && "${images}" -eq 20000 ]] || {
  echo "Invalid IPC100 archive: classes=${classes}, images=${images}" >&2
  exit 1
}
touch "${RUN_ROOT}/download_complete"

run_lpld() {
  local out="${RUN_ROOT}/lpld"
  cd "${LPLD_ROOT}/relabel_and_validate"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u generate_soft_label_pruning.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
    --gpus 0 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u train_FKD_label_pruning_batch.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" \
    --granularity batch_to_epoch --prune_metric random --gpus 0 \
    --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
    --run_name lpld_tiny_ipc100_10x --exp_name tiny_ipc100_10x \
    > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpqld() {
  local out="${RUN_ROOT}/lpqld"
  cd "${LPQLD_ROOT}/relabel_and_validate"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u generate_soft_label_pruning_batch.py \
    --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
    --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
    --gpus 0 --prune_ratio "${PRUNE_RATIO}" > "${out}/relabel.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u train_FKD_LPQLD.py \
    --model resnet18 --prune_ratio "${PRUNE_RATIO}" --sample_metric random \
    --gpus 0 --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
    --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
    --run_name lpqld_tiny_ipc100_10x --exp_name tiny_ipc100_10x \
    --temperature 20 --temp_scheduler step --temp_step_size 30 \
    --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
  touch "${out}/complete"
}

run_lpld
run_lpqld
touch "${RUN_ROOT}/complete"
