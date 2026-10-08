#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
SRE2L_ROOT="${SRE2L_ROOT:-/data2/home/ypliu/SRe2L}"
LPLD_ROOT="${LPLD_ROOT:-${SRE2L_ROOT}/LPLD}"
LPQLD_ROOT="${LPQLD_ROOT:-${SRE2L_ROOT}/LPQLD}"
CONFIG="${CONFIG:-${ROOT}/TM/configs/tiny_1x_fkd.yaml}"
SOURCE="${SOURCE:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102}"
TEACHER="${TEACHER:-${SOURCE}/teacher/checkpoint_best.pth}"
CLASS_TEACHER="${CLASS_TEACHER:-${SOURCE}/class_bn/resnet18_tiny_0.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_ipc200/Tiny_IPC200_LPLD_LPQLD_10-40x_gpu01_${STAMP}}"
IMAGES="${RUN_ROOT}/images"
IPC=200
CLASSES=200
RECOVERY_WORKERS_PER_GPU="${RECOVERY_WORKERS_PER_GPU:-4}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/recover_logs" "${IMAGES}"
exec >> "${RUN_ROOT}/pipeline.log" 2>&1

echo "[$(date --iso-8601=seconds)] IPC200 LPLD/LPQLD pipeline start pid=$$"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] pipeline failed status=$status"; touch "${RUN_ROOT}/pipeline_failed"; exit "$status"' ERR
rm -f "${RUN_ROOT}/pipeline_failed"

test -x "${PYTHON}"
test -f "${CONFIG}"
test -f "${TEACHER}"
test -f "${CLASS_TEACHER}"
test -f "${LPLD_ROOT}/recover/data_synthesis_tiny_class.py"

cat > "${RUN_ROOT}/settings.txt" <<EOF
dataset=Tiny-ImageNet
ipc=200
images=shared_official_LPLD_recovery
recovery=CE_plus_BN0.05_firstBN10_RRC_flip_jitter4_4000iter
recovery_batch_size=200
recovery_workers_per_gpu=${RECOVERY_WORKERS_PER_GPU}
gpu0=LPLD_10x_20x_30x_40x
gpu1=LPQLD_10x_20x_30x_40x
gpu2_gpu3=unused
prune_ratios=0.9,0.95,0.97,0.975
downstream_epochs=100
optimizer=SGD_lr0.2_momentum0.9_weight_decay1e-4
temperature=20
lpqld_DKR=step30_gamma0.7
EOF

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

recover_images() {
  if [[ -f "${RUN_ROOT}/recovery_complete" ]]; then
    echo "[$(date --iso-8601=seconds)] recovery already complete; skip"
    return
  fi

  echo "[$(date --iso-8601=seconds)] recovery start: 8 shards on GPUs 0/1"
  local total_workers=$((2 * RECOVERY_WORKERS_PER_GPU))
  local classes_per_worker=$((CLASSES / total_workers))
  local pids=()
  for ((worker = 0; worker < total_workers; worker++)); do
    local gpu=$((worker / RECOVERY_WORKERS_PER_GPU))
    local start=$((worker * classes_per_worker))
    local end=$((start + classes_per_worker))
    if [[ "${worker}" -eq $((total_workers - 1)) ]]; then
      end="${CLASSES}"
    fi
    echo "[$(date --iso-8601=seconds)] worker=${worker} gpu=${gpu} classes=[${start},${end})"
    (
      cd "${LPLD_ROOT}/recover"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u data_synthesis_tiny_class.py \
        --arch-name resnet18 --arch-path "${CLASS_TEACHER}" \
        --syn-data-path "${RUN_ROOT}" --exp-name images \
        --ipc "${IPC}" --batch-size "${IPC}" --iteration 4000 \
        --ipc-start "${start}" --ipc-end "${end}" \
        --lr 0.1 --jitter 4 --r-bn 0.05 --first-bn-multiplier 10 \
        --bn-hook-type class_stats_training --store-last-images \
        > "${RUN_ROOT}/recover_logs/worker${worker}_gpu${gpu}.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ "${failed}" -eq 0 ]]

  local class_count image_count
  class_count=$(find "${IMAGES}" -mindepth 1 -maxdepth 1 -type d | wc -l)
  image_count=$(find "${IMAGES}" -type f -name '*.jpg' | wc -l)
  [[ "${class_count}" -eq "${CLASSES}" && "${image_count}" -eq $((CLASSES * IPC)) ]] || {
    echo "invalid recovery output: classes=${class_count}, images=${image_count}" >&2
    return 1
  }
  touch "${RUN_ROOT}/recovery_complete"
  echo "[$(date --iso-8601=seconds)] recovery complete classes=${class_count} images=${image_count}"
}

prune_ratio_for() {
  case "$1" in
    10) echo 0.9 ;;
    20) echo 0.95 ;;
    30) echo 0.97 ;;
    40) echo 0.975 ;;
    *) return 2 ;;
  esac
}

run_lpld_queue() {
  for compression in 10 20 30 40; do
    local ratio out
    ratio=$(prune_ratio_for "${compression}")
    out="${RUN_ROOT}/lpld_${compression}x"
    mkdir -p "${out}"
    if [[ -f "${out}/complete" ]]; then continue; fi
    echo "[$(date --iso-8601=seconds)] LPLD ${compression}x relabel start gpu=0"
    cd "${LPLD_ROOT}/relabel_and_validate"
    "${PYTHON}" -u generate_soft_label_pruning.py \
      --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
      --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
      --gpus 0 --prune_ratio "${ratio}" > "${out}/relabel.log" 2>&1
    echo "[$(date --iso-8601=seconds)] LPLD ${compression}x train start gpu=0"
    "${PYTHON}" -u train_FKD_label_pruning_batch.py \
      --model resnet18 --prune_ratio "${ratio}" \
      --granularity batch_to_epoch --prune_metric random --gpus 0 \
      --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
      --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
      --run_name "lpld_tiny_ipc200_${compression}x" \
      --exp_name "tiny_ipc200_${compression}x" > "${out}/train.log" 2>&1
    touch "${out}/complete"
    echo "[$(date --iso-8601=seconds)] LPLD ${compression}x complete"
  done
  touch "${RUN_ROOT}/lpld_complete"
}

run_lpqld_queue() {
  for compression in 10 20 30 40; do
    local ratio out
    ratio=$(prune_ratio_for "${compression}")
    out="${RUN_ROOT}/lpqld_${compression}x"
    mkdir -p "${out}"
    if [[ -f "${out}/complete" ]]; then continue; fi
    echo "[$(date --iso-8601=seconds)] LPQLD ${compression}x relabel start gpu=1"
    cd "${LPQLD_ROOT}/relabel_and_validate"
    "${PYTHON}" -u generate_soft_label_pruning_batch.py \
      --cfg_yaml "${CONFIG}" --teacher_ckpt "${TEACHER}" \
      --fkd_path "${out}/labels" --train_dir "${IMAGES}" \
      --gpus 1 --prune_ratio "${ratio}" > "${out}/relabel.log" 2>&1
    echo "[$(date --iso-8601=seconds)] LPQLD ${compression}x train start gpu=1"
    "${PYTHON}" -u train_FKD_LPQLD.py \
      --model resnet18 --prune_ratio "${ratio}" --sample_metric random \
      --gpus 1 --cfg_yaml "${CONFIG}" --fkd_path "${out}/labels" \
      --train_dir "${IMAGES}" --output_dir "${out}/downstream" \
      --run_name "lpqld_tiny_ipc200_${compression}x" \
      --exp_name "tiny_ipc200_${compression}x" \
      --temperature 20 --temp_scheduler step --temp_step_size 30 \
      --temp_step_gamma 0.7 > "${out}/train.log" 2>&1
    touch "${out}/complete"
    echo "[$(date --iso-8601=seconds)] LPQLD ${compression}x complete"
  done
  touch "${RUN_ROOT}/lpqld_complete"
}

recover_images
echo "[$(date --iso-8601=seconds)] downstream queues start"
run_lpld_queue & lpld_pid=$!
run_lpqld_queue & lpqld_pid=$!
wait "${lpld_pid}"
wait "${lpqld_pid}"
touch "${RUN_ROOT}/pipeline_complete"
echo "[$(date --iso-8601=seconds)] pipeline complete"
