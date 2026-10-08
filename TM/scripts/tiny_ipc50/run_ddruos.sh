#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_DDRUO="${PYTHON_DDRUO:-python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-python}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?Set TEACHER to the ResNet18-BN checkpoint}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_ddruo/Tiny_IPC50_DDRUOS_noLabelKL_seed${SEED}_${STAMP}}"

IPC=50
CIM_FACTOR=2
CIM_MIPC=300
IMAGE_BUDGET_KIB=90
LABEL_BUDGET_KIB=210
IMAGE_MODEL_OVERHEAD_KIB=9.66
LABEL_GROUPS=30
LABEL_STEP=0.35
JOINT_ITERS=400

test -d "${DATA_PATH}/train"
test -f "${TEACHER}"
mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/run.conf" <<EOF
method=ROSD
dataset=Tiny-ImageNet
ipc=${IPC}
cim_factor=${CIM_FACTOR}
cim_candidate_ipc=${CIM_MIPC}
image_budget_kib_per_class=${IMAGE_BUDGET_KIB}
label_budget_kib_per_class=${LABEL_BUDGET_KIB}
label_groups=${LABEL_GROUPS}
label_step=${LABEL_STEP}
hard_ce_weight=0
label_kl_weight=0
joint_iterations=${JOINT_ITERS}
seed=${SEED}
EOF

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_DDRUO}" -u "${ROOT}/TM/cim_ddruo_tensorpool.py" \
  --dataset Tiny --data_path "${DATA_PATH}" --ipc "${IPC}" \
  --teacher_path "${TEACHER}" --save_path "${RUN_DIR}" \
  --cim_factor "${CIM_FACTOR}" --cim_mipc "${CIM_MIPC}" \
  --cim_class_batch 12 --cim_selection_batch 1024 \
  --cim_selection_workers 12 --cim_augmentation crop_cutout_flip \
  --optimize_label_rate --label_step "${LABEL_STEP}" \
  --label_groups "${LABEL_GROUPS}" --label_entropy_lr 0.001 \
  --label_entropy_warmup 20 --label_feature_chunk 1536 \
  --label_entropy_batch 128 --label_model_bits 16 \
  --hard_ce_weight 0 --label_kl_weight 0 \
  --separate_rate_budgets --image_target_kib "${IMAGE_BUDGET_KIB}" \
  --label_target_kib "${LABEL_BUDGET_KIB}" \
  --image_model_overhead_kib "${IMAGE_MODEL_OVERHEAD_KIB}" \
  --image_dual_lr 0.0001 --label_dual_lr 0.0001 \
  --warmup_rate_control dual --warmup_target_kib "${IMAGE_BUDGET_KIB}" \
  --warmup_rate_margin 1 --warmup_dual_lr 0.001 \
  --fast_single_warmup --fast_warmup_iterations 200 \
  --ldb 0.1 --codec_lr 0.001 --lr_it 1000 --rate_control dual \
  --latent_target_kib 80.34 --stage1_iterations "${JOINT_ITERS}" \
  --stage2_iterations 0 --layers_v v5 --arm 32 --dim 4 \
  --log_every 5 --checkpoint_every 100 --network_mse_threshold 5e-7 \
  --enable_cudnn --enable_codec_scheduler --seed "${SEED}" \
  > "${RUN_DIR}/stdout.log" 2>&1

ROOT="${ROOT}" PYTHON_SRE2L="${PYTHON_SRE2L}" DATA_PATH="${DATA_PATH}" \
TEACHER="${TEACHER}" RUN_DIR="${RUN_DIR}" GPU="${GPU}" SEED="${SEED}" \
LABEL_GROUPS="${LABEL_GROUPS}" LABEL_CODEC="${RUN_DIR}/label_codec_${JOINT_ITERS}.pt" \
bash "${ROOT}/TM/scripts/tiny_ipc50/eval_ddruos_fkd.sh"

touch "${RUN_DIR}/.complete"
printf 'complete run_dir=%s\n' "${RUN_DIR}"
