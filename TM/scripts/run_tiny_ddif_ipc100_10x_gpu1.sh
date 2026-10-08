#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON_DDRUO="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_ipc100/Tiny_IPC100_DDiF_width20_10x_gpu1_seed0_${STAMP}}"
DOWNSTREAM="${RUN_DIR}/downstream_fkd_10x_seed0"
LABELS="${DOWNSTREAM}/fkd_pool"

mkdir -p "${RUN_DIR}" "${DOWNSTREAM}"
cat > "${RUN_DIR}/run.conf" <<EOF
method=DDiF+SRe2L_utility
gpu=1
ipc=100
field_width=20
field_bits=32
image_kib_per_class=212.109375
label_compression=10x
expected_label_kib_per_class=432.542
downstream_optimizer=SGD
downstream_learning_rate=0.2
downstream_warmup_epochs=5
temperature=20
temperature_squared_scaling=true
EOF

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 "${PYTHON_DDRUO}" -u "${ROOT}/baselines/ddif_sre2l_tiny.py" \
  --dataset Tiny --data_path "${DATA_PATH}" --teacher_path "${TEACHER}" \
  --save_path "${RUN_DIR}" --ipc 100 --field_width 20 --field_bits 32 \
  --image_budget_kib 213 --expected_label_kib 432.542 \
  --init_iterations 2000 --init_pixels 4096 --init_field_chunk 5000 \
  --init_lr 5e-4 --iterations 1000 --field_lr 1e-4 --bn_weight 0.05 \
  --aug_mild_min_crop 0.5 --aug_full_min_crop 0.08 \
  --aug_mild_fraction 0.5 --aug_flip_probability 0.5 --aug_eval_trials 5 \
  --utility_slots_per_class 20 --field_chunk 2000 --export_chunk 512 \
  --log_every 10 --checkpoint_every 100 --seed 0 \
  > "${RUN_DIR}/stdout.log" 2>&1

CUDA_VISIBLE_DEVICES=1 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode relabel_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${LABELS}" \
  --output_path "${DOWNSTREAM}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs 100 --pool_compression 10 --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 256 --workers 12 --fkd_seed 42 \
  --seed 0 --device_ids 0 > "${DOWNSTREAM}/relabel.stdout.log" 2>&1

CUDA_VISIBLE_DEVICES=1 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode train_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${LABELS}" \
  --output_path "${DOWNSTREAM}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs 100 --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
  --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
  --warmup_epochs 5 --warmup_start_factor 0.01 \
  --scale_loss_by_temperature_squared --eval_every 10 --seed 0 --device_ids 0 \
  > "${DOWNSTREAM}/train.stdout.log" 2>&1

touch "${RUN_DIR}/complete"
