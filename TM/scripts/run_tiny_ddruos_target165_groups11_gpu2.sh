#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON_DDRUO="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/tiny_imagenet_ddruo/Tiny_IPC50_DDRUOS_groups11_step0p35_sepX90_Y75_total165_joint400_gpu2_seed0_${STAMP}}"
DOWNSTREAM="${RUN_DIR}/downstream_fkd_groups11_seed0"
LABELS="${DOWNSTREAM}/fkd_pool"

mkdir -p "${RUN_DIR}" "${DOWNSTREAM}"
cat > "${RUN_DIR}/run.conf" <<EOF
method=ROSD
dataset=Tiny-ImageNet
ipc=50
cim_factor=2
image_target_kib_per_class=90
label_target_kib_per_class=75
target_total_kib_per_class=165
label_groups=11
label_step=0.35
label_pool_compression=9.09091x
hard_ce_weight=0
label_kl_weight=0
joint_iterations=400
gpu=2
EOF

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=2 "${PYTHON_DDRUO}" -u "${ROOT}/TM/cim_ddruo_tensorpool.py" \
  --dataset Tiny --data_path "${DATA_PATH}" --ipc 50 \
  --teacher_path "${TEACHER}" --save_path "${RUN_DIR}" \
  --cim_factor 2 --cim_mipc 300 --cim_class_batch 12 \
  --cim_selection_batch 1024 --cim_selection_workers 12 \
  --cim_augmentation crop_cutout_flip --optimize_label_rate \
  --label_step 0.35 --label_groups 11 --label_entropy_lr 0.001 \
  --label_entropy_warmup 20 --label_feature_chunk 1536 \
  --label_entropy_batch 128 --label_model_bits 16 \
  --hard_ce_weight 0 --label_kl_weight 0 --separate_rate_budgets \
  --image_target_kib 90 --label_target_kib 75 \
  --image_model_overhead_kib 9.66 --image_dual_lr 0.0001 \
  --label_dual_lr 0.0001 --warmup_rate_control dual \
  --warmup_target_kib 90 --warmup_rate_margin 1 --warmup_dual_lr 0.001 \
  --fast_single_warmup --fast_warmup_iterations 200 --ldb 0.1 \
  --codec_lr 0.001 --lr_it 1000 --rate_control dual \
  --latent_target_kib 80.34 --stage1_iterations 400 --stage2_iterations 0 \
  --layers_v v5 --arm 32 --dim 4 --log_every 5 --checkpoint_every 100 \
  --network_mse_threshold 5e-7 --enable_cudnn --enable_codec_scheduler --seed 0 \
  > "${RUN_DIR}/stdout.log" 2>&1

CUDA_VISIBLE_DEVICES=2 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode relabel_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --codec_checkpoint "${RUN_DIR}/label_codec_400.pt" \
  --fkd_path "${LABELS}" --output_path "${DOWNSTREAM}" \
  --data_path "${DATA_PATH}" --dataset Tiny --epochs 100 \
  --pool_compression 9.09091 --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 128 --workers 12 --fkd_seed 42 \
  --seed 0 --device_ids 0 > "${DOWNSTREAM}/relabel.stdout.log" 2>&1

CUDA_VISIBLE_DEVICES=2 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode train_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${LABELS}" \
  --output_path "${DOWNSTREAM}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs 100 --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
  --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
  --warmup_epochs 5 --warmup_start_factor 0.01 \
  --scale_loss_by_temperature_squared --eval_every 10 --seed 0 --device_ids 0 \
  > "${DOWNSTREAM}/train.stdout.log" 2>&1

touch "${RUN_DIR}/complete"
