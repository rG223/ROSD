#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON_DDRUO="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiny_imagenet_target210/Tiny_IPC50_DDiF_vs_DDRUOS_${STAMP}}"

mkdir -p "${RUN_ROOT}/ddif" "${RUN_ROOT}/ddruos"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_official_fkd() {
  local gpu=$1 synthetic=$2 labels=$3 output=$4
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
    --mode train_pool --synthetic_path "${synthetic}" \
    --teacher_path "${TEACHER}" --fkd_path "${labels}" \
    --output_path "${output}" --data_path "${DATA_PATH}" --dataset Tiny \
    --epochs 100 --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
    --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --eval_every 10 --seed 0 --device_ids 0 \
    > "${output}/train.stdout.log" 2>&1
}

run_ddif() {
  local out="${RUN_ROOT}/ddif"
  local downstream="${out}/downstream_fkd_20x_seed0"
  local labels="${downstream}/fkd_pool"
  mkdir -p "${downstream}"
  cat > "${out}/run.conf" <<EOF
method=DDiF+SRe2L_utility
ipc=50
field_width=20
field_bits=32
image_kib_per_class=106.0546875
label_groups=5
pool_compression=20x
expected_label_kib_per_class=108.14
expected_total_kib_per_class=214.19
EOF

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_DDRUO}" -u "${ROOT}/baselines/ddif_sre2l_tiny.py" \
    --dataset Tiny --data_path "${DATA_PATH}" --teacher_path "${TEACHER}" \
    --save_path "${out}" --ipc 50 --field_width 20 --field_bits 32 \
    --image_budget_kib 107 --expected_label_kib 108.14 \
    --init_iterations 2000 --init_pixels 4096 --init_field_chunk 5000 \
    --init_lr 5e-4 --iterations 1000 --field_lr 1e-4 --bn_weight 0.05 \
    --aug_mild_min_crop 0.5 --aug_full_min_crop 0.08 \
    --aug_mild_fraction 0.5 --aug_flip_probability 0.5 --aug_eval_trials 5 \
    --utility_slots_per_class 20 --field_chunk 2000 --export_chunk 512 \
    --log_every 10 --checkpoint_every 100 --seed 0 > "${out}/stdout.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
    --mode relabel_pool --synthetic_path "${out}/synthetic.pt" \
    --teacher_path "${TEACHER}" --fkd_path "${labels}" \
    --output_path "${downstream}" --data_path "${DATA_PATH}" --dataset Tiny \
    --epochs 100 --pool_compression 20 --crop_size 64 --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 256 --workers 12 --fkd_seed 42 \
    --seed 0 --device_ids 0 > "${downstream}/relabel.stdout.log" 2>&1

  train_official_fkd 0 "${out}/synthetic.pt" "${labels}" "${downstream}"
  touch "${out}/complete"
}

run_ddruos() {
  local out="${RUN_ROOT}/ddruos"
  local downstream="${out}/downstream_fkd_groups17_seed0"
  local labels="${downstream}/fkd_pool"
  mkdir -p "${downstream}"
  cat > "${out}/run.conf" <<EOF
method=ROSD
ipc=50
cim_factor=2
image_target_kib_per_class=90
label_target_kib_per_class=115
label_groups=17
label_step=0.35
label_pool_compression=5.882353x
hard_ce_weight=0
label_kl_weight=0
joint_iterations=400
EOF

  CUDA_VISIBLE_DEVICES=1 "${PYTHON_DDRUO}" -u "${ROOT}/TM/cim_ddruo_tensorpool.py" \
    --dataset Tiny --data_path "${DATA_PATH}" --ipc 50 \
    --teacher_path "${TEACHER}" --save_path "${out}" \
    --cim_factor 2 --cim_mipc 300 --cim_class_batch 12 \
    --cim_selection_batch 1024 --cim_selection_workers 12 \
    --cim_augmentation crop_cutout_flip --optimize_label_rate \
    --label_step 0.35 --label_groups 17 --label_entropy_lr 0.001 \
    --label_entropy_warmup 20 --label_feature_chunk 1536 \
    --label_entropy_batch 128 --label_model_bits 16 \
    --hard_ce_weight 0 --label_kl_weight 0 --separate_rate_budgets \
    --image_target_kib 90 --label_target_kib 115 \
    --image_model_overhead_kib 9.66 --image_dual_lr 0.0001 \
    --label_dual_lr 0.0001 --warmup_rate_control dual \
    --warmup_target_kib 90 --warmup_rate_margin 1 --warmup_dual_lr 0.001 \
    --fast_single_warmup --fast_warmup_iterations 200 --ldb 0.1 \
    --codec_lr 0.001 --lr_it 1000 --rate_control dual \
    --latent_target_kib 80.34 --stage1_iterations 400 --stage2_iterations 0 \
    --layers_v v5 --arm 32 --dim 4 --log_every 5 --checkpoint_every 100 \
    --network_mse_threshold 5e-7 --enable_cudnn --enable_codec_scheduler --seed 0 \
    > "${out}/stdout.log" 2>&1

  CUDA_VISIBLE_DEVICES=1 "${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
    --mode relabel_pool --synthetic_path "${out}/synthetic.pt" \
    --teacher_path "${TEACHER}" --codec_checkpoint "${out}/label_codec_400.pt" \
    --fkd_path "${labels}" --output_path "${downstream}" \
    --data_path "${DATA_PATH}" --dataset Tiny --epochs 100 \
    --pool_compression 5.882353 --crop_size 64 --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 128 --workers 12 --fkd_seed 42 \
    --seed 0 --device_ids 0 > "${downstream}/relabel.stdout.log" 2>&1

  train_official_fkd 1 "${out}/synthetic.pt" "${labels}" "${downstream}"
  touch "${out}/complete"
}

run_ddif & p0=$!
run_ddruos & p1=$!
printf 'run_root=%s ddif_pid=%s ddruos_pid=%s\n' "${RUN_ROOT}" "${p0}" "${p1}" \
  | tee "${RUN_ROOT}/pids.txt"
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
printf 'exit_status=%s\n' "${status}" | tee "${RUN_ROOT}/finished.txt"
exit "${status}"
