#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/home/ypliu/DD-RUO-1}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-${ROOT}/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
SOURCE="${SOURCE:-${ROOT}/results/tiny_imagenet_target210/Tiny_IPC50_DDiF_vs_DDRUOS_20260810_234148/ddif}"
SYNTHETIC="${SOURCE}/synthetic.pt"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SOURCE}/label_compression_ablation_${STAMP}}"

mkdir -p "${RUN_ROOT}"/{label10x,label30x}
cat > "${RUN_ROOT}/settings.txt" <<EOF
fixed_synthetic=${SYNTHETIC}
field_width=20
image_kib_per_class=106.0546875
existing_20x_best_acc=0.3757
downstream_optimizer=SGD
learning_rate=0.2
warmup_epochs=5
temperature=20
temperature_squared_scaling=true
EOF

export PYTHONUNBUFFERED=1

run_eval() {
  local gpu=$1 compression=$2
  local out="${RUN_ROOT}/label${compression}x"
  local labels="${out}/fkd_pool"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/TM/sre2l_fkd.py" \
    --mode relabel_pool --synthetic_path "${SYNTHETIC}" \
    --teacher_path "${TEACHER}" --fkd_path "${labels}" --output_path "${out}" \
    --data_path "${DATA_PATH}" --dataset Tiny --epochs 100 \
    --pool_compression "${compression}" --crop_size 64 --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 256 --workers 12 --fkd_seed 42 \
    --seed 0 --device_ids 0 > "${out}/relabel.stdout.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/TM/sre2l_fkd.py" \
    --mode train_pool --synthetic_path "${SYNTHETIC}" \
    --teacher_path "${TEACHER}" --fkd_path "${labels}" --output_path "${out}" \
    --data_path "${DATA_PATH}" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --eval_every 10 --seed 0 --device_ids 0 \
    > "${out}/train.stdout.log" 2>&1
  touch "${out}/complete"
}

run_eval 0 10 & p0=$!
run_eval 1 30 & p1=$!
printf 'run_root=%s gpu0_10x=%s gpu1_30x=%s\n' "${RUN_ROOT}" "${p0}" "${p1}" \
  | tee "${RUN_ROOT}/pids.txt"
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
printf 'exit_status=%s\n' "${status}" | tee "${RUN_ROOT}/finished.txt"
exit "${status}"
