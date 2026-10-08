#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/home/ypliu/dd-ruo-github-latest"
SOURCE_RUN="${1:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_ipc100/LPLD_DDRUOS_IPC100_BN0p05_4GPU_lambdaImg1e-5_gain32_step0p03125_targets610_440_330_295_seed0_20260820_144602}"
PYTHON="/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python"
TEACHER="/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt"
DATA_PATH="/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200"
OUTPUT="$SOURCE_RUN/downstream_target610kib_unquantized_oracle3494_seed0"
POOL="$OUTPUT/fkd_pool"
TRAIN="$OUTPUT/seed0"
SOURCE_POOL="$SOURCE_RUN/downstream_target610kib_lpld_seed0/fkd_pool"

mkdir -p "$POOL" "$TRAIN"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$OUTPUT/relabel_complete" ]]; then
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool_oracle --synthetic_path "$SOURCE_RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --source_fkd_path "$SOURCE_POOL" \
    --pool_batches_limit 3494 --fkd_path "$POOL" --output_path "$OUTPUT" \
    --data_path "$DATA_PATH" --dataset Tiny --crop_size 64 --temperature 20 \
    --workers 8 --log_every 50 --seed 0 --device_ids 0 \
    > "$OUTPUT/relabel_pool_oracle.stdout.log" 2>&1
  touch "$OUTPUT/relabel_complete"
fi

if [[ ! -f "$OUTPUT/train_complete" ]]; then
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$SOURCE_RUN/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$POOL" --output_path "$TRAIN" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --fast_downstream --cache_replay_batches --eval_every 10 --seed 0 --device_ids 0 \
    > "$TRAIN/train_pool.stdout.log" 2>&1
  touch "$OUTPUT/train_complete"
fi

touch "$OUTPUT/complete"
