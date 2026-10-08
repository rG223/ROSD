#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data/home/yerg/dd-ruo}"
PYTHON_DDRUO="${PYTHON_DDRUO:-/mnt/data/home/yerg/anaconda3/envs/dd_ruo/bin/python}"
TORCHRUN="${TORCHRUN:-/mnt/data/home/yerg/anaconda3/envs/dd_ruo/bin/torchrun}"
DATA_PATH="${DATA_PATH:-/mnt/data/home/yerg/tiny-imagenet-200}"
TEACHER="${TEACHER:-/mnt/data1/big_file/yerg/dd-ruo-ipc100/models/rn18_50ep/ddruo_teacher.pt}"
RUN_DIR="${1:?usage: $0 RUN_DIR}"

ITERATIONS="${ITERATIONS:-4000}"
LABEL_GROUPS="${LABEL_GROUPS:-30}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/pipeline.log" 2>&1

echo "[$(date --iso-8601=seconds)] pipeline start pid=$$"
echo "run_dir=$RUN_DIR"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] pipeline failed status=$status"; touch "$RUN_DIR/pipeline_failed"; exit "$status"' ERR
rm -f "$RUN_DIR/pipeline_failed"

test -x "$PYTHON_DDRUO"
test -x "$TORCHRUN"
test -d "$DATA_PATH"
test -f "$TEACHER"

cat > "$RUN_DIR/run.conf" <<EOF
method=ROSD
utility=SRe2L_CE_plus_1.0_BN
stored_ipc=50
seed=0
iterations=$ITERATIONS
augmentation=RRC_scale_0.08_1.0_flip_0.5_jitter_4
bn_weight=1.0
bn_loss=sre2l_L2_sum
first_bn_multiplier=10
initialization=random_TensorPool_codec
posthoc_clamp=false
optimization_pixel_projection=false
lambda_image=0.0001
lambda_label=0.0001
label_gradient_cap=0.1
label_groups=$LABEL_GROUPS
synthesis_parallelism=four_class_shards_one_seed
downstream_pool_compressions=10,20,30,40
downstream_parallelism=one_compression_per_gpu
downstream_cutmix=true
downstream_dkr=step_30_gamma_0.7_min_2
downstream_ca_dynamic=true
EOF

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

if [[ ! -f "$RUN_DIR/joint_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] synthesis start"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN" --standalone --nproc-per-node=4 \
    TM/ddruos_sre2l_joint_distributed.py \
    --data_path "$DATA_PATH" --teacher_path "$TEACHER" --save_path "$RUN_DIR" \
    --ipc 50 --iterations "$ITERATIONS" --seed 0 \
    --bn_weight 1.0 --bn_loss_mode sre2l_l2_sum \
    --first_bn_multiplier 10 --jitter 4 \
    --lambda_image 0.0001 --lambda_label 0.0001 \
    --label_gradient_cap 0.1 --label_step 0.35 --label_groups "$LABEL_GROUPS" \
    --utility_batch_size 1000 --label_feature_chunk 1000 \
    --label_entropy_batch 1000 --codec_workers 12 \
    --log_every 10 --checkpoint_every 1000
  echo "[$(date --iso-8601=seconds)] synthesis complete"
else
  echo "[$(date --iso-8601=seconds)] synthesis already complete; skip"
fi

if [[ ! -f "$RUN_DIR/postquant_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] post-quantization start"
  mkdir -p "$RUN_DIR/postquant"
  IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
  if [[ "${#gpu_array[@]}" -ne 4 ]]; then
    echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
    exit 2
  fi
  quant_pids=()
  for rank in 0 1 2 3; do
    class_start=$((rank * 50))
    class_end=$((class_start + 50))
    CUDA_VISIBLE_DEVICES="${gpu_array[$rank]}" "$PYTHON_DDRUO" -u \
      TM/quantize_decode_tensorpool_shard.py \
      --input "$RUN_DIR/rank${rank}_${class_start}_${class_end}/pool_final_global_keys.pt" \
      --output "$RUN_DIR/postquant/shard${rank}.pt" \
      --quantized_output "$RUN_DIR/postquant/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --ipc 50 --workers 12 --mse_threshold 5e-7 \
      --teacher_path "$TEACHER" \
      --label_codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
      --postquant_label_rate_samples 1 \
      > "$RUN_DIR/postquant/shard${rank}.log" 2>&1 &
    quant_pids+=("$!")
  done
  for pid in "${quant_pids[@]}"; do
    wait "$pid"
  done
  "$PYTHON_DDRUO" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "$RUN_DIR/postquant/shard0.pt" "$RUN_DIR/postquant/shard1.pt" \
             "$RUN_DIR/postquant/shard2.pt" "$RUN_DIR/postquant/shard3.pt" \
    --output "$RUN_DIR/synthetic.pt" --ipc 50 \
    --label_kib 190.52 --label_model_kib 4.19 \
    > "$RUN_DIR/postquant/merge.log" 2>&1
  touch "$RUN_DIR/postquant_complete"
  echo "[$(date --iso-8601=seconds)] post-quantization complete"
else
  echo "[$(date --iso-8601=seconds)] post-quantization already complete; skip"
fi

eval_one() {
  local compression="$1"
  local gpu="$2"
  local base="$RUN_DIR/downstream_${compression}x_cutmix_dkrca_seed0"
  local fkd="$base/fkd_pool"
  local output="$base/seed0"
  mkdir -p "$fkd" "$output"
  if [[ -f "$base/complete" ]]; then
    echo "[$(date --iso-8601=seconds)] ${compression}x already complete; skip"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] ${compression}x relabel start gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_DDRUO" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$fkd" --output_path "$base" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
    --pool_compression "$compression" --crop_size 64 --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 128 --workers 8 \
    --fkd_seed 42 --seed 0 --device_ids 0 \
    > "$base/relabel_pool.stdout.log" 2>&1

  echo "[$(date --iso-8601=seconds)] ${compression}x train start gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_DDRUO" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$fkd" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 \
    --crop_size 64 --temperature 20 --train_batch 64 --workers 8 \
    --optimizer sgd --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --eval_every 10 \
    --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
    --dkr_min_temperature 2 --ca_dynamic \
    --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] ${compression}x complete gpu=$gpu"
}

echo "[$(date --iso-8601=seconds)] downstream evaluations start"
eval_pids=()
eval_one 10 0 & eval_pids+=("$!")
eval_one 20 1 & eval_pids+=("$!")
eval_one 30 2 & eval_pids+=("$!")
eval_one 40 3 & eval_pids+=("$!")

eval_status=0
for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then
    eval_status=1
  fi
done
if [[ "$eval_status" -ne 0 ]]; then
  echo "one or more downstream evaluations failed" >&2
  exit 1
fi

touch "$RUN_DIR/downstream_complete" "$RUN_DIR/pipeline_complete"
echo "[$(date --iso-8601=seconds)] pipeline complete"
