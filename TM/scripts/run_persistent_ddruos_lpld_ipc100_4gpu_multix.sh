#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_DDRUO="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
TORCHRUN="${TORCHRUN:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/torchrun}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
CLASS_TEACHER="${CLASS_TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/class_bn/resnet18_tiny_0.pth}"
RUN_DIR="${1:?usage: $0 RUN_DIR}"

IPC="${IPC:-100}"
ITERATIONS="${ITERATIONS:-4000}"
LABEL_GROUPS="${LABEL_GROUPS:-100}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
WORLD_SIZE="${#gpu_array[@]}"
CLASSES=200
if [[ "$WORLD_SIZE" -ne 4 ]]; then
  echo "This strict LPLD launcher requires four comma-separated GPU IDs" >&2
  exit 2
fi
CLASSES_PER_SHARD=$((CLASSES / WORLD_SIZE))

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/pipeline.log" 2>&1

echo "[$(date --iso-8601=seconds)] LPLD-ROSD pipeline start pid=$$"
echo "root=$ROOT run_dir=$RUN_DIR gpu_ids=$GPU_IDS"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] pipeline failed status=$status"; touch "$RUN_DIR/pipeline_failed"; exit "$status"' ERR
rm -f "$RUN_DIR/pipeline_failed"

test -x "$PYTHON_DDRUO"
test -x "$TORCHRUN"
test -d "$DATA_PATH/train"
test -d "$DATA_PATH/val"
test -f "$TEACHER"
test -f "$CLASS_TEACHER"
[[ "$IPC" -eq 100 ]]

cat > "$RUN_DIR/run.conf" <<EOF
method=LPLD-ROSD
dataset=Tiny-ImageNet
ipc=$IPC
seed=0
iterations=$ITERATIONS
lpld_upstream_commit=63eee186e15e09a6889dda07805ce921c439aef4
official_lpld_recovery=classwise_ipc100_rrc0p08to1_flip_jitter4_ce_plus_bn0p05_firstbn10
official_lpld_downstream=epochs100_sgd_lr0p2_wd1e-4_warmup5_cosine_T20_T2_no_cutmix_sampling_with_replacement
recovery_batching=class_wise_100_images
recovery_augmentation=shared_RRC_0.08_1.0_flip_0.5_jitter_4
recovery_utility=CE_plus_0.05_class_conditional_BN_L2_sum
first_bn_multiplier=10
initialization=random_TensorPool_codec
ddruos_additions=tensorpool_multiscale_codec_image_entropy_soft_label_entropy_fixed_label_step0p35
optimization_pixel_projection=false_decoder_output_is_differentiable_and_not_a_raw_image_parameter
lambda_image=0.0001
lambda_label=0.0001
label_gradient_cap=0.1
label_step=0.35
label_groups=$LABEL_GROUPS
synthesis_parallelism=4_class_shards_one_seed
classes_per_shard=$CLASSES_PER_SHARD
downstream_epochs=100
downstream_batch=64
downstream_optimizer=SGD_lr0.2_momentum0.9_wd1e-4
downstream_scheduler=linear_warmup5_start0.01_then_cosine
downstream_temperature=20_with_T_squared
downstream_augmentation=RRC_0.08_1.0_flip_no_CutMix
downstream_pruning=random_batch_to_epoch_with_replacement
downstream_ratios=0.9,0.95,0.97,0.975
downstream_names=10x,20x,30x,40x
downstream_DKR=false
downstream_CA=false
EOF

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

if [[ ! -f "$RUN_DIR/joint_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] class-wise synthesis start"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN" --standalone --nproc-per-node=4 \
    TM/ddruos_sre2l_joint_distributed.py \
    --data_path "$DATA_PATH" --teacher_path "$TEACHER" \
    --class_teacher_path "$CLASS_TEACHER" --utility_backend lpld_class_bn \
    --save_path "$RUN_DIR" --ipc "$IPC" --iterations "$ITERATIONS" --seed 0 \
    --bn_weight 0.05 --bn_loss_mode sre2l_l2_sum \
    --first_bn_multiplier 10 --jitter 4 \
    --lambda_image 0.0001 --lambda_label 0.0001 \
    --label_gradient_cap 0.1 --label_step 0.35 --label_groups "$LABEL_GROUPS" \
    --utility_batch_size 400 --label_feature_chunk 2000 \
    --label_entropy_batch 1000 --codec_workers 12 \
    --log_every 10 --checkpoint_every 1000
  echo "[$(date --iso-8601=seconds)] class-wise synthesis complete"
else
  echo "[$(date --iso-8601=seconds)] synthesis already complete; skip"
fi

if [[ ! -f "$RUN_DIR/postquant_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] post-quantization start"
  mkdir -p "$RUN_DIR/postquant"
  quant_pids=()
  shard_outputs=()
  for ((rank = 0; rank < 4; rank++)); do
    class_start=$((rank * CLASSES_PER_SHARD))
    class_end=$((class_start + CLASSES_PER_SHARD))
    shard_outputs+=("$RUN_DIR/postquant/shard${rank}.pt")
    CUDA_VISIBLE_DEVICES="${gpu_array[$rank]}" "$PYTHON_DDRUO" -u \
      TM/quantize_decode_tensorpool_shard.py \
      --input "$RUN_DIR/rank${rank}_${class_start}_${class_end}/pool_final_global_keys.pt" \
      --output "$RUN_DIR/postquant/shard${rank}.pt" \
      --quantized_output "$RUN_DIR/postquant/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --classes_per_shard "$CLASSES_PER_SHARD" \
      --ipc "$IPC" --workers 12 --mse_threshold 5e-7 \
      --teacher_path "$TEACHER" \
      --label_codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
      --postquant_label_rate_samples 1 \
      > "$RUN_DIR/postquant/shard${rank}.log" 2>&1 &
    quant_pids+=("$!")
  done
  for pid in "${quant_pids[@]}"; do wait "$pid"; done
  "$PYTHON_DDRUO" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "${shard_outputs[@]}" --output "$RUN_DIR/synthetic.pt" --ipc "$IPC" \
    > "$RUN_DIR/postquant/merge.log" 2>&1
  touch "$RUN_DIR/postquant_complete"
  echo "[$(date --iso-8601=seconds)] post-quantization complete"
else
  echo "[$(date --iso-8601=seconds)] post-quantization already complete; skip"
fi

official_prune_ratio() {
  case "$1" in
    10) echo 0.9 ;;
    20) echo 0.95 ;;
    30) echo 0.97 ;;
    40) echo 0.975 ;;
    *) return 2 ;;
  esac
}

eval_one() {
  local name="$1" gpu="$2" prune_ratio
  prune_ratio="$(official_prune_ratio "$name")"
  local base="$RUN_DIR/downstream_${name}x_lpld_seed0"
  local fkd="$base/fkd_pool" output="$base/seed0"
  mkdir -p "$fkd" "$output"
  if [[ -f "$base/complete" ]]; then
    echo "[$(date --iso-8601=seconds)] ${name}x already complete; skip"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] ${name}x relabel start gpu=$gpu prune_ratio=$prune_ratio"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_DDRUO" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$fkd" --output_path "$base" --data_path "$DATA_PATH" \
    --dataset Tiny --epochs 100 --prune_ratio "$prune_ratio" \
    --crop_size 64 --min_crop_scale 0.08 --temperature 20 --disable_cutmix \
    --loader_batch 64 --workers 8 --fkd_seed 42 --seed 0 --device_ids 0 \
    > "$base/relabel_pool.stdout.log" 2>&1
  echo "[$(date --iso-8601=seconds)] ${name}x train start gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_DDRUO" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$fkd" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --pool_sampling_with_replacement \
    --eval_every 10 --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] ${name}x complete gpu=$gpu"
}

echo "[$(date --iso-8601=seconds)] downstream evaluations start"
eval_pids=()
compressions=(10 20 30 40)
for index in 0 1 2 3; do
  eval_one "${compressions[$index]}" "${gpu_array[$index]}" &
  eval_pids+=("$!")
done
eval_status=0
for pid in "${eval_pids[@]}"; do wait "$pid" || eval_status=1; done
if [[ "$eval_status" -ne 0 ]]; then
  echo "one or more downstream evaluations failed" >&2
  exit 1
fi

touch "$RUN_DIR/downstream_complete" "$RUN_DIR/pipeline_complete"
echo "[$(date --iso-8601=seconds)] LPLD-ROSD pipeline complete"
