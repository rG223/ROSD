#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
TORCHRUN="${TORCHRUN:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/torchrun}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"
CLASS_TEACHER="${CLASS_TEACHER:-/data2/home/ypliu/DD-RUO-1/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/class_bn/resnet18_tiny_0.pth}"
RUN_DIR="${1:?usage: $0 RUN_DIR}"

UTILITY_PROFILE="${UTILITY_PROFILE:-lpld}"
IPC="${IPC:-100}"
ITERATIONS="${ITERATIONS:-4000}"
LABEL_GROUPS="${LABEL_GROUPS:-100}"
ENCODER_GAIN="${ENCODER_GAIN:-16}"
GPU_IDS="${GPU_IDS:-0,1}"
DOWNSTREAM_GPU_IDS="${DOWNSTREAM_GPU_IDS:-0,1,2,3}"
TARGETS_CSV="${TARGETS_CSV:-610,440,330,295}"
EVAL_LABEL_GROUPS="${EVAL_LABEL_GROUPS:-}"
LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.00001}"
LAMBDA_LABEL="${LAMBDA_LABEL:-0.0001}"
COMBINED_DUAL="${COMBINED_DUAL:-0}"
COMBINED_TARGET_KIB="${COMBINED_TARGET_KIB:-600}"
DUAL_INIT="${DUAL_INIT:-0.0001}"
DUAL_LR="${DUAL_LR:-0.00001}"
DUAL_RHO="${DUAL_RHO:-0.00005}"
DUAL_EMA_DECAY="${DUAL_EMA_DECAY:-0.95}"
DUAL_UPDATE_EVERY="${DUAL_UPDATE_EVERY:-10}"
DUAL_DEADBAND="${DUAL_DEADBAND:-0.02}"
DUAL_MIN="${DUAL_MIN:--0.00005}"
DUAL_MAX="${DUAL_MAX:-0.0005}"
IMAGE_RATE_GRADIENT_WEIGHT="${IMAGE_RATE_GRADIENT_WEIGHT:-1.0}"
LABEL_RATE_GRADIENT_WEIGHT="${LABEL_RATE_GRADIENT_WEIGHT:-1.0}"
LABEL_FEATURE_CHUNK="${LABEL_FEATURE_CHUNK:-2000}"
LABEL_ENTROPY_BATCH="${LABEL_ENTROPY_BATCH:-1000}"
UTILITY_BATCH_SIZE="${UTILITY_BATCH_SIZE:-400}"
ALLOW_TF32="${ALLOW_TF32:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
RESUME_ROOT="${RESUME_ROOT:-$RUN_DIR}"
RESUME_ITERATION="${RESUME_ITERATION:-0}"
RESUME_WORLD_SIZE="${RESUME_WORLD_SIZE:-0}"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
IFS=',' read -r -a downstream_gpu_array <<< "$DOWNSTREAM_GPU_IDS"
WORLD_SIZE="${#gpu_array[@]}"
CLASSES=200
[[ "$IPC" -gt 0 ]] || { echo "IPC must be positive" >&2; exit 2; }
[[ "$LABEL_GROUPS" -gt 0 ]] || { echo "LABEL_GROUPS must be positive" >&2; exit 2; }
[[ "$RESUME_ITERATION" =~ ^[0-9]+$ ]] || { echo "RESUME_ITERATION must be a non-negative integer" >&2; exit 2; }
[[ "$RESUME_WORLD_SIZE" =~ ^[0-9]+$ ]] || { echo "RESUME_WORLD_SIZE must be a non-negative integer" >&2; exit 2; }
if [[ "$RESUME_ITERATION" -gt 0 && "$RESUME_WORLD_SIZE" -lt 1 ]]; then
  echo "RESUME_WORLD_SIZE must be positive when resuming" >&2
  exit 2
fi
[[ "$WORLD_SIZE" -ge 2 && "$WORLD_SIZE" -le 4 ]] || { echo "GPU_IDS must contain two, three, or four GPUs" >&2; exit 2; }
[[ "${#downstream_gpu_array[@]}" -ge 1 && "${#downstream_gpu_array[@]}" -le 4 ]] || {
  echo "DOWNSTREAM_GPU_IDS must contain one to four GPUs" >&2
  exit 2
}
if [[ "$COMBINED_DUAL" == "1" ]]; then
  awk -v target="$COMBINED_TARGET_KIB" 'BEGIN { exit !(target > 0) }' || {
    echo "COMBINED_TARGET_KIB must be positive" >&2
    exit 2
  }
fi
case "$UTILITY_PROFILE" in
  lpld)
    METHOD_NAME="LPLD-ROSD"
    UTILITY_BACKEND="lpld_class_bn"
    BN_WEIGHT="0.05"
    UTILITY_DESCRIPTION="LPLD_class_conditional_CE_plus_BN0.05_firstBN10"
    DOWNSTREAM_DESCRIPTION="LPLD_epochs100_SGD_lr0.2_warmup5_cosine_T20_T2"
    ;;
  sre2l)
    METHOD_NAME="SRe2L-ROSD"
    UTILITY_BACKEND="sre2l"
    BN_WEIGHT="1.0"
    UTILITY_DESCRIPTION="SRe2L_CE_plus_BN1.0_firstBN10"
    DOWNSTREAM_DESCRIPTION="SRe2L_epochs100_SGD_lr0.2_warmup5_cosine_T20_T2_CutMix_DKR_CA"
    ;;
  *)
    echo "UTILITY_PROFILE must be lpld or sre2l" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/pipeline.log" 2>&1
echo "[$(date --iso-8601=seconds)] ${WORLD_SIZE}-GPU ${METHOD_NAME} start pid=$$"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] pipeline failed status=$status"; touch "$RUN_DIR/pipeline_failed"; exit "$status"' ERR
rm -f "$RUN_DIR/pipeline_failed"

test -x "$PYTHON"
test -x "$TORCHRUN"
test -d "$DATA_PATH/train"
test -d "$DATA_PATH/val"
test -f "$TEACHER"
if [[ "$UTILITY_PROFILE" == "lpld" ]]; then
  test -f "$CLASS_TEACHER"
fi

cat > "$RUN_DIR/run.conf" <<EOF
method=$METHOD_NAME
dataset=Tiny-ImageNet
ipc=$IPC
seed=0
iterations=$ITERATIONS
gpus=$GPU_IDS
downstream_gpus=$DOWNSTREAM_GPU_IDS
class_sharding=balanced_integer_boundaries
utility=$UTILITY_DESCRIPTION
utility_profile=$UTILITY_PROFILE
lambda_image=$LAMBDA_IMAGE
lambda_label=$LAMBDA_LABEL
combined_dual=$COMBINED_DUAL
combined_target_kib_per_class=$COMBINED_TARGET_KIB
dual_init=$DUAL_INIT
dual_lr=$DUAL_LR
dual_rho=$DUAL_RHO
dual_ema_decay=$DUAL_EMA_DECAY
dual_update_every=$DUAL_UPDATE_EVERY
dual_deadband=$DUAL_DEADBAND
dual_bounds=$DUAL_MIN,$DUAL_MAX
image_rate_gradient_weight=$IMAGE_RATE_GRADIENT_WEIGHT
label_rate_gradient_weight=$LABEL_RATE_GRADIENT_WEIGHT
label_rate_gradient_normalization=global_classes_over_local_classes
label_step=0.35
label_groups=$LABEL_GROUPS
label_feature_chunk=$LABEL_FEATURE_CHUNK
label_entropy_batch=$LABEL_ENTROPY_BATCH
utility_batch_size=$UTILITY_BATCH_SIZE
allow_tf32=$ALLOW_TF32
checkpoint_every=$CHECKPOINT_EVERY
encoder_gain=$ENCODER_GAIN
latent_step=$(awk -v gain="$ENCODER_GAIN" 'BEGIN { printf "%.8f", 1.0 / gain }')
target_total_kib_per_class=$TARGETS_CSV
evaluation_label_groups=${EVAL_LABEL_GROUPS:-target_driven}
target_accounting=postquant_image_entropy_plus_label_model_plus_FKD_label_entropy
downstream=$DOWNSTREAM_DESCRIPTION
resume_root=$RESUME_ROOT
resume_iteration=$RESUME_ITERATION
resume_world_size=$RESUME_WORLD_SIZE
EOF

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

if [[ ! -f "$RUN_DIR/joint_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] joint optimization start"
  utility_args=(--utility_backend "$UTILITY_BACKEND")
  if [[ "$UTILITY_PROFILE" == "lpld" ]]; then
    utility_args+=(--class_teacher_path "$CLASS_TEACHER")
  fi
  dual_args=()
  if [[ "$COMBINED_DUAL" == "1" ]]; then
    dual_args=(
      --combined_dual --combined_target_kib "$COMBINED_TARGET_KIB"
      --dual_init "$DUAL_INIT" --dual_lr "$DUAL_LR" --dual_rho "$DUAL_RHO"
      --dual_ema_decay "$DUAL_EMA_DECAY"
      --dual_update_every "$DUAL_UPDATE_EVERY"
      --dual_deadband "$DUAL_DEADBAND"
      --dual_min "$DUAL_MIN" --dual_max "$DUAL_MAX"
      --image_rate_gradient_weight "$IMAGE_RATE_GRADIENT_WEIGHT"
      --label_rate_gradient_weight "$LABEL_RATE_GRADIENT_WEIGHT"
    )
  fi
  resume_args=()
  if [[ "$RESUME_ITERATION" -gt 0 ]]; then
    resume_args=(
      --resume_root "$RESUME_ROOT"
      --resume_iteration "$RESUME_ITERATION"
      --resume_world_size "$RESUME_WORLD_SIZE"
    )
  fi
  precision_args=()
  if [[ "$ALLOW_TF32" == "1" ]]; then
    precision_args=(--allow_tf32)
  fi
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN" --standalone --nproc-per-node="$WORLD_SIZE" \
    TM/ddruos_sre2l_joint_distributed.py \
    --data_path "$DATA_PATH" --teacher_path "$TEACHER" \
    "${utility_args[@]}" \
    --save_path "$RUN_DIR" --ipc "$IPC" --iterations "$ITERATIONS" --seed 0 \
    --bn_weight "$BN_WEIGHT" --bn_loss_mode sre2l_l2_sum \
    --first_bn_multiplier 10 --jitter 4 \
    --lambda_image "$LAMBDA_IMAGE" --lambda_label "$LAMBDA_LABEL" \
    --label_gradient_cap 0.1 --label_step 0.35 --label_groups "$LABEL_GROUPS" \
    --encoder_gain "$ENCODER_GAIN" \
    --utility_batch_size "$UTILITY_BATCH_SIZE" --label_feature_chunk "$LABEL_FEATURE_CHUNK" \
    --label_entropy_batch "$LABEL_ENTROPY_BATCH" --codec_workers 12 \
    --log_every 10 --checkpoint_every "$CHECKPOINT_EVERY" \
    "${dual_args[@]}" "${resume_args[@]}" "${precision_args[@]}"
  echo "[$(date --iso-8601=seconds)] joint optimization complete"
fi

if [[ ! -f "$RUN_DIR/postquant_complete" ]]; then
  echo "[$(date --iso-8601=seconds)] post-quantization start"
  mkdir -p "$RUN_DIR/postquant"
  pids=()
  shard_outputs=()
  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    class_start=$((CLASSES * rank / WORLD_SIZE))
    class_end=$((CLASSES * (rank + 1) / WORLD_SIZE))
    classes_in_shard=$((class_end - class_start))
    shard_outputs+=("$RUN_DIR/postquant/shard${rank}.pt")
    CUDA_VISIBLE_DEVICES="${gpu_array[$rank]}" "$PYTHON" -u \
      TM/quantize_decode_tensorpool_shard.py \
      --input "$RUN_DIR/rank${rank}_${class_start}_${class_end}/pool_final_global_keys.pt" \
      --output "$RUN_DIR/postquant/shard${rank}.pt" \
      --quantized_output "$RUN_DIR/postquant/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --classes_per_shard "$classes_in_shard" \
      --ipc "$IPC" --workers 12 --mse_threshold 5e-7 \
      --encoder_gain "$ENCODER_GAIN" \
      --teacher_path "$TEACHER" \
      --label_codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
      --postquant_label_rate_samples 1 \
      > "$RUN_DIR/postquant/shard${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "${shard_outputs[@]}" --output "$RUN_DIR/synthetic.pt" --ipc "$IPC" \
    --joint_image_lambda "$LAMBDA_IMAGE" --joint_label_lambda "$LAMBDA_LABEL" \
    --label_kib "$($PYTHON -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)["config"]; print(c.get("estimated_label_kib_per_class",417.28))' "$RUN_DIR/label_codec_${ITERATIONS}.pt")" \
    --label_model_kib "$($PYTHON -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)["config"]; print(c.get("label_model_kib_per_class",7.31232421875))' "$RUN_DIR/label_codec_${ITERATIONS}.pt")" \
    --encoder_gain "$ENCODER_GAIN" \
    --label_groups "$LABEL_GROUPS" \
    > "$RUN_DIR/postquant/merge.log" 2>&1
  touch "$RUN_DIR/postquant_complete"
  echo "[$(date --iso-8601=seconds)] post-quantization complete"
fi

eval_target() {
  local target="$1" gpu="$2"
  local base="$RUN_DIR/downstream_target${target}kib_${UTILITY_PROFILE}_seed0"
  local pool="$base/fkd_pool" output="$base/seed0"
  mkdir -p "$pool" "$output"
  if [[ -f "$base/complete" ]]; then return 0; fi

  echo "[$(date --iso-8601=seconds)] target=${target} relabel start gpu=$gpu"
  relabel_profile_args=()
  train_profile_args=()
  if [[ "$UTILITY_PROFILE" == "lpld" ]]; then
    relabel_profile_args=(--disable_cutmix)
    train_profile_args=(--pool_sampling_with_replacement)
  else
    train_profile_args=(
      --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7
      --dkr_min_temperature 2 --ca_dynamic
    )
  fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$pool" --output_path "$base" --data_path "$DATA_PATH" \
    --dataset Tiny --epochs 100 --target_total_kib_per_class "$target" \
    --crop_size 64 --min_crop_scale 0.08 --temperature 20 \
    --loader_batch 64 --workers 16 --fkd_seed 42 --seed 0 --device_ids 0 \
    "${relabel_profile_args[@]}" \
    > "$base/relabel_pool.stdout.log" 2>&1
  echo "[$(date --iso-8601=seconds)] target=${target} train start gpu=$gpu"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --fast_downstream \
    "${train_profile_args[@]}" \
    --eval_every 10 --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] target=${target} complete gpu=$gpu"
}

eval_groups() {
  local groups="$1" gpu="$2"
  local compression
  compression="$(awk -v groups="$groups" 'BEGIN { printf "%.10g", 100.0 / groups }')"
  local base="$RUN_DIR/downstream_g${groups}_${UTILITY_PROFILE}_seed0"
  local pool="$base/fkd_pool" output="$base/seed0"
  mkdir -p "$pool" "$output"
  if [[ -f "$base/complete" ]]; then return 0; fi

  echo "[$(date --iso-8601=seconds)] groups=${groups} relabel start gpu=$gpu"
  relabel_profile_args=()
  train_profile_args=()
  if [[ "$UTILITY_PROFILE" == "lpld" ]]; then
    relabel_profile_args=(--disable_cutmix)
    train_profile_args=(--pool_sampling_with_replacement)
  else
    train_profile_args=(
      --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7
      --dkr_min_temperature 2 --ca_dynamic
    )
  fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN_DIR/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$pool" --output_path "$base" --data_path "$DATA_PATH" \
    --dataset Tiny --epochs 100 --pool_compression "$compression" \
    --crop_size 64 --min_crop_scale 0.08 --temperature 20 \
    --loader_batch 64 --workers 16 --fkd_seed 42 --seed 0 --device_ids 0 \
    "${relabel_profile_args[@]}" \
    > "$base/relabel_pool.stdout.log" 2>&1
  echo "[$(date --iso-8601=seconds)] groups=${groups} train start gpu=$gpu"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode train_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
    --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
    --data_path "$DATA_PATH" --dataset Tiny --epochs 100 --crop_size 64 \
    --temperature 20 --train_batch 64 --workers 8 --optimizer sgd \
    --learning_rate 0.2 --momentum 0.9 --weight_decay 0.0001 \
    --warmup_epochs 5 --warmup_start_factor 0.01 \
    --scale_loss_by_temperature_squared --fast_downstream \
    "${train_profile_args[@]}" \
    --eval_every 10 --seed 0 --device_ids 0 \
    > "$output/train_pool.stdout.log" 2>&1
  touch "$base/complete"
  echo "[$(date --iso-8601=seconds)] groups=${groups} complete gpu=$gpu"
}

if [[ -n "$EVAL_LABEL_GROUPS" ]]; then
  eval_groups "$EVAL_LABEL_GROUPS" "${downstream_gpu_array[0]}"
  touch "$RUN_DIR/downstream_complete" "$RUN_DIR/pipeline_complete"
  echo "[$(date --iso-8601=seconds)] pipeline complete"
  exit 0
fi

IFS=',' read -r -a targets <<< "$TARGETS_CSV"
if [[ "${#targets[@]}" -gt "${#downstream_gpu_array[@]}" ]]; then
  echo "TARGETS_CSV has more targets than DOWNSTREAM_GPU_IDS has GPUs" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] downstream target queues start targets=$TARGETS_CSV"
pids=()
for index in "${!targets[@]}"; do
  eval_target "${targets[$index]}" "${downstream_gpu_array[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]]
touch "$RUN_DIR/downstream_complete" "$RUN_DIR/pipeline_complete"
echo "[$(date --iso-8601=seconds)] pipeline complete"
