#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA3_ROOT="${DATA3_ROOT:-/data3/ypliu/DD-RUO-1}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
TEACHER="${TEACHER:-$DATA3_ROOT/assets/imagenet1k/resnet18-f37072fd.pth}"
CLASS_TEACHER="${CLASS_TEACHER:-/data3/ypliu/LPLD-official-data/models/resnet18_class_bn_in1k.pth}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ITERATIONS="${ITERATIONS:-4000}"
IPC="${IPC:-50}"
LABEL_GROUPS="${LABEL_GROUPS:-20}"
COMBINED_TARGET_KIB="${COMBINED_TARGET_KIB:-610}"
CLASS_BATCH_SIZE="${CLASS_BATCH_SIZE:-10}"
UTILITY_BATCH_SIZE="${UTILITY_BATCH_SIZE:-200}"
LABEL_FEATURE_CHUNK="${LABEL_FEATURE_CHUNK:-128}"
LABEL_ENTROPY_BATCH="${LABEL_ENTROPY_BATCH:-64}"
CODEC_WORKERS="${CODEC_WORKERS:-6}"
TARGETS_CSV="${TARGETS_CSV:-1160,620,290,150}"
SOURCE_POOL_COMPRESSION="${SOURCE_POOL_COMPRESSION:-3}"
WORKERS="${WORKERS:-12}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN="${RUN_DIR:-$DATA3_ROOT/results/imagenet1k_ipc${IPC}/LPLD_DDRUOS_IPC${IPC}_G${LABEL_GROUPS}_dual${COMBINED_TARGET_KIB}_4GPU_seed0_${STAMP}}"
POSTQUANT="$RUN/postquant_imagenet1k_iter${ITERATIONS}"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter${ITERATIONS}.pt"
DOWNSTREAM="$RUN/downstream_multirate_${TARGETS_CSV//,/_}_seed0"
SOURCE_POOL="$DOWNSTREAM/fkd_pool_source_c${SOURCE_POOL_COMPRESSION}"

mkdir -p "$RUN" "$POSTQUANT" "$DOWNSTREAM"
exec >>"$RUN/pipeline.log" 2>&1
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

for required in "$TEACHER" "$CLASS_TEACHER"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required teacher: $required" >&2
    exit 1
  fi
done

cat > "$RUN/run.conf" <<EOF
method=LPLD-ROSD
dataset=ImageNet-1K
image_size=224
ipc=$IPC
iterations=$ITERATIONS
utility_backend=lpld_class_bn
bn_weight=1
first_bn_multiplier=10
jitter=32
label_groups=$LABEL_GROUPS
label_step=0.35
combined_target_kib_per_class=$COMBINED_TARGET_KIB
class_batch_size_per_rank=$CLASS_BATCH_SIZE
utility_batch_size=$UTILITY_BATCH_SIZE
label_feature_chunk=$LABEL_FEATURE_CHUNK
label_entropy_batch=$LABEL_ENTROPY_BATCH
codec_workers_per_rank=$CODEC_WORKERS
downstream_targets_kib_per_class=$TARGETS_CSV
downstream_order=target_major_lpld_then_lpqld
downstream_lpld=official_in1k_T20_no_DKR_no_CA
downstream_lpqld=official_in1k_DKR_step30_gamma0p7_minT2_CA
outputs=$RUN
EOF

echo "[$(date --iso-8601=seconds)] LPLD-ROSD joint optimization start"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "$ROOT/TM/ddruos_sre2l_joint_distributed.py" \
  --dataset ImageNet1K --num_classes 1000 --image_size 224 \
  --data_path "$DATA_PATH" --teacher_path "$TEACHER" \
  --class_teacher_path "$CLASS_TEACHER" --utility_backend lpld_class_bn \
  --save_path "$RUN" --ipc "$IPC" --iterations "$ITERATIONS" --seed 0 \
  --bn_weight 1 --first_bn_multiplier 10 --jitter 32 \
  --combined_dual --combined_target_kib "$COMBINED_TARGET_KIB" \
  --dual_init 0.0001 --dual_lr 0.002 --dual_rho 0.00005 \
  --dual_ema_decay 0.95 --dual_update_every 10 --dual_deadband 0.02 \
  --dual_min -0.00005 --dual_max 1 \
  --image_rate_gradient_weight 0.25 --label_rate_gradient_weight 1 \
  --label_gradient_cap 0.1 --label_step 0.35 --label_groups "$LABEL_GROUPS" \
  --label_entropy_lr 0.001 --class_batch_size "$CLASS_BATCH_SIZE" \
  --utility_batch_size "$UTILITY_BATCH_SIZE" \
  --label_feature_chunk "$LABEL_FEATURE_CHUNK" \
  --label_entropy_batch "$LABEL_ENTROPY_BATCH" \
  --codec_workers "$CODEC_WORKERS" --codec_lr 0.001 --encoder_gain 16 \
  --ldb 0.1 --lr_it 1000 --allow_tf32 \
  --log_every 1 --checkpoint_every 1000 --skip_final_merge
touch "$RUN/joint_complete"
echo "[$(date --iso-8601=seconds)] LPLD-ROSD joint optimization complete"

echo "[$(date --iso-8601=seconds)] post-quantization decode start"
for wave in "0 1" "2 3"; do
  pids=()
  for rank in $wave; do
    class_start=$((rank * 250))
    class_end=$((class_start + 250))
    shard="$POSTQUANT/shard${rank}.pt"
    if [[ -f "$shard" ]]; then
      continue
    fi
    gpu=$(echo "$GPU_IDS" | cut -d, -f$((rank + 1)))
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u TM/quantize_decode_tensorpool_shard.py \
      --input "$RUN/rank${rank}_${class_start}_${class_end}/pool_${ITERATIONS}_global_keys.pt" \
      --output "$shard" \
      --quantized_output "$POSTQUANT/pool_quantized_shard${rank}.pt" \
      --class_start "$class_start" --classes_per_shard 250 --ipc "$IPC" \
      --image_size 224 --max_iter "$ITERATIONS" --workers "$WORKERS" \
      --mse_threshold 5e-7 --encoder_gain 16 \
      --teacher_path "$TEACHER" \
      --label_codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
      --postquant_label_rate_samples 1 \
      > "$POSTQUANT/shard${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

if [[ ! -f "$SYNTHETIC" ]]; then
  read -r LABEL_KIB LABEL_MODEL_KIB < <(
    "$PYTHON" - "$RUN/label_codec_${ITERATIONS}.pt" <<'PY'
import sys
import torch
config = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
print(config["estimated_label_kib_per_class"], config["label_model_kib_per_class"])
PY
  )
  "$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "$POSTQUANT/shard0.pt" "$POSTQUANT/shard1.pt" \
             "$POSTQUANT/shard2.pt" "$POSTQUANT/shard3.pt" \
    --output "$SYNTHETIC" --ipc "$IPC" --num_classes 1000 \
    --image_size 224 --label_kib "$LABEL_KIB" \
    --label_model_kib "$LABEL_MODEL_KIB" --encoder_gain 16 \
    --label_groups "$LABEL_GROUPS" --utility_mode lpld_class_bn \
    > "$POSTQUANT/merge.log" 2>&1
fi
touch "$POSTQUANT/complete"

# The default compression 3 builds 100 independent augmentation groups.
# The four target-rate pools below are hard-link subsets, so they do not
# duplicate the multi-gigabyte label payload on disk.
if [[ ! -f "$SOURCE_POOL/pool_summary.pt" ]]; then
  mkdir -p "$SOURCE_POOL"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$SOURCE_POOL" --output_path "$DOWNSTREAM/relabel_g100" \
    --data_path "$DATA_PATH" --dataset ImageNet1K --epochs 300 \
    --pool_compression "$SOURCE_POOL_COMPRESSION" --crop_size 224 --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 128 --workers "$WORKERS" \
    --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 \
    --fast_downstream --log_every 100 \
    > "$DOWNSTREAM/relabel_g100.stdout.log" 2>&1
fi

IFS=',' read -r -a targets <<< "$TARGETS_CSV"
for target in "${targets[@]}"; do
  pool="$DOWNSTREAM/fkd_pool_target${target}"
  mkdir -p "$pool"
  if [[ ! -f "$pool/pool_summary.pt" ]]; then
    "$PYTHON" TM/build_fkd_pool_target.py \
      --sources "$SOURCE_POOL" --output "$pool" \
      --target_total_kib "$target" --num_classes 1000 \
      > "$DOWNSTREAM/build_target${target}.log" 2>&1
  fi

  # Evaluate the same images and FKD pool under the two official protocols.
  # LPLD uses fixed T=20. LPQLD adds DKR and calibrated alignment (CA).
  for protocol in lpld lpqld; do
    output="$DOWNSTREAM/${protocol}_target${target}"
    mkdir -p "$output"
    protocol_args=(--dkr_schedule none)
    if [[ "$protocol" == "lpqld" ]]; then
      protocol_args=(
        --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7
        --dkr_min_temperature 2 --ca_dynamic
      )
    fi

    echo "[$(date --iso-8601=seconds)] downstream protocol=${protocol} target=${target} KiB/class start"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
      --mode train_pool --synthetic_path "$SYNTHETIC" \
      --teacher_path "$TEACHER" --fkd_path "$pool" --output_path "$output" \
      --data_path "$DATA_PATH" --dataset ImageNet1K --epochs 300 \
      --crop_size 224 --temperature 20 --train_batch 128 --workers "$WORKERS" \
      --optimizer adamw --learning_rate 0.001 --weight_decay 0.01 \
      --eval_every 10 --seed 0 --device_ids 0 1 2 3 \
      --fast_downstream "${protocol_args[@]}" \
      > "$output/train_pool.stdout.log" 2>&1
    touch "$output/complete"
    echo "[$(date --iso-8601=seconds)] downstream protocol=${protocol} target=${target} complete"
  done
done

touch "$RUN/complete"
echo "[$(date --iso-8601=seconds)] LPLD-ROSD multi-rate pipeline complete"
