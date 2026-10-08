#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/home/ypliu/DD-RUO-1"
PYTHON_DDRUO="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
PYTHON_SRE2L="${PYTHON_SRE2L:-/data2/home/ypliu/anaconda3/envs/sre2l/bin/python}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu/datasets/tiny_ddruo/tiny-imagenet-200}"
TEACHER="${TEACHER:-$ROOT/results/tiny_imagenet_1x/Tiny_IPC50_SRe2L_CDA_LPLD_LPQLD_20260808_200102/teacher/ddruo_teacher.pt}"

IPC="${IPC:-50}"
CIM_FACTOR="${CIM_FACTOR:-2}"
CIM_MIPC="${CIM_MIPC:-300}"
CIM_CLASS_BATCH="${CIM_CLASS_BATCH:-12}"
IMAGE_BUDGET_KIB="${IMAGE_BUDGET_KIB:-90}"
LABEL_BUDGET_KIB="${LABEL_BUDGET_KIB:-210}"
IMAGE_MODEL_OVERHEAD_KIB="${IMAGE_MODEL_OVERHEAD_KIB:-9.66}"
IMAGE_PAYLOAD_BUDGET_KIB="${IMAGE_PAYLOAD_BUDGET_KIB:-80.34}"
LABEL_GROUPS="${LABEL_GROUPS:-30}"
LABEL_STEP="${LABEL_STEP:-0.35}"
LABEL_FEATURE_CHUNK="${LABEL_FEATURE_CHUNK:-1536}"
LABEL_ENTROPY_BATCH="${LABEL_ENTROPY_BATCH:-128}"
JOINT_ITERS="${JOINT_ITERS:-400}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
HARD_CE_WEIGHT="${HARD_CE_WEIGHT:-0}"
LABEL_KL_WEIGHT="${LABEL_KL_WEIGHT:-0}"
LABEL_KL_TARGET_FRACTION="${LABEL_KL_TARGET_FRACTION:-0.05}"
LABEL_KL_TEMPERATURE="${LABEL_KL_TEMPERATURE:-2}"
LABEL_RATE_DECODER_ONLY="${LABEL_RATE_DECODER_ONLY:-0}"
POOL_COMPRESSION="${POOL_COMPRESSION:-1}"
SEED="${SEED:-0}"
VISIBLE_GPUS="${VISIBLE_GPUS:-0}"
RESUME_POOL="${RESUME_POOL:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LABEL_STEP_TAG="${LABEL_STEP//./p}"
RUN_DIR="${RUN_DIR:-$ROOT/results/tiny_imagenet_ddruo/Tiny_IPC${IPC}_DDRUOS_officialCIM_f${CIM_FACTOR}_mipc${CIM_MIPC}_groups${LABEL_GROUPS}_step${LABEL_STEP_TAG}_sepX${IMAGE_BUDGET_KIB}_Y${LABEL_BUDGET_KIB}_joint${JOINT_ITERS}_gpu0_seed${SEED}_${STAMP}}"
OUTPUT="$RUN_DIR/downstream_fkd_full_seed${SEED}"
LABELS="$OUTPUT/fkd_pool"

test -d "$DATA_PATH/train"
test -d "$DATA_PATH/val"
test -f "$TEACHER"
mkdir -p "$RUN_DIR" "$OUTPUT"
printf '%s\n' "$RUN_DIR" > "$ROOT/results/tiny_imagenet_ddruo/latest_ddruos_official_cim_ipc50.txt"

cat > "$RUN_DIR/run.conf" <<EOF
method=ROSD
cim_implementation=official_semantics
dataset=Tiny-ImageNet
stored_ipc=$IPC
cim_factor=$CIM_FACTOR
decoded_views_per_class=$((IPC * CIM_FACTOR * CIM_FACTOR))
cim_candidate_ipc=$CIM_MIPC
cim_class_batch=$CIM_CLASS_BATCH
image_budget_kib_per_class=$IMAGE_BUDGET_KIB
label_budget_kib_per_class=$LABEL_BUDGET_KIB
total_budget_kib_per_class=$((IMAGE_BUDGET_KIB + LABEL_BUDGET_KIB))
image_model_overhead_kib_per_class=$IMAGE_MODEL_OVERHEAD_KIB
label_groups_for_rate=$LABEL_GROUPS
label_step=$LABEL_STEP
label_feature_chunk=$LABEL_FEATURE_CHUNK
label_entropy_batch=$LABEL_ENTROPY_BATCH
downstream_epochs=$TRAIN_EPOCHS
hard_ce_weight=$HARD_CE_WEIGHT
label_kl_weight=$LABEL_KL_WEIGHT
label_kl_target_fraction=$LABEL_KL_TARGET_FRACTION
label_kl_temperature=$LABEL_KL_TEMPERATURE
label_rate_decoder_only=$LABEL_RATE_DECODER_ONLY
fkd_pool_compression=${POOL_COMPRESSION}x
visible_gpus=$VISIBLE_GPUS
resume_pool=${RESUME_POOL:-none}
EOF

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

resume_args=()
if [[ -n "$RESUME_POOL" ]]; then
  test -f "$RESUME_POOL"
  resume_args=(--resume_pool "$RESUME_POOL")
fi

gradient_routing_args=()
if [[ "$LABEL_RATE_DECODER_ONLY" == "1" ]]; then
  gradient_routing_args=(--label_rate_decoder_only)
fi

cd "$ROOT"
"$PYTHON_DDRUO" -u TM/cim_ddruo_tensorpool.py \
  --dataset Tiny --data_path "$DATA_PATH" --ipc "$IPC" \
  --teacher_path "$TEACHER" --save_path "$RUN_DIR" \
  --cim_factor "$CIM_FACTOR" --cim_mipc "$CIM_MIPC" \
  --cim_class_batch "$CIM_CLASS_BATCH" \
  --cim_selection_batch 1024 --cim_selection_workers 12 \
  --cim_augmentation crop_cutout_flip --optimize_label_rate \
  --label_step "$LABEL_STEP" --label_groups "$LABEL_GROUPS" \
  --label_entropy_lr 0.001 --label_entropy_warmup 20 \
  --label_feature_chunk "$LABEL_FEATURE_CHUNK" \
  --label_entropy_batch "$LABEL_ENTROPY_BATCH" --label_model_bits 16 \
  --hard_ce_weight "$HARD_CE_WEIGHT" \
  --label_kl_weight "$LABEL_KL_WEIGHT" \
  --label_kl_target_fraction "$LABEL_KL_TARGET_FRACTION" \
  --label_kl_weight_min 0 --label_kl_weight_max 20 --label_kl_ema_decay 0.9 \
  --label_kl_temperature "$LABEL_KL_TEMPERATURE" \
  --separate_rate_budgets --image_target_kib "$IMAGE_BUDGET_KIB" \
  --label_target_kib "$LABEL_BUDGET_KIB" \
  --image_model_overhead_kib "$IMAGE_MODEL_OVERHEAD_KIB" \
  --image_dual_lr 0.0001 --label_dual_lr 0.0001 \
  --warmup_rate_control dual --warmup_target_kib "$IMAGE_BUDGET_KIB" \
  --warmup_rate_margin 1 --warmup_dual_lr 0.001 \
  --fast_single_warmup --fast_warmup_iterations 200 \
  --ldb 0.1 --codec_lr 0.001 --lr_it 1000 \
  --rate_control dual --latent_target_kib "$IMAGE_PAYLOAD_BUDGET_KIB" \
  --stage1_iterations "$JOINT_ITERS" --stage2_iterations 0 \
  --layers_v v5 --arm 32 --dim 4 --log_every 5 --checkpoint_every 100 \
  --network_mse_threshold 5e-7 --enable_cudnn --enable_codec_scheduler --seed "$SEED" \
  "${gradient_routing_args[@]}" \
  "${resume_args[@]}" \
  > "$RUN_DIR/stdout.log" 2>&1

CODEC="$RUN_DIR/label_codec_${JOINT_ITERS}.pt"
"$PYTHON_SRE2L" -u TM/sre2l_fkd.py \
  --mode relabel_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
  --teacher_path "$TEACHER" --codec_checkpoint "$CODEC" \
  --fkd_path "$LABELS" --output_path "$OUTPUT" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs "$TRAIN_EPOCHS" \
  --pool_compression "$POOL_COMPRESSION" --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 128 --workers 12 \
  --fkd_seed 42 --seed "$SEED" --device_ids 0 \
  > "$OUTPUT/relabel_pool.stdout.log" 2>&1

"$PYTHON_SRE2L" -u TM/sre2l_fkd.py \
  --mode train_pool --synthetic_path "$RUN_DIR/synthetic.pt" \
  --teacher_path "$TEACHER" --fkd_path "$LABELS" --output_path "$OUTPUT" \
  --data_path "$DATA_PATH" --dataset Tiny --epochs "$TRAIN_EPOCHS" \
  --crop_size 64 --temperature 20 --train_batch 256 --workers 12 \
  --learning_rate 0.001 --weight_decay 0.01 --eval_every 10 \
  --seed "$SEED" --device_ids 0 \
  > "$OUTPUT/train_pool.stdout.log" 2>&1

touch "$RUN_DIR/.complete"
printf 'complete run_dir=%s\n' "$RUN_DIR"
