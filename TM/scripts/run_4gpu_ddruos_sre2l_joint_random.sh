#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TORCHRUN="${TORCHRUN:-torchrun}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?set TEACHER to the ResNet18-BN checkpoint}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BN_WEIGHT="${BN_WEIGHT:-0.1}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
OUT="${OUT:-$RESULTS_ROOT/${STAMP}/ddruos_sre2l_joint_random_ipc100_10x_4gpu}"
ITERATIONS="${ITERATIONS:-4000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
mkdir -p "$OUT"

cat > "$OUT/run.conf" <<EOF
method=ROSD
utility=SRe2L_CE_plus_${BN_WEIGHT}_BN
initialization=random_TensorPool_codec
iterations=$ITERATIONS
augmentation=RRC_scale_0.08_1.0_flip_0.5_jitter_4
bn_loss=sre2l_L2_sum
first_bn_multiplier=10
posthoc_clamp=false
optimization_pixel_projection=false
lambda_image=0.0001
lambda_label=0.0001
label_gradient_cap=0.1
label_groups=30
stored_ipc=100
parallelism=four_class_shards_one_seed
utility_batch_size=1000
label_feature_chunk=1000
EOF

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

"$TORCHRUN" --standalone --nproc-per-node=4 \
  TM/ddruos_sre2l_joint_distributed.py \
  --data_path "$DATA_PATH" --teacher_path "$TEACHER" --save_path "$OUT" \
  --ipc 100 --iterations "$ITERATIONS" --seed 0 \
  --bn_weight "$BN_WEIGHT" --bn_loss_mode sre2l_l2_sum \
  --first_bn_multiplier 10 --jitter 4 \
  --lambda_image 0.0001 --lambda_label 0.0001 \
  --label_gradient_cap 0.1 --label_step 0.35 --label_groups 30 \
  --utility_batch_size 1000 --label_feature_chunk 1000 \
  --label_entropy_batch 1000 --codec_workers 12 \
  --log_every 10 --checkpoint_every 1000
