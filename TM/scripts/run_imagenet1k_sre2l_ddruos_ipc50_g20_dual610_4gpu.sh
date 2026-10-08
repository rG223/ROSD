#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA3_ROOT="${DATA3_ROOT:-/data3/ypliu/DD-RUO-1}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
SOURCE_TEACHER="${SOURCE_TEACHER:-/data2/home/ypliu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
TEACHER="${TEACHER:-$DATA3_ROOT/assets/imagenet1k/resnet18-f37072fd.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
IPC="${IPC:-50}"
ITERATIONS="${ITERATIONS:-4000}"
LABEL_GROUPS="${LABEL_GROUPS:-20}"
COMBINED_TARGET_KIB="${COMBINED_TARGET_KIB:-610}"
CLASS_BATCH_SIZE="${CLASS_BATCH_SIZE:-5}"
RUN_DIR="${1:-$DATA3_ROOT/results/imagenet1k_ipc${IPC}/SRe2L_DDRUOS_IPC${IPC}_G${LABEL_GROUPS}_dual${COMBINED_TARGET_KIB}_4GPU_seed0_${STAMP}}"
PYTHON="${PYTHON:-${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "$(dirname "$TEACHER")" "$RUN_DIR"
if [[ ! -f "$TEACHER" ]]; then
  cp --reflink=auto "$SOURCE_TEACHER" "$TEACHER.tmp"
  mv "$TEACHER.tmp" "$TEACHER"
fi

cat > "$RUN_DIR/run.conf" <<EOF
dataset=ImageNet-1K
image_size=224
ipc=$IPC
iterations=$ITERATIONS
label_groups=$LABEL_GROUPS
label_step=0.35
combined_target_kib_per_class=$COMBINED_TARGET_KIB
class_batch_size_per_rank=$CLASS_BATCH_SIZE
teacher=torchvision_resnet18_imagenet1k
outputs=$RUN_DIR
EOF

printf '%s\n' "$RUN_DIR" > "$DATA3_ROOT/latest_imagenet1k_sre2l_ddruos_run.txt"
echo "[$(date --iso-8601=seconds)] ImageNet-1K joint optimization start" | tee "$RUN_DIR/pipeline.log"

CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=4 \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$ROOT/TM/ddruos_sre2l_joint_distributed.py" \
  --dataset ImageNet1K --num_classes 1000 --image_size 224 \
  --data_path "$DATA_PATH" --teacher_path "$TEACHER" \
  --utility_backend sre2l --save_path "$RUN_DIR" \
  --ipc "$IPC" --iterations "$ITERATIONS" --seed 0 \
  --bn_weight 1 --bn_loss_mode sre2l_l2_sum --first_bn_multiplier 10 \
  --jitter 0 --combined_dual --combined_target_kib "$COMBINED_TARGET_KIB" \
  --dual_init 0.0001 --dual_lr 0.002 --dual_rho 0.00005 \
  --dual_ema_decay 0.95 --dual_update_every 10 --dual_deadband 0.02 \
  --dual_min -0.00005 --dual_max 1 \
  --image_rate_gradient_weight 0.25 --label_rate_gradient_weight 1 \
  --label_gradient_cap 0.1 --label_step 0.35 --label_groups "$LABEL_GROUPS" \
  --label_entropy_lr 0.001 --class_batch_size "$CLASS_BATCH_SIZE" \
  --utility_batch_size 64 --label_feature_chunk 64 --label_entropy_batch 64 \
  --codec_workers 3 --codec_lr 0.001 --encoder_gain 16 \
  --ldb 0.1 --lr_it 1000 --allow_tf32 \
  --log_every 1 --checkpoint_every 1000 --skip_final_merge \
  2>&1 | tee -a "$RUN_DIR/pipeline.log"

echo "[$(date --iso-8601=seconds)] ImageNet-1K joint optimization complete" \
  | tee -a "$RUN_DIR/pipeline.log"
