#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_DDRUO:-/data2/home/ypliu/anaconda3/envs/dd_ruo/bin/python}"
RUN="${RUN_DIR:-/data3/ypliu/DD-RUO-1/results/imagenet1k_ipc50/SRe2L_DDRUOS_IPC50_G20_dual610_4GPU_seed0_20260826_124447}"
DATA_PATH="${DATA_PATH:-/data2/home/ypliu}"
TEACHER="${TEACHER:-/data3/ypliu/DD-RUO-1/assets/imagenet1k/resnet18-f37072fd.pth}"
ITERATIONS="${ITERATIONS:-4000}"
IPC="${IPC:-50}"
CLASSES="${CLASSES:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
LABEL_GROUPS="${LABEL_GROUPS:-20}"
EPOCHS="${EPOCHS:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS="${WORKERS:-12}"
OUT="$RUN/downstream_official_lpqld_300e_seed0"
POSTQUANT="$RUN/postquant_imagenet1k_iter${ITERATIONS}"
FKD="$OUT/fkd_pool_g${LABEL_GROUPS}"
TRAIN_OUT="$OUT/train_seed0"
SYNTHETIC="$RUN/synthetic_postquant_imagenet1k_iter${ITERATIONS}.pt"

mkdir -p "$POSTQUANT" "$FKD" "$TRAIN_OUT"
test -f "$RUN/label_codec_${ITERATIONS}.pt"
test -f "$TEACHER"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

echo "[$(date --iso-8601=seconds)] official ImageNet-1K downstream start" \
  | tee -a "$OUT/pipeline.log"

# Decode two 250-class shards at a time. Four concurrent torch.load + decoded
# tensors can exceed host RAM; downstream relabeling and training still use all
# four GPUs with the official global batch size of 128.
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
      --image_size "$IMAGE_SIZE" --max_iter "$ITERATIONS" \
      --workers 12 --mse_threshold 5e-7 --encoder_gain 16 \
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
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
print(c["estimated_label_kib_per_class"], c["label_model_kib_per_class"])
PY
  )
  "$PYTHON" -u TM/merge_ddruos_quantized_shards.py \
    --inputs "$POSTQUANT/shard0.pt" "$POSTQUANT/shard1.pt" \
             "$POSTQUANT/shard2.pt" "$POSTQUANT/shard3.pt" \
    --output "$SYNTHETIC" --ipc "$IPC" --num_classes "$CLASSES" \
    --image_size "$IMAGE_SIZE" --label_kib "$LABEL_KIB" \
    --label_model_kib "$LABEL_MODEL_KIB" --encoder_gain 16 \
    --label_groups "$LABEL_GROUPS" \
    > "$POSTQUANT/merge.log" 2>&1
fi
touch "$POSTQUANT/complete"

# Twenty stored augmentation groups over a 300-epoch schedule correspond to a
# 15x LPLD batch-pool compression. The label codec reconstructs quantized
# pre-softmax logits and records their entropy-estimated payload.
if [[ ! -f "$FKD/pool_summary.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
    --mode relabel_pool --synthetic_path "$SYNTHETIC" \
    --teacher_path "$TEACHER" --codec_checkpoint "$RUN/label_codec_${ITERATIONS}.pt" \
    --fkd_path "$FKD" --output_path "$OUT" \
    --data_path "$DATA_PATH" --dataset ImageNet1K --epochs "$EPOCHS" \
    --pool_compression 15 --crop_size "$IMAGE_SIZE" --min_crop_scale 0.08 \
    --temperature 20 --loader_batch 128 --workers "$WORKERS" \
    --fkd_seed 42 --seed 0 --device_ids 0 1 2 3 \
    --fast_downstream --log_every 100 \
    > "$OUT/relabel_pool.stdout.log" 2>&1
fi

# Official LPQLD ImageNet-1K validation recipe: ResNet-18, 300 epochs,
# AdamW(lr=1e-3, wd=1e-2), global batch 128, cosine LR, RRC/flip/CutMix,
# DKR T:20->2 (x0.7/30 epochs), and calibrated student alignment. The paper
# does not apply the Tiny-ImageNet-only T^2 multiplier on ImageNet-1K.
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u TM/sre2l_fkd.py \
  --mode train_pool --synthetic_path "$SYNTHETIC" \
  --teacher_path "$TEACHER" --fkd_path "$FKD" --output_path "$TRAIN_OUT" \
  --data_path "$DATA_PATH" --dataset ImageNet1K --epochs "$EPOCHS" \
  --crop_size "$IMAGE_SIZE" --temperature 20 --train_batch 128 \
  --workers "$WORKERS" --optimizer adamw --learning_rate 0.001 \
  --weight_decay 0.01 --eval_every 10 --seed 0 --device_ids 0 1 2 3 \
  --dkr_schedule step --dkr_step_size 30 --dkr_step_gamma 0.7 \
  --dkr_min_temperature 2 --ca_dynamic --fast_downstream \
  > "$TRAIN_OUT/train_pool.stdout.log" 2>&1

"$PYTHON" - "$TRAIN_OUT" "$FKD" <<'PY'
import re
import sys
from pathlib import Path
import torch
out, fkd = map(Path, sys.argv[1:])
text = (out / "train_pool.log.txt").read_text(errors="replace")
values = [float(x) * 100 for x in re.findall(r"test_acc=([0-9.]+)", text)]
pool = torch.load(fkd / "pool_summary.pt", map_location="cpu", weights_only=False)
(out / "summary.txt").write_text(
    f"best_top1={max(values):.4f}\nfinal_top1={values[-1]:.4f}\n"
    f"image_kib_per_class={pool['image_kib_per_class']:.4f}\n"
    f"label_kib_per_class={pool['entropy_label_kib_per_class']:.4f}\n"
    f"label_model_kib_per_class={pool['label_model_kib_per_class']:.4f}\n"
    f"total_kib_per_class={pool['achieved_total_kib_per_class']:.4f}\n"
)
PY

touch "$OUT/complete"
echo "[$(date --iso-8601=seconds)] official ImageNet-1K downstream complete" \
  | tee -a "$OUT/pipeline.log"
cat "$TRAIN_OUT/summary.txt"
