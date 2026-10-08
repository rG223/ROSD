#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_SRE2L="${PYTHON_SRE2L:-python}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to tiny-imagenet-200}"
TEACHER="${TEACHER:?Set TEACHER to the ResNet18-BN checkpoint}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed ROSD run}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
LABEL_GROUPS="${LABEL_GROUPS:-30}"
LABEL_CODEC="${LABEL_CODEC:-${RUN_DIR}/label_codec_400.pt}"
OUTPUT="${OUTPUT:-${RUN_DIR}/downstream_fkd_official_sgd_t2_seed${SEED}}"
FKD_POOL="${OUTPUT}/fkd_pool"

test -f "${RUN_DIR}/synthetic.pt"
test -f "${LABEL_CODEC}"
test -f "${TEACHER}"
mkdir -p "${OUTPUT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

"${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode relabel_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --codec_checkpoint "${LABEL_CODEC}" \
  --fkd_path "${FKD_POOL}" --output_path "${OUTPUT}" \
  --data_path "${DATA_PATH}" --dataset Tiny --epochs "${EPOCHS}" \
  --pool_compression 1 --crop_size 64 --min_crop_scale 0.08 \
  --temperature 20 --loader_batch 128 --workers 8 \
  --fkd_seed 42 --seed "${SEED}" --device_ids 0 \
  > "${OUTPUT}/relabel.stdout.log" 2>&1

"${PYTHON_SRE2L}" -u "${ROOT}/TM/sre2l_fkd.py" \
  --mode train_pool --synthetic_path "${RUN_DIR}/synthetic.pt" \
  --teacher_path "${TEACHER}" --fkd_path "${FKD_POOL}" \
  --output_path "${OUTPUT}" --data_path "${DATA_PATH}" --dataset Tiny \
  --epochs "${EPOCHS}" --crop_size 64 --temperature 20 \
  --train_batch 64 --workers 8 --optimizer sgd --learning_rate 0.2 \
  --momentum 0.9 --weight_decay 0.0001 --warmup_epochs 5 \
  --warmup_start_factor 0.01 --scale_loss_by_temperature_squared \
  --eval_every 10 --seed "${SEED}" --device_ids 0 \
  > "${OUTPUT}/train.stdout.log" 2>&1

"${PYTHON_SRE2L}" - "${OUTPUT}" "${LABEL_GROUPS}" <<'PY'
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
groups = int(sys.argv[2])
best = float("-inf")
final = float("nan")
for line in (output / "log.txt").read_text(errors="replace").splitlines():
    match = re.search(r"test_acc=([0-9.]+)", line)
    if match:
        final = 100.0 * float(match.group(1))
        best = max(best, final)
(output / "summary.txt").write_text(
    f"label_groups={groups}\nbest_top1={best:.4f}\nfinal_top1={final:.4f}\n"
)
PY

cat "${OUTPUT}/summary.txt"
