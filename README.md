# ROSD

This repository contains the training and downstream evaluation entry points
for ROSD. Experimental outputs and checkpoints are intentionally excluded.

## Installation

Create the environment from the provided specification:

```bash
conda env create -f TM/scripts/environment.yml
conda activate dd_ruo
```

Set the Python executables when DD-RUO and downstream FKD evaluation use
different environments:

```bash
export PYTHON_DDRUO=/path/to/dd_ruo/bin/python
export PYTHON_SRE2L=/path/to/sre2l/bin/python
```

Run every command from the repository root.

## Required Paths

The launchers accept path overrides through environment variables. At minimum,
set the dataset and teacher checkpoint paths:

```bash
export DATA_PATH=/path/to/dataset
export TEACHER=/path/to/resnet18_teacher.pth
```

For LPLD-based runs, also provide the class-conditional BN teacher:

```bash
export CLASS_TEACHER=/path/to/resnet18_class_bn.pth
```

Use `RUN_DIR` to place checkpoints and generated payloads on a large disk:

```bash
export RUN_DIR=/data3/$USER/rosd/my_run
```

## Tiny-ImageNet

Run the IPC-50 pipeline on one GPU:

```bash
export DATA_PATH=/path/to/tiny-imagenet-200
export TEACHER=/path/to/tiny_resnet18_bn.pth
export GPU=0

bash TM/scripts/run_rosd_tiny_ipc50.sh
```

Common overrides:

```bash
IPC=100 \
LABEL_GROUPS=20 \
IMAGE_BUDGET_KIB=190 \
LABEL_BUDGET_KIB=430 \
JOINT_ITERS=400 \
RUN_DIR=/data3/$USER/rosd/tiny_ipc100 \
bash TM/scripts/run_rosd_tiny_ipc50.sh
```

Rerun downstream evaluation from a completed run:

```bash
export RUN_DIR=/data3/$USER/rosd/tiny_ipc100
export LABEL_CODEC="$RUN_DIR/label_codec_400.pt"
export DATA_PATH=/path/to/tiny-imagenet-200
export TEACHER=/path/to/tiny_resnet18_bn.pth
export GPU=0

bash TM/scripts/tiny_ipc50/eval_ddruos_fkd.sh
```

## ImageNet-1K: SRe2L Initialization

Launch class-sharded joint optimization on four GPUs:

```bash
export DATA_PATH=/path/to/imagenet
export TEACHER=/path/to/resnet18_teacher.pth
export GPU_IDS=0,1,2,3
export IPC=50
export LABEL_GROUPS=20
export COMBINED_TARGET_KIB=610
export RUN_DIR=/data3/$USER/rosd/imagenet1k_sre2l_ipc50

bash TM/scripts/run_rosd_imagenet1k_sre2l_4gpu.sh "$RUN_DIR"
```

## ImageNet-1K: LPLD Initialization

Launch LPLD-initialized joint optimization and downstream evaluation:

```bash
export DATA_PATH=/path/to/imagenet
export TEACHER=/path/to/resnet18_teacher.pth
export CLASS_TEACHER=/path/to/resnet18_class_bn.pth
export GPU_IDS=0,1,2,3
export IPC=50
export LABEL_GROUPS=20
export COMBINED_TARGET_KIB=610
export TARGETS_CSV=1160,620,290,150
export RUN_DIR=/data3/$USER/rosd/imagenet1k_lpld_ipc50

bash TM/scripts/run_rosd_imagenet1k_lpld_4gpu.sh
```

## Variable IPC and Target Rate

The public launcher forwards IPC, label-group, and target-rate settings to the
joint optimization and evaluation pipeline:

```bash
export DATA_PATH=/path/to/dataset
export TEACHER=/path/to/resnet18_teacher.pth
export CLASS_TEACHER=/path/to/resnet18_class_bn.pth
export GPU_IDS=0,1,2,3
export IPC=100
export LABEL_GROUPS=40
export COMBINED_TARGET_KIB=600
export TARGETS_CSV=610,440,330,295

bash TM/scripts/run_rosd_variable_ipc_target_kib.sh \
  /data3/$USER/rosd/ipc100_target600
```

## Multi-Rate Evaluation

Evaluate a completed ImageNet-1K run at several target payloads:

```bash
export RUN_DIR=/data3/$USER/rosd/imagenet1k_sre2l_ipc50
export DATA_PATH=/path/to/imagenet
export TEACHER=/path/to/resnet18_teacher.pth
export GPU_IDS=0,1,2,3
export TARGETS_CSV=1160,620,290,150

bash TM/scripts/eval_rosd_imagenet1k_multirate_4gpu.sh
```

Network post-quantization is enabled by these launchers. Set
`--postquant_label_rate_samples 0` only when invoking
`TM/quantize_decode_tensorpool_shard.py` directly and intentionally using the
legacy image-codec-only selection rule.

## Outputs

Each run directory contains the following artifacts when its corresponding
stage completes:

```text
run.conf
log.txt or pipeline.log
pool_*.pt
label_codec_*.pt
postquant/
synthetic*.pt
downstream*/
```

Use the completion marker files in the run directory to resume scripts without
repeating finished stages.
