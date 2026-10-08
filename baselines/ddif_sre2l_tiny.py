"""Tiny-ImageNet DDiF parameterization trained with SRe2L CE+BN utility.

This is an isolated DDiF baseline. Each synthetic slot owns an independent
two-hidden-layer SIREN. Independent fields are represented by batched tensors
to make the official parameterization practical for 200 classes; no DD-RUO,
entropy model, parameter quantization, or soft-label rate gradient is used.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import get_time, save_and_print, set_seed
from TM.sre2l_bn import (
    BNFeatureLoss,
    build_resnet18_bn,
    cda_augment,
    dataset_context,
    real_initialization,
)


class BatchedSiren(nn.Module):
    """One independent official-style SIREN per synthetic image."""

    def __init__(self, count, hidden, w0_initial=30.0, w0=10.0):
        super().__init__()
        self.count = int(count)
        self.hidden = int(hidden)
        self.w0_initial = float(w0_initial)
        self.w0 = float(w0)
        self.weight1 = nn.Parameter(torch.empty(count, hidden, 2))
        self.bias1 = nn.Parameter(torch.empty(count, hidden))
        self.weight2 = nn.Parameter(torch.empty(count, hidden, hidden))
        self.bias2 = nn.Parameter(torch.empty(count, hidden))
        self.weight3 = nn.Parameter(torch.empty(count, 3, hidden))
        self.bias3 = nn.Parameter(torch.empty(count, 3))
        self.reset_parameters()

    @property
    def parameters_per_field(self):
        return self.hidden**2 + 7 * self.hidden + 3

    def reset_parameters(self):
        first_bound = 0.5
        hidden_bound = math.sqrt(6.0 / self.hidden) / self.w0
        nn.init.uniform_(self.weight1, -first_bound, first_bound)
        nn.init.uniform_(self.bias1, -first_bound, first_bound)
        nn.init.uniform_(self.weight2, -hidden_bound, hidden_bound)
        nn.init.uniform_(self.bias2, -hidden_bound, hidden_bound)
        nn.init.uniform_(self.weight3, -hidden_bound, hidden_bound)
        nn.init.uniform_(self.bias3, -hidden_bound, hidden_bound)

    def forward(self, indices, coordinates):
        hidden = torch.einsum("pi,bhi->bph", coordinates, self.weight1[indices])
        hidden = torch.sin(
            self.w0_initial * (hidden + self.bias1[indices, None])
        )
        hidden = torch.einsum("bpi,bhi->bph", hidden, self.weight2[indices])
        hidden = torch.sin(self.w0 * (hidden + self.bias2[indices, None]))
        return (
            torch.einsum("bpi,boi->bpo", hidden, self.weight3[indices])
            + self.bias3[indices, None]
        )


def make_coordinates(height, width, device):
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def decode(fields, indices, coordinates, image_size):
    height, width = image_size
    values = fields(indices, coordinates)
    return values.reshape(-1, height, width, 3).permute(0, 3, 1, 2)


def decode_chunked(fields, indices, coordinates, image_size, chunk):
    outputs = []
    for start in range(0, indices.numel(), chunk):
        outputs.append(
            decode(
                fields, indices[start:start + chunk], coordinates, image_size
            )
        )
    return torch.cat(outputs)


def balanced_slot_indices(classes, ipc, slot_start, slot_count, device):
    class_ids = torch.arange(classes, device=device)
    slots = torch.arange(slot_start, slot_start + slot_count, device=device)
    return (class_ids[:, None] * ipc + slots[None]).reshape(-1)


def warmup_fields(args, fields, targets, coordinates):
    target_values = targets.permute(0, 2, 3, 1).reshape(targets.shape[0], -1, 3)
    optimizer = torch.optim.Adam(fields.parameters(), lr=args.init_lr)
    start_time = time.time()
    for iteration in range(1, args.init_iterations + 1):
        sampled_gpu = torch.randperm(
            coordinates.shape[0], device=coordinates.device
        )[:args.init_pixels]
        sampled_cpu = sampled_gpu.cpu()
        sampled_coordinates = coordinates[sampled_gpu]
        optimizer.zero_grad(set_to_none=True)
        average_mse = 0.0
        for start in range(0, fields.count, args.init_field_chunk):
            end = min(start + args.init_field_chunk, fields.count)
            indices = torch.arange(start, end, device=coordinates.device)
            predicted = fields(indices, sampled_coordinates)
            expected = target_values[start:end, sampled_cpu].to(
                coordinates.device, non_blocking=True
            )
            weight = (end - start) / fields.count
            loss = F.mse_loss(predicted, expected) * weight
            loss.backward()
            average_mse += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(fields.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % args.log_every == 0:
            save_and_print(
                args.log_path,
                f"warmup iter={iteration:04d}/{args.init_iterations} "
                f"sampled_mse={average_mse:.8f} "
                f"sec_per_iter={(time.time() - start_time) / iteration:.3f}",
            )


def utility_min_crop(iteration, total_iterations, mild_min, full_min,
                     mild_fraction):
    """Hold a mild crop first, then broaden smoothly to the FKD crop range."""
    progress = (iteration - 1) / max(1, total_iterations - 1)
    if progress <= mild_fraction:
        return mild_min
    transition = (progress - mild_fraction) / max(1e-8, 1.0 - mild_fraction)
    return mild_min + transition * (full_min - mild_min)


@torch.no_grad()
def teacher_accuracy(fields, teacher, labels, coordinates, image_size,
                     classes, ipc, slots_per_class, field_chunk,
                     min_crop=None, flip_probability=0.0, trials=1):
    correct = 0
    count = 0
    for _ in range(trials):
        for slot_start in range(0, ipc, slots_per_class):
            slot_count = min(slots_per_class, ipc - slot_start)
            indices = balanced_slot_indices(
                classes, ipc, slot_start, slot_count, coordinates.device
            )
            images = decode_chunked(
                fields, indices, coordinates, image_size, field_chunk
            )
            if min_crop is not None:
                images = cda_augment(
                    images, min_crop, 1.0, image_size, flip_probability, 0
                )
            batch_labels = labels[indices]
            correct += (teacher(images).argmax(1) == batch_labels).sum().item()
            count += batch_labels.numel()
    return correct / count


def save_fields(path, fields, args, classes, image_kib):
    torch.save(
        {
            "method": "DDiF with SRe2L utility",
            "fields": fields.state_dict(),
            "classes": classes,
            "ipc": args.ipc,
            "field_width": args.field_width,
            "field_bits": args.field_bits,
            "image_kib_per_class": image_kib,
        },
        path,
    )


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    args.log_path = os.path.join(args.save_path, "log.txt")
    args.device = "cuda:0"
    args.zca = False
    set_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    _, image_size, classes, _, mean, std, train_set, _, _, _, _, _ = dataset_context(args)
    if tuple(image_size) != (64, 64):
        raise ValueError(f"Expected Tiny-ImageNet 64x64, got {image_size}")
    checkpoint = torch.load(args.teacher_path, map_location="cpu", weights_only=False)
    teacher = build_resnet18_bn(classes, image_size).to(args.device)
    teacher.load_state_dict(checkpoint["state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    targets, labels = real_initialization(train_set, args.ipc, classes)
    targets = targets.pin_memory()
    labels = labels.to(args.device)
    fields = BatchedSiren(
        classes * args.ipc, args.field_width, args.w0_initial, args.w0
    ).to(args.device)
    coordinates = make_coordinates(*image_size, args.device)
    image_kib = (
        args.ipc * fields.parameters_per_field * args.field_bits / 8192.0
    )
    if image_kib > args.image_budget_kib:
        raise ValueError(
            f"Field payload {image_kib:.4f} KiB/class exceeds "
            f"budget {args.image_budget_kib:.4f}"
        )

    save_and_print(args.log_path, f"begin={get_time()}")
    save_and_print(
        args.log_path,
        "method=DDiF_SRe2L objective=teacher_CE_plus_BN "
        "image_entropy=false image_quantization=false soft_label_training=false "
        f"classes={classes} ipc={args.ipc} fields={classes * args.ipc} "
        f"field_width={args.field_width} params_per_field={fields.parameters_per_field} "
        f"field_bits={args.field_bits} image_kib_per_class={image_kib:.4f} "
        f"image_budget_kib={args.image_budget_kib:.4f} "
        f"expected_10x_label_kib={args.expected_label_kib:.2f} "
        f"expected_total_kib={image_kib + args.expected_label_kib:.2f}",
    )

    warmup_fields(args, fields, targets, coordinates)
    save_fields(
        os.path.join(args.save_path, "fields_warmup.pt"),
        fields, args, classes, image_kib,
    )
    del targets

    optimizer = torch.optim.Adam(fields.parameters(), lr=args.field_lr)
    bn_loss = BNFeatureLoss(teacher)
    micro_slots = args.utility_slots_per_class
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        min_crop = utility_min_crop(
            iteration,
            args.iterations,
            args.aug_mild_min_crop,
            args.aug_full_min_crop,
            args.aug_mild_fraction,
        )
        utility_sum = 0.0
        ce_sum = 0.0
        bn_sum = 0.0
        correct = 0
        count = 0
        for slot_start in range(0, args.ipc, micro_slots):
            slot_count = min(micro_slots, args.ipc - slot_start)
            indices = balanced_slot_indices(
                classes, args.ipc, slot_start, slot_count, args.device
            )
            images = decode_chunked(
                fields, indices, coordinates, image_size, args.field_chunk
            )
            teacher_inputs = cda_augment(
                images,
                min_crop,
                1.0,
                image_size,
                args.aug_flip_probability,
                0,
            )
            batch_labels = labels[indices]
            bn_loss.clear()
            logits = teacher(teacher_inputs)
            ce = F.cross_entropy(logits, batch_labels)
            bn = bn_loss.value()
            micro_weight = slot_count / args.ipc
            utility = (ce + args.bn_weight * bn) * micro_weight
            utility.backward()
            utility_sum += float(utility.detach())
            ce_sum += float(ce.detach()) * micro_weight
            bn_sum += float(bn.detach()) * micro_weight
            correct += (logits.argmax(1) == batch_labels).sum().item()
            count += batch_labels.numel()
            del images, logits, utility

        torch.nn.utils.clip_grad_norm_(fields.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            save_and_print(
                args.log_path,
                f"iter={iteration:04d}/{args.iterations} utility={utility_sum:.6f} "
                f"ce={ce_sum:.6f} bn={bn_sum:.6f} teacher_acc={correct / count:.4f} "
                f"aug_min_crop={min_crop:.4f} "
                f"sec_per_iter={elapsed / iteration:.2f} "
                f"eta_hours={(elapsed / iteration) * (args.iterations - iteration) / 3600:.2f}",
            )
        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            save_fields(
                os.path.join(args.save_path, f"fields_{iteration}.pt"),
                fields, args, classes, image_kib,
            )

    bn_loss.close()
    clean_teacher_acc = teacher_accuracy(
        fields, teacher, labels, coordinates, image_size, classes, args.ipc,
        args.utility_slots_per_class, args.field_chunk,
    )
    augmented_teacher_acc = teacher_accuracy(
        fields, teacher, labels, coordinates, image_size, classes, args.ipc,
        args.utility_slots_per_class, args.field_chunk,
        min_crop=args.aug_full_min_crop,
        flip_probability=args.aug_flip_probability,
        trials=args.aug_eval_trials,
    )
    save_and_print(
        args.log_path,
        f"final_teacher_diagnostics clean_acc={clean_teacher_acc:.4f} "
        f"aug_acc={augmented_teacher_acc:.4f} "
        f"aug_trials={args.aug_eval_trials} "
        f"aug_min_crop={args.aug_full_min_crop:.4f}",
    )
    decoded = []
    with torch.no_grad():
        all_indices = torch.arange(fields.count, device=args.device)
        for start in range(0, fields.count, args.export_chunk):
            decoded.append(
                decode(
                    fields,
                    all_indices[start:start + args.export_chunk],
                    coordinates,
                    image_size,
                ).cpu()
            )
    torch.save(
        {
            "images": torch.cat(decoded),
            "labels": labels.cpu(),
            "mean": mean,
            "std": std,
            "ipc": args.ipc,
            "parameterization": "DDiF independent SIREN",
            "image_kib_per_class": image_kib,
            "clean_teacher_acc": clean_teacher_acc,
            "augmented_teacher_acc": augmented_teacher_acc,
        },
        os.path.join(args.save_path, "synthetic.pt"),
    )
    save_and_print(
        args.log_path,
        f"complete={get_time()} image_kib_per_class={image_kib:.4f}",
    )


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", default="Tiny")
    result.add_argument("--subset", default="")
    result.add_argument("--data_path", required=True)
    result.add_argument("--teacher_path", required=True)
    result.add_argument("--save_path", required=True)
    result.add_argument("--ipc", type=int, default=50)
    result.add_argument("--batch_real", type=int, default=512)
    result.add_argument("--field_width", type=int, default=18)
    result.add_argument("--w0_initial", type=float, default=30.0)
    result.add_argument("--w0", type=float, default=10.0)
    result.add_argument("--field_bits", type=int, default=32)
    result.add_argument("--image_budget_kib", type=float, default=88.54)
    result.add_argument("--expected_label_kib", type=float, default=221.46)
    result.add_argument("--init_iterations", type=int, default=500)
    result.add_argument("--init_pixels", type=int, default=1024)
    result.add_argument("--init_field_chunk", type=int, default=5000)
    result.add_argument("--init_lr", type=float, default=5e-4)
    result.add_argument("--iterations", type=int, default=2000)
    result.add_argument("--field_lr", type=float, default=1e-4)
    result.add_argument("--bn_weight", type=float, default=0.05)
    result.add_argument("--aug_mild_min_crop", type=float, default=0.5)
    result.add_argument("--aug_full_min_crop", type=float, default=0.08)
    result.add_argument("--aug_mild_fraction", type=float, default=0.5)
    result.add_argument("--aug_flip_probability", type=float, default=0.5)
    result.add_argument("--aug_eval_trials", type=int, default=5)
    result.add_argument("--utility_slots_per_class", type=int, default=20)
    result.add_argument("--field_chunk", type=int, default=2000)
    result.add_argument("--export_chunk", type=int, default=512)
    result.add_argument("--log_every", type=int, default=10)
    result.add_argument("--checkpoint_every", type=int, default=100)
    result.add_argument("--seed", type=int, default=0)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
