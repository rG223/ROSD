"""Faithful SRe2L FKD relabeling and downstream evaluation for tensor datasets."""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TM.sre2l_bn import build_resnet18_bn, dataset_context
from TM.cim_label_codec import AugmentedSlotStepLabelEntropy, logistic_bits
from core.utils import save_and_print, set_seed


_SHARED_SYNTHETIC = {}


class RelabelDataset(Dataset):
    def __init__(self, images, labels, crop_size, crop_scale,
                 augmentation_mode="rrc_cutmix"):
        self.images = images
        self.labels = labels
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.augmentation_mode = augmentation_mode

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        if self.augmentation_mode == "cifar_mixup":
            padded = TF.pad(image, [4, 4, 4, 4])
            top = int(torch.randint(0, 9, ()))
            left = int(torch.randint(0, 9, ()))
            crop = TF.crop(padded, top, left, self.crop_size, self.crop_size)
            flip = bool(torch.rand(()) < 0.5)
            if flip:
                crop = TF.hflip(crop)
            coords = torch.tensor(
                (top, left, self.crop_size, self.crop_size), dtype=torch.int32
            )
            return crop, self.labels[index], index, coords, flip
        top, left, height, width = RandomResizedCrop.get_params(
            image, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        flip = bool(torch.rand(()) < 0.5)
        crop = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [self.crop_size, self.crop_size],
            InterpolationMode.BILINEAR,
            antialias=True,
        )
        if flip:
            crop = TF.hflip(crop)
        coords = torch.tensor((top, left, height, width), dtype=torch.int32)
        return crop, self.labels[index], index, coords, flip


def replay_crop(image, coords, flip, crop_size, augmentation_mode="rrc_cutmix"):
    top, left, height, width = (int(value) for value in coords)
    if augmentation_mode == "cifar_mixup":
        crop = TF.crop(TF.pad(image, [4, 4, 4, 4]), top, left, crop_size, crop_size)
    else:
        crop = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [crop_size, crop_size],
            InterpolationMode.BILINEAR,
            antialias=True,
        )
    return TF.hflip(crop) if bool(flip) else crop


class ReplayDataset(Dataset):
    def __init__(self, images, labels, config, crop_size):
        self.images = images
        self.labels = labels
        self.order = config["order"].long()
        self.coords = config["coords"].int()
        self.flips = config["flips"].bool()
        self.mix_index = config["mix_index"].long()
        self.bbox = tuple(int(value) for value in config["bbox"])
        self.augmentation_mode = config.get("augmentation_mode", "rrc_cutmix")
        self.mix_mode = config.get("mix_mode", "cutmix")
        self.mix_lambda = float(config.get("mix_lambda", 1.0))
        if "logit_values" in config:
            values = config["logit_values"]
            indices = config["logit_indices"].long()
            classes = int(config["num_classes"])
            if values.shape != indices.shape or values.ndim != 2:
                raise ValueError("Invalid MR-k value/index shapes")
            if indices.min() < 0 or indices.max() >= classes:
                raise ValueError("MR-k class index is out of range")
            # -inf gives exactly zero probability to logits removed by MR-k.
            self.logits = values.new_full((values.shape[0], classes), -torch.inf)
            self.logits.scatter_(1, indices, values)
        else:
            self.logits = config["logits"]
        self.crop_size = crop_size

    def __len__(self):
        return self.order.numel()

    def __getitem__(self, position):
        image = replay_crop(
            self.images[int(self.order[position])],
            self.coords[position],
            self.flips[position],
            self.crop_size,
            self.augmentation_mode,
        )
        original_index = int(self.order[position])
        return image, self.labels[original_index], self.logits[position]


class MutableBatchSampler:
    """Keep DataLoader workers alive while replacing epoch batch indices."""

    def __init__(self):
        self.batches = []

    def set_batches(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def rand_bbox(size, lam, generator):
    cut_ratio = math.sqrt(1.0 - lam)
    cut_height = int(size * cut_ratio)
    cut_width = int(size * cut_ratio)
    center_x = int(torch.randint(size, (), generator=generator))
    center_y = int(torch.randint(size, (), generator=generator))
    x1 = max(center_x - cut_height // 2, 0)
    y1 = max(center_y - cut_width // 2, 0)
    x2 = min(center_x + cut_height // 2, size)
    y2 = min(center_y + cut_width // 2, size)
    return x1, y1, x2, y2


def dkr_temperature(args, epoch):
    if args.dkr_schedule == "none":
        return float(args.temperature)
    if args.dkr_schedule == "step":
        temperature = args.temperature * (
            args.dkr_step_gamma ** (epoch // args.dkr_step_size)
        )
        return max(float(args.dkr_min_temperature), float(temperature))
    if args.dkr_schedule == "cosine":
        progress = epoch / max(args.epochs - 1, 1)
        return float(
            args.dkr_min_temperature
            + 0.5
            * (args.temperature - args.dkr_min_temperature)
            * (1.0 + math.cos(math.pi * progress))
        )
    raise ValueError(f"Unsupported DKR schedule: {args.dkr_schedule}")


@torch.inference_mode()
def calibrate_student_temperature_ratio(
    student_logits, teacher_probabilities, teacher_temperature, grid_points,
):
    """LPQLD CA: calibrate student temperature relative to teacher temperature."""
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    ratios = torch.linspace(
        0.01, 1.0, grid_points, device=student_logits.device
    )
    losses = torch.stack(
        [
            F.kl_div(
                F.log_softmax(
                    student_logits / (ratio * teacher_temperature), dim=1
                ),
                teacher_probabilities,
                reduction="batchmean",
            )
            for ratio in ratios
        ]
    )
    return float(ratios[losses.argmin()].item())


def load_synthetic(path):
    cached = _SHARED_SYNTHETIC.get(os.path.abspath(path))
    if cached is not None:
        return cached
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "images_raw" in payload:
        images = payload["images_raw"]
        if images.dtype != torch.float32:
            images = images.float()
        images.clamp_(0.0, 1.0)
        if not images.is_contiguous():
            images = images.contiguous()
    elif "images" in payload:
        images = payload["images"].contiguous()
    else:
        raise KeyError("Synthetic payload must contain images_raw or images")
    label_key = "labels" if "labels" in payload else "hard_labels"
    labels = payload[label_key].long().contiguous()
    if images.ndim != 4 or labels.shape != (images.shape[0],):
        raise ValueError("Invalid synthetic payload shapes")
    return images, labels


def cache_synthetic_for_fork(path):
    """Load one CPU tensor copy for forked downstream workers."""
    key = os.path.abspath(path)
    images, labels = load_synthetic(path)
    _SHARED_SYNTHETIC[key] = (images, labels)
    return images, labels


def load_teacher(path, num_classes, device_ids, crop_size):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    if "fc.weight" in state:
        converted = {}
        for key, value in state.items():
            key = key.replace(".downsample.", ".shortcut.")
            if key.startswith("fc."):
                key = "classifier." + key.removeprefix("fc.")
            converted[key] = value
        state = converted
    teacher = build_resnet18_bn(num_classes, (crop_size, crop_size))
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return nn.DataParallel(teacher, device_ids=device_ids).cuda()


def hard_reference_logits(logits, hard_labels):
    classes = logits.shape[1]
    reference = logits.gather(1, hard_labels[:, None])
    relative = logits - reference
    keep = torch.arange(classes, device=logits.device)[None] != hard_labels[:, None]
    return relative[keep].view(logits.shape[0], classes - 1)


def reconstruct_hard_reference_logits(relative, hard_labels, classes):
    logits = relative.new_zeros((relative.shape[0], classes))
    keep = torch.arange(classes, device=relative.device)[None] != hard_labels[:, None]
    logits[keep] = relative.reshape(-1)
    return logits


class FixedCIMLabelCodec(nn.Module):
    """Fixed-step augmented label codec trained with CIM+DD-RUO."""

    def __init__(self, checkpoint):
        super().__init__()
        config = checkpoint["config"]
        self.label_step = float(config["label_step"])
        self.label_entropy = AugmentedSlotStepLabelEntropy(
            config["num_slots"], classes=config["classes"],
            dims=config["classes"] - 1,
        )
        self.label_entropy.load_state_dict(checkpoint["label_entropy"])

    @torch.inference_mode()
    def quantize(
        self, teacher_logits, hard_labels, order, mix_index, coords, flips,
        bbox, image_size,
    ):
        device = teacher_logits.device
        labels = hard_labels.to(device=device, non_blocking=True)
        slots = order.long().to(device=device, non_blocking=True)
        partners = mix_index.long().to(device=device, non_blocking=True)
        partner_labels = labels[partners]
        partner_slots = slots[partners]
        relative = hard_reference_logits(teacher_logits, labels)
        symbols = torch.round(relative / self.label_step)
        crop_heights = coords[:, 2].to(
            device=device, dtype=relative.dtype, non_blocking=True
        )
        crop_widths = coords[:, 3].to(
            device=device, dtype=relative.dtype, non_blocking=True
        )
        flips_cuda = flips.to(
            device=device, dtype=relative.dtype, non_blocking=True
        )
        x1, y1, x2, y2 = bbox
        partner_fraction = relative.new_tensor(
            ((x2 - x1) * (y2 - y1)) / float(image_size * image_size)
        )
        augmentation_features = torch.stack(
            [
                crop_heights * crop_widths / float(image_size * image_size),
                torch.log(crop_widths / crop_heights),
                flips_cuda,
                partner_fraction.expand_as(crop_heights),
            ],
            dim=1,
        )
        log_steps = relative.new_full((relative.shape[0], 1), self.label_step).log()
        mean, log_scale = self.label_entropy(
            labels, log_steps, slots, partner_labels, partner_slots,
            augmentation_features,
        )
        bits = logistic_bits(symbols, mean, log_scale).sum().item()
        reconstructed = reconstruct_hard_reference_logits(
            symbols * self.label_step, labels, teacher_logits.shape[1]
        )
        return reconstructed, bits


def load_label_codec(args):
    if not args.codec_checkpoint:
        return None, None

    checkpoint = torch.load(args.codec_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_type") in {
        "cim_ddruo_fixed_label_rate",
        "ddruos_fixed_label_rate",
        "ddruos_combined_dual",
        "rosd_fixed_label_rate",
        "rosd_combined_dual",
    }:
        codec = FixedCIMLabelCodec(checkpoint).cuda().eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        return codec, None

    raise ValueError(
        "Unsupported label codec checkpoint experiment_type="
        f"{checkpoint.get('experiment_type')!r}."
    )


def relabel(args):
    images, labels = load_synthetic(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    teacher = load_teacher(args.teacher_path, num_classes, args.device_ids, args.crop_size)
    label_codec, _ = load_label_codec(args)
    dataset = RelabelDataset(
        images, labels, args.crop_size, (args.min_crop_scale, 1.0),
        args.augmentation_mode,
    )
    sampler_generator = torch.Generator().manual_seed(args.fkd_seed)
    sampler = RandomSampler(dataset, generator=sampler_generator)
    loader = DataLoader(
        dataset,
        batch_size=args.loader_batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 4} if args.workers > 0 else {}),
    )
    mix_generator = torch.Generator().manual_seed(args.fkd_seed + 1)
    np_rng = np.random.default_rng(args.fkd_seed + 2)
    Path(args.fkd_path).mkdir(parents=True, exist_ok=True)
    save_and_print(
        args.log_path,
        f"relabel start images={len(dataset)} epochs={args.epochs} crop={args.crop_size} "
        f"scale=[{args.min_crop_scale},1] flip=0.5 "
        f"cutmix={'off' if args.disable_cutmix else args.cutmix_alpha} "
        f"temperature_used_at_training={args.temperature}",
    )

    start = time.time()
    total_label_bits = 0.0
    for epoch in range(args.epochs):
        crops, hard_labels, order, coords, flips = [], [], [], [], []
        for batch in loader:
            batch_crops, batch_labels, batch_order, batch_coords, batch_flips = batch
            crops.append(batch_crops)
            hard_labels.append(batch_labels)
            order.append(batch_order)
            coords.append(batch_coords)
            flips.append(batch_flips)
        crops = torch.cat(crops).cuda(non_blocking=True)
        hard_labels = torch.cat(hard_labels)
        order = torch.cat(order)
        coords = torch.cat(coords)
        flips = torch.cat(flips)

        if args.disable_cutmix:
            mix_index = torch.arange(crops.shape[0])
            lam = 1.0
            bbox = (0, 0, 0, 0)
        else:
            mix_index = torch.randperm(crops.shape[0], generator=mix_generator)
            lam = float(np_rng.beta(args.cutmix_alpha, args.cutmix_alpha))
            bbox = rand_bbox(args.crop_size, lam, mix_generator)
            x1, y1, x2, y2 = bbox
            crops[:, :, x1:x2, y1:y2] = crops[mix_index, :, x1:x2, y1:y2]
        with torch.inference_mode():
            teacher_logits = teacher(crops)
            label_bits = 0.0
            if label_codec is not None:
                teacher_logits, label_bits = label_codec.quantize(
                    teacher_logits, hard_labels, order, mix_index,
                    coords, flips, bbox, args.crop_size,
                )
            logits = teacher_logits.half().cpu()
        total_label_bits += label_bits
        config = {
            "order": order,
            "coords": coords,
            "flips": flips,
            "mix_index": mix_index,
            "mix_lambda": lam,
            "bbox": torch.tensor(bbox, dtype=torch.int16),
            "logits": logits,
            "label_bits": label_bits,
        }
        torch.save(config, os.path.join(args.fkd_path, f"epoch_{epoch:03d}.pt"))
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            agreement = (logits.float().argmax(1) == hard_labels).float().mean().item()
            save_and_print(
                args.log_path,
                f"relabel epoch={epoch + 1:03d}/{args.epochs} teacher_hard_agreement={agreement:.4f} "
                f"elapsed={time.time() - start:.1f}s",
            )
        del crops, logits
    bits_per_class = total_label_bits / num_classes
    save_and_print(
        args.log_path,
        f"relabel complete path={args.fkd_path} total_label_bits={total_label_bits:.0f} "
        f"label_bits_per_class={bits_per_class:.0f} label_kib_per_class={bits_per_class/8192.0:.4f}",
    )
    torch.save(
        {
            "epochs": args.epochs,
            "temperature": args.temperature,
            "total_label_bits": total_label_bits,
            "label_bits_per_class": bits_per_class,
        },
        os.path.join(args.fkd_path, "rate_summary.pt"),
    )


def relabel_pool(args):
    """Generate a pruned LPLD/LPQLD pool of self-contained FKD batches."""
    images, labels = load_synthetic(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    teacher = load_teacher(args.teacher_path, num_classes, args.device_ids, args.crop_size)
    label_codec, _ = load_label_codec(args)
    dataset = RelabelDataset(
        images, labels, args.crop_size, (args.min_crop_scale, 1.0),
        args.augmentation_mode,
    )
    sampler_generator = torch.Generator().manual_seed(args.fkd_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.loader_batch,
        sampler=RandomSampler(dataset, generator=sampler_generator),
        drop_last=not args.keep_last_pool_batch,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 4} if args.workers > 0 else {}),
    )
    batches_per_epoch = len(loader)
    full_batches = args.epochs * batches_per_epoch
    target_label_bits = None
    image_kib_per_class = None
    label_model_kib_per_class = None
    if label_codec is not None:
        payload = torch.load(
            args.synthetic_path, map_location="cpu", weights_only=False
        )
        image_kib_per_class = float(payload["image_kib_per_class"])
        label_model_kib_per_class = float(
            payload.get("label_model_kib_per_class", 0.0)
        )
        del payload
    if args.target_total_kib_per_class is not None:
        if label_codec is None:
            raise ValueError("target_total_kib_per_class requires a label codec")
        available_label_kib = (
            args.target_total_kib_per_class
            - image_kib_per_class
            - label_model_kib_per_class
        )
        if available_label_kib <= 0:
            raise ValueError(
                f"target {args.target_total_kib_per_class:g} KiB/class does not "
                f"cover image ({image_kib_per_class:.3f}) and label-model "
                f"({label_model_kib_per_class:.3f}) overhead"
            )
        target_label_bits = available_label_kib * num_classes * 8192.0
        # The actual number of retained batches is selected online from the
        # codec's entropy estimate. full_batches is only an upper bound.
        pool_batches = full_batches
        requested_compression = None
    elif args.prune_ratio is not None:
        if not 0.0 <= args.prune_ratio < 1.0:
            raise ValueError("prune_ratio must be in [0, 1)")
        # Match official LPLD exactly, including its floor conversion.
        pool_batches = max(1, int((1.0 - args.prune_ratio) * full_batches))
        requested_compression = full_batches / pool_batches
    else:
        pool_batches = max(1, math.ceil(full_batches / args.pool_compression))
        requested_compression = args.pool_compression
    pool_batches = min(pool_batches, args.pool_batches_limit)
    mix_generator = torch.Generator().manual_seed(args.fkd_seed + 1)
    np_rng = np.random.default_rng(args.fkd_seed + 2)
    Path(args.fkd_path).mkdir(parents=True, exist_ok=True)
    save_and_print(
        args.log_path,
        f"LPLD relabel pool start images={len(dataset)} train_epochs={args.epochs} "
        f"batch={args.loader_batch} batches_per_epoch={batches_per_epoch} "
        f"full_batches={full_batches} pool_batches_limit={pool_batches} "
        f"prune_ratio={args.prune_ratio} "
        f"requested_compression={requested_compression}x mr_topk={args.mr_topk} "
        f"target_total_kib_per_class={args.target_total_kib_per_class}",
    )

    start = time.time()
    saved = 0
    total_label_bits = 0.0

    if args.relabel_forward_accum < 1:
        raise ValueError("relabel_forward_accum must be at least 1")

    def flush_pending(pending):
        nonlocal saved, total_label_bits
        if not pending:
            return
        batch_sizes = [item[0].shape[0] for item in pending]
        with torch.inference_mode():
            all_logits = teacher(torch.cat([item[0] for item in pending], dim=0))
        logits_chunks = all_logits.split(batch_sizes)
        for logits, item in zip(logits_chunks, pending):
            if saved >= pool_batches or (
                target_label_bits is not None
                and total_label_bits >= target_label_bits
            ):
                break
            (
                crops,
                hard_labels,
                order,
                coords,
                flips,
                mix_index,
                lam,
                bbox,
                mix_mode,
            ) = item
            label_bits = 0.0
            with torch.inference_mode():
                if label_codec is not None:
                    logits, label_bits = label_codec.quantize(
                        logits, hard_labels, order, mix_index,
                        coords, flips, bbox, args.crop_size,
                    )
                    total_label_bits += label_bits
                if args.mr_topk > 0:
                    if args.mr_topk > num_classes:
                        raise ValueError(
                            f"mr_topk={args.mr_topk} exceeds classes={num_classes}"
                        )
                    topk = logits.topk(args.mr_topk, dim=1)
                    stored_labels = {
                        "logit_values": topk.values.half().cpu(),
                        "logit_indices": topk.indices.to(torch.int16).cpu(),
                        "num_classes": num_classes,
                    }
                    predictions = topk.indices[:, 0].cpu()
                else:
                    stored_labels = {"logits": logits.half().cpu()}
                    predictions = logits.argmax(1).cpu()
            config = {
                "order": order,
                "coords": coords,
                "flips": flips,
                "mix_index": mix_index,
                "mix_lambda": lam,
                "bbox": torch.tensor(bbox, dtype=torch.int16),
                "augmentation_mode": args.augmentation_mode,
                "mix_mode": mix_mode,
                "label_bits": label_bits,
                **stored_labels,
            }
            torch.save(
                config,
                os.path.join(args.fkd_path, f"pool_batch_{saved:05d}.pt"),
            )
            saved += 1
            if saved == 1 or saved % args.log_every == 0 or saved == pool_batches:
                agreement = (predictions == hard_labels).float().mean().item()
                save_and_print(
                    args.log_path,
                    f"LPLD pool batch={saved:04d}/{pool_batches} "
                    f"teacher_hard_agreement={agreement:.4f} "
                    f"forward_accum={args.relabel_forward_accum} "
                    f"elapsed={time.time() - start:.1f}s",
                )

    while saved < pool_batches and (
        target_label_bits is None or total_label_bits < target_label_bits
    ):
        pending = []
        for crops, hard_labels, order, coords, flips in loader:
            if saved + len(pending) >= pool_batches or (
                target_label_bits is not None
                and total_label_bits >= target_label_bits
            ):
                break
            crops = crops.cuda(non_blocking=True)
            if args.disable_cutmix:
                mix_index = torch.arange(crops.shape[0])
                lam = 1.0
                bbox = (0, 0, 0, 0)
                mix_mode = "none"
            elif args.augmentation_mode == "cifar_mixup":
                mix_index = torch.randperm(
                    crops.shape[0], generator=mix_generator
                )
                lam = float(np_rng.beta(args.mixup_alpha, args.mixup_alpha))
                bbox = (0, 0, 0, 0)
                crops = lam * crops + (1.0 - lam) * crops[mix_index]
                mix_mode = "mixup"
            else:
                mix_index = torch.randperm(
                    crops.shape[0], generator=mix_generator
                )
                lam = float(np_rng.beta(args.cutmix_alpha, args.cutmix_alpha))
                bbox = rand_bbox(args.crop_size, lam, mix_generator)
                x1, y1, x2, y2 = bbox
                crops[:, :, x1:x2, y1:y2] = crops[mix_index, :, x1:x2, y1:y2]
                mix_mode = "cutmix"
            pending.append(
                (
                    crops,
                    hard_labels,
                    order,
                    coords,
                    flips,
                    mix_index,
                    lam,
                    bbox,
                    mix_mode,
                )
            )
            if len(pending) == args.relabel_forward_accum:
                flush_pending(pending)
                pending = []
        flush_pending(pending)

    pool_batches = saved
    requested_compression = full_batches / pool_batches
    payload_bytes = sum(
        os.path.getsize(os.path.join(args.fkd_path, f"pool_batch_{index:05d}.pt"))
        for index in range(pool_batches)
    )
    actual_compression = requested_compression
    entropy_label_kib_per_class = total_label_bits / num_classes / 8192.0
    achieved_total_kib_per_class = None
    if image_kib_per_class is not None:
        achieved_total_kib_per_class = (
            image_kib_per_class
            + label_model_kib_per_class
            + entropy_label_kib_per_class
        )
    summary = {
        "method": (
            "LPQLD random batch pruning with MR-k label quantization"
            if args.mr_topk > 0
            else "LPLD random batch pruning with batch-to-epoch reuse"
        ),
        "train_epochs": args.epochs,
        "batches_per_epoch": batches_per_epoch,
        "pool_batches": pool_batches,
        "full_batches": full_batches,
        "prune_ratio": args.prune_ratio,
        "requested_compression": requested_compression,
        "actual_compression": actual_compression,
        "payload_bytes": payload_bytes,
        "payload_kib_per_class": payload_bytes / 1024.0 / num_classes,
        "entropy_label_bits": total_label_bits,
        "entropy_label_kib_per_class": entropy_label_kib_per_class,
        "target_total_kib_per_class": args.target_total_kib_per_class,
        "achieved_total_kib_per_class": achieved_total_kib_per_class,
        "image_kib_per_class": image_kib_per_class,
        "label_model_kib_per_class": label_model_kib_per_class,
        "codec_quantized": label_codec is not None,
        "temperature": args.temperature,
        "mr_topk": args.mr_topk,
    }
    torch.save(summary, os.path.join(args.fkd_path, "pool_summary.pt"))
    save_and_print(
        args.log_path,
        f"{'LPQLD' if args.mr_topk > 0 else 'LPLD'} pool complete "
        f"actual_compression={actual_compression:.3f}x mr_topk={args.mr_topk} "
        f"payload_kib_per_class={summary['payload_kib_per_class']:.3f} "
        f"entropy_label_kib_per_class={summary['entropy_label_kib_per_class']:.3f} "
        f"achieved_total_kib_per_class={achieved_total_kib_per_class} "
        f"codec_quantized={summary['codec_quantized']}",
    )


def relabel_pool_oracle(args):
    """Replace coded labels in an existing FKD pool with raw teacher logits."""
    if not args.source_fkd_path:
        raise ValueError("--source_fkd_path is required for relabel_pool_oracle")
    images, labels = load_synthetic(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    teacher = load_teacher(args.teacher_path, num_classes, args.device_ids, args.crop_size)
    source_summary = torch.load(
        os.path.join(args.source_fkd_path, "pool_summary.pt"),
        map_location="cpu",
        weights_only=False,
    )
    pool_batches = min(args.pool_batches_limit, int(source_summary["pool_batches"]))
    configs = [
        torch.load(
            os.path.join(args.source_fkd_path, f"pool_batch_{index:05d}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        for index in range(pool_batches)
    ]
    replays = [ReplayDataset(images, labels, config, args.crop_size) for config in configs]
    offsets = []
    offset = 0
    for replay in replays:
        offsets.append(offset)
        offset += len(replay)
    batch_sampler = MutableBatchSampler()
    batch_sampler.set_batches(
        [list(range(start, start + len(replay))) for start, replay in zip(offsets, replays)]
    )
    loader = DataLoader(
        ConcatDataset(replays),
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 4} if args.workers > 0 else {}),
    )
    Path(args.fkd_path).mkdir(parents=True, exist_ok=True)
    save_and_print(
        args.log_path,
        f"oracle relabel start source={args.source_fkd_path} pool_batches={pool_batches} "
        f"workers={args.workers} label_storage=raw_teacher_FP16",
    )
    start = time.time()
    total_label_bits = 0.0
    for index, (batch_images, batch_hard, _) in enumerate(loader):
        with torch.inference_mode():
            logits = teacher(batch_images.cuda(non_blocking=True))
        output_config = dict(configs[index])
        output_config.pop("logit_values", None)
        output_config.pop("logit_indices", None)
        output_config.pop("num_classes", None)
        output_config["logits"] = logits.half().cpu()
        label_bits = float(logits.numel() * 16)
        output_config["label_bits"] = label_bits
        total_label_bits += label_bits
        torch.save(output_config, os.path.join(args.fkd_path, f"pool_batch_{index:05d}.pt"))
        if index == 0 or (index + 1) % args.log_every == 0 or index + 1 == pool_batches:
            agreement = (logits.argmax(1).cpu() == batch_hard).float().mean().item()
            save_and_print(
                args.log_path,
                f"oracle pool batch={index + 1:05d}/{pool_batches} "
                f"teacher_hard_agreement={agreement:.4f} elapsed={time.time() - start:.1f}s",
            )
    payload_bytes = sum(
        os.path.getsize(os.path.join(args.fkd_path, f"pool_batch_{index:05d}.pt"))
        for index in range(pool_batches)
    )
    summary = dict(source_summary)
    summary.update(
        {
            "method": "LPLD pool with unquantized teacher FP16 logits",
            "pool_batches": pool_batches,
            "actual_compression": source_summary["full_batches"] / pool_batches,
            "payload_bytes": payload_bytes,
            "payload_kib_per_class": payload_bytes / 1024.0 / num_classes,
            "entropy_label_bits": total_label_bits,
            "entropy_label_kib_per_class": total_label_bits / num_classes / 8192.0,
            "codec_quantized": False,
            "mr_topk": 0,
        }
    )
    torch.save(summary, os.path.join(args.fkd_path, "pool_summary.pt"))
    save_and_print(
        args.log_path,
        f"oracle pool complete actual_compression={summary['actual_compression']:.3f}x "
        f"raw_label_kib_per_class={summary['entropy_label_kib_per_class']:.3f} "
        f"payload_kib_per_class={summary['payload_kib_per_class']:.3f}",
    )


@torch.inference_mode()
def evaluate(model, testloader, device):
    model.eval()
    correct = total = 0
    for images, labels in testloader:
        labels = labels.to(device, non_blocking=True)
        predictions = model(images.to(device, non_blocking=True)).argmax(1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total


def official_sgd_epoch_lr(args, epoch):
    """LPLD/LPQLD linear warmup followed by cosine decay."""
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        progress = epoch / args.warmup_epochs
        factor = args.warmup_start_factor + (
            1.0 - args.warmup_start_factor
        ) * progress
        return args.learning_rate * factor
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / cosine_epochs
    return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_student_optimizer(args, parameters):
    # PyTorch's fused optimizers have produced illegal-memory-access failures
    # with this process-local DataParallel training path. TF32, channels-last,
    # cuDNN autotuning, pinned transfers, and persistent workers still provide
    # the safe fast path; use the standard optimizer for multi-GPU runs.
    use_fused = args.fast_downstream and len(args.device_ids) == 1
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            **({"fused": True} if use_fused else {}),
        )
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        **({"fused": True} if use_fused else {}),
    )


def train(args):
    images, labels = load_synthetic(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    _, _, dataset_classes, _, _, _, _, _, testloader, _, _, _ = dataset_context(args)
    if dataset_classes != num_classes:
        raise ValueError(f"Class mismatch: synthetic={num_classes}, dataset={dataset_classes}")

    student = build_resnet18_bn(num_classes, (args.crop_size, args.crop_size))
    student = nn.DataParallel(student, device_ids=args.device_ids).cuda()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_accuracy = 0.0
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    save_and_print(
        args.log_path,
        f"train start epochs={args.epochs} global_batch={len(labels)} micro_batch={args.train_batch} "
        f"optimizer=AdamW lr={args.learning_rate} wd={args.weight_decay} T={args.temperature} "
        f"optimizer_step={'minibatch' if args.optimizer_step_per_batch else 'augmentation_group'} "
        f"augmentation=replayed_RRC_flip_CutMix",
    )

    for epoch in range(args.epochs):
        config_path = os.path.join(args.fkd_path, f"epoch_{epoch:03d}.pt")
        config = torch.load(config_path, map_location="cpu", weights_only=False)
        dataset = ReplayDataset(images, labels, config, args.crop_size)
        loader = DataLoader(
            dataset,
            batch_size=args.train_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        student.train()
        if not args.optimizer_step_per_batch:
            optimizer.zero_grad(set_to_none=True)
        loss_sum = correct = total = 0
        for batch_images, batch_hard, teacher_logits in loader:
            batch_images = batch_images.cuda(non_blocking=True)
            batch_hard = batch_hard.cuda(non_blocking=True)
            teacher_logits = teacher_logits.cuda(non_blocking=True)
            student_logits = student(batch_images)
            loss = criterion(
                F.log_softmax(student_logits / args.temperature, dim=1),
                F.softmax(teacher_logits / args.temperature, dim=1),
            )
            if args.optimizer_step_per_batch:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                scaled_loss = loss * (batch_hard.numel() / len(dataset))
                scaled_loss.backward()
            loss_sum += loss.item() * batch_hard.numel()
            correct += (student_logits.argmax(1) == batch_hard).sum().item()
            total += batch_hard.numel()
        if not args.optimizer_step_per_batch:
            optimizer.step()
        scheduler.step()

        should_eval = epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if should_eval:
            accuracy = evaluate(student, testloader, "cuda")
            best_accuracy = max(best_accuracy, accuracy)
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": student.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
                "temperature": args.temperature,
            }
            torch.save(checkpoint, os.path.join(args.output_path, "checkpoint.pt"))
            if accuracy >= best_accuracy:
                torch.save(checkpoint, os.path.join(args.output_path, "model_best.pt"))
            save_and_print(
                args.log_path,
                f"train epoch={epoch + 1:03d}/{args.epochs} lr={scheduler.get_last_lr()[0]:.7f} "
                f"kd={loss_sum / total:.6f} hard_train_acc={correct / total:.4f} "
                f"test_acc={accuracy:.4f} best={best_accuracy:.4f}",
            )
    save_and_print(args.log_path, f"train complete best_test_acc={best_accuracy:.4f}")


def train_pool(args):
    """Train while reusing an LPLD/LPQLD FKD batch pool."""
    images, labels = load_synthetic(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    _, _, dataset_classes, _, _, _, _, _, testloader, _, _, _ = dataset_context(args)
    if dataset_classes != num_classes:
        raise ValueError(f"Class mismatch: synthetic={num_classes}, dataset={dataset_classes}")
    summary = torch.load(os.path.join(args.fkd_path, "pool_summary.pt"), map_location="cpu", weights_only=False)
    pool = [
        torch.load(
            os.path.join(args.fkd_path, f"pool_batch_{index:05d}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        for index in range(summary["pool_batches"])
    ]
    generator = torch.Generator().manual_seed(args.seed + 9103)
    use_channels_last = args.fast_downstream and len(args.device_ids) == 1
    if args.fast_downstream:
        # cuDNN FIND can fail to select a convolution engine when process-local
        # DataParallel replicas concurrently benchmark channels-last tensors.
        # Keep the aggressive layout/autotuning path for one GPU and use the
        # stable contiguous path for multi-GPU DataParallel.
        torch.backends.cudnn.benchmark = len(args.device_ids) == 1
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    student_core = build_resnet18_bn(num_classes, (args.crop_size, args.crop_size)).cuda()
    if use_channels_last:
        student_core = student_core.to(memory_format=torch.channels_last)
    student = (
        nn.DataParallel(student_core, device_ids=args.device_ids)
        if len(args.device_ids) > 1 else student_core
    )
    optimizer = build_student_optimizer(args, student.parameters())
    scheduler = (
        None
        if args.optimizer == "sgd"
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_accuracy = 0.0
    start_epoch = 0
    # CA is a ratio relative to the current DKR teacher temperature. Treating
    # it as an absolute temperature makes the student distribution about 20x
    # sharper at the default teacher T=20 and destroys validation transfer.
    ca_student_temperature_ratio = 1.0
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        student_core.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        ca_student_temperature_ratio = float(
            checkpoint.get("next_ca_ratio", 1.0)
        )
        # Pool indices are the only draws from this generator. Replaying them
        # restores the exact selection stream without storing a large state.
        for _ in range(start_epoch):
            if args.pool_sampling_with_replacement:
                torch.randint(
                    summary["pool_batches"],
                    (summary["batches_per_epoch"],),
                    generator=generator,
                )
            else:
                torch.randperm(summary["pool_batches"], generator=generator)
        save_and_print(
            args.log_path,
            f"resume checkpoint={args.resume_checkpoint} start_epoch={start_epoch} "
            f"best_test_acc={best_accuracy:.4f}",
        )
    save_and_print(
        args.log_path,
        f"{'LPQLD' if summary.get('mr_topk', 0) > 0 else 'LPLD'} pool train start "
        f"epochs={args.epochs} updates_per_epoch={summary['batches_per_epoch']} "
        f"pool_batches={summary['pool_batches']} actual_compression={summary['actual_compression']:.3f}x "
        f"mr_topk={summary.get('mr_topk', 0)} optimizer={args.optimizer.upper()} "
        f"lr={args.learning_rate} momentum={args.momentum} wd={args.weight_decay} "
        f"warmup_epochs={args.warmup_epochs} "
        f"warmup_start_factor={args.warmup_start_factor} teacher_T={args.temperature} "
        f"pool_sampling={'with_replacement' if args.pool_sampling_with_replacement else 'without_replacement'} "
        f"fast_downstream={args.fast_downstream} workers={args.workers} "
        f"channels_last={use_channels_last} cudnn_benchmark={torch.backends.cudnn.benchmark} "
        f"dkr={args.dkr_schedule} ca_dynamic={args.ca_dynamic} "
        f"temperature_loss_scale={args.scale_loss_by_temperature_squared}",
    )

    # Pool contents and replay geometry are fixed. Construct them once so the
    # DataLoader workers survive across epochs; only sampled batch indices vary.
    all_replays = [
        ReplayDataset(images, labels, config, args.crop_size) for config in pool
    ]
    replay_offsets = []
    offset = 0
    for replay in all_replays:
        replay_offsets.append(offset)
        offset += len(replay)
    batch_sampler = MutableBatchSampler()
    loader = DataLoader(
        ConcatDataset(all_replays),
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 4} if args.workers > 0 else {}),
    )
    cached_batches = None
    if args.cache_replay_batches:
        cached_batches = [[] for _ in all_replays]
        cache_pool_indices = []
        cache_batch_indices = []
        for pool_index, replay in enumerate(all_replays):
            offset = replay_offsets[pool_index]
            for start in range(0, len(replay), args.train_batch):
                stop = min(start + args.train_batch, len(replay))
                cache_pool_indices.append(pool_index)
                cache_batch_indices.append(list(range(offset + start, offset + stop)))
        batch_sampler.set_batches(cache_batch_indices)
        cache_start = time.time()
        cache_bytes = 0
        for pool_index, batch in zip(cache_pool_indices, loader):
            copied = []
            for tensor in batch:
                host_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                host_tensor.copy_(tensor)
                copied.append(host_tensor)
                cache_bytes += host_tensor.numel() * host_tensor.element_size()
            cached_batches[pool_index].append(tuple(copied))
        save_and_print(
            args.log_path,
            f"replay cache complete batches={len(cache_batch_indices)} "
            f"size_gib={cache_bytes / (1024 ** 3):.3f} elapsed={time.time() - cache_start:.1f}s",
        )

    for epoch in range(start_epoch, args.epochs):
        if args.optimizer == "sgd":
            epoch_lr = official_sgd_epoch_lr(args, epoch)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = epoch_lr
        else:
            epoch_lr = optimizer.param_groups[0]["lr"]
        teacher_temperature = dkr_temperature(args, epoch)
        student_temperature = (
            ca_student_temperature_ratio * teacher_temperature
            if args.ca_dynamic else teacher_temperature
        )
        if summary["pool_batches"] < summary["batches_per_epoch"]:
            raise ValueError(
                "The pruned label pool must contain at least one epoch of unique batches"
            )
        if args.pool_sampling_with_replacement:
            selected = torch.randint(
                summary["pool_batches"],
                (summary["batches_per_epoch"],),
                generator=generator,
            ).tolist()
        else:
            selected = torch.randperm(
                summary["pool_batches"], generator=generator
            )[: summary["batches_per_epoch"]].tolist()
        student.train()
        loss_sum = correct = total = 0
        last_student_logits = last_teacher_probabilities = None
        # Keep minibatches from crossing pool-entry boundaries and preserve the
        # official per-minibatch zero_grad/backward/step optimizer semantics.
        batch_indices = []
        batch_pool_indices = []
        for pool_index in selected:
            replay = all_replays[pool_index]
            if len(replay) != args.train_batch:
                raise ValueError(
                    "Batch-level FKD replay requires train_batch to match the "
                    f"stored pool batch: train_batch={args.train_batch}, "
                    f"pool_batch={len(replay)}"
                )
            offset = replay_offsets[pool_index]
            for start in range(0, len(replay), args.train_batch):
                stop = min(start + args.train_batch, len(replay))
                batch_indices.append(
                    list(range(offset + start, offset + stop))
                )
                batch_pool_indices.append(pool_index)
        if cached_batches is None:
            batch_sampler.set_batches(batch_indices)
            epoch_batches = loader
        else:
            epoch_batches = (
                batch for pool_index in selected for batch in cached_batches[pool_index]
            )
        epoch_start = time.time()
        for update_index, (pool_index, batch) in enumerate(
            zip(batch_pool_indices, epoch_batches)
        ):
            batch_images, batch_hard, teacher_logits = batch
            optimizer.zero_grad(set_to_none=True)
            batch_images = batch_images.cuda(non_blocking=True)
            if use_channels_last:
                batch_images = batch_images.to(memory_format=torch.channels_last)
            # Match official FKD replay: transform every image once, then apply
            # the saved MixUp/CutMix permutation to the complete batch.
            replay = all_replays[pool_index]
            mix_index = replay.mix_index.to(device="cuda", non_blocking=True)
            mixed_source = batch_images[mix_index]
            if replay.mix_mode == "mixup":
                batch_images = (
                    replay.mix_lambda * batch_images
                    + (1.0 - replay.mix_lambda) * mixed_source
                )
            else:
                x1, y1, x2, y2 = replay.bbox
                batch_images[:, :, x1:x2, y1:y2] = mixed_source[
                    :, :, x1:x2, y1:y2
                ]
            batch_hard = batch_hard.cuda(non_blocking=True)
            teacher_logits = teacher_logits.cuda(non_blocking=True).float()
            student_logits = student(batch_images)
            teacher_probabilities = F.softmax(
                teacher_logits / teacher_temperature, dim=1
            )
            loss = criterion(
                F.log_softmax(student_logits / student_temperature, dim=1),
                teacher_probabilities,
            )
            if args.scale_loss_by_temperature_squared:
                loss = loss * (teacher_temperature ** 2)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_hard.numel()
            correct += (student_logits.argmax(1) == batch_hard).sum().item()
            total += batch_hard.numel()
            last_student_logits = student_logits.detach()
            last_teacher_probabilities = teacher_probabilities.detach()
            if epoch == start_epoch and (
                update_index == 0 or (update_index + 1) % 100 == 0
            ):
                save_and_print(
                    args.log_path,
                    f"first_epoch update={update_index + 1:03d}/{len(batch_pool_indices)} "
                    f"elapsed={time.time() - epoch_start:.1f}s",
                )
        if args.ca_dynamic:
            ca_student_temperature_ratio = calibrate_student_temperature_ratio(
                last_student_logits,
                last_teacher_probabilities,
                teacher_temperature,
                args.ca_grid_points,
            )
        if scheduler is not None:
            scheduler.step()

        should_eval = epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if should_eval:
            accuracy = evaluate(student, testloader, "cuda")
            best_accuracy = max(best_accuracy, accuracy)
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": student_core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
                "temperature": args.temperature,
                "teacher_temperature": teacher_temperature,
                "student_temperature": student_temperature,
                "next_ca_ratio": ca_student_temperature_ratio,
                "next_ca_student_temperature": (
                    ca_student_temperature_ratio * teacher_temperature
                ),
                "dkr_schedule": args.dkr_schedule,
                "ca_dynamic": args.ca_dynamic,
                "lpld_pool_summary": summary,
            }
            torch.save(checkpoint, os.path.join(args.output_path, "checkpoint.pt"))
            if accuracy >= best_accuracy:
                torch.save(checkpoint, os.path.join(args.output_path, "model_best.pt"))
            save_and_print(
                args.log_path,
                f"{'LPQLD' if summary.get('mr_topk', 0) > 0 else 'LPLD'} train "
                f"epoch={epoch + 1:03d}/{args.epochs} lr={epoch_lr:.7f} "
                f"teacher_T={teacher_temperature:.4f} student_T={student_temperature:.4f} "
                f"next_ca_ratio={ca_student_temperature_ratio:.4f} "
                f"next_student_T={ca_student_temperature_ratio * teacher_temperature:.4f} "
                f"kd={loss_sum / total:.6f} hard_train_acc={correct / total:.4f} "
                f"test_acc={accuracy:.4f} best={best_accuracy:.4f}",
            )
    save_and_print(
        args.log_path,
        f"{'LPQLD' if summary.get('mr_topk', 0) > 0 else 'LPLD'} train complete "
        f"best_test_acc={best_accuracy:.4f}",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("relabel", "train", "relabel_pool", "relabel_pool_oracle", "train_pool"),
        required=True,
    )
    parser.add_argument("--synthetic_path", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--fkd_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--dataset", default="ImageNet")
    parser.add_argument("--subset", default="imagefruit")
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--min_crop_scale", type=float, default=0.08)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument(
        "--augmentation_mode",
        choices=("rrc_cutmix", "cifar_mixup"),
        default="rrc_cutmix",
    )
    parser.add_argument("--disable_cutmix", action="store_true")
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--codec_checkpoint", default="")
    parser.add_argument("--loader_batch", type=int, default=128)
    parser.add_argument("--train_batch", type=int, default=256)
    parser.add_argument("--optimizer_step_per_batch", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_start_factor", type=float, default=0.01)
    parser.add_argument("--scale_loss_by_temperature_squared", action="store_true")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--fkd_seed", type=int, default=42)
    parser.add_argument("--pool_compression", type=float, default=40.0)
    parser.add_argument("--prune_ratio", type=float, default=None)
    parser.add_argument("--target_total_kib_per_class", type=float, default=None)
    parser.add_argument("--keep_last_pool_batch", action="store_true")
    parser.add_argument("--pool_sampling_with_replacement", action="store_true")
    parser.add_argument("--fast_downstream", action="store_true")
    parser.add_argument("--cache_replay_batches", action="store_true")
    parser.add_argument("--source_fkd_path", default="")
    parser.add_argument("--pool_batches_limit", type=int, default=2**31 - 1)
    parser.add_argument("--relabel_forward_accum", type=int, default=1)
    parser.add_argument("--mr_topk", type=int, default=0)
    parser.add_argument(
        "--dkr_schedule", choices=("none", "step", "cosine"), default="none"
    )
    parser.add_argument("--dkr_step_size", type=int, default=30)
    parser.add_argument("--dkr_step_gamma", type=float, default=0.7)
    parser.add_argument("--dkr_min_temperature", type=float, default=2.0)
    parser.add_argument("--ca_dynamic", action="store_true")
    parser.add_argument("--ca_grid_points", type=int, default=100)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_ids", type=int, nargs="+", default=(0, 1, 2, 3))
    args = parser.parse_args(argv)
    args.log_path = os.path.join(args.output_path, f"{args.mode}.log.txt")
    args.save_path = args.output_path
    args.zca = False
    return args


if __name__ == "__main__":
    parsed = parse_args()
    Path(parsed.output_path).mkdir(parents=True, exist_ok=True)
    set_seed(parsed.seed)
    {
        "relabel": relabel,
        "train": train,
        "relabel_pool": relabel_pool,
        "relabel_pool_oracle": relabel_pool_oracle,
        "train_pool": train_pool,
    }[parsed.mode](parsed)
