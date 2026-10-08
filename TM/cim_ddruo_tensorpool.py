"""ROSD training on top of the official DD-RUO TensorPool codec.

Each stored sample is a mosaic of ``factor**2`` high-confidence real anchors.
DD-RUO compresses the mosaics, while CIM decodes each reconstructed mosaic
into full-resolution views and matches them to the anchors in the observer's
penultimate feature space.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ts.tensor_pool import TensorPool
from core.utils import DiffAugment, ParamDiffAug, get_time, save_and_print, set_seed
from TM.sre2l_bn import build_resnet18_bn, dataset_context
from TM.sre2l_official_tensorpool import (
    BNFeatureLoss,
    build_parser as build_base_parser,
    initialize_pool_with_dual_warmup,
    install_per_gpu_tensorpool_locks,
    load_reference_images,
    normalize_images,
    teacher_metrics_and_backward,
    validate_initialized_pool,
)
from TM.cim_label_codec import (
    AugmentedSlotStepLabelEntropy,
    logistic_bits,
    sample_shared_augmentation,
    ste_round,
)
from TM.postquant_rate import build_class_scorers
from TM.sre2l_fkd import FixedCIMLabelCodec


def unnormalize_images(images, mean, std):
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    return (images * std_tensor + mean_tensor).clamp(0.0, 1.0)


def current_codec_lr(pool):
    """Read the optimizer LR from one slice; every slice shares the schedule."""
    for key in pool.key_list:
        solver = pool.slice_pool[key]["param"].solver
        if solver is not None:
            return float(solver.param_groups[0]["lr"])
    return float("nan")


def decode_mosaic_views(mosaics, factor):
    """Decode mosaics in the quadrant-major order used by official CIM."""
    if mosaics.ndim != 4 or mosaics.shape[-2] % factor or mosaics.shape[-1] % factor:
        raise ValueError(f"Invalid mosaic shape {tuple(mosaics.shape)} for factor={factor}")
    height, width = mosaics.shape[-2:]
    crop_h, crop_w = height // factor, width // factor
    views = []
    for row in range(factor):
        for column in range(factor):
            crop = mosaics[
                :, :, row * crop_h:(row + 1) * crop_h,
                column * crop_w:(column + 1) * crop_w,
            ]
            views.append(F.interpolate(
                crop, size=(height, width), mode="bilinear", align_corners=False
            ))
    return torch.cat(views, dim=0)


def hard_reference_logits(logits, hard_labels):
    """Represent K logits by K-1 differences to the known hard-class logit."""
    classes = logits.shape[1]
    reference = logits.gather(1, hard_labels[:, None])
    relative = logits - reference
    keep = torch.arange(classes, device=logits.device)[None] != hard_labels[:, None]
    return relative[keep].view(logits.shape[0], classes - 1)


def reconstruct_hard_reference_logits(relative, hard_labels, classes):
    """Restore the canonical K-logit vector whose hard-class entry is zero."""
    reconstructed = relative.new_zeros((relative.shape[0], classes))
    keep = torch.arange(classes, device=relative.device)[None] != hard_labels[:, None]
    reconstructed[keep] = relative.reshape(-1)
    return reconstructed


def apply_fkd_augmentation_chunk(images, augmentation, indices, output_size):
    """Replay RRC/flip/CutMix for a chunk without materializing all crops."""
    _, channels, height, width = images.shape
    dtype = images.dtype

    def crop(source_indices, parameter_indices):
        count = parameter_indices.numel()
        theta = images.new_zeros((count, 2, 3))
        crop_widths = augmentation["widths"][parameter_indices].to(dtype)
        crop_heights = augmentation["heights"][parameter_indices].to(dtype)
        lefts = augmentation["lefts"][parameter_indices].to(dtype)
        tops = augmentation["tops"][parameter_indices].to(dtype)
        theta[:, 0, 0] = crop_widths / width
        theta[:, 1, 1] = crop_heights / height
        theta[:, 0, 2] = (2.0 * lefts + crop_widths) / width - 1.0
        theta[:, 1, 2] = (2.0 * tops + crop_heights) / height - 1.0
        grid = F.affine_grid(
            theta,
            torch.Size((count, channels, output_size, output_size)),
            align_corners=False,
        )
        result = F.grid_sample(
            images[source_indices], grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        flips = augmentation["flips"][parameter_indices]
        return torch.where(flips[:, None, None, None], result.flip(-1), result)

    primary = crop(indices, indices)
    partner_indices = augmentation["mix_index"][indices]
    partner = crop(partner_indices, partner_indices)
    x1, y1, x2, y2 = augmentation["bbox"]
    if x2 > x1 and y2 > y1:
        primary = primary.clone()
        primary[:, :, x1:x2, y1:y2] = partner[:, :, x1:x2, y1:y2]
    return primary, partner_indices


def teacher_forward_to(teacher, normalized_images, output_device):
    """Run a frozen teacher on its primary device and gather its logits."""
    teacher_device = next(teacher.parameters()).device
    inputs = normalized_images.to(teacher_device, non_blocking=True)
    return teacher(inputs).to(output_device, non_blocking=True)


def augmented_sre2l_metrics_and_backward(
    images, labels, teacher, bn_feature, mean, std, bn_weight,
    crop_scale_min=0.08, flip_probability=0.5, batch_size=256,
    backward_scale=1.0, jitter=0,
):
    """Optimize SRe2L CE+BN on per-image differentiable RRC/flip views."""
    augmentation = sample_shared_augmentation(
        images,
        crop_scale_min=crop_scale_min,
        flip_probability=flip_probability,
        cutmix_alpha=0.0,
    )
    count = images.shape[0]
    batch_size = min(max(1, batch_size), count)
    utility_sum = images.new_zeros(())
    ce_sum = images.new_zeros(())
    bn_sum = images.new_zeros(())
    correct = images.new_zeros(())
    if jitter > 0:
        jitter_offsets = tuple(
            int(value) for value in torch.randint(
                -jitter, jitter + 1, (2,), device=images.device
            ).tolist()
        )
    else:
        jitter_offsets = (0, 0)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        weight = (end - start) / count
        indices = torch.arange(start, end, device=images.device)
        augmented, _ = apply_fkd_augmentation_chunk(
            images, augmentation, indices, images.shape[-1]
        )
        if jitter_offsets != (0, 0):
            augmented = torch.roll(
                augmented, shifts=jitter_offsets, dims=(2, 3)
            )
        bn_feature.clear()
        logits = teacher_forward_to(
            teacher, normalize_images(augmented, mean, std), images.device
        )
        ce = F.cross_entropy(logits, labels[start:end])
        bn = bn_feature.value()
        utility = ce + bn_weight * bn
        (backward_scale * weight * utility).backward()
        utility_sum += weight * utility.detach()
        ce_sum += weight * ce.detach()
        bn_sum += weight * bn.detach()
        correct += (logits.detach().argmax(1) == labels[start:end]).sum()
    return utility_sum, ce_sum, bn_sum, correct / count


def encode_anchor_mosaics(anchors, factor):
    """Resize Q anchors into one factor-by-factor mosaic per storage slot."""
    slots, views, channels, height, width = anchors.shape
    if views != factor**2:
        raise ValueError(f"Expected {factor**2} anchors per slot, got {views}")
    crop_h, crop_w = height // factor, width // factor
    mosaics = anchors.new_empty(slots, channels, height, width)
    view_index = 0
    for row in range(factor):
        for column in range(factor):
            resized = F.interpolate(
                anchors[:, view_index], size=(crop_h, crop_w), mode="nearest",
            )
            mosaics[
                :, :, row * crop_h:(row + 1) * crop_h,
                column * crop_w:(column + 1) * crop_w,
            ] = resized
            view_index += 1
    return mosaics


def build_or_load_cim_references(
    cache_path, train_set, teacher, classes, ipc, factor, mean, std,
    image_size, batch_size, workers, device, log_path, candidate_ipc, seed,
):
    """Select low-CE anchors and cache compact uint8 mosaics and anchor views."""
    required = ipc * factor**2
    if os.path.isfile(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = (classes * ipc, factor**2, 3, *image_size)
        if tuple(payload["anchors_uint8"].shape) != expected:
            raise ValueError(
                f"CIM cache shape {tuple(payload['anchors_uint8'].shape)} != {expected}"
            )
        save_and_print(log_path, f"cim_reference_cache_loaded={cache_path}")
    else:
        save_and_print(
            log_path,
            f"cim_anchor_selection_start={get_time()} required_per_class={required}",
        )
        candidates = [[] for _ in range(classes)]
        loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=True,
        )
        offset = 0
        teacher.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels_device = labels.to(device, non_blocking=True)
                scores = F.cross_entropy(
                    teacher(images), labels_device, reduction="none"
                ).cpu()
                for local_index, (label, score) in enumerate(zip(labels, scores)):
                    candidates[int(label)].append((float(score), offset + local_index))
                offset += labels.numel()

        selected = []
        selected_score_ranges = []
        for class_index, items in enumerate(candidates):
            if len(items) < required:
                raise RuntimeError(
                    f"Class {class_index} has {len(items)} examples; {required} required"
                )
            if candidate_ipc > 0:
                if candidate_ipc < required:
                    raise ValueError(
                        f"cim_mipc={candidate_ipc} is smaller than the "
                        f"required {required} anchors per class"
                    )
                generator = torch.Generator().manual_seed(seed + class_index)
                permutation = torch.randperm(len(items), generator=generator)
                items = [items[index] for index in permutation[:candidate_ipc]]
            items.sort(key=lambda item: item[0])
            selected.append([index for _, index in items[:required]])
            selected_score_ranges.append((items[0][0], items[required - 1][0]))

        all_anchors = []
        all_mosaics = []
        for class_index, indices in enumerate(selected):
            normalized = torch.stack([train_set[index][0] for index in indices])
            raw = unnormalize_images(normalized, mean, std)
            anchors = raw.reshape(factor**2, ipc, 3, *image_size).transpose(0, 1)
            mosaics = encode_anchor_mosaics(anchors, factor)
            all_anchors.append((anchors * 255.0).round().to(torch.uint8))
            all_mosaics.append((mosaics * 255.0).round().to(torch.uint8))
            ce_min, ce_max_selected = selected_score_ranges[class_index]
            save_and_print(
                log_path,
                f"cim_anchor_class={class_index} ce_min={ce_min:.6f} "
                f"ce_max_selected={ce_max_selected:.6f}",
            )

        payload = {
            "anchors_uint8": torch.cat(all_anchors),
            "mosaics_uint8": torch.cat(all_mosaics),
            "labels": torch.arange(classes).repeat_interleave(ipc),
            "factor": factor,
            "ipc": ipc,
            "candidate_ipc": candidate_ipc,
            "selection_seed": seed,
        }
        torch.save(payload, cache_path)
        save_and_print(log_path, f"cim_reference_cache_saved={cache_path}")

    references = {}
    mosaics = payload["mosaics_uint8"]
    for class_index in range(classes):
        start = class_index * ipc
        references[class_index] = mosaics[start:start + ipc].float().div_(255.0)
    return references, payload["anchors_uint8"], payload["labels"].long()


def observer_penultimate(observer, images):
    """Return the input to the classifier, matching official CIM depth=-1."""
    features = F.relu(observer.bn1(observer.conv1(images)))
    if hasattr(observer, "maxpool"):
        features = observer.maxpool(features)
    features = observer.layer1(features)
    features = observer.layer2(features)
    features = observer.layer3(features)
    features = observer.layer4(features)
    return F.adaptive_avg_pool2d(features, 1).flatten(1)


def cim_utility_and_backward(
    mosaics, anchors_uint8, observer, mean, std, factor, ipc,
    class_batch_size, iteration, strategy,
):
    """Backpropagate official class-paired CIM matching into TensorPool images."""
    if mosaics.shape[0] % ipc:
        raise ValueError("CIM mosaics are not class-major IPC blocks")
    classes = mosaics.shape[0] // ipc
    total_views = mosaics.shape[0] * factor**2
    utility_sum = mosaics.new_zeros(())

    for class_start in range(0, classes, class_batch_size):
        class_end = min(class_start + class_batch_size, classes)
        real_batches = []
        synthetic_batches = []
        for class_index in range(class_start, class_end):
            start = class_index * ipc
            end = start + ipc
            synthetic_views = decode_mosaic_views(mosaics[start:end], factor)
            real_views = anchors_uint8[start:end].to(
                mosaics.device, non_blocking=True
            ).float().div_(255.0).transpose(0, 1).flatten(0, 1)
            paired = normalize_images(
                torch.cat((real_views, synthetic_views), dim=0), mean, std
            )
            aug_param = ParamDiffAug()
            aug_param.aug_mode = "M"
            paired = DiffAugment(
                paired, strategy=strategy,
                seed=iteration * 100000 + class_index + 1,
                param=aug_param,
            )
            real_aug, synthetic_aug = paired.chunk(2)
            real_batches.append(real_aug)
            synthetic_batches.append(synthetic_aug)

        real_aug = torch.cat(real_batches)
        synthetic_aug = torch.cat(synthetic_batches)
        with torch.no_grad():
            real_features = observer_penultimate(observer, real_aug)
        synthetic_features = observer_penultimate(observer, synthetic_aug)

        # This follows the released CIM implementation's per-example mean L1.
        per_view = (synthetic_features - real_features).abs().flatten(1).mean(1)
        chunk_utility = per_view.sum() / total_views
        chunk_utility.backward()
        utility_sum = utility_sum + chunk_utility.detach()

    return utility_sum


def soft_label_rate_step(
    mosaics, mosaic_labels, observer, entropy_model, entropy_optimizer,
    mean, std, label_step, label_groups, classes, ipc,
    feature_chunk_size, entropy_batch_size, dual_lambda_eff,
    hard_ce_weight=0.0, label_kl_weight=0.0, label_kl_temperature=2.0,
    train_entropy=True, slot_offset=0,
):
    """Fit the label entropy model and backpropagate its rate to mosaics.

    One sampled FKD augmentation realizes an unbiased estimate of one label
    group. Its bit count is multiplied by label_groups in the total payload.
    There is intentionally no CE or label-distortion term in this ablation.
    """
    # Soft labels belong to the stored mosaics. The factor-decoded views are
    # internal to CIM synthesis and must not multiply the FKD payload IPC.
    views = mosaics
    hard_labels = mosaic_labels
    slot_ids = (
        torch.arange(views.shape[0], device=mosaics.device, dtype=torch.long)
        + int(slot_offset)
    )
    augmentation = sample_shared_augmentation(
        views, crop_scale_min=0.08, flip_probability=0.5, cutmix_alpha=1.0
    )
    log_steps = views.new_full((views.shape[0], 1), float(label_step)).log()
    denominator = float(classes * ipc * mosaics.shape[-2] * mosaics.shape[-1])

    total_bits = 0.0
    agreement_correct = 0
    count = 0
    margin_sum = 0.0
    zero_symbols = 0
    symbol_count = 0
    hard_ce_sum = 0.0
    label_kl_sum = 0.0
    utility_count = float(views.shape[0])

    for start in range(0, views.shape[0], feature_chunk_size):
        end = min(start + feature_chunk_size, views.shape[0])
        indices = torch.arange(start, end, device=mosaics.device)
        augmented, partner_indices = apply_fkd_augmentation_chunk(
            views, augmentation, indices, mosaics.shape[-1]
        )
        logits = teacher_forward_to(
            observer, normalize_images(augmented, mean, std), mosaics.device
        )
        labels = hard_labels[indices]
        slots = slot_ids[indices]
        partner_labels = hard_labels[partner_indices]
        partner_slots = slot_ids[partner_indices]
        augmentation_features = augmentation["features"][indices]
        relative = hard_reference_logits(logits, labels)
        symbols = ste_round(relative / label_step)

        for batch_start in range(0, end - start, entropy_batch_size):
            batch_end = min(batch_start + entropy_batch_size, end - start)
            batch = slice(batch_start, batch_end)
            batch_indices = indices[batch]

            if train_entropy:
                entropy_optimizer.zero_grad(set_to_none=True)
                fit_mean, fit_log_scale = entropy_model(
                    labels[batch], log_steps[batch_indices], slots[batch],
                    partner_labels[batch], partner_slots[batch],
                    augmentation_features[batch],
                )
                fit_loss = logistic_bits(
                    symbols[batch].detach(), fit_mean, fit_log_scale
                ).mean()
                fit_loss.backward()
                torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 5.0)
                entropy_optimizer.step()

            for parameter in entropy_model.parameters():
                parameter.requires_grad_(False)
            rate_mean, rate_log_scale = entropy_model(
                labels[batch], log_steps[batch_indices], slots[batch],
                partner_labels[batch], partner_slots[batch],
                augmentation_features[batch],
            )
            bits = logistic_bits(
                symbols[batch], rate_mean, rate_log_scale
            ).sum()
            batch_logits = logits[batch]
            batch_labels = labels[batch]
            batch_partner_labels = partner_labels[batch]
            partner_fraction = augmentation["partner_fraction"]
            hard_ce = (
                (1.0 - partner_fraction)
                * F.cross_entropy(batch_logits, batch_labels, reduction="sum")
                + partner_fraction
                * F.cross_entropy(
                    batch_logits, batch_partner_labels, reduction="sum"
                )
            ) / utility_count
            quantized_relative = symbols[batch] * label_step
            quantized_logits = reconstruct_hard_reference_logits(
                quantized_relative, batch_labels, classes,
            )
            temperature = float(label_kl_temperature)
            label_kl = (
                F.kl_div(
                    F.log_softmax(quantized_logits / temperature, dim=1),
                    F.softmax(batch_logits.detach() / temperature, dim=1),
                    reduction="sum",
                )
                * temperature**2
                / utility_count
            )
            auxiliary_utility = (
                hard_ce_weight * hard_ce + label_kl_weight * label_kl
            )
            if auxiliary_utility.requires_grad and (
                hard_ce_weight > 0.0 or label_kl_weight > 0.0
            ):
                auxiliary_utility.backward(
                    retain_graph=(
                        dual_lambda_eff != 0.0 or batch_end < end - start
                    )
                )
            if dual_lambda_eff != 0.0:
                label_equivalent_bpp = bits * label_groups / denominator
                (dual_lambda_eff * label_equivalent_bpp).backward(
                    retain_graph=batch_end < end - start
                )
            for parameter in entropy_model.parameters():
                parameter.requires_grad_(True)

            with torch.no_grad():
                batch_relative = relative[batch]
                hard_quantized_logits = reconstruct_hard_reference_logits(
                    torch.round(batch_relative / label_step) * label_step,
                    batch_labels, classes,
                )
                agreement_correct += (
                    hard_quantized_logits.argmax(1) == batch_logits.argmax(1)
                ).sum().item()
                top_two = batch_logits.topk(2, dim=1).values
                margin_sum += (top_two[:, 0] - top_two[:, 1]).sum().item()
                hard_symbols = torch.round(batch_relative / label_step)
                zero_symbols += (hard_symbols == 0).sum().item()
                symbol_count += hard_symbols.numel()
                count += batch_end - batch_start
                total_bits += bits.item()
                hard_ce_sum += hard_ce.item() * utility_count
                label_kl_sum += label_kl.item() * utility_count

    label_kib_per_class = total_bits * label_groups / classes / 8192.0
    return {
        "group_bits": total_bits,
        "label_kib_per_class": label_kib_per_class,
        "quantized_top1_agreement": agreement_correct / count,
        "teacher_margin": margin_sum / count,
        "zero_symbol_fraction": zero_symbols / symbol_count,
        "hard_ce": hard_ce_sum / count,
        "label_kl": label_kl_sum / count,
    }


@torch.no_grad()
def mosaic_teacher_diagnostics(mosaics, labels, teacher, mean, std, batch_size=128):
    loss_sum = correct = count = 0
    for start in range(0, mosaics.shape[0], batch_size):
        end = min(start + batch_size, mosaics.shape[0])
        logits = teacher_forward_to(
            teacher, normalize_images(mosaics[start:end], mean, std),
            mosaics.device,
        )
        batch_labels = labels[start:end]
        loss_sum += F.cross_entropy(logits, batch_labels, reduction="sum").item()
        correct += (logits.argmax(1) == batch_labels).sum().item()
        count += end - start
    return loss_sum / count, correct / count


def export_cim_payload(
    pool, teacher, mean, std, ipc, factor, args, rate_kind,
    total_bpp=None, rate_components=None, label_rate=None,
):
    pool.test()
    mosaics, labels, latent_bpp = pool.get_data()
    labels = labels.to(mosaics.device)
    ce, accuracy = mosaic_teacher_diagnostics(
        mosaics, labels, teacher, mean, std
    )
    reported_bpp = float(latent_bpp) if total_bpp is None else float(total_bpp)
    kib_per_class = reported_bpp * ipc * mosaics.shape[-2] * mosaics.shape[-1] / 8192.0
    label_kib = 0.0 if label_rate is None else float(label_rate["label_kib_per_class"])
    label_model_kib = 0.0 if label_rate is None else float(label_rate["model_kib_per_class"])
    total_kib = kib_per_class + label_kib + label_model_kib
    payload = {
        "images": normalize_images(mosaics, mean, std).detach().cpu(),
        "labels": labels.detach().cpu(),
        "mean": mean,
        "std": std,
        "teacher": "ResNet18ImageNetBN",
        "loss": (
            "CE + alpha * BN"
            if args.utility_mode == "sre2l"
            else "CIM paired layer3 feature L1"
        ),
        "codec": "official DD-RUO TensorPool",
        "rate_kind": rate_kind,
        "latent_bpp": float(latent_bpp),
        "bpp": reported_bpp,
        "kib_per_class": kib_per_class,
        "image_kib_per_class": kib_per_class,
        "label_kib_per_class": label_kib,
        "label_model_kib_per_class": label_model_kib,
        "total_kib_per_class": total_kib,
        "rate_components": rate_components,
        "stored_ipc": ipc,
        "decoded_views_per_class": ipc * factor**2,
        "cim_factor": factor if args.utility_mode == "cim" else None,
        "utility_mode": args.utility_mode,
        "joint_image_lambda": args.joint_image_lambda,
        "joint_label_lambda": args.joint_label_lambda,
    }
    torch.save(payload, args.synthetic_path)
    save_and_print(
        args.log_path,
        f"final rate_kind={rate_kind} latent_bpp={float(latent_bpp):.6f} "
        f"total_bpp={reported_bpp:.6f} kib_per_class={kib_per_class:.2f} "
        f"label_kib_per_class={label_kib:.2f} "
        f"label_model_kib_per_class={label_model_kib:.2f} "
        f"combined_kib_per_class={total_kib:.2f} "
        f"stored_ipc={ipc} decoded_views_per_class={ipc * factor**2} "
        f"utility_mode={args.utility_mode} "
        f"mosaic_ce={ce:.6f} mosaic_teacher_acc={accuracy:.4f} "
        f"output={args.synthetic_path}",
    )


def main(args):
    set_seed(args.seed)
    if args.utility_mode == "cim":
        if args.cim_factor < 1:
            raise ValueError("--cim_factor must be positive")
        if args.cim_class_batch < 1:
            raise ValueError("--cim_class_batch must be positive")
        if args.cim_mipc < args.ipc * args.cim_factor**2:
            raise ValueError("--cim_mipc must cover ipc * factor**2 anchors")
    elif not args.init_path and not args.random_pool_init:
        raise ValueError("--init_path is required for SRe2L utility")
    if args.random_pool_init and args.resume_pool:
        raise ValueError("--random_pool_init and --resume_pool are mutually exclusive")
    if args.joint_rate_optimization:
        if args.rate_control != "fixed":
            raise ValueError("joint optimization requires --rate_control fixed")
        if args.joint_image_lambda < 0 or args.joint_label_lambda < 0:
            raise ValueError("joint lambdas must be non-negative")
    if args.rate_control == "dual" and args.ldb <= 0:
        raise ValueError("--ldb must be positive for dual rate conversion")
    if args.hard_ce_weight < 0 or args.label_kl_weight < 0:
        raise ValueError("Auxiliary utility weights must be non-negative")
    if args.label_kl_target_fraction < 0:
        raise ValueError("--label_kl_target_fraction must be non-negative")
    if not 0.0 <= args.label_kl_ema_decay < 1.0:
        raise ValueError("--label_kl_ema_decay must be in [0, 1)")
    if args.label_kl_weight_min < 0 or args.label_kl_weight_max <= 0:
        raise ValueError("Adaptive label-KL weight bounds must be positive")
    if args.label_kl_weight_min > args.label_kl_weight_max:
        raise ValueError("label KL minimum weight exceeds maximum weight")
    if not 0.0 < args.sre2l_crop_scale_min <= 1.0:
        raise ValueError("--sre2l_crop_scale_min must be in (0, 1]")
    if not 0.0 <= args.sre2l_flip_probability <= 1.0:
        raise ValueError("--sre2l_flip_probability must be in [0, 1]")
    if args.label_rate_gradient_ratio_cap < 0.0:
        raise ValueError("--label_rate_gradient_ratio_cap must be non-negative")
    if args.label_rate_decoder_only and args.label_rate_gradient_ratio_cap > 0.0:
        raise ValueError(
            "Use either --label_rate_decoder_only or "
            "--label_rate_gradient_ratio_cap, not both"
        )
    if args.label_kl_temperature <= 0:
        raise ValueError("--label_kl_temperature must be positive")
    if args.separate_rate_budgets:
        if not args.optimize_label_rate:
            raise ValueError("--separate_rate_budgets requires --optimize_label_rate")
        if args.image_target_kib <= args.image_model_overhead_kib:
            raise ValueError(
                "--image_target_kib must exceed --image_model_overhead_kib"
            )
        if args.label_target_kib <= 0:
            raise ValueError("--label_target_kib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("At least one visible GPU is required")
    if not args.tensorpool_device_ids:
        args.tensorpool_device_ids = list(range(device_count))
    if any(index < 0 or index >= device_count
           for index in args.tensorpool_device_ids):
        raise ValueError("--tensorpool_device_ids contains a non-visible GPU")
    if args.teacher_device_ids is None:
        args.teacher_device_ids = list(range(device_count))
    if any(index < 0 or index >= device_count
           for index in args.teacher_device_ids):
        raise ValueError("--teacher_device_ids contains a non-visible GPU")
    if args.teacher_data_parallel and not args.teacher_device_ids:
        raise ValueError("--teacher_data_parallel needs teacher devices")

    os.makedirs(args.save_path, exist_ok=True)
    args.log_path = os.path.join(args.save_path, "log.txt")
    args.synthetic_path = os.path.join(args.save_path, "synthetic.pt")
    args.zca = False
    args.device = "cuda:0"

    _, image_size, classes, _, mean, std, train_set, _, _, _, _, _ = dataset_context(
        args, remap_labels=(args.utility_mode == "cim")
    )
    checkpoint = torch.load(args.teacher_path, map_location="cpu", weights_only=False)
    teacher_primary = (
        args.teacher_device_ids[0] if args.teacher_data_parallel else 0
    )
    teacher = build_resnet18_bn(classes, image_size).to(
        f"cuda:{teacher_primary}"
    )
    teacher.load_state_dict(checkpoint["state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    save_and_print(args.log_path, f"begin={get_time()}")
    save_and_print(
        args.log_path,
        "implementation=joint_label_rate+official_DD-RUO "
        f"utility_mode={args.utility_mode} "
        f"cim_factor={args.cim_factor if args.utility_mode == 'cim' else 1} "
        f"stored_ipc={args.ipc} "
        f"decoded_views_per_class={args.ipc * (args.cim_factor**2 if args.utility_mode == 'cim' else 1)} "
        f"joint_rate_optimization={args.joint_rate_optimization} "
        f"joint_image_lambda={args.joint_image_lambda:g} "
        f"joint_label_lambda={args.joint_label_lambda:g} "
        f"sre2l_crop_scale_min={args.sre2l_crop_scale_min:g} "
        f"sre2l_flip_probability={args.sre2l_flip_probability:g} "
        f"label_rate_gradient_ratio_cap={args.label_rate_gradient_ratio_cap:g} "
        f"latent_target_kib={args.latent_target_kib:g} stage1={args.stage1_iterations} "
        f"stage2={args.stage2_iterations}",
    )
    bn_feature = None
    anchors_uint8 = None
    if args.utility_mode == "cim":
        reference_cache = os.path.join(args.save_path, "cim_references.pt")
        references, anchors_uint8, reference_labels = build_or_load_cim_references(
            reference_cache, train_set, teacher, classes, args.ipc, args.cim_factor,
            mean, std, image_size, args.cim_selection_batch,
            args.cim_selection_workers,
            args.device, args.log_path, args.cim_mipc, args.seed,
        )
    else:
        references = None
        if not args.random_pool_init:
            references = load_reference_images(
                args.init_path, classes, args.ipc, mean, std
            )
        reference_labels = torch.arange(classes).repeat_interleave(args.ipc)
        bn_feature = BNFeatureLoss(teacher)
        if args.teacher_data_parallel and len(args.teacher_device_ids) > 1:
            teacher = nn.DataParallel(
                teacher, device_ids=args.teacher_device_ids,
                output_device=args.teacher_device_ids[0],
            )
        save_and_print(
            args.log_path,
            f"sre2l_initialization={'random_codec' if args.random_pool_init else args.init_path} "
            f"bn_weight={args.bn_weight:g} "
            f"teacher_data_parallel={isinstance(teacher, nn.DataParallel)} "
            f"teacher_device_ids={args.teacher_device_ids} "
            f"visible_gpus={device_count}",
        )

    # Match the stable scheduling used by the existing official TensorPool run.
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.backends.cudnn.benchmark = args.enable_cudnn
    locks = install_per_gpu_tensorpool_locks(device_count)
    pool = TensorPool(
        classes, 1000, [args.ipc] * classes, args.tensorpool_device_ids,
        nthread=classes,
        ldb=args.ldb, img_size=image_size,
        max_iter=args.stage1_iterations + args.stage2_iterations,
        channel=3, lr=args.codec_lr, layers_v=args.layers_v,
        arm=args.arm, dim=args.dim,
    )

    pool_init_path = os.path.join(args.save_path, "pool_init.pt")
    if args.resume_pool:
        pool.load_slice_pool(args.resume_pool)
        validate_initialized_pool(pool)
        save_and_print(args.log_path, f"pool_loaded={args.resume_pool}")
    elif os.path.isfile(pool_init_path):
        pool.load_slice_pool(pool_init_path)
        validate_initialized_pool(pool)
        save_and_print(args.log_path, f"pool_loaded={pool_init_path}")
    elif args.random_pool_init:
        pool.free_model()
        for index, key in enumerate(pool.key_list):
            model = pool.get_model()
            parameters = model.produce_parameters(pool.slice_nums[index])
            pool.slice_pool[key]["param"].set_params(
                parameters.get_params(), "cpu"
            )
        pool.free_model()
        validate_initialized_pool(pool)
        pool.save_slice_pool(pool_init_path)
        save_and_print(
            args.log_path,
            f"random_pool_initialized seed={args.seed} output={pool_init_path}",
        )
    else:
        save_and_print(args.log_path, f"candidate_warmup_start={get_time()}")
        if args.warmup_rate_control == "dual":
            target_bpp = (args.warmup_target_kib * args.warmup_rate_margin) / (
                args.ipc * image_size[0] * image_size[1] / 8192.0
            )
            initialize_pool_with_dual_warmup(
                pool, references, target_bpp, args.warmup_dual_lr,
                args.enforce_pixel_range, args.pixel_range_weight, locks,
                args.fast_single_warmup, args.fast_warmup_iterations,
            )
        else:
            pool.init_from_data(references)
        validate_initialized_pool(pool)
        pool.save_slice_pool(pool_init_path)
        save_and_print(
            args.log_path,
            f"candidate_warmup_done={get_time()} output={pool_init_path}",
        )

    pool.test()
    warmup_images, warmup_labels, warmup_bpp = pool.get_data()
    warmup_labels = warmup_labels.to(warmup_images.device)
    warmup_ce, warmup_acc = mosaic_teacher_diagnostics(
        warmup_images, warmup_labels, teacher, mean, std
    )
    kib_per_bpp = args.ipc * image_size[0] * image_size[1] / 8192.0
    save_and_print(
        args.log_path,
        f"warmup_hard_check latent_bpp={float(warmup_bpp):.6f} "
        f"latent_kib_per_class={float(warmup_bpp) * kib_per_bpp:.2f} "
        f"mosaic_ce={warmup_ce:.6f} mosaic_teacher_acc={warmup_acc:.4f}",
    )

    label_entropy = None
    label_entropy_optimizer = None
    label_rate = None
    label_model_kib = 0.0
    if args.optimize_label_rate:
        label_entropy = AugmentedSlotStepLabelEntropy(
            args.ipc * classes,
            classes=classes,
            dims=classes - 1,
        ).to(args.device)
        label_entropy_optimizer = torch.optim.AdamW(
            label_entropy.parameters(), lr=args.label_entropy_lr,
            weight_decay=1e-5,
        )
        label_model_kib = (
            sum(parameter.numel() for parameter in label_entropy.parameters())
            * args.label_model_bits
            / classes / 8192.0
        )
        save_and_print(
            args.log_path,
            f"label_entropy_start step={args.label_step:g} groups={args.label_groups} "
            f"target_total_kib={args.total_target_kib:g} "
            f"separate_rate_budgets={args.separate_rate_budgets} "
            f"image_target_kib={args.image_target_kib:g} "
            f"label_target_kib={args.label_target_kib:g} "
            f"image_model_overhead_kib={args.image_model_overhead_kib:g} "
            f"model_kib_per_class={label_model_kib:.4f} "
            f"hard_ce_weight={args.hard_ce_weight:g} "
            f"label_kl_weight={args.label_kl_weight:g} "
            f"label_kl_target_fraction={args.label_kl_target_fraction:g} "
            f"label_kl_temperature={args.label_kl_temperature:g} "
            "reference=hard_class",
        )
        pool.test()
        entropy_images, entropy_labels, _ = pool.get_data()
        # Entropy-model warmup does not update images. Detaching avoids keeping
        # the full TensorPool decode graph alive across all label chunks.
        entropy_images = entropy_images.detach()
        entropy_labels = entropy_labels.to(entropy_images.device)
        for warmup_iteration in range(1, args.label_entropy_warmup + 1):
            hook_context = (
                bn_feature.suspended() if bn_feature is not None else nullcontext()
            )
            with hook_context:
                label_rate = soft_label_rate_step(
                    entropy_images, entropy_labels, teacher, label_entropy,
                    label_entropy_optimizer, mean, std,
                    args.label_step, args.label_groups, classes, args.ipc,
                    args.label_feature_chunk, args.label_entropy_batch, 0.0,
                    train_entropy=True,
                )
            if warmup_iteration == 1 or warmup_iteration % 5 == 0:
                save_and_print(
                    args.log_path,
                    f"label_entropy_warmup={warmup_iteration:03d}/"
                    f"{args.label_entropy_warmup:03d} "
                    f"label_kib_per_class={label_rate['label_kib_per_class']:.2f} "
                    f"top1_agreement={label_rate['quantized_top1_agreement']:.4f} "
                    f"teacher_margin={label_rate['teacher_margin']:.4f} "
                    f"zero_symbol_fraction={label_rate['zero_symbol_fraction']:.4f}",
                )
        del entropy_images, entropy_labels
        torch.cuda.empty_cache()

    if not torch.equal(reference_labels, pool.label):
        raise RuntimeError("CIM reference and TensorPool label ordering differ")
    pool.set_training_phase(0)
    pool.init_solvers()
    total_iterations = args.stage1_iterations + args.stage2_iterations
    dual_lambda_eff = 0.0
    image_dual_lambda_eff = 0.0
    label_dual_lambda_eff = 0.0
    label_kl_ema = None if label_rate is None else float(label_rate["label_kl"])
    start_time = time.time()

    for iteration in range(1, total_iterations + 1):
        pool.train()
        mosaics, pool_labels, bpp = pool.get_data()
        if not torch.equal(pool_labels.cpu(), pool.label):
            raise RuntimeError("TensorPool label ordering changed unexpectedly")
        pool.data.grad = None
        if args.utility_mode == "cim":
            utility = cim_utility_and_backward(
                mosaics, anchors_uint8, teacher, mean, std, args.cim_factor,
                args.ipc, args.cim_class_batch, iteration, args.cim_augmentation,
            )
            utility_detail = f"cim_feature_l1={utility.item():.8f}"
        else:
            utility, sre2l_ce, sre2l_bn, sre2l_acc = (
                augmented_sre2l_metrics_and_backward(
                    mosaics, pool_labels.to(mosaics.device), teacher, bn_feature,
                    mean, std, args.bn_weight,
                    crop_scale_min=args.sre2l_crop_scale_min,
                    flip_probability=args.sre2l_flip_probability,
                    batch_size=args.utility_batch_size,
                )
            )
            utility_detail = (
                f"sre2l_utility={utility.item():.8f} ce={sre2l_ce.item():.6f} "
                f"bn={sre2l_bn.item():.6f} teacher_acc={sre2l_acc.item():.4f}"
            )
        preserve_utility_gradient = (
            args.label_rate_decoder_only
            or args.label_rate_gradient_ratio_cap > 0.0
        )
        utility_gradient = (
            pool.data.grad.detach().clone()
            if preserve_utility_gradient else None
        )
        if args.optimize_label_rate:
            effective_label_kl_weight = args.label_kl_weight
            if args.label_kl_target_fraction > 0.0:
                denominator = max(
                    label_kl_ema if label_kl_ema is not None else 0.0, 1e-12
                )
                effective_label_kl_weight = min(
                    args.label_kl_weight_max,
                    max(
                        args.label_kl_weight_min,
                        args.label_kl_target_fraction * utility.item() / denominator,
                    ),
                )
            if args.joint_rate_optimization:
                label_rate_multiplier = args.joint_label_lambda
            else:
                label_rate_multiplier = (
                    label_dual_lambda_eff
                    if args.separate_rate_budgets else dual_lambda_eff
                )
            hook_context = (
                bn_feature.suspended() if bn_feature is not None else nullcontext()
            )
            with hook_context:
                label_rate = soft_label_rate_step(
                    mosaics, pool_labels.to(mosaics.device), teacher, label_entropy,
                    label_entropy_optimizer, mean, std,
                    args.label_step, args.label_groups, classes, args.ipc,
                    args.label_feature_chunk, args.label_entropy_batch,
                    label_rate_multiplier,
                    hard_ce_weight=args.hard_ce_weight,
                    label_kl_weight=effective_label_kl_weight,
                    label_kl_temperature=args.label_kl_temperature,
                    train_entropy=True,
                )
            current_label_kl = float(label_rate["label_kl"])
            if label_kl_ema is None:
                label_kl_ema = current_label_kl
            else:
                label_kl_ema = (
                    args.label_kl_ema_decay * label_kl_ema
                    + (1.0 - args.label_kl_ema_decay) * current_label_kl
                )
        if pool.data.grad is None or not torch.isfinite(pool.data.grad).all():
            raise FloatingPointError("Non-finite or missing CIM image gradient")
        gradient_l1 = pool.data.grad.abs().sum().item()
        utility_gradient_l1 = (
            gradient_l1
            if utility_gradient is None else utility_gradient.abs().sum().item()
        )
        decoder_only_gradient = None
        label_rate_gradient_l1 = 0.0
        label_rate_gradient_scale = 1.0
        if args.label_rate_decoder_only:
            decoder_only_gradient = pool.data.grad.detach() - utility_gradient
            if not torch.isfinite(decoder_only_gradient).all():
                raise FloatingPointError("Non-finite decoder-only label-rate gradient")
            label_rate_gradient_l1 = decoder_only_gradient.abs().sum().item()
            # The ordinary output gradient now contains utility only. The
            # separated label-rate gradient is applied to decoder parameters
            # explicitly inside TensorPool.backward().
            pool.data.grad = utility_gradient
        elif args.label_rate_gradient_ratio_cap > 0.0 and args.optimize_label_rate:
            label_gradient = pool.data.grad.detach() - utility_gradient
            utility_norm = utility_gradient.norm()
            label_norm = label_gradient.norm()
            maximum_label_norm = (
                args.label_rate_gradient_ratio_cap * utility_norm
            )
            label_rate_gradient_scale = min(
                1.0,
                float(maximum_label_norm / label_norm.clamp_min(1e-12)),
            )
            label_gradient.mul_(label_rate_gradient_scale)
            pool.data.grad = utility_gradient + label_gradient
            label_rate_gradient_l1 = label_gradient.abs().sum().item()
        pool.fill_data_diff()
        latent_kib = float(bpp) * kib_per_bpp
        if args.joint_rate_optimization:
            # TensorPool's internal rate gradient includes qp.ldb. This is the
            # exact conversion for lr_it * (L_sre2l + lambda_1 * image_bpp).
            ldb_it = args.joint_image_lambda * args.lr_it / args.ldb
        elif args.rate_control == "dual":
            image_rate_multiplier = (
                image_dual_lambda_eff
                if args.separate_rate_budgets else dual_lambda_eff
            )
            ldb_it = image_rate_multiplier * args.lr_it / args.ldb
        else:
            ldb_it = (
                args.stage1_ldb_it if iteration <= args.stage1_iterations
                else args.stage2_ldb_it
            )
        pool.backward(
            args.lr_it, ldb_it, decoder_only_grad=decoder_only_gradient,
        )
        if args.enable_codec_scheduler:
            pool.validate(iteration - 1)
        if args.optimize_label_rate:
            combined_kib = (
                latent_kib + label_rate["label_kib_per_class"] + label_model_kib
            )
            if args.separate_rate_budgets:
                image_accounted_kib = latent_kib + args.image_model_overhead_kib
                label_accounted_kib = (
                    label_rate["label_kib_per_class"] + label_model_kib
                )
                image_violation = image_accounted_kib / args.image_target_kib - 1.0
                label_violation = label_accounted_kib / args.label_target_kib - 1.0
                violation = max(image_violation, label_violation)
            else:
                violation = combined_kib / args.total_target_kib - 1.0
        else:
            combined_kib = latent_kib
            violation = latent_kib / args.latent_target_kib - 1.0
        if args.rate_control == "dual":
            if args.separate_rate_budgets:
                image_dual_lambda_eff = max(
                    0.0,
                    image_dual_lambda_eff + args.image_dual_lr * image_violation,
                )
                label_dual_lambda_eff = max(
                    0.0,
                    label_dual_lambda_eff + args.label_dual_lr * label_violation,
                )
            else:
                dual_lambda_eff = max(
                    0.0, dual_lambda_eff + args.dual_lr * violation
                )

        if iteration == 1 or iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            seconds_per_iter = elapsed / iteration
            eta_hours = seconds_per_iter * (total_iterations - iteration) / 3600.0
            save_and_print(
                args.log_path,
                f"iter={iteration:05d}/{total_iterations:05d} "
                f"{utility_detail} latent_bpp={float(bpp):.6f} "
                f"latent_kib_per_class={latent_kib:.2f} image_grad_l1={gradient_l1:.4e} "
                f"utility_grad_l1={utility_gradient_l1:.4e} "
                f"label_rate_grad_l1={label_rate_gradient_l1:.4e} "
                f"label_rate_grad_scale={label_rate_gradient_scale:.6f} "
                + (
                    f"label_kib_per_class={label_rate['label_kib_per_class']:.2f} "
                    f"combined_kib_per_class={combined_kib:.2f} "
                    f"label_top1_agreement={label_rate['quantized_top1_agreement']:.4f} "
                    f"hard_ce={label_rate['hard_ce']:.6f} "
                    f"label_kl={label_rate['label_kl']:.6f} "
                    f"label_kl_weight_eff={effective_label_kl_weight:.6g} "
                    f"teacher_margin={label_rate['teacher_margin']:.4f} "
                    f"zero_symbol_fraction={label_rate['zero_symbol_fraction']:.4f} "
                    if args.optimize_label_rate else ""
                )
                +
                (
                    f"image_accounted_kib={image_accounted_kib:.2f} "
                    f"label_accounted_kib={label_accounted_kib:.2f} "
                    f"image_dual={image_dual_lambda_eff:.8g} "
                    f"label_dual={label_dual_lambda_eff:.8g} "
                    f"image_violation={image_violation:.6f} "
                    f"label_violation={label_violation:.6f} "
                    if args.separate_rate_budgets and not args.joint_rate_optimization else
                    f"dual_lambda_eff={dual_lambda_eff:.8g} "
                    f"dual_violation={violation:.6f} "
                )
                +
                (
                    f"joint_image_lambda={args.joint_image_lambda:g} "
                    f"joint_label_lambda={args.joint_label_lambda:g} "
                    if args.joint_rate_optimization else ""
                )
                +
                f"ldb_it={ldb_it:g} codec_lr={current_codec_lr(pool):.7g} "
                f"sec_per_iter={seconds_per_iter:.2f} "
                f"eta_hours={eta_hours:.2f}",
            )

        if iteration % args.checkpoint_every == 0 or iteration == total_iterations:
            checkpoint_path = os.path.join(args.save_path, f"pool_{iteration}.pt")
            pool.save_slice_pool(checkpoint_path)
            if args.optimize_label_rate:
                torch.save(
                    {
                            "experiment_type": "rosd_fixed_label_rate",
                        "label_entropy": label_entropy.state_dict(),
                        "config": {
                            "num_slots": args.ipc * classes,
                            "classes": classes,
                            "label_step": args.label_step,
                            "label_groups": args.label_groups,
                            "reference_mode": "hard_class",
                            "utility_mode": args.utility_mode,
                        },
                    },
                    os.path.join(args.save_path, f"label_codec_{iteration}.pt"),
                )
            save_and_print(args.log_path, f"checkpoint={checkpoint_path}")

    pool.save_slice_pool(os.path.join(args.save_path, "pool_pre_net_quant.pt"))
    rate_kind = "latent_entropy"
    total_bpp = None
    rate_components = None
    if not args.skip_net_quantization:
        decoder_scorers = None
        if args.optimize_label_rate and args.postquant_label_rate_samples > 0:
            label_checkpoint = {
                "experiment_type": "rosd_fixed_label_rate",
                "label_entropy": label_entropy.state_dict(),
                "config": {
                    "num_slots": args.ipc * classes,
                    "classes": classes,
                    "label_step": args.label_step,
                    "label_groups": args.label_groups,
                    "reference_mode": "hard_class",
                    "utility_mode": args.utility_mode,
                },
            }
            fixed_label_codec = FixedCIMLabelCodec(label_checkpoint).cuda().eval()
            decoder_scorers = build_class_scorers(
                teacher,
                fixed_label_codec,
                0,
                classes,
                args.ipc,
                args.label_groups,
                args.postquant_label_rate_samples,
                mean=mean,
                std=std,
            )
        result = pool.quantize_net(
            args.network_mse_threshold,
            decoder_soft_label_rate_fns=decoder_scorers,
        )
        if result is None:
            raise RuntimeError("DD-RUO network post-quantization failed")
        total_bpp, rate_components = result
        pool.save_slice_pool(os.path.join(args.save_path, "pool_quantized.pt"))
        save_and_print(
            args.log_path,
            f"network_quantization total_bpp={float(total_bpp):.6f} "
            f"components={rate_components}",
        )
        rate_kind = "latent_entropy_after_network_quantization"

    export_cim_payload(
        pool, teacher, mean, std, args.ipc,
        args.cim_factor if args.utility_mode == "cim" else 1, args,
        rate_kind, total_bpp, rate_components,
        None if label_rate is None else {
            **label_rate, "model_kib_per_class": label_model_kib,
        },
    )
    if bn_feature is not None:
        bn_feature.close()
    save_and_print(args.log_path, f"complete={get_time()}")


def build_parser():
    parser = build_base_parser()
    parser.description = "CIM feature matching with the official DD-RUO codec"
    # CIM initializes from selected real-image references rather than an SRe2L
    # synthetic-image checkpoint. Keep the inherited option optional so this
    # entry point can run independently.
    for action in parser._actions:
        if action.dest == "init_path":
            action.required = False
            action.default = ""
    parser.add_argument("--cim_factor", type=int, default=2)
    parser.add_argument("--utility_mode", choices=("cim", "sre2l"), default="cim")
    parser.add_argument("--cim_mipc", type=int, default=300)
    parser.add_argument("--cim_class_batch", type=int, default=2)
    parser.add_argument("--cim_feature_chunk", type=int, default=16)
    parser.add_argument("--cim_selection_batch", type=int, default=128)
    parser.add_argument("--cim_selection_workers", type=int, default=8)
    parser.add_argument("--cim_augmentation", default="crop_cutout_flip")
    parser.add_argument("--sre2l_crop_scale_min", type=float, default=0.08)
    parser.add_argument("--sre2l_flip_probability", type=float, default=0.5)
    parser.add_argument("--teacher_data_parallel", action="store_true")
    parser.add_argument("--teacher_device_ids", type=int, nargs="+", default=None)
    parser.add_argument("--tensorpool_device_ids", type=int, nargs="+", default=None)
    parser.add_argument("--random_pool_init", action="store_true")
    parser.add_argument("--optimize_label_rate", action="store_true")
    parser.add_argument("--label_step", type=float, default=0.5)
    parser.add_argument("--label_groups", type=int, default=300)
    parser.add_argument("--label_entropy_lr", type=float, default=1e-3)
    parser.add_argument("--label_entropy_warmup", type=int, default=20)
    parser.add_argument("--label_feature_chunk", type=int, default=128)
    parser.add_argument("--label_entropy_batch", type=int, default=128)
    parser.add_argument("--hard_ce_weight", type=float, default=0.0)
    parser.add_argument("--label_kl_weight", type=float, default=0.0)
    parser.add_argument("--label_kl_target_fraction", type=float, default=0.0)
    parser.add_argument("--label_kl_ema_decay", type=float, default=0.9)
    parser.add_argument("--label_kl_weight_min", type=float, default=0.0)
    parser.add_argument("--label_kl_weight_max", type=float, default=20.0)
    parser.add_argument("--label_kl_temperature", type=float, default=2.0)
    parser.add_argument("--label_model_bits", type=float, default=16.0)
    parser.add_argument("--total_target_kib", type=float, default=1300.0)
    parser.add_argument("--separate_rate_budgets", action="store_true")
    parser.add_argument("--image_target_kib", type=float, default=100.0)
    parser.add_argument("--label_target_kib", type=float, default=1200.0)
    parser.add_argument("--image_model_overhead_kib", type=float, default=0.0)
    parser.add_argument("--image_dual_lr", type=float, default=1e-4)
    parser.add_argument("--label_dual_lr", type=float, default=1e-4)
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--enable_codec_scheduler", action="store_true")
    parser.add_argument("--label_rate_decoder_only", action="store_true")
    parser.add_argument("--label_rate_gradient_ratio_cap", type=float, default=0.0)
    parser.add_argument("--joint_rate_optimization", action="store_true")
    parser.add_argument("--joint_image_lambda", type=float, default=0.0)
    parser.add_argument("--joint_label_lambda", type=float, default=0.0)
    parser.add_argument("--postquant_label_rate_samples", type=int, default=1)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
