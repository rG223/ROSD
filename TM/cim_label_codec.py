"""Fixed-step soft-label quantization and entropy modeling for CIM-DD-RUO."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_round(values):
    """Round in the forward pass and use an identity backward pass."""
    return values + (torch.round(values) - values).detach()


def logistic_bits(symbols, mean, log_scale):
    """Estimate bits under a discretized logistic distribution."""
    bounded_log_scale = log_scale.clamp(math.log(0.05), math.log(100.0))
    scale = bounded_log_scale.exp()
    upper = (symbols + 0.5 - mean) / scale
    lower = (symbols - 0.5 - mean) / scale
    log_mass = (
        F.logsigmoid(upper)
        + F.logsigmoid(-lower)
        + torch.log(-torch.expm1(-1.0 / scale))
    )
    return -log_mass.clamp_min(math.log(1e-9)) / math.log(2.0)


class AugmentedSlotStepLabelEntropy(nn.Module):
    """Model FKD symbols from image slots and augmentation metadata."""

    def __init__(self, num_slots, classes=10, dims=9, slot_dim=16, hidden=96):
        super().__init__()
        self.primary_class = nn.Embedding(classes, hidden)
        self.partner_class = nn.Embedding(classes, hidden)
        self.primary_slot = nn.Embedding(num_slots, slot_dim)
        self.partner_slot = nn.Embedding(num_slots, slot_dim)
        self.slot_project = nn.Linear(2 * slot_dim, hidden)
        self.step_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.augmentation_embed = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dims),
        )

    def forward(
        self,
        labels,
        log_steps,
        slot_ids,
        partner_labels,
        partner_slot_ids,
        augmentation_features,
    ):
        slots = torch.cat(
            [self.primary_slot(slot_ids), self.partner_slot(partner_slot_ids)],
            dim=1,
        )
        feature = (
            self.primary_class(labels)
            + self.partner_class(partner_labels)
            + self.slot_project(slots)
            + self.step_embed(log_steps)
            + self.augmentation_embed(augmentation_features)
        )
        return self.out(feature).chunk(2, dim=-1)


def sample_shared_augmentation(
    images,
    crop_scale_min=0.08,
    crop_ratio_min=3.0 / 4.0,
    crop_ratio_max=4.0 / 3.0,
    flip_probability=0.5,
    cutmix_alpha=1.0,
):
    """Sample replayable RRC, flip, and CutMix parameters for one FKD group."""
    count, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    area = float(height * width)
    log_ratio_min = math.log(crop_ratio_min)
    log_ratio_max = math.log(crop_ratio_max)

    crop_heights = torch.full((count,), height, device=device, dtype=torch.long)
    crop_widths = torch.full((count,), width, device=device, dtype=torch.long)
    unresolved = torch.ones(count, device=device, dtype=torch.bool)
    for _ in range(10):
        target_area = (
            crop_scale_min
            + (1.0 - crop_scale_min) * torch.rand(count, device=device, dtype=dtype)
        ) * area
        aspect = torch.exp(
            log_ratio_min
            + (log_ratio_max - log_ratio_min)
            * torch.rand(count, device=device, dtype=dtype)
        )
        proposed_widths = torch.round(torch.sqrt(target_area * aspect)).long()
        proposed_heights = torch.round(torch.sqrt(target_area / aspect)).long()
        valid = (
            unresolved
            & (proposed_widths > 0)
            & (proposed_widths <= width)
            & (proposed_heights > 0)
            & (proposed_heights <= height)
        )
        crop_widths[valid] = proposed_widths[valid]
        crop_heights[valid] = proposed_heights[valid]
        unresolved &= ~valid
        if not unresolved.any():
            break

    max_top = height - crop_heights
    max_left = width - crop_widths
    tops = torch.floor(
        torch.rand(count, device=device, dtype=dtype) * (max_top + 1).to(dtype)
    ).long()
    lefts = torch.floor(
        torch.rand(count, device=device, dtype=dtype) * (max_left + 1).to(dtype)
    ).long()
    flips = torch.rand(count, device=device) < flip_probability
    mix_index = torch.randperm(count, device=device)

    if cutmix_alpha > 0:
        mix_lambda = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample()
        mix_lambda = mix_lambda.to(device=device, dtype=dtype)
        cut_ratio = torch.sqrt(1.0 - mix_lambda)
        cut_height = int(height * cut_ratio.item())
        cut_width = int(width * cut_ratio.item())
        center_x = int(torch.randint(height, (), device=device).item())
        center_y = int(torch.randint(width, (), device=device).item())
        x1 = max(center_x - cut_height // 2, 0)
        y1 = max(center_y - cut_width // 2, 0)
        x2 = min(center_x + cut_height // 2, height)
        y2 = min(center_y + cut_width // 2, width)
    else:
        x1 = y1 = x2 = y2 = 0

    partner_fraction = images.new_tensor(
        ((x2 - x1) * (y2 - y1)) / float(height * width)
    )
    augmentation_features = torch.stack(
        [
            (crop_heights * crop_widths).to(dtype) / area,
            torch.log(crop_widths.to(dtype) / crop_heights.to(dtype)),
            flips.to(dtype),
            partner_fraction.expand(count),
        ],
        dim=1,
    )
    return {
        "tops": tops,
        "lefts": lefts,
        "heights": crop_heights,
        "widths": crop_widths,
        "flips": flips,
        "mix_index": mix_index,
        "bbox": (x1, y1, x2, y2),
        "partner_fraction": partner_fraction,
        "features": augmentation_features,
    }
