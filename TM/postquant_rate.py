"""Rate scorers used during ROSD network post-quantization."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from TM.cim_label_codec import sample_shared_augmentation


def normalize_images(images, mean, std):
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


def apply_fkd_augmentation_chunk(images, augmentation, indices, output_size):
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
    return primary


@dataclass
class DecoderSoftLabelRateScorer:
    """Estimate stored label bits for one decoded class payload."""

    teacher: torch.nn.Module
    label_codec: torch.nn.Module
    hard_class: int
    global_slot_start: int
    label_groups: int
    samples: int = 1
    crop_scale_min: float = 0.08
    flip_probability: float = 0.5
    cutmix_alpha: float = 1.0
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    _augmentations: list = field(default_factory=list, init=False, repr=False)

    def _fixed_augmentations(self, decoded_images):
        if self._augmentations:
            return self._augmentations
        device = decoded_images.device
        devices = []
        if device.type == "cuda":
            device_index = (
                torch.cuda.current_device()
                if device.index is None
                else device.index
            )
            devices = [device_index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1729 + self.hard_class)
            if device.type == "cuda":
                torch.cuda.manual_seed(1729 + self.hard_class)
            self._augmentations = [
                sample_shared_augmentation(
                    decoded_images,
                    crop_scale_min=self.crop_scale_min,
                    flip_probability=self.flip_probability,
                    cutmix_alpha=self.cutmix_alpha,
                )
                for _ in range(self.samples)
            ]
        return self._augmentations

    @torch.inference_mode()
    def __call__(self, decoded_images: torch.Tensor) -> float:
        if decoded_images.ndim != 4:
            raise ValueError(
                f"Expected NCHW decoded images, got {tuple(decoded_images.shape)}"
            )
        device = decoded_images.device
        codec_device = next(self.label_codec.parameters()).device
        if codec_device != device:
            self.label_codec.to(device)
        count = decoded_images.shape[0]
        hard_labels = torch.full(
            (count,), self.hard_class, dtype=torch.long, device=device
        )
        global_slots = torch.arange(
            self.global_slot_start,
            self.global_slot_start + count,
            dtype=torch.long,
            device=device,
        )
        sample_bits = []
        for augmentation in self._fixed_augmentations(decoded_images):
            indices = torch.arange(count, device=device)
            augmented = apply_fkd_augmentation_chunk(
                decoded_images, augmentation, indices, decoded_images.shape[-1]
            )
            teacher_device = next(self.teacher.parameters()).device
            logits = self.teacher(
                normalize_images(augmented, self.mean, self.std).to(teacher_device)
            ).to(device)
            _, bits = self.label_codec.quantize(
                logits,
                hard_labels,
                global_slots,
                augmentation["mix_index"],
                torch.stack(
                    [
                        augmentation["tops"],
                        augmentation["lefts"],
                        augmentation["heights"],
                        augmentation["widths"],
                    ],
                    dim=1,
                ),
                augmentation["flips"],
                augmentation["bbox"],
                decoded_images.shape[-1],
            )
            sample_bits.append(float(bits))
        return sum(sample_bits) / len(sample_bits) * self.label_groups


def build_class_scorers(
    teacher,
    label_codec,
    class_start,
    classes_per_shard,
    ipc,
    label_groups,
    samples,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return {
        f"{local_class}_0": DecoderSoftLabelRateScorer(
            teacher=teacher,
            label_codec=label_codec,
            hard_class=class_start + local_class,
            global_slot_start=(class_start + local_class) * ipc,
            label_groups=label_groups,
            samples=samples,
            mean=tuple(mean),
            std=tuple(std),
        )
        for local_class in range(classes_per_shard)
    }
