"""LPLD class-wise BatchNorm utility for ImageNet recovery."""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF


class ClassAwareBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with the class-statistic buffers stored by official LPLD."""

    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__(num_features, **kwargs)
        self.register_buffer(
            "class_running_mean", torch.zeros(num_classes, num_features)
        )
        self.register_buffer(
            "class_running_var", torch.ones(num_classes, num_features)
        )


def _replace_batch_norm(module, num_classes):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = ClassAwareBatchNorm2d(
                child.num_features,
                num_classes,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child, num_classes)


def build_lpld_teacher(num_classes, image_size):
    """Build the ResNet-18 architecture expected by an LPLD checkpoint."""
    model = models.resnet18(num_classes=num_classes)
    if int(image_size) <= 64:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    _replace_batch_norm(model, num_classes)
    return model


def build_lpld_tiny_teacher(num_classes=200):
    """Backward-compatible Tiny-ImageNet teacher constructor."""
    return build_lpld_teacher(num_classes, image_size=64)


class ClassConditionalBNFeatureLoss:
    """Official LPLD class-statistic L2 loss with a reusable hook set."""

    def __init__(self, model, first_bn_multiplier=10.0):
        self.class_indices = None
        self.samples_per_class = None
        self.enabled = True
        self.records = []
        self.first_bn_multiplier = float(first_bn_multiplier)
        self.modules = [
            module for module in model.modules()
            if isinstance(module, ClassAwareBatchNorm2d)
        ]
        if not self.modules:
            raise ValueError("LPLD utility requires class-aware BatchNorm layers")
        self.handles = [
            module.register_forward_hook(self._make_hook(index))
            for index, module in enumerate(self.modules)
        ]

    def _make_hook(self, layer_index):
        def hook(module, inputs, _output):
            if not self.enabled:
                return
            feature = inputs[0]
            class_count = self.class_indices.numel()
            expected = class_count * self.samples_per_class
            if feature.shape[0] != expected:
                raise RuntimeError(
                    f"Expected {expected} grouped LPLD samples, got {feature.shape[0]}"
                )
            grouped = feature.reshape(
                class_count, self.samples_per_class, *feature.shape[1:]
            )
            mean = grouped.mean((1, 3, 4))
            # Reduce in the existing NCHW-derived layout. The previous
            # permute/contiguous/reshape path copied every activation at every
            # BN layer and became prohibitively expensive for larger batches.
            var = grouped.var((1, 3, 4), unbiased=False)
            self.records.append((layer_index, mean, var, module))

        return hook

    def set_class(self, class_index):
        self.set_classes([class_index], 1)

    def set_classes(self, class_indices, samples_per_class):
        device = self.modules[0].class_running_mean.device
        self.class_indices = torch.as_tensor(
            class_indices, device=device, dtype=torch.long
        )
        self.samples_per_class = int(samples_per_class)

    def clear(self):
        self.records.clear()

    def value(self):
        if self.class_indices is None or not self.records:
            raise RuntimeError("No LPLD class or BN features were captured")
        losses = []
        for layer_index, mean, var, module in self.records:
            target_mean = module.class_running_mean[self.class_indices]
            target_var = module.class_running_var[self.class_indices]
            loss = torch.linalg.vector_norm(
                target_var - var, ord=2, dim=1
            ) + torch.linalg.vector_norm(target_mean - mean, ord=2, dim=1)
            if layer_index == 0:
                loss = loss * self.first_bn_multiplier
            losses.append(loss.sum())
        return torch.stack(losses).sum()

    @contextmanager
    def suspended(self):
        previous = self.enabled
        self.enabled = False
        self.clear()
        try:
            yield
        finally:
            self.clear()
            self.enabled = previous

    def close(self):
        for handle in self.handles:
            handle.remove()


def _official_lpld_augmentation(images, jitter):
    top, left, height, width = RandomResizedCrop.get_params(
        images, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
    )
    augmented = TF.resized_crop(
        images,
        top,
        left,
        height,
        width,
        images.shape[-2:],
        antialias=True,
    )
    if torch.rand((), device=images.device) < 0.5:
        augmented = augmented.flip(-1)
    if jitter > 0:
        offsets = torch.randint(
            -jitter, jitter + 1, (2,), device=images.device
        ).tolist()
        augmented = torch.roll(
            augmented, shifts=(int(offsets[0]), int(offsets[1])), dims=(2, 3)
        )
    return augmented


def _normalize(images, mean, std):
    mean = images.new_tensor(mean).view(1, 3, 1, 1)
    std = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean) / std


def lpld_classwise_metrics_and_backward(
    images, labels, teacher, bn_feature, mean, std, bn_weight, jitter=4,
    max_images_per_forward=400,
):
    """Apply official LPLD recovery utility independently to every class.

    Each class loss is backpropagated without averaging it across classes. This
    matches official LPLD, where every class owns an independent image tensor
    and optimizer, while still allowing DD-RUO to store all class gradients in
    one TensorPool.
    """
    classes = labels.unique(sorted=True)
    utility_sum = images.new_zeros(())
    ce_sum = images.new_zeros(())
    bn_sum = images.new_zeros(())
    correct = images.new_zeros(())
    count = 0
    samples_per_class = int((labels == classes[0]).sum().item())
    if samples_per_class <= 0:
        raise RuntimeError("LPLD class batch is empty")
    classes_per_forward = max(1, int(max_images_per_forward) // samples_per_class)
    for start in range(0, classes.numel(), classes_per_forward):
        chunk_classes = classes[start:start + classes_per_forward]
        augmented_chunks = []
        label_chunks = []
        for class_index_tensor in chunk_classes:
            indices = torch.nonzero(
                labels == class_index_tensor, as_tuple=False
            ).flatten()
            if indices.numel() != samples_per_class:
                raise RuntimeError("LPLD vectorization requires equal IPC per class")
            augmented_chunks.append(
                _official_lpld_augmentation(images[indices], jitter)
            )
            label_chunks.append(labels[indices])
        augmented = torch.cat(augmented_chunks)
        chunk_labels = torch.cat(label_chunks)
        bn_feature.set_classes(chunk_classes, samples_per_class)
        bn_feature.clear()
        logits = teacher(_normalize(augmented, mean, std))
        ce_per_class = F.cross_entropy(
            logits, chunk_labels, reduction="none"
        ).reshape(chunk_classes.numel(), samples_per_class).mean(1)
        ce = ce_per_class.sum()
        bn = bn_feature.value()
        utility = ce + float(bn_weight) * bn
        utility.backward()
        utility_sum += utility.detach()
        ce_sum += ce.detach()
        bn_sum += bn.detach()
        correct += (logits.detach().argmax(1) == chunk_labels).sum()
        count += chunk_labels.numel()
    denominator = max(1, classes.numel())
    return (
        utility_sum / denominator,
        ce_sum / denominator,
        bn_sum / denominator,
        correct / max(1, count),
    )
