"""Strict Eq. (3) SRe2L baseline with a BatchNorm ResNet18 teacher."""
import argparse
from contextlib import contextmanager
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import ParamDiffAug, evaluate_synset, get_dataset, get_network, get_time, save_and_print, set_seed
from core.networks import BasicBlock, ResNet18BN, ResNetImageNet


def build_resnet18_bn(num_classes, im_size=(128, 128)):
    """Select the BN ResNet-18 stem and pooling rule for the input resolution."""
    if max(im_size) <= 64:
        return ResNet18BN(channel=3, num_classes=num_classes)
    return ResNetImageNet(
        BasicBlock, [2, 2, 2, 2], channel=3,
        num_classes=num_classes, norm="batchnorm"
    )


class BNFeatureLoss:
    def __init__(self, model, mode="mse_mean", first_bn_multiplier=1.0):
        if mode not in {"mse_mean", "sre2l_l2_sum"}:
            raise ValueError(f"Unsupported BN feature-loss mode: {mode}")
        self.losses = []
        self.enabled = True
        self.mode = mode
        self.first_bn_multiplier = float(first_bn_multiplier)
        modules = [module for module in model.modules()
                   if isinstance(module, nn.BatchNorm2d)]
        self.handles = [
            module.register_forward_hook(self._make_hook(index))
            for index, module in enumerate(modules)
        ]
        if not self.handles:
            raise ValueError("SRe2L Eq. (3) requires BatchNorm layers")

    def _make_hook(self, layer_index):
        def hook(module, inputs, _output):
            if not self.enabled:
                return
            feature = inputs[0]
            dimensions = (0, 2, 3)
            count = feature.shape[0] * feature.shape[2] * feature.shape[3]
            # Sufficient statistics let value() reconstruct the exact global
            # batch moments when a frozen teacher is data-parallel.  Moving
            # these small channel vectors also preserves gradients to each
            # input shard.
            self.losses.append((
                layer_index,
                feature.sum(dimensions),
                feature.square().sum(dimensions),
                count,
                module.running_mean,
                module.running_var,
            ))
        return hook

    def clear(self):
        self.losses.clear()

    @contextmanager
    def suspended(self):
        """Disable BN-stat capture while preserving ordinary model gradients."""
        previous = self.enabled
        self.enabled = False
        self.clear()
        try:
            yield
        finally:
            self.clear()
            self.enabled = previous

    def value(self):
        if not self.losses:
            raise RuntimeError("No BatchNorm features were captured")
        # value() runs on DataParallel's output device (cuda:0 here), even if
        # another replica happened to finish first and append the first entry.
        output_device = torch.device("cuda", torch.cuda.current_device())
        layer_losses = []
        layer_indices = sorted({entry[0] for entry in self.losses})
        for layer_index in layer_indices:
            entries = [entry for entry in self.losses if entry[0] == layer_index]
            count = sum(entry[3] for entry in entries)
            feature_sum = sum(entry[1].to(output_device) for entry in entries)
            square_sum = sum(entry[2].to(output_device) for entry in entries)
            mean = feature_sum / count
            var = square_sum / count - mean.square()
            running_mean = entries[0][4].to(output_device)
            running_var = entries[0][5].to(output_device)
            if self.mode == "sre2l_l2_sum":
                loss = torch.norm(running_var - var, 2) + torch.norm(
                    running_mean - mean, 2
                )
                if layer_index == layer_indices[0]:
                    loss = loss * self.first_bn_multiplier
            else:
                loss = F.mse_loss(mean, running_mean) + F.mse_loss(
                    var, running_var
                )
            layer_losses.append(loss)
        stacked = torch.stack(layer_losses)
        return stacked.sum() if self.mode == "sre2l_l2_sum" else stacked.mean()

    def close(self):
        for handle in self.handles:
            handle.remove()


class RemappedDataset(Dataset):
    """Apply ImageNet-subset labels exactly as DD-RUO's tensor cache does."""
    def __init__(self, dataset, class_map):
        self.dataset = dataset
        self.class_map = class_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = int(label)
        if label in self.class_map:
            label = self.class_map[label]
        return image, label


def build_args_for_dataset(args):
    args.zca = False
    args.log_path = os.path.join(args.save_path, "log.txt")
    return args


def dataset_context(args, remap_labels=True):
    args = build_args_for_dataset(args)
    context = list(get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args))
    class_map = context[10]
    if remap_labels and class_map is not None:
        context[6] = RemappedDataset(context[6], class_map)
        context[7] = RemappedDataset(context[7], class_map)
        context[8] = DataLoader(context[7], batch_size=args.batch_real, shuffle=False,
                                num_workers=8, pin_memory=True)
    return tuple(context)


def train_teacher(args):
    _, im_size, num_classes, _, mean, std, train_set, _, testloader, _, _, _ = dataset_context(args)
    if args.teacher_augmentation == "cifar" and args.dataset.startswith("CIFAR"):
        base_dataset = train_set.dataset if isinstance(train_set, RemappedDataset) else train_set
        base_dataset.transform = transforms.Compose([
            transforms.RandomCrop(im_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    teacher = build_resnet18_bn(num_classes, im_size).to(args.device)
    optimizer = torch.optim.SGD(teacher.parameters(), lr=args.teacher_lr, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.teacher_epochs)
    loader = DataLoader(train_set, batch_size=args.batch_real, shuffle=True, num_workers=8, pin_memory=True)
    for epoch in range(1, args.teacher_epochs + 1):
        teacher.train(); correct = total = 0; loss_sum = 0.
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * labels.numel(); correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.teacher_epochs:
            teacher.eval(); test_correct = test_total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    logits = teacher(images.to(args.device)); labels = labels.to(args.device)
                    test_correct += (logits.argmax(1) == labels).sum().item(); test_total += labels.numel()
            save_and_print(args.log_path, f"teacher epoch={epoch:03d} train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f} test_acc={test_correct/test_total:.4f}")
    torch.save({"model": "ResNet18ImageNetBN", "state_dict": teacher.state_dict(), "im_size": im_size, "num_classes": num_classes}, args.teacher_path)
    save_and_print(args.log_path, f"teacher saved: {args.teacher_path}")


def real_initialization(train_set, ipc, num_classes):
    slots = [[] for _ in range(num_classes)]
    for image, label in train_set:
        label = int(label)
        if len(slots[label]) < ipc:
            slots[label].append(image)
        if all(len(items) == ipc for items in slots):
            break
    if not all(len(items) == ipc for items in slots):
        raise RuntimeError("Could not collect IPC real images for every class")
    return torch.cat([torch.stack(items) for items in slots]), torch.arange(num_classes).repeat_interleave(ipc)


def gaussian_initialization(ipc, num_classes, im_size):
    """Initialize every class/IPC slot independently in normalized image space."""
    images = torch.randn(num_classes * ipc, 3, *im_size)
    labels = torch.arange(num_classes).repeat_interleave(ipc)
    return images, labels


def prepare_initialization(args):
    _, im_size, num_classes, _, _, _, train_set, _, _, _, _, _ = dataset_context(args)
    if args.initialization == "gaussian":
        images, labels = gaussian_initialization(args.ipc, num_classes, im_size)
    else:
        images, labels = real_initialization(train_set, args.ipc, num_classes)
    torch.save(
        {
            "images": images.cpu(),
            "labels": labels.cpu(),
            "ipc": args.ipc,
            "initialization": args.initialization,
        },
        args.init_path,
    )
    save_and_print(
        args.log_path,
        f"saved fixed {args.initialization} initialization: {args.init_path} "
        f"({len(labels)} images)",
    )


def cda_min_crop_scale(iteration, total_iterations, minimum, maximum, milestone, mode):
    """Return CDA's global-to-local lower crop bound for the current iteration."""
    curriculum_iterations = max(1, int(total_iterations * milestone))
    progress = min(max(iteration / curriculum_iterations, 0.0), 1.0)
    if mode == "step":
        return maximum if progress < 1.0 else minimum
    if mode == "linear":
        weight = 1.0 - progress
    elif mode == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown CDA scheduler: {mode}")
    return minimum + (maximum - minimum) * weight


def cda_augment(images, min_crop, max_crop, output_size, flip_probability, jitter):
    """Apply the same differentiable CDA crop to a class-balanced image batch."""
    top, left, height, width = RandomResizedCrop.get_params(
        images,
        scale=(min_crop, max_crop),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    )
    augmented = TF.resized_crop(
        images,
        top,
        left,
        height,
        width,
        output_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if torch.rand((), device=images.device).item() < flip_probability:
        augmented = TF.hflip(augmented)
    if jitter > 0:
        shift_y = random.randint(0, jitter)
        shift_x = random.randint(0, jitter)
        augmented = torch.roll(augmented, shifts=(shift_y, shift_x), dims=(-2, -1))
    return augmented


def synthesize(args):
    _, im_size, num_classes, _, mean, std, train_set, _, testloader, _, _, _ = dataset_context(args)
    checkpoint = torch.load(args.teacher_path, map_location="cpu", weights_only=False)
    teacher = build_resnet18_bn(num_classes, im_size).to(args.device)
    teacher.load_state_dict(checkpoint["state_dict"]); teacher.eval()
    for param in teacher.parameters(): param.requires_grad_(False)
    if args.initialization == "gaussian":
        images, labels = gaussian_initialization(args.ipc, num_classes, im_size)
    elif args.init_path and os.path.isfile(args.init_path):
        initialization = torch.load(args.init_path, map_location="cpu", weights_only=False)
        images, labels = initialization["images"], initialization["labels"]
    else:
        images, labels = real_initialization(train_set, args.ipc, num_classes)
    images = nn.Parameter(images.to(args.device))
    if args.initialization == "real":
        images.data.add_(.01 * torch.randn_like(images))
    labels = labels.to(args.device)
    optimizer = torch.optim.Adam([images], lr=args.image_lr)
    scheduler = None
    if args.recover_augmentation == "cda":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.Iteration), eta_min=0.0
        )
    bn_loss = BNFeatureLoss(teacher)
    lower = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=args.device).view(1, 3, 1, 1)
    upper = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=args.device).view(1, 3, 1, 1)
    for iteration in range(args.Iteration + 1):
        bn_loss.clear(); optimizer.zero_grad(set_to_none=True)
        min_crop = 1.0
        teacher_inputs = images
        if args.recover_augmentation == "cda":
            min_crop = cda_min_crop_scale(
                iteration,
                args.Iteration,
                args.cda_min_crop_scale,
                args.cda_max_crop_scale,
                args.cda_milestone,
                args.cda_mode,
            )
            teacher_inputs = cda_augment(
                images,
                min_crop,
                args.cda_max_crop_scale,
                im_size,
                args.cda_flip_probability,
                args.cda_jitter,
            )
        logits = teacher(teacher_inputs)
        ce = F.cross_entropy(logits, labels)
        bn = bn_loss.value()
        loss = ce + args.bn_weight * bn
        image_lr = optimizer.param_groups[0]["lr"]
        loss.backward(); optimizer.step()
        if scheduler is not None and iteration < args.Iteration:
            scheduler.step()
        with torch.no_grad(): images.clamp_(lower, upper)
        if iteration % args.log_it == 0:
            save_and_print(
                args.log_path,
                f"iter={iteration:05d} recover_aug={args.recover_augmentation} "
                f"crop_min={min_crop:.5f} image_lr={image_lr:.8f} "
                f"total={loss.item():.6f} ce={ce.item():.6f} bn={bn.item():.6f} "
                f"teacher_acc={(logits.argmax(1)==labels).float().mean().item():.4f}",
            )
    bn_loss.close()
    payload = {"images": images.detach().cpu(), "labels": labels.cpu(), "mean": mean, "std": std,
               "teacher": "ResNet18ImageNetBN", "loss": "CE + alpha * BN", "alpha": args.bn_weight,
               "initialization": args.initialization,
               "recover_augmentation": args.recover_augmentation,
               "cda": {"mode": args.cda_mode, "milestone": args.cda_milestone,
                       "min_crop_scale": args.cda_min_crop_scale,
                       "max_crop_scale": args.cda_max_crop_scale,
                       "flip_probability": args.cda_flip_probability,
                       "jitter": args.cda_jitter}}
    torch.save(payload, os.path.join(args.save_path, "synthetic.pt"))
    teacher.eval(); correct = total = 0
    with torch.no_grad():
        for real, real_labels in testloader:
            pred = teacher(real.to(args.device)).argmax(1); real_labels = real_labels.to(args.device)
            correct += (pred == real_labels).sum().item(); total += real_labels.numel()
    save_and_print(args.log_path, f"synthesis complete; fixed_teacher_test_acc={correct/total:.4f}; output=synthetic.pt")


def evaluate(args):
    # Keep test labels in their original ImageNet indices: evaluate_synset maps them internally.
    channel, im_size, num_classes, _, _, _, _, _, testloader, _, _, _ = dataset_context(args, remap_labels=False)
    payload = torch.load(args.synthetic_path, map_location="cpu", weights_only=False)
    images, labels = payload["images"], payload["labels"]
    args.device = args.device
    args.lr_net = args.eval_lr
    args.epoch_eval_train = args.eval_epochs
    args.batch_train = args.eval_batch
    args.dsa = args.eval_dsa
    args.dsa_strategy = args.eval_dsa_strategy
    args.dsa_param = ParamDiffAug() if args.dsa else None
    accuracies = []
    for trial in range(args.num_eval):
        set_seed(args.seed + trial)
        net = build_resnet18_bn(num_classes, im_size).to(args.device)
        _, _, acc = evaluate_synset(trial, net, images, labels, testloader, args)
        accuracies.append(acc)
        save_and_print(args.log_path, f"downstream trial={trial} resnet18bn_test_acc={acc:.4f}")
    result = torch.tensor(accuracies)
    save_and_print(args.log_path, f"downstream summary model=ResNet18ImageNetBN n={len(accuracies)} mean={result.mean().item():.4f} std={result.std(unbiased=False).item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train_teacher", "prepare_init", "synthesize", "evaluate"), required=True)
    parser.add_argument("--dataset", default="ImageNet"); parser.add_argument("--subset", default="imagefruit")
    parser.add_argument("--data_path", required=True); parser.add_argument("--save_path", required=True)
    parser.add_argument("--teacher_path", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init_path", default="")
    parser.add_argument("--initialization", choices=("real", "gaussian"), default="real")
    parser.add_argument("--synthetic_path", default="")
    parser.add_argument("--ipc", type=int, default=102); parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--teacher_epochs", type=int, default=100); parser.add_argument("--teacher_lr", type=float, default=.05)
    parser.add_argument("--teacher_augmentation", choices=("none", "cifar"), default="none")
    parser.add_argument("--Iteration", type=int, default=5000); parser.add_argument("--image_lr", type=float, default=.1)
    parser.add_argument("--bn_weight", type=float, default=.01); parser.add_argument("--log_it", type=int, default=50); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover_augmentation", choices=("clean", "cda"), default="clean")
    parser.add_argument("--cda_mode", choices=("step", "linear", "cosine"), default="cosine")
    parser.add_argument("--cda_milestone", type=float, default=1.0)
    parser.add_argument("--cda_min_crop_scale", type=float, default=.08)
    parser.add_argument("--cda_max_crop_scale", type=float, default=1.0)
    parser.add_argument("--cda_flip_probability", type=float, default=.5)
    parser.add_argument("--cda_jitter", type=int, default=16)
    parser.add_argument("--eval_epochs", type=int, default=1000); parser.add_argument("--eval_lr", type=float, default=.01)
    parser.add_argument("--eval_batch", type=int, default=256); parser.add_argument("--num_eval", type=int, default=1)
    parser.add_argument("--eval_dsa", action="store_true"); parser.add_argument("--eval_dsa_strategy", default="color_crop_cutout_flip_scale_rotate")
    args = parser.parse_args()
    if not 0 < args.cda_min_crop_scale <= args.cda_max_crop_scale <= 1:
        parser.error("CDA crop scales must satisfy 0 < min <= max <= 1")
    if not 0 < args.cda_milestone <= 1:
        parser.error("--cda_milestone must be in (0, 1]")
    if not 0 <= args.cda_flip_probability <= 1:
        parser.error("--cda_flip_probability must be in [0, 1]")
    os.makedirs(args.save_path, exist_ok=True); set_seed(args.seed)
    {"train_teacher": train_teacher, "prepare_init": prepare_initialization, "synthesize": synthesize, "evaluate": evaluate}[args.mode](args)
