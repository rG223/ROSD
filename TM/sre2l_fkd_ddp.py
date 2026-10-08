"""Distributed FKD pool training with rank-local replay and mmap images."""

import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils import save_and_print, set_seed
from TM import sre2l_fkd
from TM.sre2l_bn import build_resnet18_bn, dataset_context


class RankReplayDataset(sre2l_fkd.ReplayDataset):
    """Replay a target and its global mixing source inside one DDP rank."""

    def __init__(
        self, images, labels, config, crop_size, clamp_images=False,
        position_start=0,
    ):
        super().__init__(images, labels, config, crop_size)
        self.clamp_images = clamp_images
        self.position_start = position_start

    def __len__(self):
        return self.logits.shape[0]

    def replay_image(self, index):
        image = self.images[index].float()
        return image.clamp(0.0, 1.0) if self.clamp_images else image

    def __getitem__(self, position):
        local_position = position
        position += self.position_start
        source_position = int(self.mix_index[position])
        target = sre2l_fkd.replay_crop(
            self.replay_image(int(self.order[position])),
            self.coords[position],
            self.flips[position],
            self.crop_size,
            self.augmentation_mode,
        )
        source = sre2l_fkd.replay_crop(
            self.replay_image(int(self.order[source_position])),
            self.coords[source_position],
            self.flips[source_position],
            self.crop_size,
            self.augmentation_mode,
        )
        if self.mix_mode == "mixup":
            target = self.mix_lambda * target + (1.0 - self.mix_lambda) * source
        else:
            x1, y1, x2, y2 = self.bbox
            target[:, x1:x2, y1:y2] = source[:, x1:x2, y1:y2]
        original_index = int(self.order[position])
        return target, self.labels[original_index], self.logits[local_position]


def load_synthetic_mmap(path):
    payload = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    if "images_raw" in payload:
        images = payload["images_raw"]
        clamp_images = True
    elif "images" in payload:
        images = payload["images"]
        clamp_images = False
    else:
        raise KeyError("Synthetic payload contains neither images_raw nor images")
    label_key = "labels" if "labels" in payload else "hard_labels"
    return images, payload[label_key].long(), clamp_images


def reduce_metrics(loss_sum, correct, total, device):
    metrics = torch.tensor(
        [loss_sum, correct, total], dtype=torch.float64, device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics.tolist()


@torch.inference_mode()
def distributed_evaluate(model, testloader, device):
    model.eval()
    correct = total = 0
    for images, labels in testloader:
        labels = labels.to(device, non_blocking=True)
        predictions = model(
            images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
        ).argmax(1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    metrics = torch.tensor([correct, total], dtype=torch.int64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return float(metrics[0]) / float(metrics[1])


def gather_last_batch(student_logits, teacher_probabilities):
    gathered_student = [torch.empty_like(student_logits) for _ in range(dist.get_world_size())]
    gathered_teacher = [
        torch.empty_like(teacher_probabilities) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_student, student_logits)
    dist.all_gather(gathered_teacher, teacher_probabilities)
    return torch.cat(gathered_student), torch.cat(gathered_teacher)


def build_resume_scheduler(args, optimizer, checkpoint):
    if args.optimizer == "sgd":
        return None
    if checkpoint is not None and checkpoint.get("scheduler") is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scheduler.load_state_dict(checkpoint["scheduler"])
        return scheduler

    # Older pool checkpoints did not persist scheduler state. Infer the phase
    # from the current optimizer LR so a DP-to-DDP switch does not restart the
    # cosine decay. The original pool trainer uses eta_min=0 and T_max=epochs.
    current_lr = float(optimizer.param_groups[0]["lr"])
    base_lr = float(args.learning_rate)
    ratio = min(1.0, max(0.0, current_lr / base_lr))
    completed_steps = int(
        round(args.epochs * math.acos(2.0 * ratio - 1.0) / math.pi)
    )
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", base_lr)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=completed_steps - 1
    )


def train_pool_ddp(args):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed)

    images, labels, clamp_images = load_synthetic_mmap(args.synthetic_path)
    num_classes = int(labels.max()) + 1
    _, _, dataset_classes, _, _, _, _, _, base_testloader, _, _, _ = dataset_context(args)
    if dataset_classes != num_classes:
        raise ValueError(
            f"Class mismatch: synthetic={num_classes}, dataset={dataset_classes}"
        )
    test_sampler = DistributedSampler(
        base_testloader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    testloader = DataLoader(
        base_testloader.dataset,
        batch_size=base_testloader.batch_size,
        sampler=test_sampler,
        num_workers=max(1, 8 // world_size),
        pin_memory=True,
        persistent_workers=True,
    )

    summary = torch.load(
        os.path.join(args.fkd_path, "pool_summary.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if args.train_batch % world_size:
        raise ValueError(
            f"Global train_batch={args.train_batch} must divide world_size={world_size}"
        )
    local_batch = args.train_batch // world_size
    local_start = rank * local_batch
    local_stop = local_start + local_batch
    pool = []
    for index in range(summary["pool_batches"]):
        config = torch.load(
            os.path.join(args.fkd_path, f"pool_batch_{index:05d}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        # Only logits are large. Keep global crop/mixing metadata, but retain
        # this rank's contiguous logits slice so four ranks together occupy
        # roughly the same host memory as the single-process trainer.
        config["logits"] = config["logits"][local_start:local_stop].clone()
        pool.append(config)
    generator = torch.Generator().manual_seed(args.seed + 9103)

    if args.fast_downstream:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    student_core = build_resnet18_bn(
        num_classes, (args.crop_size, args.crop_size)
    ).to(device, memory_format=torch.channels_last)
    optimizer = sre2l_fkd.build_student_optimizer(args, student_core.parameters())
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_accuracy = 0.0
    start_epoch = 0
    ca_student_temperature_ratio = 1.0
    checkpoint = None
    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        student_core.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        ca_student_temperature_ratio = float(checkpoint.get("next_ca_ratio", 1.0))

    student = DistributedDataParallel(
        student_core,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
    )
    scheduler = build_resume_scheduler(args, optimizer, checkpoint)

    for _ in range(start_epoch):
        if args.pool_sampling_with_replacement:
            torch.randint(
                summary["pool_batches"],
                (summary["batches_per_epoch"],),
                generator=generator,
            )
        else:
            torch.randperm(summary["pool_batches"], generator=generator)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if args.resume_checkpoint:
            save_and_print(
                args.log_path,
                f"DDP resume checkpoint={args.resume_checkpoint} "
                f"start_epoch={start_epoch} best_test_acc={best_accuracy:.4f}",
            )
        save_and_print(
            args.log_path,
            f"DDP pool train start world_size={world_size} "
            f"global_batch={args.train_batch} local_batch={local_batch} "
            f"epochs={args.epochs} updates_per_epoch={summary['batches_per_epoch']} "
            f"pool_batches={summary['pool_batches']} workers_per_rank={args.workers}",
        )

    all_replays = [
        RankReplayDataset(
            images,
            labels,
            config,
            args.crop_size,
            clamp_images,
            local_start,
        )
        for config in pool
    ]
    replay_offsets = []
    offset = 0
    for replay in all_replays:
        replay_offsets.append(offset)
        offset += len(replay)
    batch_sampler = sre2l_fkd.MutableBatchSampler()
    loader = DataLoader(
        ConcatDataset(all_replays),
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=sre2l_fkd.seed_worker,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 2} if args.workers > 0 else {}),
    )

    for epoch in range(start_epoch, args.epochs):
        if args.optimizer == "sgd":
            epoch_lr = sre2l_fkd.official_sgd_epoch_lr(args, epoch)
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
        else:
            epoch_lr = optimizer.param_groups[0]["lr"]
        teacher_temperature = sre2l_fkd.dkr_temperature(args, epoch)
        student_temperature = (
            ca_student_temperature_ratio * teacher_temperature
            if args.ca_dynamic else teacher_temperature
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

        batches = []
        for pool_index in selected:
            replay = all_replays[pool_index]
            if len(replay) != local_batch:
                raise ValueError(
                    f"Rank-local pool batch={len(replay)} differs from "
                    f"expected local_batch={local_batch}"
                )
            offset = replay_offsets[pool_index]
            batches.append(list(range(offset, offset + local_batch)))
        batch_sampler.set_batches(batches)

        student.train()
        loss_sum = correct = total = 0.0
        last_student_logits = last_teacher_probabilities = None
        epoch_start = time.time()
        for update, (batch_images, batch_hard, teacher_logits) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            batch_images = batch_images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            batch_hard = batch_hard.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True).float()
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
            if rank == 0 and epoch == start_epoch and (
                update == 0 or (update + 1) % 100 == 0
            ):
                save_and_print(
                    args.log_path,
                    f"DDP first_epoch update={update + 1:03d}/{len(batches)} "
                    f"elapsed={time.time() - epoch_start:.1f}s",
                )

        loss_sum, correct, total = reduce_metrics(
            loss_sum, correct, total, device
        )
        if args.ca_dynamic:
            all_student, all_teacher = gather_last_batch(
                last_student_logits, last_teacher_probabilities
            )
            ratio = torch.zeros((), dtype=torch.float64, device=device)
            if rank == 0:
                ratio.fill_(
                    sre2l_fkd.calibrate_student_temperature_ratio(
                        all_student,
                        all_teacher,
                        teacher_temperature,
                        args.ca_grid_points,
                    )
                )
            dist.broadcast(ratio, src=0)
            ca_student_temperature_ratio = float(ratio)
        if scheduler is not None:
            scheduler.step()

        should_eval = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch + 1 == args.epochs
        )
        if should_eval:
            accuracy_tensor = torch.zeros((), dtype=torch.float64, device=device)
            accuracy_tensor.fill_(
                distributed_evaluate(student_core, testloader, device)
            )
            accuracy = float(accuracy_tensor)
            is_best = accuracy >= best_accuracy
            best_accuracy = max(best_accuracy, accuracy)
            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": student_core.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
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
                    "distributed_world_size": world_size,
                }
                torch.save(state, os.path.join(args.output_path, "checkpoint.pt"))
                if is_best:
                    torch.save(state, os.path.join(args.output_path, "model_best.pt"))
                save_and_print(
                    args.log_path,
                    f"DDP train epoch={epoch + 1:03d}/{args.epochs} "
                    f"lr={epoch_lr:.7f} teacher_T={teacher_temperature:.4f} "
                    f"student_T={student_temperature:.4f} "
                    f"next_ca_ratio={ca_student_temperature_ratio:.4f} "
                    f"kd={loss_sum / total:.6f} "
                    f"hard_train_acc={correct / total:.4f} "
                    f"test_acc={accuracy:.4f} best={best_accuracy:.4f}",
                )
        dist.barrier()

    if rank == 0:
        save_and_print(
            args.log_path,
            f"DDP pool train complete best_test_acc={best_accuracy:.4f}",
        )
    dist.destroy_process_group()


def main():
    args = sre2l_fkd.parse_args()
    if args.mode != "train_pool":
        raise ValueError("DDP entry point only supports --mode train_pool")
    train_pool_ddp(args)


if __name__ == "__main__":
    main()
