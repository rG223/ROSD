"""Class-sharded ROSD joint optimization for Tiny/ImageNet-1K."""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ts.tensor_pool import TensorPool
from core.utils import save_and_print, set_seed
from TM.cim_ddruo_tensorpool import (
    augmented_sre2l_metrics_and_backward,
    soft_label_rate_step,
)
from TM.cim_label_codec import AugmentedSlotStepLabelEntropy
from TM.sre2l_bn import BNFeatureLoss, build_resnet18_bn, dataset_context
from TM.lpld_bn import (
    ClassConditionalBNFeatureLoss,
    build_lpld_teacher,
    lpld_classwise_metrics_and_backward,
)
from TM.sre2l_official_tensorpool import install_per_gpu_tensorpool_locks


def load_sre2l_teacher(path, classes, image_size):
    """Load either a local SRe2L checkpoint or torchvision ResNet-18 weights."""
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
    teacher = build_resnet18_bn(classes, image_size).cuda().eval()
    teacher.load_state_dict(state, strict=True)
    return teacher


def reduce_mean(value):
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor)
    return (tensor / dist.get_world_size()).item()


def reduce_sum(value):
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor)
    return tensor.item()


def reduce_weighted_mean(value, weight):
    numerator = torch.as_tensor(
        float(value) * float(weight), dtype=torch.float64, device="cuda"
    )
    denominator = torch.as_tensor(
        float(weight), dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(numerator)
    dist.all_reduce(denominator)
    return (numerator / denominator).item()


def update_combined_dual(args, dual_value, violation_ema, violation, iteration):
    """Update a bounded signed multiplier for an equality rate budget."""
    if violation_ema is None:
        violation_ema = float(violation)
    else:
        violation_ema = (
            args.dual_ema_decay * violation_ema
            + (1.0 - args.dual_ema_decay) * float(violation)
        )
    controlled_violation = (
        0.0 if abs(violation_ema) <= args.dual_deadband else violation_ema
    )
    if iteration % args.dual_update_every == 0:
        dual_value = min(
            args.dual_max,
            max(args.dual_min, dual_value + args.dual_lr * controlled_violation),
        )
    dual_effective = min(
        args.dual_max,
        max(
            args.dual_min,
            dual_value + args.dual_rho * controlled_violation,
        ),
    )
    return dual_value, dual_effective, violation_ema


def save_global_key_shard(pool, class_start, path):
    for key in pool.key_list:
        pool.slice_pool[key]["noise"] = None
    global_state = {
        f"{class_start + local_class}_0": pool.slice_pool[f"{local_class}_0"]
        for local_class in range(pool.nclass)
    }
    torch.save(global_state, path)


def load_repartitioned_pool_checkpoint(
    pool, resume_root, resume_iteration, resume_world_size,
    class_start, class_end, rank_dir, total_classes,
):
    """Load global-key shards and repartition them for the current world size."""
    selected = {}
    for old_rank in range(resume_world_size):
        old_start = total_classes * old_rank // resume_world_size
        old_end = total_classes * (old_rank + 1) // resume_world_size
        overlap_start = max(class_start, old_start)
        overlap_end = min(class_end, old_end)
        if overlap_start >= overlap_end:
            continue
        path = os.path.join(
            resume_root,
            f"rank{old_rank}_{old_start}_{old_end}",
            f"pool_{resume_iteration}_global_keys.pt",
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        for global_class in range(overlap_start, overlap_end):
            selected[f"{global_class - class_start}_0"] = state[
                f"{global_class}_0"
            ]
        del state
    expected = set(pool.key_list)
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise RuntimeError(
            f"Resume checkpoint does not cover current shard; missing={missing[:5]}"
        )
    local_path = os.path.join(rank_dir, f"resume_{resume_iteration}.local.pt")
    torch.save(selected, local_path)
    pool.load_slice_pool(local_path)
    os.remove(local_path)


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    available_cpus = sorted(os.sched_getaffinity(0))
    cpu_affinity = set(available_cpus[rank::world])
    try:
        os.sched_setaffinity(0, cpu_affinity)
    except OSError:
        pass
    set_seed(args.seed + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_save_path = args.save_path
    if args.dataset == "ImageNet1K":
        image_size = (args.image_size, args.image_size)
        classes = args.num_classes
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        args.subset = "imagefruit"
        args.batch_real = 256
        args.zca = False
        args.save_path = root_save_path
        _, image_size, classes, _, mean, std, _, _, _, _, _, _ = dataset_context(
            args, remap_labels=False
        )
    class_start = classes * rank // world
    class_end = classes * (rank + 1) // world
    classes_per_rank = class_end - class_start
    rank_dir = os.path.join(root_save_path, f"rank{rank}_{class_start}_{class_end}")
    os.makedirs(rank_dir, exist_ok=True)
    log_path = os.path.join(rank_dir, "log.txt")

    args.save_path = rank_dir
    if args.utility_backend == "lpld_class_bn":
        checkpoint = torch.load(
            args.class_teacher_path, map_location="cpu", weights_only=False
        )
        teacher = build_lpld_teacher(classes, image_size[0]).cuda().eval()
        teacher.load_state_dict(checkpoint["model"])
    else:
        teacher = load_sre2l_teacher(args.teacher_path, classes, image_size)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if args.utility_backend == "lpld_class_bn":
        bn_feature = ClassConditionalBNFeatureLoss(
            teacher, first_bn_multiplier=args.first_bn_multiplier
        )
    else:
        bn_feature = BNFeatureLoss(
            teacher,
            mode=args.bn_loss_mode,
            first_bn_multiplier=args.first_bn_multiplier,
        )

    install_per_gpu_tensorpool_locks(torch.cuda.device_count())
    pool = TensorPool(
        classes_per_rank, 1000, [args.ipc] * classes_per_rank, [rank],
        nthread=args.codec_workers,
        ldb=args.ldb, img_size=image_size, max_iter=args.iterations,
        channel=3, lr=args.codec_lr, layers_v="v5", arm=32, dim=4,
        encoder_gain=args.encoder_gain,
    )
    pool.free_model()
    for index, key in enumerate(pool.key_list):
        model = pool.get_model()
        parameters = model.produce_parameters(pool.slice_nums[index])
        pool.slice_pool[key]["param"].set_params(parameters.get_params(), "cpu")
    pool.free_model()
    pool.set_training_phase(0)
    if args.resume_iteration > 0:
        load_repartitioned_pool_checkpoint(
            pool, args.resume_root, args.resume_iteration,
            args.resume_world_size, class_start, class_end, rank_dir, classes,
        )
    else:
        pool.init_solvers()

    entropy = AugmentedSlotStepLabelEntropy(
        classes * args.ipc, classes=classes, dims=classes - 1,
    ).cuda()
    resume_label_checkpoint = None
    if args.resume_iteration > 0:
        resume_label_checkpoint = torch.load(
            os.path.join(
                args.resume_root, f"label_codec_{args.resume_iteration}.pt"
            ),
            map_location="cpu", weights_only=False,
        )
        entropy.load_state_dict(resume_label_checkpoint["label_entropy"])
    entropy = DistributedDataParallel(entropy, device_ids=[rank])
    entropy_optimizer = torch.optim.AdamW(
        entropy.parameters(), lr=args.label_entropy_lr, weight_decay=1e-5,
    )
    label_model_kib = (
        sum(parameter.numel() for parameter in entropy.module.parameters())
        * 16.0 / classes / 8192.0
    )
    start_time = time.time()
    save_and_print(
        log_path,
        f"rank={rank} classes=[{class_start},{class_end}) seed={args.seed} "
        f"cpu_affinity={min(cpu_affinity)}-{max(cpu_affinity)} "
        + (
            f"initialization=resume iteration={args.resume_iteration} "
            f"resume_world_size={args.resume_world_size} "
            if args.resume_iteration > 0 else
            "initialization=random_codec warmup=none "
        )
        + f"objective={args.utility_backend}_RRC_flip_jitter{args.jitter}_CE_plus_{args.bn_weight:g}_BN "
        f"bn_loss_mode={'class_l2_sum' if args.utility_backend == 'lpld_class_bn' else args.bn_loss_mode} "
        f"first_bn_multiplier={args.first_bn_multiplier:g} "
        + (
            f"+combined_dual(image_rate+label_rate<->{args.combined_target_kib:g}KiB) "
            f"dual_init={args.dual_init:g} dual_lr={args.dual_lr:g} "
            f"dual_rho={args.dual_rho:g} dual_bounds=[{args.dual_min:g},{args.dual_max:g}] "
            f"rate_gradient_weights=[image:{args.image_rate_gradient_weight:g},"
            f"label:{args.label_rate_gradient_weight:g}]"
            if args.combined_dual else
            f"+{args.lambda_image:g}_image_rate+{args.lambda_label:g}_label_rate"
        ),
    )

    last_label_rate = None
    dual_value = float(args.dual_init)
    dual_effective = float(args.dual_init)
    if resume_label_checkpoint is not None:
        resume_config = resume_label_checkpoint.get("config", {})
        dual_value = float(resume_config.get("final_dual_value", dual_value))
        dual_effective = float(
            resume_config.get("final_dual_effective", dual_effective)
        )
    violation_ema = None
    completed_after_resume = 0
    class_batch_size = (
        classes_per_rank if args.class_batch_size <= 0
        else min(args.class_batch_size, classes_per_rank)
    )
    local_class_ranges = [
        (start, min(start + class_batch_size, classes_per_rank))
        for start in range(0, classes_per_rank, class_batch_size)
    ]
    for iteration in range(args.resume_iteration + 1, args.iterations + 1):
        completed_after_resume += 1
        label_rate_multiplier_base = (
            dual_effective * args.label_rate_gradient_weight
            if args.combined_dual else args.lambda_label
        )
        # soft_label_rate_step normalizes by the global class count, while each
        # rank backpropagates only its local class shard. Restore the local-mean
        # gradient scale so changing world size does not weaken label rate.
        label_rate_world_scale = classes / classes_per_rank
        rate_multiplier = label_rate_multiplier_base * label_rate_world_scale
        image_rate_multiplier = (
            dual_effective * args.image_rate_gradient_weight
            if args.combined_dual else args.lambda_image
        )
        ldb_it = image_rate_multiplier * args.lr_it / args.ldb
        local_image_bpp_sum = 0.0
        local_label_kib = 0.0
        local_utility_sum = local_ce_sum = local_bn_sum = 0.0
        local_accuracy_sum = local_scale_sum = 0.0
        for chunk_index, (local_start, local_end) in enumerate(local_class_ranges):
            chunk_classes = local_end - local_start
            pool.train()
            pool.set_active_class_range(local_start, local_end)
            images, local_labels, bpp = pool.get_data()
            global_labels = local_labels.to(images.device) + class_start
            pool.data.grad = None
            if args.utility_backend == "lpld_class_bn":
                utility, ce, bn, accuracy = lpld_classwise_metrics_and_backward(
                    images, global_labels, teacher, bn_feature, mean, std,
                    args.bn_weight, jitter=args.jitter,
                    max_images_per_forward=args.utility_batch_size,
                )
                pool.data.grad.mul_(chunk_classes / classes_per_rank)
            else:
                utility, ce, bn, accuracy = augmented_sre2l_metrics_and_backward(
                    images, global_labels, teacher, bn_feature, mean, std,
                    args.bn_weight, crop_scale_min=0.08, flip_probability=0.5,
                    batch_size=args.utility_batch_size,
                    backward_scale=chunk_classes / classes,
                    jitter=args.jitter,
                )
            utility_gradient = pool.data.grad.detach().clone()
            with bn_feature.suspended():
                last_label_rate = soft_label_rate_step(
                    images, global_labels, teacher, entropy, entropy_optimizer,
                    mean, std, args.label_step, args.label_groups, classes, args.ipc,
                    args.label_feature_chunk, args.label_entropy_batch,
                    rate_multiplier, hard_ce_weight=0.0, label_kl_weight=0.0,
                    train_entropy=True,
                    slot_offset=(class_start + local_start) * args.ipc,
                )
            label_gradient = pool.data.grad.detach() - utility_gradient
            utility_norm = utility_gradient.norm()
            label_norm = label_gradient.norm()
            scale = min(
                1.0,
                float(args.label_gradient_cap * utility_norm
                      / label_norm.clamp_min(1e-12)),
            )
            pool.data.grad = utility_gradient + scale * label_gradient
            if not torch.isfinite(pool.data.grad).all():
                raise FloatingPointError("Non-finite joint image gradient")
            pool.fill_data_diff()
            pool.backward(
                args.lr_it, ldb_it,
                advance_epoch=chunk_index == len(local_class_ranges) - 1,
            )
            local_image_bpp_sum += float(bpp) * chunk_classes
            local_label_kib += last_label_rate["label_kib_per_class"]
            local_utility_sum += utility.item() * chunk_classes
            local_ce_sum += ce.item() * chunk_classes
            local_bn_sum += bn.item() * chunk_classes
            local_accuracy_sum += accuracy.item() * chunk_classes
            local_scale_sum += scale * chunk_classes
        pool.validate(iteration - 1)

        global_image_bpp = reduce_sum(local_image_bpp_sum) / classes
        global_image_kib = (
            global_image_bpp * args.ipc * image_size[0] * image_size[1] / 8192.0
        )
        global_label_kib = reduce_sum(local_label_kib)
        combined_rate_kib = global_image_kib + global_label_kib
        violation = combined_rate_kib / args.combined_target_kib - 1.0
        if args.combined_dual:
            dual_value, dual_effective, violation_ema = update_combined_dual(
                args, dual_value, violation_ema, violation, iteration,
            )

        if iteration == 1 or iteration % args.log_every == 0:
            global_utility = reduce_sum(local_utility_sum) / classes
            global_ce = reduce_sum(local_ce_sum) / classes
            global_bn = reduce_sum(local_bn_sum) / classes
            global_accuracy = reduce_sum(local_accuracy_sum) / classes
            global_scale = reduce_sum(local_scale_sum) / classes
            elapsed = time.time() - start_time
            eta_hours = (
                elapsed / completed_after_resume
                * (args.iterations - iteration) / 3600.0
            )
            save_and_print(
                log_path,
                f"iter={iteration:04d}/{args.iterations:04d} "
                f"global_utility={global_utility:.7f} ce={global_ce:.7f} "
                f"bn={global_bn:.7f} augmented_teacher_acc={global_accuracy:.4f} "
                f"image_latent_kib_per_class={global_image_kib:.2f} "
                f"label_kib_per_class={global_label_kib:.2f} "
                f"combined_rate_kib_per_class={combined_rate_kib:.2f} "
                f"label_model_kib_per_class={label_model_kib:.2f} "
                f"label_grad_scale={global_scale:.6f} "
                + (
                    f"dual_value={dual_value:.8g} dual_effective={dual_effective:.8g} "
                    f"image_rate_multiplier={image_rate_multiplier:.8g} "
                    f"label_rate_multiplier_base={label_rate_multiplier_base:.8g} "
                    f"label_rate_world_scale={label_rate_world_scale:.6g} "
                    f"label_rate_multiplier={rate_multiplier:.8g} "
                    f"dual_violation={violation:.6f} dual_violation_ema={violation_ema:.6f} "
                    if args.combined_dual else ""
                )
                + f"eta_hours={eta_hours:.2f}",
            )

        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            shard_path = os.path.join(rank_dir, f"pool_{iteration}_global_keys.pt")
            save_global_key_shard(pool, class_start, shard_path)
            if rank == 0:
                torch.save(
                    {
                        "experiment_type": (
                            "rosd_combined_dual"
                            if args.combined_dual else "rosd_fixed_label_rate"
                        ),
                        "label_entropy": entropy.module.state_dict(),
                        "config": {
                            "num_slots": classes * args.ipc,
                            "classes": classes,
                            "label_step": args.label_step,
                            "label_groups": args.label_groups,
                            "reference_mode": "hard_class",
                            "utility_mode": args.utility_backend,
                            "rate_control": (
                                "combined_dual" if args.combined_dual else "fixed_lambda"
                            ),
                            "combined_target_kib": args.combined_target_kib,
                            "image_rate_gradient_weight": (
                                args.image_rate_gradient_weight
                            ),
                            "label_rate_gradient_weight": (
                                args.label_rate_gradient_weight
                            ),
                            "estimated_image_kib_per_class": global_image_kib,
                            "estimated_label_kib_per_class": global_label_kib,
                            "estimated_combined_rate_kib_per_class": combined_rate_kib,
                            "label_model_kib_per_class": label_model_kib,
                            "final_dual_value": dual_value,
                            "final_dual_effective": dual_effective,
                            "resume_iteration": args.resume_iteration,
                            "resume_world_size": args.resume_world_size,
                        },
                    },
                    os.path.join(root_save_path, f"label_codec_{iteration}.pt"),
                )
        dist.barrier()

    final_shard = os.path.join(rank_dir, "pool_final_global_keys.pt")
    final_iteration_shard = os.path.join(
        rank_dir, f"pool_{args.iterations}_global_keys.pt"
    )
    if os.path.lexists(final_shard):
        os.remove(final_shard)
    if os.path.exists(final_iteration_shard):
        os.link(final_iteration_shard, final_shard)
    else:
        save_global_key_shard(pool, class_start, final_shard)
    dist.barrier()
    if rank == 0 and not args.skip_final_merge:
        merged = {}
        for other_rank in range(world):
            start = classes * other_rank // world
            end = classes * (other_rank + 1) // world
            path = os.path.join(
                root_save_path,
                f"rank{other_rank}_{start}_{end}",
                "pool_final_global_keys.pt",
            )
            merged.update(torch.load(path, map_location="cpu", weights_only=False))
        if len(merged) != classes:
            raise RuntimeError(f"Expected {classes} merged codecs, got {len(merged)}")
        torch.save(merged, os.path.join(root_save_path, "pool_joint_merged.pt"))
    if rank == 0:
        open(os.path.join(root_save_path, "joint_complete"), "w").close()
    dist.barrier()
    bn_feature.close()
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("Tiny", "ImageNet1K"), default="Tiny"
    )
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--class_teacher_path", default="")
    parser.add_argument(
        "--utility_backend",
        choices=("sre2l", "lpld_class_bn"),
        default="sre2l",
    )
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ipc", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bn_weight", type=float, default=0.05)
    parser.add_argument(
        "--bn_loss_mode", choices=("mse_mean", "sre2l_l2_sum"),
        default="mse_mean",
    )
    parser.add_argument("--first_bn_multiplier", type=float, default=1.0)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument("--lambda_image", type=float, default=1e-4)
    parser.add_argument("--lambda_label", type=float, default=1e-4)
    parser.add_argument("--combined_dual", action="store_true")
    parser.add_argument("--combined_target_kib", type=float, default=600.0)
    parser.add_argument("--dual_init", type=float, default=1e-4)
    parser.add_argument("--dual_lr", type=float, default=1e-5)
    parser.add_argument("--dual_rho", type=float, default=5e-5)
    parser.add_argument("--dual_ema_decay", type=float, default=0.95)
    parser.add_argument("--dual_update_every", type=int, default=10)
    parser.add_argument("--dual_deadband", type=float, default=0.02)
    parser.add_argument("--dual_min", type=float, default=-5e-5)
    parser.add_argument("--dual_max", type=float, default=5e-4)
    parser.add_argument("--image_rate_gradient_weight", type=float, default=1.0)
    parser.add_argument("--label_rate_gradient_weight", type=float, default=1.0)
    parser.add_argument("--label_gradient_cap", type=float, default=0.1)
    parser.add_argument("--label_step", type=float, default=0.35)
    parser.add_argument("--label_groups", type=int, default=30)
    parser.add_argument("--label_entropy_lr", type=float, default=1e-3)
    parser.add_argument("--utility_batch_size", type=int, default=1000)
    parser.add_argument(
        "--class_batch_size", type=int, default=0,
        help="Local classes decoded together; 0 keeps the full rank shard.",
    )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--label_feature_chunk", type=int, default=1000)
    parser.add_argument("--label_entropy_batch", type=int, default=1000)
    parser.add_argument("--codec_workers", type=int, default=12)
    parser.add_argument("--codec_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_gain", type=int, default=16)
    parser.add_argument("--resume_root", default="")
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--resume_world_size", type=int, default=0)
    parser.add_argument("--ldb", type=float, default=0.1)
    parser.add_argument("--lr_it", type=float, default=1000.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument(
        "--skip_final_merge", action="store_true",
        help="Keep global-key rank shards without materializing one large file.",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.resume_iteration > 0:
        if not parsed.resume_root:
            raise ValueError("--resume_root is required with --resume_iteration")
        if parsed.resume_world_size <= 0:
            raise ValueError(
                "--resume_world_size must be positive with --resume_iteration"
            )
        if parsed.resume_iteration >= parsed.iterations:
            raise ValueError("--resume_iteration must be smaller than --iterations")
    main(parsed)
