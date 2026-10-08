"""Official DD-RUO TensorPool with SRe2L CE/BN synthesis utility.

This entry point deliberately uses the unconditioned TensorPool path.  The
codec initialization, quantizer schedule, entropy model, same-noise replay,
manual utility/rate backward, and network post-quantization are inherited from
the released DD-RUO implementation.  Only the external utility gradient is
changed from trajectory matching to SRe2L's CE + alpha * BN objective.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.ts.tensor_pool as tensor_pool_module
from core.ts.tensor_pool import TensorPool
from core.ts.tensor_data_func_v6 import TrainingHelper, pretrain
from core.ts.training import LossFunctionOutput, loss_function
from core.utils import get_time, save_and_print, set_seed
from TM.sre2l_bn import BNFeatureLoss, build_resnet18_bn, dataset_context


def install_per_gpu_tensorpool_locks(device_count):
    """Serialize codec work per GPU while retaining four-GPU parallelism.

    Upstream submits one thread per class. With fewer GPUs than classes, modern
    cuDNN can reject simultaneous calls from two host threads on one device.
    The wrappers alter scheduling only; every upstream codec function and
    optimization step remains unchanged.
    """
    locks = {f"cuda:{index}": threading.Lock() for index in range(device_count)}
    function_names = (
        "run_warmup",
        "run_model",
        "run_model_test",
        "run_model_backward",
        "run_quantize_net",
    )
    for function_name in function_names:
        original = getattr(tensor_pool_module, function_name)

        def locked(model, *args, _original=original, **kwargs):
            with locks[str(model.device)]:
                # CUDA's current device is thread-local. TensorPool creates up
                # to one worker per class, so every worker must enter the
                # model's device context before helper code allocates temporary
                # CUDA tensors. Without this, multi-GPU pools intermittently
                # launch kernels with cuda:0 state against another GPU's data.
                with torch.cuda.device(model.device):
                    return _original(model, *args, **kwargs)

        setattr(tensor_pool_module, function_name, locked)
    return locks


def bounded_pixels_and_penalty(images):
    """Return legal pixels and a differentiable penalty for decoder overflow."""
    penalty = (F.relu(-images).square() + F.relu(images - 1.0).square()).mean()
    return images.clamp(0.0, 1.0), penalty


class DualReconstructionLoss:
    """MSE reconstruction under a per-class latent-rate upper bound."""

    def __init__(self, target_bpp, dual_lr, enforce_pixel_range,
                 pixel_range_weight):
        self.target_bpp = float(target_bpp)
        self.dual_lr = float(dual_lr)
        self.enforce_pixel_range = bool(enforce_pixel_range)
        self.pixel_range_weight = float(pixel_range_weight)
        self.dual_lambda = 0.0

    def __call__(self, decoded_image, rate_latent_bit, target_image, lmbda=None):
        del lmbda
        mse = F.mse_loss(decoded_image, target_image)
        range_penalty = decoded_image.new_zeros(())
        if self.enforce_pixel_range:
            _, range_penalty = bounded_pixels_and_penalty(decoded_image)
        pixels = decoded_image.shape[0] * decoded_image.shape[-2] * decoded_image.shape[-1]
        rate_bpp = rate_latent_bit.sum() / pixels
        loss = (
            mse + self.pixel_range_weight * range_penalty
            + self.dual_lambda * rate_bpp
        )
        if torch.is_grad_enabled():
            violation = float(rate_bpp.detach()) / self.target_bpp - 1.0
            self.dual_lambda = max(
                0.0, self.dual_lambda + self.dual_lr * violation
            )
        return LossFunctionOutput(
            loss=loss,
            mse=max(float(mse.detach()), 1e-12),
            rate_latent_bpp=float(rate_bpp.detach()),
        )


def run_dual_warmup(model, ref, target_bpp, dual_lr, enforce_pixel_range,
                    pixel_range_weight, fast_single_candidate=False,
                    fast_warmup_iterations=400):
    """Run the released 5-to-2 candidate warmup with a dual rate bound."""
    helper = TrainingHelper(0.0, 5000)
    candidates = []
    phases = helper.manager.preset.warmup.phases
    if fast_single_candidate:
        phases[0].candidates = 1
        phases[0].training_phase.max_itr = fast_warmup_iterations
        phases[0].training_phase.freq_valid = fast_warmup_iterations
        phase_count = 1
    else:
        phase_count = len(phases)
    count = phases[0].candidates
    for candidate_id in range(count):
        candidates.append({
            "id": candidate_id,
            "dp": model.produce_parameters(ref.shape[0]),
            "metrics": None,
            "loss": DualReconstructionLoss(
                target_bpp, dual_lr, enforce_pixel_range, pixel_range_weight
            ),
        })

    for phase_index in range(phase_count):
        keep = phases[phase_index].candidates
        candidates = candidates[:keep]
        for candidate in candidates:
            helper.set_training_phase(phase_index, warmup=True)
            helper.init_solver(candidate["dp"])
            candidate["dp"], candidate["metrics"] = pretrain(
                model, candidate["dp"], candidate["loss"], ref, helper
            )

        # Prefer feasible candidates, then choose the lowest reconstruction MSE.
        candidates.sort(key=lambda item: (
            max(0.0, item["metrics"].rate_latent_bpp - target_bpp),
            item["metrics"].mse,
        ))
        for item in candidates:
            print(
                f"dual_warmup phase={phase_index} candidate={item['id']} "
                f"mse={item['metrics'].mse:.8f} "
                f"latent_bpp={item['metrics'].rate_latent_bpp:.6f} "
                f"dual_lambda={item['loss'].dual_lambda:.8g}"
            )

    winner = candidates[0]
    print(f"dual_warmup winner={winner['id']}")
    return winner["dp"], ref


def initialize_pool_with_dual_warmup(
    pool, references, target_bpp, dual_lr, enforce_pixel_range,
    pixel_range_weight, locks, fast_single_candidate=False,
    fast_warmup_iterations=400,
):
    pool.free_model()
    pool.executor.clear_futures()
    for class_index in references:
        for slice_index in range(pool.slice_per_class[class_index]):
            ref = references[class_index][
                slice_index * pool.slice_size:(slice_index + 1) * pool.slice_size
            ]
            model = pool.get_model()
            key = f"{class_index}_{slice_index}"

            def locked_warmup(current_model, current_ref, _device=str(model.device)):
                with locks[_device]:
                    return run_dual_warmup(
                        current_model, current_ref, target_bpp, dual_lr,
                        enforce_pixel_range, pixel_range_weight,
                        fast_single_candidate, fast_warmup_iterations,
                    )

            pool.executor.submit_task(key, locked_warmup, model, ref)
    pool.executor.all_tasks_done()
    pool.free_model()
    for future, key in pool.executor.futures.items():
        parameters, _ = future.result()
        pool.slice_pool[key]["param"].set_params(parameters.get_params(), "cpu")


def run_fixed_single_warmup(model, ref, ldb, iterations):
    """Warm up one randomly initialized codec without candidate selection."""
    helper = TrainingHelper(ldb, 5000)
    phase = helper.manager.preset.warmup.phases[0]
    phase.candidates = 1
    phase.training_phase.max_itr = iterations
    phase.training_phase.freq_valid = iterations
    parameters = model.produce_parameters(ref.shape[0])
    helper.set_training_phase(0, warmup=True)
    helper.init_solver(parameters)
    parameters, metrics = pretrain(
        model, parameters, loss_function, ref, helper
    )
    print(
        f"fixed_single_warmup mse={metrics.mse:.8f} "
        f"latent_bpp={metrics.rate_latent_bpp:.6f}"
    )
    return parameters, ref


def initialize_pool_with_fixed_single_warmup(
    pool, references, locks, iterations,
):
    pool.free_model()
    pool.executor.clear_futures()
    for class_index in references:
        for slice_index in range(pool.slice_per_class[class_index]):
            ref = references[class_index][
                slice_index * pool.slice_size:(slice_index + 1) * pool.slice_size
            ]
            model = pool.get_model()
            key = f"{class_index}_{slice_index}"

            def locked_warmup(current_model, current_ref, _device=str(model.device)):
                with locks[_device]:
                    return run_fixed_single_warmup(
                        current_model, current_ref, pool.ldb, iterations
                    )

            pool.executor.submit_task(key, locked_warmup, model, ref)
    pool.executor.all_tasks_done()
    pool.free_model()
    for future, key in pool.executor.futures.items():
        parameters, _ = future.result()
        pool.slice_pool[key]["param"].set_params(parameters.get_params(), "cpu")


def validate_initialized_pool(pool):
    missing = []
    for key in pool.key_list:
        parameter_pool = pool.slice_pool[key]["param"].pool
        if any(parameter_pool[name] is None for name in ("grids", "ap", "up", "sp")):
            missing.append(key)
    if missing:
        raise RuntimeError(f"TensorPool initialization incomplete for slices: {missing}")


def normalize_images(images, mean, std):
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


def load_reference_images(path, classes, ipc, mean, std):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "images_raw" in payload:
        raw_images = payload["images_raw"]
        if raw_images.dtype == torch.uint8:
            raw_images = raw_images.float().div_(255.0)
        else:
            raw_images = raw_images.float()
        raw_images = raw_images.clamp(0.0, 1.0)
    elif "images" in payload:
        images = payload["images"].float()
        mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
        std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
        raw_images = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    else:
        raise KeyError("Reference checkpoint must contain 'images' or 'images_raw'")
    labels = payload["labels"].long()
    expected_labels = torch.arange(classes).repeat_interleave(ipc)
    if raw_images.shape[0] != classes * ipc:
        raise ValueError(
            f"Expected {classes * ipc} reference images, got {raw_images.shape[0]}"
        )
    if not torch.equal(labels, expected_labels):
        raise ValueError("Reference images must use class-major slot ordering")
    return {
        class_index: raw_images[class_index * ipc:(class_index + 1) * ipc]
        for class_index in range(classes)
    }


def teacher_metrics_and_backward(images, labels, teacher, bn_feature, mean, std,
                                 bn_weight, backward, enforce_pixel_range=False,
                                 pixel_range_weight=0.0, batch_size=0):
    """Evaluate CE+BN in chunks and accumulate an equivalent mean gradient.

    DD-RUO materializes every decoded image in one tensor. Sending the full
    Tiny-ImageNet IPC-50 payload through ResNet18 retains a 10,000-image
    autograd graph and exceeds an 80 GiB GPU. CE is exactly accumulated over
    chunks; BN matching is the sample-weighted mean of chunk-level statistics.
    """
    count = images.shape[0]
    batch_size = count if batch_size <= 0 else min(batch_size, count)
    totals = [images.new_zeros(()) for _ in range(5)]
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        weight = (end - start) / count
        chunk = images[start:end]
        chunk_range = images.new_zeros(())
        with torch.set_grad_enabled(backward):
            if enforce_pixel_range:
                chunk, chunk_range = bounded_pixels_and_penalty(chunk)
            normalized = normalize_images(chunk, mean, std)
            bn_feature.clear()
            logits = teacher(normalized)
            chunk_ce = F.cross_entropy(logits, labels[start:end])
            chunk_bn = bn_feature.value()
            chunk_utility = (
                chunk_ce + bn_weight * chunk_bn
                + pixel_range_weight * chunk_range
            )
            if backward:
                (weight * chunk_utility).backward()
        totals[0] += weight * chunk_utility.detach()
        totals[1] += weight * chunk_ce.detach()
        totals[2] += weight * chunk_bn.detach()
        totals[3] += (logits.detach().argmax(1) == labels[start:end]).sum()
        totals[4] += weight * chunk_range.detach()
    totals[3] /= count
    return tuple(totals)


def export_payload(pool, teacher, bn_feature, mean, std, classes, ipc, output_path,
                   log_path, rate_kind, total_bpp=None, rate_components=None):
    pool.test()
    raw_images, labels, bpp = pool.get_data()
    if output_path.enforce_pixel_range:
        raw_images = raw_images.clamp(0.0, 1.0)
    labels = labels.to(raw_images.device)
    with torch.no_grad():
        utility, ce, bn, accuracy, range_penalty = teacher_metrics_and_backward(
            raw_images, labels, teacher, bn_feature, mean, std,
            bn_weight=output_path.bn_weight, backward=False,
            enforce_pixel_range=False, pixel_range_weight=0.0,
            batch_size=output_path.utility_batch_size,
        )
    normalized = normalize_images(raw_images, mean, std).detach().cpu()
    latent_bpp = float(bpp)
    reported_bpp = latent_bpp if total_bpp is None else float(total_bpp)
    kib_per_class = reported_bpp * ipc * raw_images.shape[-2] * raw_images.shape[-1] / 8192.0
    payload = {
        "images": normalized,
        "labels": labels.detach().cpu(),
        "mean": mean,
        "std": std,
        "teacher": "ResNet18ImageNetBN",
        "loss": "CE + alpha * BN",
        "alpha": output_path.bn_weight,
        "codec": "official DD-RUO TensorPool",
        "rate_kind": rate_kind,
        "latent_bpp": latent_bpp,
        "bpp": reported_bpp,
        "kib_per_class": kib_per_class,
        "rate_components": rate_components,
    }
    torch.save(payload, output_path.synthetic_path)
    save_and_print(
        log_path,
        f"final rate_kind={rate_kind} latent_bpp={latent_bpp:.6f} "
        f"total_bpp={reported_bpp:.6f} kib_per_class={kib_per_class:.2f} "
        f"utility={utility.item():.6f} ce={ce.item():.6f} bn={bn.item():.6f} "
        f"range_penalty={range_penalty.item():.6g} "
        f"teacher_acc={accuracy.item():.4f} output={output_path.synthetic_path}",
    )


def main(args):
    set_seed(args.seed)
    if args.fast_warmup_iterations <= 0:
        raise ValueError("--fast_warmup_iterations must be positive")
    if not 0.0 < args.warmup_rate_margin <= 1.0:
        raise ValueError("--warmup_rate_margin must be in (0, 1]")
    if args.pixel_range_weight < 0.0:
        raise ValueError("--pixel_range_weight must be non-negative")
    if args.utility_batch_size < 0:
        raise ValueError("--utility_batch_size must be non-negative")
    if args.rate_control == "dual":
        if args.ldb <= 0:
            raise ValueError("--ldb must be positive for dual rate-gradient conversion")
        if args.latent_target_kib <= 0:
            raise ValueError("--latent_target_kib must be positive")
        if args.dual_lr <= 0:
            raise ValueError("--dual_lr must be positive")
    # Upstream's ten host threads can trigger cuDNN stream/JIT faults when
    # several class codecs share one GPU. Per-device locks retain four-way
    # parallelism; disabling cuDNN removes the remaining PyTorch 2.6 hazard.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("At least one visible GPU is required")
    locks = install_per_gpu_tensorpool_locks(device_count)

    os.makedirs(args.save_path, exist_ok=True)
    args.log_path = os.path.join(args.save_path, "log.txt")
    args.synthetic_path = os.path.join(args.save_path, "synthetic.pt")
    args.zca = False
    args.device = "cuda:0"

    _, image_size, classes, _, mean, std, _, _, _, _, _, _ = dataset_context(
        args, remap_labels=False
    )
    checkpoint = torch.load(args.teacher_path, map_location="cpu", weights_only=False)
    teacher = build_resnet18_bn(classes, image_size).to(args.device)
    teacher.load_state_dict(checkpoint["state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    bn_feature = BNFeatureLoss(teacher)

    sample_per_class = [args.ipc] * classes
    pool = TensorPool(
        classes,
        1000,
        sample_per_class,
        list(range(device_count)),
        nthread=classes,
        ldb=args.ldb,
        img_size=image_size,
        max_iter=args.stage1_iterations + args.stage2_iterations,
        channel=3,
        lr=args.codec_lr,
        layers_v=args.layers_v,
        arm=args.arm,
        dim=args.dim,
    )

    save_and_print(args.log_path, f"begin={get_time()}")
    save_and_print(
        args.log_path,
        "implementation=official TensorPool utility=SRe2L(CE+alpha*BN) "
        "backend_compat=per_gpu_lock+cudnn_disabled "
        f"alpha={args.bn_weight:g} ldb={args.ldb:g} codec_lr={args.codec_lr:g} "
        f"lr_it={args.lr_it:g} stage1_ldb_it={args.stage1_ldb_it:g} "
        f"stage2_ldb_it={args.stage2_ldb_it:g} stage1={args.stage1_iterations} "
        f"stage2={args.stage2_iterations} rate_control={args.rate_control} "
        f"latent_target_kib={args.latent_target_kib:g} dual_lr={args.dual_lr:g} "
        f"warmup_rate_control={args.warmup_rate_control} "
        f"warmup_target_kib={args.warmup_target_kib:g} "
        f"warmup_rate_margin={args.warmup_rate_margin:g} "
        f"warmup_dual_lr={args.warmup_dual_lr:g} "
        f"fast_single_warmup={args.fast_single_warmup} "
        f"fast_warmup_iterations={args.fast_warmup_iterations} "
        f"enforce_pixel_range={args.enforce_pixel_range} "
        f"pixel_range_weight={args.pixel_range_weight:g} "
        f"utility_batch_size={args.utility_batch_size}",
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
    else:
        references = load_reference_images(
            args.init_path, classes, args.ipc, mean, std
        )
        save_and_print(args.log_path, f"candidate_warmup_start={get_time()}")
        if args.warmup_rate_control == "dual":
            warmup_target_bpp = (
                args.warmup_target_kib * args.warmup_rate_margin
            ) / (
                args.ipc * image_size[0] * image_size[1] / 8192.0
            )
            initialize_pool_with_dual_warmup(
                pool, references, warmup_target_bpp, args.warmup_dual_lr,
                args.enforce_pixel_range, args.pixel_range_weight, locks,
                args.fast_single_warmup, args.fast_warmup_iterations,
            )
        elif args.fast_single_warmup:
            initialize_pool_with_fixed_single_warmup(
                pool, references, locks, args.fast_warmup_iterations
            )
        else:
            pool.init_from_data(references)
        validate_initialized_pool(pool)
        pool.save_slice_pool(pool_init_path)
        save_and_print(args.log_path, f"candidate_warmup_done={get_time()} output={pool_init_path}")

    if args.warmup_rate_control == "dual" and not args.resume_pool:
        pool.test()
        warmup_images, warmup_labels, warmup_bpp = pool.get_data()
        warmup_labels = warmup_labels.to(warmup_images.device)
        warmup_utility, warmup_ce, warmup_bn, warmup_accuracy, warmup_range = teacher_metrics_and_backward(
            warmup_images, warmup_labels, teacher, bn_feature, mean, std,
            args.bn_weight, backward=False,
            enforce_pixel_range=args.enforce_pixel_range,
            pixel_range_weight=args.pixel_range_weight,
            batch_size=args.utility_batch_size,
        )
        warmup_kib = float(warmup_bpp) * args.ipc * image_size[0] * image_size[1] / 8192.0
        save_and_print(
            args.log_path,
            f"warmup_hard_check latent_bpp={float(warmup_bpp):.6f} "
            f"latent_kib_per_class={warmup_kib:.2f} utility={warmup_utility.item():.6f} "
            f"ce={warmup_ce.item():.6f} bn={warmup_bn.item():.6f} "
            f"range_penalty={warmup_range.item():.6g} "
            f"teacher_acc={warmup_accuracy.item():.4f}",
        )
        if warmup_accuracy.item() < args.warmup_min_teacher_acc:
            raise RuntimeError(
                f"Warmup teacher accuracy {warmup_accuracy.item():.4f} is below "
                f"required {args.warmup_min_teacher_acc:.4f}"
            )

    pool.set_training_phase(0)
    pool.init_solvers()
    labels = pool.label.to(args.device)
    total_iterations = args.stage1_iterations + args.stage2_iterations
    kib_per_bpp = args.ipc * image_size[0] * image_size[1] / 8192.0
    dual_lambda_eff = 0.0
    start_time = time.time()

    for iteration in range(1, total_iterations + 1):
        pool.train()
        raw_images, pool_labels, bpp = pool.get_data()
        if not torch.equal(pool_labels.cpu(), pool.label):
            raise RuntimeError("TensorPool label ordering changed unexpectedly")
        pool.data.grad = None
        utility, ce, bn, accuracy, range_penalty = teacher_metrics_and_backward(
            raw_images, labels, teacher, bn_feature, mean, std,
            args.bn_weight, backward=True,
            enforce_pixel_range=args.enforce_pixel_range,
            pixel_range_weight=args.pixel_range_weight,
            batch_size=args.utility_batch_size,
        )
        if pool.data.grad is None or not torch.isfinite(pool.data.grad).all():
            raise FloatingPointError("Non-finite or missing SRe2L image gradient")
        gradient_l1 = pool.data.grad.abs().sum().item()
        pool.fill_data_diff()
        latent_kib = float(bpp) * kib_per_bpp
        if args.rate_control == "dual":
            # TensorPool scales utility gradients by lr_it. Internally its rate
            # gradient already contains qp.ldb, so this conversion implements
            # lr_it * (L_utility + dual_lambda_eff * latent_bpp).
            ldb_it = dual_lambda_eff * args.lr_it / args.ldb
        else:
            ldb_it = (
                args.stage1_ldb_it
                if iteration <= args.stage1_iterations
                else args.stage2_ldb_it
            )
        pool.backward(args.lr_it, ldb_it)

        dual_violation = latent_kib / args.latent_target_kib - 1.0
        if args.rate_control == "dual":
            dual_lambda_eff = max(
                0.0, dual_lambda_eff + args.dual_lr * dual_violation
            )

        if iteration == 1 or iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            seconds_per_iter = elapsed / iteration
            eta_hours = seconds_per_iter * (total_iterations - iteration) / 3600.0
            save_and_print(
                args.log_path,
                f"iter={iteration:05d}/{total_iterations:05d} stage="
                f"{1 if iteration <= args.stage1_iterations else 2} "
                f"utility={utility.item():.6f} ce={ce.item():.6f} bn={bn.item():.6f} "
                f"range_penalty={range_penalty.item():.6g} "
                f"teacher_acc={accuracy.item():.4f} latent_bpp={float(bpp):.6f} "
                f"latent_kib_per_class={latent_kib:.2f} image_grad_l1={gradient_l1:.4e} "
                f"ldb_it={ldb_it:g} dual_lambda_eff={dual_lambda_eff:.8g} "
                f"dual_violation={dual_violation:.6f} "
                f"sec_per_iter={seconds_per_iter:.2f} eta_hours={eta_hours:.2f}",
            )

        if iteration % args.checkpoint_every == 0 or iteration == total_iterations:
            checkpoint_path = os.path.join(args.save_path, f"pool_{iteration}.pt")
            pool.save_slice_pool(checkpoint_path)
            if args.rate_control == "dual":
                torch.save(
                    {
                        "iteration": iteration,
                        "dual_lambda_eff": dual_lambda_eff,
                        "latent_target_kib": args.latent_target_kib,
                        "dual_lr": args.dual_lr,
                    },
                    os.path.join(args.save_path, f"dual_state_{iteration}.pt"),
                )
            save_and_print(args.log_path, f"checkpoint={checkpoint_path}")

    pool.save_slice_pool(os.path.join(args.save_path, "pool_pre_net_quant.pt"))
    rate_kind = "latent_entropy"
    total_bpp = None
    rate_components = None
    if not args.skip_net_quantization:
        result = pool.quantize_net(args.network_mse_threshold)
        if result is None:
            raise RuntimeError("DD-RUO network post-quantization failed")
        total_bpp, rate_components = result
        pool.save_slice_pool(os.path.join(args.save_path, "pool_quantized.pt"))
        save_and_print(
            args.log_path,
            f"network_quantization total_bpp={float(total_bpp):.6f} components={rate_components}",
        )
        rate_kind = "latent_entropy_after_network_quantization"

    export_payload(
        pool, teacher, bn_feature, mean, std, classes, args.ipc, args,
        args.log_path, rate_kind, total_bpp, rate_components
    )
    bn_feature.close()
    save_and_print(args.log_path, f"complete={get_time()}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ImageNet")
    parser.add_argument("--subset", default="imagefruit")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--ipc", type=int, default=102)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--init_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--resume_pool", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bn_weight", type=float, default=0.01)
    parser.add_argument("--ldb", type=float, default=0.1)
    parser.add_argument("--codec_lr", type=float, default=0.001)
    parser.add_argument("--lr_it", type=float, default=1000.0)
    parser.add_argument("--stage1_ldb_it", type=float, default=10.0)
    parser.add_argument("--stage2_ldb_it", type=float, default=150.0)
    parser.add_argument("--stage1_iterations", type=int, default=4000)
    parser.add_argument("--stage2_iterations", type=int, default=4000)
    parser.add_argument("--rate_control", choices=("fixed", "dual"), default="fixed")
    parser.add_argument("--latent_target_kib", type=float, default=200.0)
    parser.add_argument("--dual_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_rate_control", choices=("fixed", "dual"), default="fixed")
    parser.add_argument("--warmup_target_kib", type=float, default=200.0)
    parser.add_argument("--warmup_rate_margin", type=float, default=1.0)
    parser.add_argument("--warmup_dual_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_min_teacher_acc", type=float, default=0.99)
    parser.add_argument("--fast_single_warmup", action="store_true")
    parser.add_argument("--fast_warmup_iterations", type=int, default=400)
    parser.add_argument("--enforce_pixel_range", action="store_true")
    parser.add_argument("--pixel_range_weight", type=float, default=1.0)
    parser.add_argument("--utility_batch_size", type=int, default=0)
    parser.add_argument("--layers_v", default="v5")
    parser.add_argument("--arm", type=int, default=32)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--network_mse_threshold", type=float, default=5e-7)
    parser.add_argument("--skip_net_quantization", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
