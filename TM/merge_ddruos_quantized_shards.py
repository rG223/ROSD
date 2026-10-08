"""Merge decoded class shards into the normalized downstream payload."""

import argparse
import os

import torch


def main(args):
    classes = args.num_classes
    images_raw = torch.empty(
        classes * args.ipc, 3, args.image_size, args.image_size,
        dtype=torch.float32,
    )
    labels = torch.empty(classes * args.ipc, dtype=torch.long)
    weighted_total_bpp = 0.0
    weighted_latent_bpp = 0.0
    weighted_components = None
    expected_class_start = 0
    for path in args.inputs:
        item = torch.load(path, map_location="cpu", weights_only=False)
        class_start = int(item["class_start"])
        count = int(item["classes_per_shard"])
        if class_start != expected_class_start:
            raise RuntimeError(
                f"Non-contiguous shard order: expected class {expected_class_start}, "
                f"got {class_start} from {path}"
            )
        if item.get("encoder_gain", 16) != args.encoder_gain:
            raise RuntimeError("Post-quantization shards use inconsistent encoder gains")
        if int(item.get("image_size", args.image_size)) != args.image_size:
            raise RuntimeError("Post-quantization shards use inconsistent image sizes")
        image_start = class_start * args.ipc
        image_end = (class_start + count) * args.ipc
        expected_shape = (
            count * args.ipc, 3, args.image_size, args.image_size
        )
        if tuple(item["images_raw"].shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected shard image shape {tuple(item['images_raw'].shape)}; "
                f"expected {expected_shape}"
            )
        images_raw[image_start:image_end].copy_(item["images_raw"])
        labels[image_start:image_end].copy_(item["labels"])
        weighted_total_bpp += float(item["total_bpp"]) * count
        weighted_latent_bpp += float(item["latent_bpp"]) * count
        if weighted_components is None:
            weighted_components = {
                key: 0.0 for key in item["rate_components"]
            }
        for key, value in item["rate_components"].items():
            weighted_components[key] += float(value) * count
        expected_class_start += count
        del item
    if expected_class_start != classes:
        raise RuntimeError(
            f"Shards cover {expected_class_start} classes, expected {classes}"
        )
    expected = torch.arange(classes).repeat_interleave(args.ipc)
    if not torch.equal(labels, expected):
        raise RuntimeError("Merged class/label ordering is invalid")

    # Tiny-ImageNet experiments in this repository use ImageNet normalization.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_tensor = images_raw.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images_raw.new_tensor(std).view(1, 3, 1, 1)
    # Match the established ROSD export path exactly. TensorPool's
    # learned continuous output is not constrained to conventional RGB range;
    # post-hoc clipping would change the optimized solution.
    images_raw.sub_(mean_tensor).div_(std_tensor)
    images = images_raw

    total_bpp = weighted_total_bpp / classes
    latent_bpp = weighted_latent_bpp / classes
    components = {
        key: value / classes for key, value in weighted_components.items()
    }
    image_kib = (
        total_bpp * args.ipc * args.image_size * args.image_size / 8192.0
    )
    total_kib = image_kib + args.label_kib + args.label_model_kib
    payload = {
        "images": images,
        "labels": labels,
        "mean": mean,
        "std": std,
        "teacher": "ResNet18ImageNetBN",
        "loss": f"RRC+flip {args.utility_mode} CE + alpha * BN",
        "codec": f"official DD-RUO TensorPool; {len(args.inputs)} class shards",
        "rate_kind": "latent_entropy_after_network_quantization",
        "latent_bpp": latent_bpp,
        "bpp": total_bpp,
        "kib_per_class": image_kib,
        "image_kib_per_class": image_kib,
        "label_kib_per_class": args.label_kib,
        "label_model_kib_per_class": args.label_model_kib,
        "total_kib_per_class": total_kib,
        "rate_components": components,
        "stored_ipc": args.ipc,
        "image_size": args.image_size,
        "decoded_views_per_class": args.ipc,
        "cim_factor": None,
        "utility_mode": args.utility_mode,
        "joint_image_lambda": args.joint_image_lambda,
        "joint_label_lambda": args.joint_label_lambda,
        "encoder_gain": args.encoder_gain,
        "latent_step": 1.0 / args.encoder_gain,
        "label_groups": args.label_groups,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"merged images={len(labels)} total_bpp={total_bpp:.8f} "
        f"image_kib_per_class={image_kib:.3f} "
        f"estimated_label_kib_per_class={args.label_kib:.3f} "
        f"label_model_kib_per_class={args.label_model_kib:.3f} "
        f"estimated_total_kib_per_class={total_kib:.3f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ipc", default=100, type=int)
    parser.add_argument("--num_classes", default=200, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--label_kib", default=417.28, type=float)
    parser.add_argument("--label_model_kib", default=7.31232421875, type=float)
    parser.add_argument(
        "--utility_mode", default="sre2l",
        choices=("sre2l", "lpld_class_bn"),
    )
    parser.add_argument("--joint_image_lambda", default=1e-4, type=float)
    parser.add_argument("--joint_label_lambda", default=1e-4, type=float)
    parser.add_argument("--encoder_gain", default=16, type=int)
    parser.add_argument("--label_groups", default=30, type=int)
    main(parser.parse_args())
