"""Post-quantize and decode one TensorPool class shard on one GPU."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ts.tensor_pool import TensorPool
from TM.ddruos_sre2l_joint_distributed import load_sre2l_teacher
from TM.postquant_rate import build_class_scorers
from TM.sre2l_fkd import FixedCIMLabelCodec
from TM.sre2l_official_tensorpool import install_per_gpu_tensorpool_locks


def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    install_per_gpu_tensorpool_locks(1)

    pool = TensorPool(
        args.classes_per_shard,
        1000,
        [args.ipc] * args.classes_per_shard,
        [0],
        nthread=args.workers,
        ldb=0.1,
        img_size=(args.image_size, args.image_size),
        max_iter=args.max_iter,
        channel=3,
        lr=0.001,
        layers_v="v5",
        arm=32,
        dim=4,
        encoder_gain=args.encoder_gain,
    )
    global_state = torch.load(args.input, map_location="cpu", weights_only=False)
    local_state = {}
    for local_class in range(args.classes_per_shard):
        global_key = f"{args.class_start + local_class}_0"
        local_key = f"{local_class}_0"
        if global_key not in global_state:
            raise KeyError(f"Missing TensorPool key {global_key}")
        local_state[local_key] = global_state[global_key]
    pool.slice_pool = local_state
    for key in pool.key_list:
        pool.slice_pool[key]["param"].load_reset()

    decoder_scorers = None
    label_groups = None
    if args.postquant_label_rate_samples > 0:
        if not args.teacher_path or not args.label_codec_checkpoint:
            raise ValueError(
                "--teacher_path and --label_codec_checkpoint are required "
                "when --postquant_label_rate_samples is positive"
            )
        checkpoint = torch.load(
            args.label_codec_checkpoint, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        classes = int(config["classes"])
        label_groups = int(config["label_groups"])
        expected_slots = classes * args.ipc
        if int(config["num_slots"]) != expected_slots:
            raise ValueError(
                f"Label codec has {config['num_slots']} slots; expected "
                f"classes={classes} * ipc={args.ipc} = {expected_slots}"
            )
        teacher = load_sre2l_teacher(
            args.teacher_path,
            classes,
            (args.image_size, args.image_size),
        ).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        label_codec = FixedCIMLabelCodec(checkpoint).cuda().eval()
        for parameter in label_codec.parameters():
            parameter.requires_grad_(False)
        decoder_scorers = build_class_scorers(
            teacher,
            label_codec,
            args.class_start,
            args.classes_per_shard,
            args.ipc,
            label_groups,
            args.postquant_label_rate_samples,
        )

    result = pool.quantize_net(
        args.mse_threshold,
        decoder_soft_label_rate_fns=decoder_scorers,
    )
    if result is None:
        raise RuntimeError("TensorPool network post-quantization failed")
    total_bpp, components = result
    pool.save_slice_pool(args.quantized_output)

    pool.test()
    images, labels, latent_bpp = pool.get_data()
    labels = labels + args.class_start
    expected_images = args.classes_per_shard * args.ipc
    if images.shape != (
        expected_images, 3, args.image_size, args.image_size
    ):
        raise RuntimeError(
            f"Decoded image shape {tuple(images.shape)} does not match "
            f"classes_per_shard={args.classes_per_shard}, ipc={args.ipc}"
        )
    expected = torch.arange(
        args.class_start,
        args.class_start + args.classes_per_shard,
    ).repeat_interleave(args.ipc)
    if not torch.equal(labels.cpu(), expected):
        raise RuntimeError("Decoded TensorPool label order is invalid")
    torch.save(
        {
            "images_raw": images.detach().cpu(),
            "labels": labels.cpu(),
            "class_start": args.class_start,
            "classes_per_shard": args.classes_per_shard,
            "ipc": args.ipc,
            "image_size": args.image_size,
            "encoder_gain": args.encoder_gain,
            "latent_step": 1.0 / args.encoder_gain,
            "latent_bpp": float(latent_bpp),
            "total_bpp": float(total_bpp),
            "rate_components": {key: float(value) for key, value in components.items()},
            "postquant_selection": {
                "arm": "quantized_arm_rate_plus_image_latent_rate",
                "decoder": (
                    "quantized_decoder_rate_plus_soft_label_rate"
                    if decoder_scorers is not None
                    else "legacy_total_image_codec_rate"
                ),
                "label_groups": label_groups,
                "label_rate_samples": args.postquant_label_rate_samples,
            },
        },
        args.output,
    )
    print(
        f"complete class_start={args.class_start} total_bpp={float(total_bpp):.8f} "
        f"latent_bpp={float(latent_bpp):.8f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantized_output", required=True)
    parser.add_argument("--class_start", required=True, type=int)
    parser.add_argument("--classes_per_shard", default=50, type=int)
    parser.add_argument("--ipc", default=100, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--max_iter", default=400, type=int)
    parser.add_argument("--workers", default=12, type=int)
    parser.add_argument("--mse_threshold", default=5e-7, type=float)
    parser.add_argument("--encoder_gain", default=16, type=int)
    parser.add_argument("--teacher_path")
    parser.add_argument("--label_codec_checkpoint")
    parser.add_argument("--postquant_label_rate_samples", default=0, type=int)
    main(parser.parse_args())
