"""Build a target-rate FKD pool from existing pools using hard links."""

import argparse
import os
from pathlib import Path

import torch


_AUGMENTATION_METADATA_KEYS = (
    "order",
    "coords",
    "flips",
    "mix_index",
    "bbox",
)


def augmentation_metadata_bits(config):
    """Return the stored tensor payload needed to replay one FKD augmentation."""
    return 8 * sum(
        config[key].numel() * config[key].element_size()
        for key in _AUGMENTATION_METADATA_KEYS
        if key in config and torch.is_tensor(config[key])
    )


def main(args):
    sources = [Path(path) for path in args.sources]
    summaries = [
        torch.load(path / "pool_summary.pt", map_location="cpu", weights_only=False)
        for path in sources
    ]
    reference = summaries[0]
    classes = args.num_classes
    base_kib = float(reference["image_kib_per_class"]) + float(
        reference["label_model_kib_per_class"]
    )
    if args.target_total_mb is not None:
        target_total_kib = args.target_total_mb * 1_000_000.0 / 1024.0 / classes
    else:
        target_total_kib = args.target_total_kib
    target_label_bits = (target_total_kib - base_kib) * classes * 8192.0
    if target_label_bits <= 0:
        raise ValueError("Target does not cover image and model overhead")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.pool_batches is not None:
        available = sum(int(summary["pool_batches"]) for summary in summaries)
        if not 1 <= args.pool_batches <= available:
            raise ValueError(
                f"pool_batches={args.pool_batches} is outside available [1, {available}]"
            )
        for stale in output.glob("pool_batch_*.pt"):
            stale.unlink()
        saved = 0
        payload_bytes = 0
        total_bits = 0.0
        for source, summary in zip(sources, summaries):
            take = min(args.pool_batches - saved, int(summary["pool_batches"]))
            for index in range(take):
                path = source / f"pool_batch_{index:05d}.pt"
                destination = output / f"pool_batch_{saved:05d}.pt"
                os.link(path, destination)
                payload_bytes += destination.stat().st_size
                saved += 1
            total_bits += float(summary["entropy_label_bits"]) * (
                take / int(summary["pool_batches"])
            )
            if saved == args.pool_batches:
                break
        achieved_label_kib = total_bits / classes / 8192.0
        achieved_total = base_kib + achieved_label_kib
        result = dict(reference)
        result.update(
            {
                "method": "fixed-size subset of FKD batch pools",
                "pool_batches": saved,
                "actual_compression": float(reference["full_batches"]) / saved,
                "payload_bytes": payload_bytes,
                "payload_kib_per_class": payload_bytes / 1024.0 / classes,
                "entropy_label_bits": total_bits,
                "entropy_label_kib_per_class": achieved_label_kib,
                "target_total_kib_per_class": target_total_kib,
                "achieved_total_kib_per_class": achieved_total,
            }
        )
        torch.save(result, output / "pool_summary.pt")
        print(
            f"pool_batches={saved} label_kib_per_class={achieved_label_kib:.4f} "
            f"total_kib_per_class={achieved_total:.4f} output={output}",
            flush=True,
        )
        return
    total_bits = 0.0
    metadata_bits = 0
    payload_bytes = 0
    saved = 0
    selection_complete = False
    for source, summary in zip(sources, summaries):
        for index in range(int(summary["pool_batches"])):
            path = source / f"pool_batch_{index:05d}.pt"
            config = torch.load(path, map_location="cpu", weights_only=False)
            bits = float(config.get("label_bits", 0.0))
            batch_metadata_bits = (
                augmentation_metadata_bits(config)
                if args.include_augmentation_metadata else 0
            )
            accounted_bits = bits + batch_metadata_bits
            if saved >= int(reference["batches_per_epoch"]) and abs(
                total_bits + metadata_bits - target_label_bits
            ) <= abs(total_bits + metadata_bits + accounted_bits - target_label_bits):
                selection_complete = True
                break
            destination = output / f"pool_batch_{saved:05d}.pt"
            if destination.exists():
                destination.unlink()
            os.link(path, destination)
            payload_bytes += destination.stat().st_size
            total_bits += bits
            metadata_bits += batch_metadata_bits
            saved += 1
        if selection_complete:
            break
        if (
            saved >= int(reference["batches_per_epoch"])
            and total_bits + metadata_bits >= target_label_bits
        ):
            selection_complete = True
            break

    if not selection_complete and total_bits + metadata_bits < target_label_bits:
        reachable_total_kib = base_kib + (
            total_bits + metadata_bits
        ) / classes / 8192.0
        reachable_total_mb = reachable_total_kib * classes * 1024.0 / 1_000_000.0
        raise ValueError(
            "Source pools cannot reach the requested target: "
            f"requested={target_total_kib:.4f} KiB/class, "
            f"maximum={reachable_total_kib:.4f} KiB/class "
            f"({reachable_total_mb:.4f} MB total)"
        )

    achieved_label_kib = total_bits / classes / 8192.0
    metadata_kib = metadata_bits / classes / 8192.0
    achieved_total = base_kib + achieved_label_kib + metadata_kib
    achieved_total_mb = achieved_total * classes * 1024.0 / 1_000_000.0
    result = dict(reference)
    result.update(
        {
            "method": "target-rate subset of FKD batch pools",
            "pool_batches": saved,
            "actual_compression": float(reference["full_batches"]) / saved,
            "payload_bytes": payload_bytes,
            "payload_kib_per_class": payload_bytes / 1024.0 / classes,
            "entropy_label_bits": total_bits,
            "entropy_label_kib_per_class": achieved_label_kib,
            "target_total_kib_per_class": target_total_kib,
            "target_total_mb": args.target_total_mb,
            "augmentation_metadata_bits": metadata_bits,
            "augmentation_metadata_kib_per_class": metadata_kib,
            "achieved_total_including_metadata_kib_per_class": achieved_total,
            "achieved_total_including_metadata_mb": achieved_total_mb,
            "achieved_total_kib_per_class": achieved_total,
        }
    )
    torch.save(result, output / "pool_summary.pt")
    print(
        f"pool_batches={saved} label_kib_per_class={achieved_label_kib:.4f} "
        f"metadata_kib_per_class={metadata_kib:.4f} "
        f"total_kib_per_class={achieved_total:.4f} total_mb={achieved_total_mb:.4f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target_total_kib", type=float)
    target.add_argument("--target_total_mb", type=float)
    parser.add_argument("--include_augmentation_metadata", action="store_true")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--pool_batches", type=int, default=None)
    main(parser.parse_args())
