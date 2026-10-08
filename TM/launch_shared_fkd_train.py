"""Launch several FKD downstream jobs sharing one forked synthetic tensor."""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils import set_seed
from TM import sre2l_fkd


def run_job(argv, stdout_path, cuda_visible_devices=None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", buffering=1) as stream:
        os.dup2(stream.fileno(), sys.stdout.fileno())
        os.dup2(stream.fileno(), sys.stderr.fileno())
        args = sre2l_fkd.parse_args(argv)
        set_seed(args.seed)
        if args.mode == "train_pool":
            sre2l_fkd.train_pool(args)
        elif args.mode == "relabel_pool":
            sre2l_fkd.relabel_pool(args)
        else:
            raise ValueError(f"Unsupported shared-launch mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_path", required=True)
    parser.add_argument("--jobs_json", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs_json).read_text())
    if not jobs:
        raise ValueError("jobs_json contains no jobs")
    images, _ = sre2l_fkd.cache_synthetic_for_fork(args.synthetic_path)
    print(
        f"shared synthetic loaded shape={tuple(images.shape)} "
        f"size_gib={images.numel() * images.element_size() / 1024**3:.3f}",
        flush=True,
    )
    context = mp.get_context("fork")
    processes = []
    for job in jobs:
        process = context.Process(
            target=run_job,
            args=(
                job["argv"],
                job["stdout_path"],
                job.get("cuda_visible_devices"),
            ),
            name=job["name"],
        )
        process.start()
        processes.append(process)
        print(f"started name={job['name']} pid={process.pid}", flush=True)
    failed = []
    for process in processes:
        process.join()
        print(
            f"finished name={process.name} exitcode={process.exitcode}", flush=True
        )
        if process.exitcode != 0:
            failed.append((process.name, process.exitcode))
    if failed:
        raise RuntimeError(f"Downstream jobs failed: {failed}")


if __name__ == "__main__":
    main()
