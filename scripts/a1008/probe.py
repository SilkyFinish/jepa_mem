"""Real NCCL and three longest/mixed-prefix AdamW updates, using the production Trainer."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


def collective(output):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        value = torch.ones(1024 * 1024, device="cuda")
        for _ in range(3):
            value.fill_(1)
            dist.all_reduce(value)
            torch.cuda.synchronize()
            if not torch.equal(value, torch.full_like(value, dist.get_world_size())):
                raise RuntimeError("NCCL sum mismatch")
        if dist.get_rank() == 0:
            Path(output).write_text(json.dumps(dict(ok=True, world_size=dist.get_world_size())) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def attention(output):
    import io
    import unittest
    from scripts.a1008.common import REPO
    if not torch.cuda.is_available():
        raise RuntimeError("FlexAttention validation requires a CUDA device")
    suite = unittest.defaultTestLoader.discover(
        str(REPO / "tests"), pattern="test_memory_flex_attention.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    Path(output).write_text(json.dumps(dict(
        ok=result.wasSuccessful() and not result.skipped, tests=result.testsRun,
        skipped=len(result.skipped), torch=torch.__version__,
        gpu=torch.cuda.get_device_name())) + "\n")
    if result.testsRun != 3 or not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Compiled FlexAttention forward/backward parity checks failed")


def stress(argv):
    from scripts.benchmark_history_training import main as benchmark
    from scripts.train_vjepa_policy import parse_args
    from scripts.a1008.common import schedule
    args = parse_args(argv)
    plan = schedule(48, int(os.environ["WORLD_SIZE"]), 1, args.batch_size)
    options = ["--benchmark-stress-updates", "3"]
    if args.batch_size >= 2:
        options.append("--benchmark-stress-mixed-episodes")
    benchmark([*options, *argv, "--max-steps", "3", "--num-workers", "0",
               "--gradient-accumulation-steps", str(plan["accumulation"])])
    if int(os.environ["RANK"]) == 0:
        report = json.loads(Path(args.output_dir, "rank0_timing.json").read_text())
        if report["status"] != "passed" or report["actual_updates"] != 3:
            raise RuntimeError("Longest/mixed history stress did not complete")
        Path(args.output_dir, "stress.json").write_text(json.dumps(dict(
            ok=True, diagnostic=True, updates=3, world_size=plan["world_size"],
            per_device_batch=args.batch_size, accumulation=plan["accumulation"],
            global_batch=16, mixed_episode_stress=args.batch_size >= 2,
            history_tokens_per_block=report["history_tokens_per_block"],
            steps=report["steps"])) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collective", "attention", "stress"), required=True)
    parser.add_argument("--output")
    args, rest = parser.parse_known_args()
    if args.mode == "collective":
        collective(args.output)
    elif args.mode == "attention":
        attention(args.output)
    else:
        stress(rest)
