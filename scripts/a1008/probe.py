"""Real NCCL and two longest-prefix AdamW updates, using the production Trainer."""
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


def stress(argv):
    from scripts import train_vjepa_policy as entry
    from vjepa_policy.train_common import build_dataset_collator
    from scripts.a1008.common import schedule
    world = int(os.environ["WORLD_SIZE"])

    class Longest:
        def __init__(self, data):
            self.data = data
            self.index = max(range(len(data)), key=lambda i: data.history_lengths[i])
            self.history_lengths = [data.history_lengths[self.index]] * 32
            self.provenance = dict(data.provenance, diagnostic="repeated longest sample, two updates")
            self.batch_schema = data.batch_schema
        def __len__(self):
            return 32
        def __getitem__(self, index):
            return self.data[self.index]

    def builder(args, **kwargs):
        data, collator = build_dataset_collator(args, **kwargs)
        return Longest(data), collator
    builder.supports_local_layout = True
    args = entry.parse_args([*argv, "--max-steps", "2", "--num-workers", "0"])
    plan = schedule(32, world, 1, args.batch_size)
    args.gradient_accumulation_steps = plan["accumulation"]
    start = time.time()
    entry.main(args=args, dataset_collator_builder=builder)
    if int(os.environ["RANK"]) == 0:
        Path(args.output_dir, "stress.json").write_text(json.dumps(dict(
            ok=True, diagnostic=True, updates=2, seconds=time.time() - start,
            world_size=world, per_device_batch=args.batch_size,
            accumulation=args.gradient_accumulation_steps, global_batch=plan["global_batch"],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collective", "stress"), required=True)
    parser.add_argument("--output")
    args, rest = parser.parse_known_args()
    if args.mode == "collective":
        collective(args.output)
    else:
        stress(rest)
