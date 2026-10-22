"""Run separately with torchrun to check real multi-process restart/teardown."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from test_memory_v0_model import model, batch
from vjepa_policy.trainer import Trainer
from vjepa_policy.train_common import seed_process


class Data:
    history_lengths = [5] * 10
    provenance = {"fixture": "DDP deterministic features v1"}
    def __len__(self):
        return 10
    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(301 + index)
            sample = batch()
        sample = {k: v[:1] if torch.is_tensor(v) else v for k, v in sample.items()}
        sample["sample_index"] = index
        return sample


class Cutoff(Trainer):
    def _stop_requested(self):
        return self.global_step >= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("continuous", "interrupt", "resume"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seed_process(11)
    seen = []
    policy = model()
    policy.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]["sample_index"]))
    cfg = dict(output_dir=str(args.output), batch_size=1, num_workers=0,
               gradient_accumulation_steps=2, mixed_precision="bf16", lr=1e-3,
               weight_decay=.01, max_steps=5, num_epochs=3, max_grad_norm=1.,
               log_every=1, save_every=1, data_seed=7, train_seed=11,
               save_training_state=True, keep_last_checkpoints=2,
               checkpoint_milestones=[2], recycle_workers_every=0, history_bucket_size=0)
    if args.mode == "resume":
        cfg["resume_from"] = str(args.output / "resume_state.pt")
    with patch.object(Trainer, "_init_wandb", return_value=None):
        cls = Cutoff if args.mode == "interrupt" else Trainer
        trainer = cls(policy, Data(), lambda values: values[0], cfg=cfg)
        trainer.train()
    (args.output / f"{args.mode}-rank{os.environ['RANK']}.json").write_text(json.dumps(dict(
        seen=seen, step=trainer.global_step, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        frozen_encoder=not any(p.grad is not None for p in policy.encoder.parameters()))) + "\n")


if __name__ == "__main__":
    main()
