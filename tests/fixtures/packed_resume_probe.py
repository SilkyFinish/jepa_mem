"""Two-GPU old-loader to compact-loader continuation, including epoch changes."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_memory_v0_model import enable_attention_gates
from test_memory_v1_spatial import model
from vjepa_policy.datasets.rmbench_history import HistoryBatchCollator
from vjepa_policy.memory_config import FullHistoryV2Config
from vjepa_policy.packed_history_features import PackedHistoryBatchCollator
from vjepa_policy.trainer import Trainer
from vjepa_policy.train_common import seed_process


class FeatureBank:
    def __init__(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1001)
            self.observations = torch.randn(3, 16, 12, 64)
            self.targets = torch.randn(3, 16, 48, 64)

    def rows(self, episode, kind, ends):
        bank = self.observations if kind == "observations" else self.targets
        return bank[int(episode), list(ends)]


class Data:
    provenance = {"fixture": "same-FP32-features-plain-to-compact-v1"}
    share_history_prefixes = True
    index = [(episode, t) for episode in range(3) for t in range(16)]
    history_lengths = [len(FullHistoryV2Config().observed_steps(t)) for _, t in index]

    def __init__(self, packed):
        self.packed, self.bank = packed, FeatureBank()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2002)
            self.states = torch.randn(3, 16, 14)
            self.language = torch.randn(3, 4, 16)
            self.actions = torch.randn(3, 16, 32, 14)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, t = self.index[index]
        ends = FullHistoryV2Config().observed_steps(t)
        state = self.states[episode, list(ends)]
        result = dict(batch_schema="rmbench_history_v0", episode_id=str(episode),
                      sample_key=f"{episode}:{t}", history_state=state,
                      history_end_step=torch.tensor(ends), history_valid=torch.ones(len(ends), dtype=torch.bool),
                      decision_step=torch.tensor(t), current_state=state[-1],
                      language=self.language[episode], language_mask=torch.tensor([True, True, False, False]),
                      future_valid=torch.ones(48, dtype=torch.bool), action=self.actions[episode, t],
                      action_is_pad=torch.arange(32) + t >= 32)
        if self.packed:
            result["history_feature_ref"] = (str(episode), t, tuple(ends))
        else:
            result["history_latents"] = self.bank.rows(str(episode), "observations", ends)
            result["future_targets"] = self.bank.rows(str(episode), "targets", [t])[0]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("continuous", "interrupt", "resume"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seed_process(11)
    policy = model(checkpointing=True)
    policy.config = FullHistoryV2Config()
    policy.share_history_prefixes = True
    policy.action_expert.checkpoint_memory_blocks = True
    enable_attention_gates(policy.action_expert)
    seen = []
    policy.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]["sample_key"]))
    data = Data(packed=args.mode != "interrupt")
    collator = HistoryBatchCollator()
    if data.packed:
        collator = PackedHistoryBatchCollator(data.bank, collator)
    cfg = dict(output_dir=str(args.output), batch_size=4, num_workers=2,
               prefetch_factor=4 if data.packed else 2, gradient_accumulation_steps=2,
               mixed_precision="bf16", lr=1e-3, weight_decay=.01, max_steps=7,
               num_epochs=3, max_grad_norm=1., log_every=1, save_every=7,
               data_seed=7, train_seed=11, save_training_state=True,
               keep_last_checkpoints=2, checkpoint_milestones=[3],
               recycle_workers_every=0, history_bucket_size=128)
    if args.mode == "interrupt":
        cfg["stop_after_step"] = 3
    elif args.mode == "resume":
        cfg["resume_from"] = str(args.output / "resume_state.pt")
    with patch.object(Trainer, "_init_wandb", return_value=None):
        trainer = Trainer(policy, data, collator, cfg=cfg)
        trainer.train()
    (args.output / f"{args.mode}-rank{os.environ['RANK']}.json").write_text(json.dumps(
        dict(seen=seen, step=trainer.global_step, frozen_encoder=not any(
            p.grad is not None for p in policy.encoder.parameters()))))


if __name__ == "__main__":
    main()
