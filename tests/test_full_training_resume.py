"""Actual AdamW continuation: weights, moments, RNG, data cursor and retention."""
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm", "h5py", "lerobot", "accelerate")):
    raise unittest.SkipTest("full resume requires the installed policy environment")

import numpy as np
import torch
from vjepa_policy.trainer import Trainer
from test_memory_v0_model import model as tiny_model, batch as tiny_batch


class Data:
    history_lengths = [5] * 10
    provenance = {"fixture": "deterministic indexes, stochastic observations"}
    def __init__(self):
        self.seen = []
    def __len__(self):
        return 10
    def __getitem__(self, index):
        # Production cached observations are deterministic. Keep DataLoader
        # lookahead from changing the model's flow-noise RNG in this fixture.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(301 + index)
            value = tiny_batch()
        value["action"] += index
        value["sample_index"] = index
        return value


class FullStateTests(unittest.TestCase):
    def test_uninterrupted_matches_resumed_across_epoch_boundary(self):
        def seed():
            random.seed(11); np.random.seed(11); torch.manual_seed(11)
        def model(data):
            result = tiny_model()
            result.register_forward_pre_hook(lambda module, args: data.seen.append(args[0]["sample_index"]))
            return result
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with tempfile.TemporaryDirectory() as root, patch.object(Trainer, "_init_wandb", return_value=None):
            cfg = dict(output_dir=root + "/continuous", batch_size=1, num_workers=0,
                       gradient_accumulation_steps=2, mixed_precision="no", lr=1e-3,
                       weight_decay=0.01, max_steps=7, num_epochs=2, max_grad_norm=1.,
                       log_every=1, save_every=1, data_seed=7, train_seed=11,
                       save_training_state=True, keep_last_checkpoints=2, checkpoint_milestones=[2],
                       recycle_workers_every=0, history_bucket_size=0)
            seed(); full_data = Data()
            continuous = Trainer(model(full_data), full_data, lambda samples: samples[0], cfg=cfg)
            continuous.train()
            expected = torch.load(Path(cfg["output_dir"]) / "resume_state.pt", weights_only=False)
            seed(); first_data = Data()
            first_cfg = dict(cfg, output_dir=root + "/partial")
            first = Trainer(model(first_data), first_data, lambda samples: samples[0],
                            cfg=dict(first_cfg, stop_after_step=3))
            first.train()
            resume_path = Path(first_cfg["output_dir"]) / "resume_state.pt"
            partial = torch.load(resume_path, weights_only=False)
            self.assertEqual(partial["training_state"]["microbatches_in_epoch"], 6)
            self.assertEqual(partial["step"], 3)
            resumed_data = Data()
            resumed = Trainer(model(resumed_data), resumed_data, lambda samples: samples[0],
                              cfg=dict(first_cfg, resume_from=str(resume_path)))
            resumed.train()
            actual = torch.load(resume_path, weights_only=False)
            self.assertEqual(full_data.seen, first_data.seen + resumed_data.seen)
            for group in ("predictor", "action_expert"):
                for key in expected[group]:
                    torch.testing.assert_close(expected[group][key], actual[group][key], rtol=0, atol=0)
            expected_state = expected["training_state"]
            actual_state = actual["training_state"]
            self.assertEqual(expected_state["scheduler"], actual_state["scheduler"])
            for index, state in expected_state["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, actual_state["optimizer"]["state"][index][key], rtol=0, atol=0)
            remaining = sorted(p.name for p in Path(first_cfg["output_dir"]).glob("checkpoint_step*.pt"))
            self.assertEqual(remaining, ["checkpoint_step000002.pt", "checkpoint_step000006.pt", "checkpoint_step000007.pt"])
            with self.assertRaisesRegex(ValueError, "same data"):
                Trainer(tiny_model(), Data(), lambda samples: samples[0],
                        cfg=dict(first_cfg, resume_from=str(resume_path), max_steps=8))
