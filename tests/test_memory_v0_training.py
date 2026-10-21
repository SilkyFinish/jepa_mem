"""M3 native data/cache, training dispatch and checkpoint integration tests."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm", "h5py", "lerobot", "accelerate")):
    raise unittest.SkipTest("history training tests require the installed policy environment")

import h5py
import numpy as np
import torch
from torch import nn

from scripts.train_vjepa_policy import parse_args
from scripts.cache_rmbench_history_features import build_cache
from vjepa_policy.datasets import DEFAULT_PROMPT
from vjepa_policy.datasets.rmbench import JOINT_FIELDS
from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset, HistoryBatchCollator
from vjepa_policy.history_features import encoder_signature, sha256_file
from vjepa_policy.memory_training import validate_history_args, validate_predictor_transfer, validate_reference_capacity
from vjepa_policy.policy_arch import (
    require_checkpoint_arch, load_policy_weights, memory_checkpoint_metadata,
    capabilities, HISTORY_CAPABILITIES,
)
from vjepa_policy.policy_serving.libero import PolicyServingConfig
from vjepa_policy.trainer import Trainer, _EpochRandomSampler
from test_memory_v0_model import model as tiny_model, batch as tiny_batch


class ToyFrozenEncoder(nn.Module):
    patch_size, tubelet_size = 16, 2

    def forward(self, video, mask=None):
        # Distinguishable temporal/view/feature channels, with actual tubelet ordering.
        values = video.mean((1, 3, 4)).reshape(len(video), -1, 2).mean(-1)
        values = values.repeat_interleave(196, dim=1)
        return values[..., None] + torch.arange(1024, device=video.device).float()[None, None] / 100


def write_fixture(root):
    codec = root / "codec.py"
    codec.write_text("import numpy as np\ndef decode_image_bit(value):\n    return np.broadcast_to(np.asarray(value,dtype=np.uint8),(240,320,3)).copy()\n")
    stats = {
        "version": 1,
        "features": {"action": {"type": "ACTION", "shape": [14]},
                     "observation.state": {"type": "STATE", "shape": [14]}},
        "normalization": {"action": "QUANTILE", "state": "QUANTILE", "clip_values": False},
        "stats": {key: {"q01": [0.] * 14, "q99": [100.] * 14}
                  for key in ("action", "observation.state")}, "metadata": {},
    }
    (root / "stats.json").write_text(json.dumps(stats))
    rows = []
    for episode, frames in ((0, 65), (1, 49)):
        path = root / f"episode{episode}.hdf5"
        state = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14) / 20
        action = np.concatenate((state[1:], state[-1:]))
        with h5py.File(path, "w") as handle:
            for group, vector in (("state", state), ("action", action)):
                for field, indices in zip(JOINT_FIELDS, (slice(0,6), slice(6,7), slice(7,13), slice(13,14))):
                    handle.create_dataset(f"{group}/{field}", data=vector[:, indices])
            for view, camera in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
                rgb = np.stack([np.full(3, 30 + view * 50 + t, np.uint8) for t in range(frames)])
                handle.create_dataset(f"vision/{camera}/colors", data=rgb)
        rows.append(dict(path=str(path), source_episode=episode, frames=frames,
                         instruction="Move the block back.", sha256=sha256_file(path)))
    manifest = dict(action_contract="aloha_absolute_joint_14", codec_path=str(codec),
                    statistics=str(root / "stats.json"), episodes=rows)
    (root / "manifest.json").write_text(json.dumps(manifest))
    text_dir = root / "text"
    text_dir.mkdir()
    digest = hashlib.sha256(DEFAULT_PROMPT.format(task=rows[0]["instruction"]).encode()).hexdigest()
    torch.save({"context": torch.ones(128,4096), "mask": torch.ones(128,dtype=torch.bool)},
               text_dir / f"{digest}.t5_len128.pt")
    (root / "encoder.pt").write_bytes(b"fixture weights; no real encoder is loaded")
    return parse_args([
        "--policy-arch", "full_history", "--checkpoint", str(root / "encoder.pt"),
        "--text-cache-dir", str(text_dir), "--dataset-dirs", str(root),
        "--output-dir", str(root / "run"),
    ])


class HistoryDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = write_fixture(self.root)
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:
            dataset.close()
        self.temporary.cleanup()

    def dataset(self, **kwargs):
        args = copy.copy(self.args)
        vars(args).update(kwargs)
        dataset = RMBenchHistoryDataset(self.root / "manifest.json", args)
        self.datasets.append(dataset)
        return dataset

    def test_native_rgb_timing_state_action_alignment_and_right_padding(self):
        data = self.dataset()
        self.assertEqual(data.index, [(0,0), (0,16), (0,32), (1,0), (1,16)])
        sample = data[1]
        self.assertEqual(sample["history_end_step"].tolist(), [0,4,8,12,16])
        self.assertEqual(sample["history_video"].shape, (5,3,3,2,224,224))
        # Head at block 16 uses frame 12 and 16; view colors also remain ordered.
        for view in range(3):
            for slot, t in enumerate((12,16)):
                expected = (30 + view * 50 + t) / 255 * 2 - 1
                torch.testing.assert_close(sample["history_video"][4,view,:,slot],
                                           torch.full((3,224,224),expected),atol=3e-7,rtol=0)
        torch.testing.assert_close(sample["current_state"], sample["history_state"][-1])
        raw_next = torch.arange(17 * 14,18 * 14).float()/20
        torch.testing.assert_close(sample["action"][0], raw_next/100*2-1)
        batch = HistoryBatchCollator()([data[0], sample])
        self.assertEqual(batch["history_valid"].tolist(), [[True,False,False,False,False],[True]*5])
        self.assertTrue((batch["history_video"][0,1:] == 0).all())
        self.assertEqual(batch["sample_key"], ["0:0","0:16"])
        self.assertEqual(batch["future_valid"].shape, (2,2352))
        torch.testing.assert_close(data[0]["history_video"][0,:,:,0], data[0]["history_video"][0,:,:,1])

    def test_selection_is_explicit_and_capacity_never_truncates(self):
        data = self.dataset(history_episode_ids=["1"], history_max_decision_step=16)
        self.assertEqual(data.history_lengths, [1,5])
        self.assertTrue(data.selection["diagnostic_subset"])
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.dataset(max_history_blocks=5)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.dataset(history_episode_ids=["999"])
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            self.dataset(history_max_decision_step=15)

    def test_cache_roundtrip_matches_raw_encoder_and_checks_provenance(self):
        data = self.dataset(history_episode_ids=["0"], history_max_decision_step=16)
        output = self.root / "features"
        signature = encoder_signature(self.args)
        metadata = build_cache(data, ToyFrozenEncoder(), output, encoder_metadata=signature, device="cpu")
        cached = self.dataset(history_episode_ids=["0"], history_max_decision_step=16,
                              history_feature_cache=str(output / "manifest.json"))
        from vjepa_policy.models.vjepa_policy import encode_video_views
        from torch.nn import functional as F
        sample, feature = data[1], cached[1]
        expected_history = encode_video_views(ToyFrozenEncoder(), sample["history_video"])
        expected_target = encode_video_views(ToyFrozenEncoder(), sample["target_video"][None])
        expected_target = F.layer_norm(expected_target,(1024,))[:,588:]
        torch.testing.assert_close(feature["history_latents"],expected_history,atol=0,rtol=0)
        torch.testing.assert_close(feature["future_targets"],expected_target[0],atol=0,rtol=0)
        self.assertGreater(float(feature["history_latents"].mean()),1.0)  # no extra observation LN
        self.assertLess(abs(float(feature["future_targets"].mean())),1e-6)
        self.assertFalse(feature["history_latents"].requires_grad)
        # This includes no trainable Predictor data.
        self.assertEqual(set(metadata["episodes"]["0"]), {"source_sha256","frames","observations","targets"})
        original = (output / "manifest.json").read_text()
        for field, wrong in (("target_layer_norm",False),("dtype","float16"),("encoder",{}),("encoding_code",{})):
            altered = json.loads(original)
            altered[field] = wrong
            (output / "manifest.json").write_text(json.dumps(altered))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.dataset(history_episode_ids=["0"],history_max_decision_step=16,
                             history_feature_cache=str(output / "manifest.json"))
        (output / "manifest.json").write_text(original)
        with self.assertRaisesRegex(ValueError,"Missing (future|observed)"):
            self.dataset(history_episode_ids=["0"],history_feature_cache=str(output / "manifest.json"))
        entry = cached.feature_cache.metadata["episodes"]["0"]["observations"]["0"]
        entry["frame_indices"] = [0,4]
        with self.assertRaisesRegex(ValueError,"frame indices"):
            cached[0]

    def test_bucket_sampler_coverage_determinism_and_worker_restarts(self):
        data = self.dataset()
        a = _EpochRandomSampler(data, seed=13,bucket_size=4,batch_size=2)
        b = _EpochRandomSampler(data, seed=13,bucket_size=4,batch_size=2)
        order = list(a)
        self.assertEqual(order,list(b))
        self.assertEqual(sorted(order),list(range(len(data))))
        a.set_position(2)
        self.assertEqual(list(a),order[2:])
        a.start_epoch(1)
        b.start_epoch(1)
        self.assertEqual(list(a),list(b))


class HistoryDispatchAndCheckpointTests(unittest.TestCase):
    def test_cli_defaults_legacy_and_explicit_v0_contract(self):
        required = ["--checkpoint","unused","--text-cache-dir","unused"]
        legacy = parse_args(required)
        self.assertEqual(legacy.policy_arch,"legacy")
        self.assertEqual((legacy.action_dim,legacy.context_len),(7,32))
        memory = parse_args([*required,"--policy-arch","full_history"])
        validate_history_args(memory)
        self.assertEqual((memory.action_dim,memory.context_len,memory.batch_size),(14,128,1))
        memory.action_dim = 7
        with self.assertRaises(ValueError):
            validate_history_args(memory)
        from vjepa_policy.models.memory_policy import build_full_history_policy
        self.assertEqual(capabilities(build_full_history_policy),HISTORY_CAPABILITIES)

    def test_legacy_builder_remains_default_with_custom_data_and_encoder_hooks(self):
        from scripts import train_vjepa_policy as entry
        from test_memory_v0_model import encoder
        from vjepa_policy.models.vjepa_policy import VJEPAPolicy
        args = parse_args([
            "--checkpoint","unused","--text-cache-dir","unused",
            "--views","head","left","right","--crop-size","32",
            "--pred-depth","3","--pred-embed-dim","128","--pred-num-heads","2",
            "--action-num-layers","3","--action-hidden-size","64","--condition-num-heads","2",
            "--lang-dim","16","--context-len","4","--proprio-dim","14","--action-dim","14",
            "--activation-checkpointing-blocks","0",
        ])
        with patch.object(entry, "run_training") as run:
            entry.main(args=args,encoder_builder=lambda args:encoder(),
                       dataset_collator_builder=lambda args,**kw:([0],None))
        built=run.call_args.args[1]
        self.assertIsInstance(built,VJEPAPolicy)
        self.assertEqual(built.predictor.proprio_dim,96)
        # Original checkpoints without new architecture markers remain valid.
        require_checkpoint_arch({"predictor":built.predictor.state_dict()},"legacy")

    def test_reference_capacity_rejects_long_prefixes_before_training(self):
        from types import SimpleNamespace
        validate_reference_capacity(SimpleNamespace(history_lengths=[1,5]),1)
        for lengths,batch in (([9],1),([5],2)):
            with self.assertRaisesRegex(ValueError,"never be truncated"):
                validate_reference_capacity(SimpleNamespace(history_lengths=lengths),batch)

    def test_schema_and_rope_rejected_before_weights_are_mutated(self):
        model = tiny_model()
        payload = {"predictor":model.predictor.state_dict(),"action_expert":model.action_expert.state_dict(),
                   **memory_checkpoint_metadata(model)}
        load_policy_weights(tiny_model(),payload)
        with self.assertRaisesRegex(ValueError,"cannot be loaded"):
            require_checkpoint_arch(payload,"legacy")
        with self.assertRaisesRegex(ValueError,"cannot be loaded"):
            load_policy_weights(model,{"predictor":model.predictor.state_dict()})
        altered = copy.deepcopy(payload)
        altered["memory_training"]["model_semantics"]["rope_options"]["interpolate_rope"] = False
        with patch.object(model.predictor,"load_state_dict") as load, self.assertRaisesRegex(ValueError,"semantics"):
            load_policy_weights(model,altered)
        load.assert_not_called()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"memory.pt"
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,"cannot be loaded"):
                PolicyServingConfig.from_checkpoint(path)

    def test_predictor_transfer_requires_rotary_and_packed_state_metadata(self):
        predictor = tiny_model().predictor
        topology = {"packed_proprio_dim":96,"predictor_rope_frequency_pairing":"legacy",
                    "predictor_interpolate_rope":True}
        validate_predictor_transfer({"model_topology":topology},predictor)
        with self.assertRaisesRegex(ValueError,"metadata"):
            validate_predictor_transfer({},predictor)
        topology["predictor_rope_frequency_pairing"]="corrected"
        with self.assertRaises(ValueError):
            validate_predictor_transfer({"model_topology":topology},predictor)

    def test_actual_trainer_dict_batch_update_save_strict_reload_and_resume(self):
        class Data:
            history_lengths = [5,5]
            def __len__(self): return 2
            def __getitem__(self,index): return tiny_batch()
        with tempfile.TemporaryDirectory() as root, patch.object(Trainer, "_init_wandb", return_value=None):
            cfg=dict(output_dir=root,batch_size=1,num_workers=0,gradient_accumulation_steps=1,
                     mixed_precision="no",lr=1e-3,weight_decay=0.01,max_steps=2,num_epochs=1,
                     max_grad_norm=1.0,log_every=1,save_every=0,data_seed=7,train_seed=7,
                     wandb_project=None,wandb_name=None,recycle_workers_every=1,history_bucket_size=4)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            m = tiny_model()
            before = m.predictor.predictor_proj.weight.detach().clone()
            trainer = Trainer(m,Data(),lambda samples:samples[0],cfg=cfg)
            trainer.train()
            self.assertEqual(trainer.global_step,2)
            self.assertFalse(torch.equal(before,m.predictor.predictor_proj.weight))
            payload = torch.load(Path(root)/"checkpoint_step000002.pt",map_location="cpu",weights_only=False)
            self.assertEqual(payload["policy_arch"],"full_history")
            self.assertEqual(payload["model_version"],"v0")
            restored = tiny_model().eval()
            load_policy_weights(restored,payload)
            sample = tiny_batch()
            noise = torch.randn(2,32,14)
            torch.testing.assert_close(m.eval().predict_action(sample,initial_noise=noise),
                                       restored.predict_action(sample,initial_noise=noise),atol=0,rtol=0)
            resume_cfg = dict(cfg, output_dir=str(Path(root)/"resumed"),max_steps=3,
                              resume_from=str(Path(root)/"checkpoint_step000002.pt"))
            resumed = Trainer(tiny_model(),Data(),lambda samples:samples[0],cfg=resume_cfg)
            self.assertEqual(resumed.global_step,2)
            resumed.train()
            self.assertEqual(resumed.global_step,3)
            self.assertTrue((Path(root)/"resumed/checkpoint_step000003.pt").exists())


if __name__ == "__main__":
    unittest.main()
