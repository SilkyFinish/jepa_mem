"""Lossless storage, compact-history gradients and unchanged resume identity."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from vjepa_policy.history_features import fingerprint, sha256_file
from vjepa_policy.packed_history_features import (
    PackedHistoryFeatureCache, PackedHistoryBatchCollator, pack_frozen_cache, warm_packed_observations,
)
from vjepa_policy.datasets.rmbench_history import HistoryBatchCollator
from vjepa_policy.memory_config import FullHistoryV2Config
from test_shared_history import shared_batch
from test_memory_v0_model import enable_attention_gates
from test_memory_v1_spatial import model


def compact_batch(batch):
    result = dict(batch)
    bank, lookup = [], {}
    indices = torch.zeros_like(batch["history_end_step"])
    for row in range(len(batch["decision_step"])):
        episode = batch["episode_id"][row]
        for column in range(int(batch["history_valid"][row].sum())):
            end = int(batch["history_end_step"][row, column])
            key = episode, end
            if key not in lookup:
                lookup[key] = len(bank)
                bank.append(batch["history_latents"][row, column])
            indices[row, column] = lookup[key]
    result["history_latents"] = torch.stack(bank)
    result["history_latent_index"] = indices
    return result


class CompactHistoryTests(unittest.TestCase):
    def test_losses_and_every_gradient_match_dense_inputs(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                torch.manual_seed(53)
                reference, candidate = model(checkpointing=True), model(checkpointing=True)
                reference.config = candidate.config = FullHistoryV2Config()
                enable_attention_gates(reference.action_expert)
                candidate.load_state_dict(reference.state_dict())
                for policy in (reference, candidate):
                    policy.share_history_prefixes = True
                    policy.action_expert.checkpoint_memory_blocks = True
                batch = shared_batch()
                if mixed:
                    batch["episode_id"] = ["a", "b", "c", "d"]
                    batch["action_is_pad"][1, 7:] = True
                    batch["action_is_pad"][2] = True
                packed = compact_batch(batch)
                self.assertLess(packed["history_latents"].numel(), batch["history_latents"].numel())
                noise, time = torch.randn_like(batch["action"]), torch.tensor([.1, .3, .5, .7])
                a, am = reference(batch, flow_noise=noise, flow_time=time)
                b, bm = candidate(packed, flow_noise=noise, flow_time=time)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                for key in am:
                    torch.testing.assert_close(am[key], bm[key], atol=0, rtol=0)
                a.backward(); b.backward()
                for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)

    def test_bad_compact_indices_fail(self):
        policy = model()
        policy.config = FullHistoryV2Config()
        batch = compact_batch(shared_batch())
        batch["history_latent_index"][0, 0] = len(batch["history_latents"])
        with self.assertRaisesRegex(ValueError, "compact"):
            policy(batch)


class PackedStorageTests(unittest.TestCase):
    def test_distributed_prewarm_rejected_on_every_rank_before_io(self):
        for rank in ("0", "1"):
            with self.subTest(rank=rank), patch.dict(os.environ, WORLD_SIZE="2", RANK=rank):
                with self.assertRaisesRegex(ValueError, "before torchrun"):
                    warm_packed_observations("/nonexistent/manifest.json")

    def test_standalone_warm_reads_only_observations_and_does_not_repack(self):
        logical, path = self.fixture(Path(self.temporary()))
        storage = json.loads(path.read_text())
        before = path.read_bytes()
        expected = sum(row["observations"]["file_bytes"] for row in storage["episodes"].values())
        # Future targets are absent: observation warmup must not read them.
        for row in storage["episodes"].values():
            (path.parent / row["targets"]["file"]).unlink()
        with patch.dict(os.environ, WORLD_SIZE="1"):
            result = warm_packed_observations(path, reserve_bytes=0)
        self.assertEqual(result["bytes"], expected)
        self.assertEqual(before, path.read_bytes())

    def temporary(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        metadata = {"format": "rmbench_frozen_history_v0", "dtype": "float32", "episodes": {}}
        torch.manual_seed(7)
        for episode in ("a", "b"):
            row = {"frames": 6, "observations": {}, "targets": {}}
            for kind, tokens in (("observations", 2), ("targets", 8)):
                for end in range(6):
                    path = source / f"{episode}_{kind}_{end}.pt"
                    value = torch.randn(tokens, 1024)
                    torch.save(value, path)
                    row[kind][str(end)] = {"file": path.name, "sha256": sha256_file(path),
                                          "frame_indices": [end]}
            metadata["episodes"][episode] = row
        manifest = source / "manifest.json"
        manifest.write_text(json.dumps(metadata))
        logical = SimpleNamespace(path=manifest, metadata=metadata, fingerprint=fingerprint(metadata),
                                  config=SimpleNamespace(tokens_per_block=2, future_tokens=8,
                                                         observation_frames=lambda end: [end], target_offsets=(0,)))
        packed = root / "packed"
        pack_frozen_cache(manifest, packed)
        return logical, packed / "manifest.json"

    def test_exact_reconstruction_and_provenance_and_resume_packing(self):
        temporary = self.temporary()
        logical, path = self.fixture(Path(temporary))
        cache = PackedHistoryFeatureCache(logical, path)
        self.assertEqual(cache.fingerprint, logical.fingerprint)
        for episode, row in logical.metadata["episodes"].items():
            for kind in ("observations", "targets"):
                for ends in ([0, 1, 2], [5, 0, 3]):
                    actual = cache.rows(episode, kind, ends)
                    expected = torch.stack([torch.load(logical.path.parent / row[kind][str(end)]["file"],
                                                      weights_only=True) for end in ends])
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        before = path.read_bytes()
        pack_frozen_cache(logical.path, path.parent)
        self.assertEqual(before, path.read_bytes())

    def test_tampering_and_mismatched_source_rejected(self):
        temporary = self.temporary()
        logical, path = self.fixture(Path(temporary))
        cache = PackedHistoryFeatureCache(logical, path)
        record = cache.storage["episodes"]["a"]["targets"]
        array = np.load(path.parent / record["file"], mmap_mode="r+")
        array[0, 0, 0] += 1
        array.flush()
        with self.assertRaisesRegex(ValueError, "Corrupt packed tensor"):
            cache.rows("a", "targets", [0])
        changed = copy.copy(logical)
        changed.fingerprint = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            PackedHistoryFeatureCache(changed, path)

    def test_collator_deduplicates_without_reordering_or_cross_episode_sharing(self):
        temporary = self.temporary()
        logical, path = self.fixture(Path(temporary))
        cache = PackedHistoryFeatureCache(logical, path)
        samples, expected = [], []
        for episode, decision, ends in (("a", 3, (0, 1, 3)), ("a", 2, (0, 1, 2)),
                                         ("b", 2, (0, 2))):
            samples.append({"batch_schema": "rmbench_history_v0", "episode_id": episode,
                            "sample_key": f"{episode}:{decision}", "history_state": torch.zeros(len(ends), 14),
                            "history_end_step": torch.tensor(ends), "history_valid": torch.ones(len(ends), dtype=torch.bool),
                            "decision_step": torch.tensor(decision), **cache.load(episode, decision, ends)})
            row = logical.metadata["episodes"][episode]
            expected.append(torch.stack([torch.load(logical.path.parent / row["observations"][str(end)]["file"],
                                                    weights_only=True) for end in ends]))
        result = PackedHistoryBatchCollator(cache, HistoryBatchCollator())(samples)
        self.assertEqual(result["sample_key"], ["a:3", "a:2", "b:2"])
        self.assertEqual(len(result["history_latents"]), 6)
        for i, value in enumerate(expected):
            actual = result["history_latents"][result["history_latent_index"][i, :len(value)]]
            torch.testing.assert_close(actual, value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
