"""Full spatial dense BERT contracts, gradients, serving and strict continuation."""
import copy
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.train_vjepa_policy import parse_args
from scripts.cache_rmbench_history_features import build_cache
from vjepa_policy.datasets import DEFAULT_PROMPT
from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
from vjepa_policy.history_features import encoder_signature
from vjepa_policy.memory_config import full_history_config, contract_from_checkpoint_metadata
from vjepa_policy.memory_training import validate_history_args
from vjepa_policy.packed_history_features import PackedHistoryFeatureCache, pack_frozen_cache
from vjepa_policy.policy_arch import load_policy_weights, memory_checkpoint_metadata
from vjepa_policy.policy_serving.memory import MemoryPolicyServing
from vjepa_policy.policy_serving.memory_client import MemorySessionClient, HistoryCollector
from vjepa_policy.text_cache import text_cache_filename
from test_memory_bert import bert_model
from test_memory_v0_model import batch, enable_attention_gates
from test_memory_v0_online import IdentityNormalizer, block
from test_memory_v0_training import write_fixture, ToyFrozenEncoder
from test_shared_history import shared_batch
from test_packed_history_features import compact_batch
import test_full_training_resume as resume_tests


def model():
    value = bert_model()
    value.config = full_history_config("v2_full", "bert")
    return value


def bert_batch():
    value = batch()
    value["language"] = torch.randn(2, 128, 768)
    value["language_mask"] = torch.ones(2, 128, dtype=torch.bool)
    return value


class FullBertTests(unittest.TestCase):
    def test_versions_roundtrip_without_redefining_existing_schemas(self):
        for version, pool in (("v0", 1), ("v1", 2), ("v2", 2), ("v2_full", 1)):
            for text, width in (("t5", 4096), ("bert", 768)):
                config = full_history_config(version, text)
                self.assertEqual((config.history_pool_size, config.text_dim), (pool, width))
                self.assertEqual(config.text_encoder, text)
                self.assertEqual(contract_from_checkpoint_metadata(config.checkpoint_metadata()), config)
        config = full_history_config("v2_full", "bert")
        self.assertEqual((config.history_tokens_per_block, config.future_tokens), (588, 2352))
        self.assertEqual(config.observed_steps(7), (0, 4, 7))
        self.assertEqual(config.decision_steps(49), tuple(range(49)))
        invalid = config.to_dict()
        invalid["history_spatial_pool"] = 2
        with self.assertRaises(ValueError):
            type(config).from_dict(invalid)

    def test_cli_selects_dense_full_grid_and_bert_together(self):
        args = parse_args(["--checkpoint", "unused", "--text-cache-dir", "unused",
                           "--policy-arch", "full_history", "--history-model-version", "v2_full",
                           "--text-encoder", "bert", "--history-attention-backend", "flex",
                           "--history-shared-prefix", "--history-action-checkpointing"])
        config = validate_history_args(args)
        self.assertEqual(config.model_version, "v2_full_bert")
        self.assertEqual(args.lang_dim, 768)
        self.assertFalse(args.require_full_future_window)
        self.assertEqual(config.execution_interval_steps, 16)

    def test_dense_bert_frozen_cache_and_packed_reader_preserve_14_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_fixture(root)
            args.history_model_version, args.text_encoder, args.lang_dim = "v2_full", "bert", 768
            args.history_episode_ids, args.history_max_decision_step = ["0"], 5
            prompt = DEFAULT_PROMPT.format(task="Move the block back.")
            torch.save(dict(context=torch.ones(128, 768), mask=torch.ones(128, dtype=torch.bool)),
                       root / "text" / text_cache_filename(prompt, 128, "bert"))
            data = cached = packed = None
            try:
                data = RMBenchHistoryDataset(root / "manifest.json", args)
                # The cache contract describes uncompressed frozen tensors and
                # is shared with pooled v2; the Predictor contract stays v2_full.
                self.assertEqual(data.config.model_version, "v2_bert")
                meta = build_cache(data, ToyFrozenEncoder(), root / "features",
                                   encoder_metadata=encoder_signature(args), device="cpu")
                self.assertEqual(set(meta["episodes"]["0"]["observations"]), set(map(str, range(6))))
                args.history_feature_cache = str(root / "features/manifest.json")
                cached = RMBenchHistoryDataset(root / "manifest.json", args)
                sample = cached[5]
                self.assertEqual(sample["history_latents"].shape, (3, 588, 1024))
                self.assertEqual(sample["language"].shape, (128, 768))
                pack_frozen_cache(args.history_feature_cache, root / "packed")
                packed = PackedHistoryFeatureCache(cached.feature_cache, root / "packed/manifest.json")
                torch.testing.assert_close(packed.rows("0", "observations", [0, 4, 5]),
                                           sample["history_latents"], atol=0, rtol=0)
                self.assertEqual(packed.fingerprint, cached.feature_cache.fingerprint)
            finally:
                if data is not None:
                    data.close()
                if cached is not None:
                    cached.close()
                del packed, cached, data
                gc.collect()

    def test_shared_compact_full_grid_bert_matches_independent_loss_and_gradients(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                torch.manual_seed(31)
                reference, candidate = model(), model()
                enable_attention_gates(reference.action_expert)
                candidate.load_state_dict(reference.state_dict())
                candidate.predictor.activation_checkpointing_blocks = 3
                candidate.action_expert.checkpoint_memory_blocks = True
                candidate.share_history_prefixes = True
                value = shared_batch()
                value["language"] = torch.randn(1, 8, 768).expand(4, -1, -1).clone()
                value["language_mask"] = torch.ones(4, 8, dtype=torch.bool)
                value["action_is_pad"][1, 7:] = True
                if mixed:
                    value["episode_id"] = ["a", "b", "c", "d"]
                noise, time = torch.randn_like(value["action"]), torch.tensor([.1, .3, .5, .7])
                a = reference(value, flow_noise=noise, flow_time=time)[0]
                b = candidate(compact_batch(value), flow_noise=noise, flow_time=time)[0]
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                a.backward()
                b.backward()
                for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, atol=6e-6, rtol=6e-5, msg=name)
                z = value["history_latents"]
                self.assertIs(candidate.predictor.compress_history(z), z)

    def test_new_checkpoint_and_online_client_negotiate_full_bert_schema(self):
        value = model().eval()
        payload = dict(predictor=value.predictor.state_dict(),
                       action_expert=value.action_expert.state_dict(),
                       **memory_checkpoint_metadata(value))
        restored = model()
        load_policy_weights(restored, payload)
        old = bert_model()
        with patch.object(old.predictor, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_policy_weights(old, payload)
            load.assert_not_called()
        text = lambda _: (torch.ones(1, 128, 768), torch.ones(1, 128, dtype=torch.bool))
        service = MemoryPolicyServing(value, IdentityNormalizer(), text)
        client = MemorySessionClient(service)
        self.assertEqual(client.config, value.config)
        collector = HistoryCollector(config=client.config)
        self.assertEqual(collector.config.history_tokens_per_block, 588)
        self.assertTrue(client.reset("episode", "whole task")["ok"])
        with patch.object(service, "_encode_block",
                          return_value=(torch.randn(1, 12, 64), torch.zeros(1, 96))):
            reply = service.infer(dict(op="append", session_id=client.session_id, request_id="extra",
                                       base_revision=0, blocks=[block(0)]))
        self.assertTrue(reply["ok"], reply)
        client.close()


class FullBertResumeTests(resume_tests.FullStateTests):
    def test_uninterrupted_matches_resumed_across_epoch_boundary(self):
        # Exercise actual AdamW/RNG/cursor/schedule restoration with the new
        # model metadata and 768-wide frozen text, including an epoch boundary.
        with patch("test_full_training_resume.tiny_model", model), \
                patch("test_full_training_resume.tiny_batch", bert_batch):
            super().test_uninterrupted_matches_resumed_across_epoch_boundary()
