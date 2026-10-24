"""Spatial compression, version safety, gradients and online equivalence."""
import copy
import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm", "accelerate", "lerobot")):
    raise unittest.SkipTest("v1 model tests require the installed policy environment")

import torch
from torch import nn

from scripts.train_vjepa_policy import parse_args
from test_memory_v0_model import predictor_options, action_options, batch, encoder, enable_attention_gates
from test_memory_v0_online import records
from vjepa_policy.memory_config import FullHistoryV0Config, FullHistoryV1Config, contract_from_checkpoint_metadata
from vjepa_policy.models.memory_predictor import MemoryWorldModelPredictor
from vjepa_policy.models.memory_policy import FullHistoryVJEPAPolicy
from vjepa_policy.models.memory_action_expert import MemoryActionExpert
from vjepa_policy.models.history_cache import HistoryKVCache
from vjepa_policy.policy_arch import memory_checkpoint_metadata, load_policy_weights
from vjepa_policy.policy_serving.memory import MemoryPolicyServing
from vjepa_policy.state import pack_normalized_state


def model(version="v1", checkpointing=False, video=False, backend="segmented"):
    config = FullHistoryV1Config() if version == "v1" else FullHistoryV0Config()
    predictor = MemoryWorldModelPredictor(
        **predictor_options(use_activation_checkpointing=checkpointing),
        history_spatial_pool=config.history_pool_size, attention_backend=backend,
    )
    expert = MemoryActionExpert(rope_options=predictor.rope_options, attention_backend=backend, **action_options())
    return FullHistoryVJEPAPolicy(encoder() if video else nn.Linear(1, 1), predictor, expert,
                                 config=config, encoder_batch_size=2)


class SpatialHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)

    def test_version_cli_and_frozen_contract(self):
        base, compressed = FullHistoryV0Config(), FullHistoryV1Config()
        self.assertEqual((base.history_tokens_per_block, compressed.history_tokens_per_block), (588, 147))
        self.assertEqual((compressed.tokens_per_block, compressed.future_tokens), (588, 2352))
        self.assertEqual(base.sample_indices("14", 353, 320), compressed.sample_indices("14", 353, 320))
        self.assertEqual(contract_from_checkpoint_metadata(compressed.checkpoint_metadata()), compressed)
        bad = compressed.to_dict()
        bad["history_spatial_pool"] = 4
        with self.assertRaises(ValueError):
            FullHistoryV1Config.from_dict(bad)
        required = ["--checkpoint", "encoder.pt", "--text-cache-dir", "text", "--dataset-dirs", "data", "--output-dir", "run"]
        self.assertEqual(parse_args([*required, "--policy-arch", "full_history"]).history_model_version, "v0")
        self.assertEqual(parse_args([*required, "--policy-arch", "full_history", "--history-model-version", "v1"]).history_model_version, "v1")
        with self.assertRaises(SystemExit):
            parse_args([*required, "--history-model-version", "v1"])

    def test_pooling_preserves_view_time_and_original_coordinate_basis(self):
        p = MemoryWorldModelPredictor(**dict(predictor_options(), img_size=64), history_spatial_pool=2)
        z = torch.arange(2*5*3*4*4*64, dtype=torch.float32).reshape(2, 5, 48, 64)
        expected = torch.empty(2, 5, 3, 2, 2, 64)
        images = z.reshape(2, 5, 3, 4, 4, 64)
        for y in range(2):
            for x in range(2):
                expected[:, :, :, y, x] = images[:, :, :, 2*y:2*y+2, 2*x:2*x+2].mean((3, 4))
        torch.testing.assert_close(p.compress_history(z), expected.reshape(2, 5, 12, 64), atol=0, rtol=0)
        ends = torch.tensor([[0, 4]])
        layout = p.history_layout(ends, torch.ones_like(ends, dtype=torch.bool))
        torch.testing.assert_close(layout.rope_coords[0, :4], torch.tensor([[0., .5, .5], [0., .5, 2.5], [0., 2.5, .5], [0., 2.5, 2.5]]))
        self.assertEqual(layout.view_id[0, :12].tolist(), [0]*4 + [1]*4 + [2]*4)
        self.assertEqual(layout.rope_coords[0, 12, 0].item(), .5)
        self.assertEqual(p.future_tokens, 192)

    def test_v0_keeps_identity_and_v1_changes_only_history_token_count(self):
        b, m0, m1 = batch(), model("v0"), model()
        m1.load_state_dict(m0.state_dict(), strict=True)
        self.assertIs(m0.predictor.compress_history(b["history_latents"]), b["history_latents"])
        with torch.no_grad():
            h0, f0, *_ = m0._encode_context(b)
            h1, f1, *_ = m1._encode_context(b)
        self.assertEqual(h0.layout.valid.shape, (2, 60))
        self.assertEqual(h1.layout.valid.shape, (2, 15))
        self.assertEqual(f0.prediction.shape, f1.prediction.shape)
        self.assertEqual(f1.prediction.shape, (2, 48, 64))

    def test_dense_segmented_checkpointing_gradients_match(self):
        b = batch()
        reference = model(backend="dense").train()
        enable_attention_gates(reference.action_expert)
        candidate = model(checkpointing=True).train()
        candidate.load_state_dict(reference.state_dict(), strict=True)
        noise, time = torch.randn_like(b["action"]), torch.tensor([.2, .8])
        a = reference(b, flow_noise=noise, flow_time=time)[0]
        c = candidate(b, flow_noise=noise, flow_time=time)[0]
        torch.testing.assert_close(a, c, atol=3e-6, rtol=3e-5)
        a.backward()
        c.backward()
        for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
            if p.grad is not None:
                self.assertTrue(torch.isfinite(q.grad).all(), name)
                torch.testing.assert_close(p.grad, q.grad, atol=4e-6, rtol=4e-5, msg=name)
        self.assertGreater(candidate.predictor.predictor_embed.weight.grad.abs().sum().item(), 0)

    def test_padding_causality_and_other_episode_isolation(self):
        b, m = batch(), model().eval()
        altered = copy.deepcopy(b)
        altered["history_latents"][1, 1:] = float("nan")
        altered["history_state"][1, 1:] = float("nan")
        noise, time = torch.randn_like(b["action"]), torch.tensor([.3, .7])
        torch.testing.assert_close(m(b, flow_noise=noise, flow_time=time)[0], m(altered, flow_noise=noise, flow_time=time)[0], atol=0, rtol=0)
        before = m._encode_history(b)[0]
        altered["history_latents"][0, 3:] += 50
        altered["history_latents"][1, 0] -= 50
        after = m._encode_history(altered)[0]
        for old, new in zip(before.layers, after.layers):
            torch.testing.assert_close(old.key[0, :, :9], new.key[0, :, :9], atol=0, rtol=0)

    def test_incremental_kv_and_actions_equal_full_recompute(self):
        m, b = model().eval(), batch()
        enable_attention_gates(m.action_expert)
        p = m.predictor
        z = b["history_latents"][:1]
        states = pack_normalized_state(b["history_state"][:1])
        lang, mask = b["language"][:1], b["language_mask"][:1]
        for chunks in ((1, 2, 3, 4, 5), (1, 5)):
            cache = HistoryKVCache(depth=3, heads=2, head_dim=64, tokens_per_block=3,
                                   device="cpu", dtype=torch.float32, max_blocks=5, max_bytes=1024**2)
            start = 0
            for stop in chunks:
                ends = torch.arange(start, stop)[None]*4
                layout = p.history_layout(ends, torch.ones_like(ends, dtype=torch.bool), block_offset=start)
                delta = p.append_history(cache.snapshot(), z[:, start:stop], states[:, start:stop], lang, mask, layout)
                cache.commit(delta, records(start, states[:, start:stop]))
                ends = torch.arange(stop)[None]*4
                with torch.no_grad():
                    full = p.forward_history(z[:, :stop], states[:, :stop], lang, mask,
                                             p.history_layout(ends, torch.ones_like(ends, dtype=torch.bool)))
                for actual, expected in zip(cache.snapshot().layers, full.layers):
                    torch.testing.assert_close(actual.key, expected.key, atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(actual.value, expected.value, atol=3e-6, rtol=3e-5)
                start = stop
            noise = torch.randn(1, 32, 14)
            args = (states[:, -1], lang, mask, torch.tensor([16]))
            a = m.predict_from_history(cache.snapshot(), *args, initial_noise=noise)
            c = m.predict_from_history(full, *args, initial_noise=noise)
            torch.testing.assert_close(a, c, atol=4e-6, rtol=4e-5)
            self.assertEqual(cache.valid_length, 15)

    def test_checkpoint_version_rejected_before_mutation_and_roundtrip(self):
        m, b = model().eval(), batch()
        payload = {"predictor": m.predictor.state_dict(), "action_expert": m.action_expert.state_dict(),
                   **memory_checkpoint_metadata(m)}
        restored = model().eval()
        load_policy_weights(restored, payload)
        noise = torch.randn_like(b["action"])
        torch.testing.assert_close(m.predict_action(b, initial_noise=noise), restored.predict_action(b, initial_noise=noise), atol=0, rtol=0)
        incompatible = model("v0")
        with patch.object(incompatible.predictor, "load_state_dict") as loader, self.assertRaisesRegex(ValueError, "contract"):
            load_policy_weights(incompatible, payload)
        loader.assert_not_called()

    def test_raw_encoder_and_cache_paths_match_with_full_resolution_targets(self):
        m, b = model(video=True).eval(), batch()
        del b["history_latents"], b["future_targets"]
        b["history_video"] = torch.randn(2, 5, 3, 3, 2, 32, 32)
        b["target_video"] = torch.randn(2, 3, 3, 10, 32, 32)
        z, target = m.encode_observed_blocks(b), m.encode_local_future_targets(b)
        self.assertEqual(z.shape, (2, 5, 12, 64))
        self.assertEqual(target.shape, (2, 48, 64))
        cached = {k: v for k, v in b.items() if k not in ("history_video", "target_video")}
        cached.update(history_latents=z, future_targets=target)
        noise, time = torch.randn_like(b["action"]), torch.tensor([.3, .7])
        raw = m(b, flow_noise=noise, flow_time=time)[0]
        saved = m(cached, flow_noise=noise, flow_time=time)[0]
        torch.testing.assert_close(raw, saved, atol=0, rtol=0)

    def test_serving_uses_compressed_layout_and_advertises_v1(self):
        m = model().eval()
        serving = MemoryPolicyServing(m, None, lambda _: None)
        self.assertEqual(serving.metadata["model_version"], "v1")
        self.assertEqual(serving._layout(5).valid.shape, (1, 15))
        language, mask = torch.ones(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
        serving.text_loader = lambda _: (language, mask)
        reply, session = serving._reset("s", {"protocol": "rmbench_memory_v0", "sampling_config": FullHistoryV0Config().to_dict(),
                                              "episode_id": "14", "instruction": "Return the block."}, None)
        self.assertEqual(session.cache.tokens_per_block, 3)
        self.assertEqual(reply["memory_schema"]["model_version"], "v1")


if __name__ == "__main__":
    unittest.main()
