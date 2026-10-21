"""M4 equivalence, transaction and session regressions (CPU, no external assets)."""
import copy
from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import patch

if importlib.util.find_spec("timm") is None:
    raise unittest.SkipTest("online memory tests need the policy environment")

import numpy as np
import torch

from test_memory_v0_model import model, batch, enable_attention_gates
from vjepa_policy.models.history_cache import HistoryKVCache, HistoryBlock, HistoryCacheError
from vjepa_policy.models.memory_layouts import visual_layout
from vjepa_policy.models.memory_types import HistoryEncoding, LayerKV
from vjepa_policy.policy_serving.memory import MemoryPolicyServing, PROTOCOL
from vjepa_policy.policy_serving.memory_client import HistoryCollector, MemorySessionClient
from vjepa_policy.state import pack_normalized_state


def layout(start, count):
    ends = torch.arange(start, start + count)[None] * 4
    return visual_layout(ends, torch.ones_like(ends, dtype=torch.bool),
                         grid_height=2, grid_width=2, num_views=3, block_offset=start)


def cache(max_blocks=5, max_bytes=1024**3):
    return HistoryKVCache(depth=3, heads=2, head_dim=64, tokens_per_block=12,
                          device="cpu", dtype=torch.float32, max_blocks=max_blocks, max_bytes=max_bytes)


def records(start, states):
    return tuple(HistoryBlock(i, i * 4, str(i), states[:, j])
                 for j, i in enumerate(range(start, start + states.shape[1])))


class OnlineOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = model().eval()
        enable_attention_gates(self.model.action_expert)
        b = batch()
        self.z = b["history_latents"][:1]
        self.states = pack_normalized_state(b["history_state"][:1])
        self.lang = b["language"][:1].masked_fill(~b["language_mask"][:1, :, None], 0)
        self.mask = b["language_mask"][:1]

    def append(self, c, start, stop):
        delta = self.model.predictor.append_history(
            c.snapshot(), self.z[:, start:stop], self.states[:, start:stop],
            self.lang, self.mask, layout(start, stop-start),
        )
        c.commit(delta, records(start, self.states[:, start:stop]))
        return delta

    def test_each_layer_future_and_action_equal_full_recompute(self):
        for chunks in ((1, 2, 3, 4, 5), (1, 5)):
            c, start = cache(), 0
            for stop in chunks:
                old = copy.deepcopy(c.snapshot())
                self.append(c, start, stop)
                with torch.no_grad():
                    ref = self.model.predictor.forward_history(self.z[:, :stop], self.states[:, :stop],
                                                               self.lang, self.mask, layout(0, stop))
                for index, (actual, expected) in enumerate(zip(c.snapshot().layers, ref.layers)):
                    torch.testing.assert_close(actual.key, expected.key, atol=2e-6, rtol=2e-5, msg=str(index))
                    torch.testing.assert_close(actual.value, expected.value, atol=2e-6, rtol=2e-5)
                    if old:
                        torch.testing.assert_close(actual.key[:, :, :start*12], old.layers[index].key, atol=0, rtol=0)
                start = stop
            before = copy.deepcopy(c.snapshot())
            noise = torch.randn(1, 32, 14)
            step = torch.tensor([16])
            with torch.no_grad():
                a = self.model.predict_from_history(c.snapshot(), self.states[:, -1], self.lang, self.mask,
                                                    step, initial_noise=noise)
                b = self.model.predict_from_history(ref, self.states[:, -1], self.lang, self.mask,
                                                    step, initial_noise=noise)
                f = self.model.predictor.query_future(c.snapshot(), self.states[:, -1], self.lang, self.mask,
                                                      step, self.model._future_layout(step))
                g = self.model.predictor.query_future(ref, self.states[:, -1], self.lang, self.mask,
                                                      step, self.model._future_layout(step))
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(f.prediction, g.prediction, atol=3e-6, rtol=3e-5)
            for x, y in zip(f.layers, g.layers):
                torch.testing.assert_close(x.key, y.key, atol=3e-6, rtol=3e-5)
            self.assertEqual(c.valid_length, 60)
            self.assertEqual(c.revision, len(chunks))
            for x, y in zip(before.layers, c.snapshot().layers):
                torch.testing.assert_close(x.key, y.key, atol=0, rtol=0)
                self.assertFalse(y.key.requires_grad)

    def test_capacity_and_failed_multilayer_growth_are_atomic(self):
        c = cache()
        self.append(c, 0, 1)
        old = c.snapshot()
        allocator = c._allocate
        calls = 0
        def failing(capacity):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected allocation failure")
            return allocator(capacity)
        with patch.object(c, "_allocate", side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.append(c, 1, 2)
        self.assertIs(c.snapshot(), old)
        self.assertEqual((c.revision, c.valid_length, c.capacity_blocks), (1, 12, 1))
        self.append(c, 1, 5)
        with self.assertRaises(HistoryCacheError):
            c.preflight(1)
        small = cache(max_bytes=100)
        with self.assertRaises(HistoryCacheError):
            self.append(small, 0, 1)
        self.assertIsNone(small.snapshot())

    def test_failed_tail_copy_leaves_committed_prefix_untouched(self):
        c = cache(max_blocks=8)
        self.append(c, 0, 1)
        self.append(c, 1, 2)
        self.append(c, 2, 3)  # capacity 4, length 3
        old = copy.deepcopy(c.snapshot())
        original = torch.Tensor.copy_
        count = 0
        def fail_copy(tensor, source, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("injected tail copy failure")
            return original(tensor, source, *args, **kwargs)
        with patch.object(torch.Tensor, "copy_", fail_copy):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.append(c, 3, 4)
        self.assertEqual((c.valid_length, c.revision), (36, 3))
        for x, y in zip(old.layers, c.snapshot().layers):
            torch.testing.assert_close(x.key, y.key, atol=0, rtol=0)
            torch.testing.assert_close(x.value, y.value, atol=0, rtol=0)
        self.append(c, 3, 4)
        self.assertEqual(c.valid_length, 48)

    def test_reject_training_mode_and_misordered_delta(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.append(cache(), 0, 1)
        self.model.eval()
        c = cache()
        with self.assertRaisesRegex(ValueError, "block IDs"):
            self.append(c, 1, 2)
        self.assertFalse(any("cache" in key for key in self.model.state_dict()))

    def test_bfloat16_incremental_close_to_recompute(self):
        c = HistoryKVCache(depth=3, heads=2, head_dim=64, tokens_per_block=12,
                           device="cpu", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.append(c, 0, 1)
            self.append(c, 1, 5)
            ref = self.model.predictor.forward_history(self.z, self.states, self.lang, self.mask, layout(0, 5))
        for x, y in zip(c.snapshot().layers, ref.layers):
            torch.testing.assert_close(x.key, y.key, atol=.02, rtol=.03)
            torch.testing.assert_close(x.value, y.value, atol=.02, rtol=.03)


class IdentityNormalizer:
    def __call__(self, values):
        return values
    def unnormalize_action(self, values):
        return values


def pixels(step=0):
    return {camera: np.full((240, 320, 3), step, np.uint8)
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")}


def block(index):
    collector = HistoryCollector()
    for step in range(index * 4 + 1):
        collector.add(step, pixels(step), np.full(14, step, np.float32))
    return collector.completed_blocks()[-1]


class SessionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        self.model = model().eval()
        enable_attention_gates(self.model.action_expert)
        self.language, self.mask = torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
        self.service = self.make_service()

    def make_service(self, **options):
        service = MemoryPolicyServing(self.model, IdentityNormalizer(),
                                      lambda _: (self.language, self.mask), max_sessions=2, **options)
        # Deterministic stand-in for expensive frozen encoder; operator tests
        # above use real Predictor/ActionExpert and independent full recompute.
        def encode(b):
            z = torch.arange(12*64, dtype=torch.float32).reshape(1, 12, 64) / 1000 + b["end_step"]/10
            state = pack_normalized_state(torch.from_numpy(b["state"])[None])
            return z, state
        service._encode_block = encode
        return service

    def reset(self, sid="a", owner=None):
        return self.service.infer_connection(dict(op="reset", session_id=sid, request_id="reset",
            episode_id="episode", instruction="whole task", protocol=PROTOCOL,
            sampling_config=self.model.config.to_dict()), owner)

    def append(self, blocks, rid="append", revision=0, sid="a"):
        return self.service.infer(dict(op="append", session_id=sid, request_id=rid,
                                       base_revision=revision, blocks=blocks))

    def infer_request(self, rid="infer", step=0, revision=1):
        return dict(op="infer", session_id="a", request_id=rid, expected_revision=revision, decision_step=step)

    def test_retry_conflict_duplicate_block_and_read_only_inference(self):
        self.assertTrue(self.reset()["ok"])
        self.assertTrue(self.append([block(0)])["ok"])
        repeated = self.append([block(0)])
        self.assertEqual(repeated["cache_revision"], 1)
        conflict = self.append([block(0)], revision=1)
        self.assertEqual(conflict["error"]["code"], "request_conflict")
        same = self.append([block(0)], rid="same-block", revision=1)
        self.assertEqual((same["cache_revision"], same["appended_blocks"]), (1, 0))
        changed = block(0)
        changed["state"][0] = 9
        self.assertEqual(self.append([changed], rid="bad-block", revision=1)["error"]["code"], "block_conflict")
        c = self.service.sessions["a"].cache
        before = copy.deepcopy(c.snapshot())
        request = self.infer_request()
        a = self.service.infer(request)
        self.assertTrue(a["ok"], a)
        a["actions"][0, 0] = 100  # returned array must not alias retry ledger
        b = self.service.infer(request)
        self.assertNotEqual(b["actions"][0, 0], 100)
        np.testing.assert_array_equal(b["actions"], self.service.infer(request)["actions"])
        self.assertEqual((c.revision, c.valid_length), (1, 12))
        for x, y in zip(before.layers, c.snapshot().layers):
            torch.testing.assert_close(x.key, y.key, atol=0, rtol=0)

    def test_bulk_failure_gap_revision_and_future_rejected_before_encoding(self):
        self.reset()
        cases = [
            ([block(1)], "block_order_conflict"),
            ([block(0), block(2)], "block_order_conflict"),
            ([dict(block(0), latent=np.zeros(1))], "invalid_request"),
        ]
        with patch.object(self.service, "_encode_block", wraps=self.service._encode_block) as encoder:
            for i, (blocks, code) in enumerate(cases):
                reply = self.append(blocks, rid=str(i))
                self.assertEqual(reply["error"]["code"], code)
                self.assertEqual(reply["cache_revision"], 0)
            encoder.assert_not_called()
        self.assertEqual(self.service.infer(self.infer_request(revision=0))["error"]["code"], "invalid_request")
        self.assertTrue(self.append([block(0)], rid="good")["ok"])
        self.assertEqual(self.append([block(1)], rid="stale")["error"]["code"], "revision_conflict")
        self.assertEqual(self.service.infer(self.infer_request(rid="future", step=16))["error"]["code"], "invalid_request")

    def test_session_owner_cleanup_capacity_and_reset_contract(self):
        self.assertTrue(self.reset("a", "connection1")["ok"])
        wrong = self.service.infer_connection(self.infer_request(), "connection2")
        self.assertEqual(wrong["error"]["code"], "session_owner_mismatch")
        self.assertTrue(self.reset("b", "connection2")["ok"])
        self.assertEqual(self.reset("c", "connection3")["error"]["code"], "session_capacity_exceeded")
        self.service.on_disconnect("connection1")
        self.assertEqual(set(self.service.sessions), {"b"})
        self.assertTrue(self.reset("a")["ok"])
        self.assertTrue(self.service.infer({"op": "close", "session_id": "a"})["closed"])
        self.assertTrue(self.service.infer({"op": "close", "session_id": "a"})["closed"])
        invalid = dict(op="reset", session_id="a", request_id="r", episode_id="e",
                       instruction="task", protocol="wrong", sampling_config=self.model.config.to_dict())
        self.assertEqual(self.service.infer(invalid)["error"]["code"], "invalid_request")

    def test_request_ledger_and_history_limits_do_not_evict(self):
        self.service = self.make_service(max_requests=2, max_history_blocks=1)
        self.reset()
        self.append([block(0)])
        result = self.service.infer(self.infer_request())
        self.assertEqual(result["error"]["code"], "request_capacity_exceeded")
        self.assertEqual(len(self.service.sessions["a"].cache.records), 1)
        self.service = self.make_service(max_history_blocks=1)
        self.reset()
        self.append([block(0)])
        result = self.append([block(1)], rid="over", revision=1)
        self.assertEqual(result["error"]["code"], "history_capacity_exceeded")
        self.assertEqual(result["cache_revision"], 1)

    def test_backends_match_and_bulk_commit_only_once(self):
        policies = (self.service, self.make_service(backend="recompute"))
        responses = []
        for service in policies:
            self.service = service
            self.reset()
            self.append([block(0)])
            result = self.append([block(i) for i in range(1, 5)], rid="bulk", revision=1)
            self.assertEqual((result["cache_revision"], result["last_block_id"]), (2, 4))
            responses.append(service.infer(self.infer_request(step=16, revision=2))["actions"])
        np.testing.assert_allclose(responses[0], responses[1], atol=3e-6, rtol=3e-5)
        self.assertEqual(policies[0].sessions["a"].features, ())

    def test_failed_compute_and_device_oom_preserve_revision(self):
        self.reset()
        self.append([block(0)])
        with patch.object(self.model.predictor, "append_history", side_effect=torch.cuda.OutOfMemoryError("injected")):
            result = self.append([block(1)], rid="oom", revision=1)
        self.assertEqual(result["error"]["code"], "device_out_of_memory")
        self.assertFalse(result["error"]["needs_reset"])
        self.assertEqual(result["cache_revision"], 1)
        self.assertTrue(self.append([block(1)], rid="recover", revision=1)["ok"])

    def test_collector_retains_unacked_frames_and_transport_retry(self):
        collector = HistoryCollector()
        for step in range(17):
            collector.add(step, pixels(step), np.full(14, step, np.float32))
        self.assertEqual(len(collector.pending), 5)
        for index, item in enumerate(collector.pending):
            self.assertEqual(item["frame_indices"], [max(0, index*4-4), index*4])
            np.testing.assert_array_equal(item["images"]["cam_head"][:, 0, 0, 0], item["frame_indices"])
        client = MemorySessionClient(self.service, "client")
        client.reset("episode", "whole task")
        original = self.service.infer
        captured = []
        def lost_ack(request):
            reply = original(request)
            captured.append(copy.deepcopy(request))
            if len(captured) == 1:
                raise TimeoutError("ack lost after server commit")
            return reply
        with patch.object(self.service, "infer", side_effect=lost_ack):
            with self.assertRaises(TimeoutError):
                client.append(collector)
            self.assertEqual(len(collector.pending), 5)
            client.append(collector)
        self.assertEqual(captured[0]["request_id"], captured[1]["request_id"])
        self.assertEqual(collector.pending, [])
        self.assertEqual(client.revision, 1)
        self.assertEqual(client.infer(16).shape, (32, 14))
        client.close()
        self.assertEqual(self.service.sessions, {})


if __name__ == "__main__":
    unittest.main()
