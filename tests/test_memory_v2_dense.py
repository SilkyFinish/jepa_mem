"""Dense anchors, tails and unchanged deployment cadence."""
import tempfile
import unittest
from pathlib import Path
import torch
from test_memory_v0_training import write_fixture, ToyFrozenEncoder
from test_memory_v0_model import batch
from test_memory_v1_spatial import model
from scripts.cache_rmbench_history_features import build_cache
from scripts.train_vjepa_policy import parse_args
from vjepa_policy.history_features import encoder_signature
from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
from vjepa_policy.memory_config import FullHistoryV2Config, contract_from_checkpoint_metadata
from vjepa_policy.memory_training import validate_history_args

class DenseTests(unittest.TestCase):
    def test_all_anchors_and_tails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_fixture(root)
            args.history_model_version = "v2"
            args.require_full_future_window = False
            data = RMBenchHistoryDataset(root / "manifest.json", args)
            try:
                self.assertEqual(len(data), 114)
                for t in (1, 7, 16, 63, 64):
                    s = data[t]
                    self.assertEqual(s["history_end_step"][-1].item(), t)
                    self.assertEqual((~s["action_is_pad"]).sum().item(), min(32, 65-t))
                    self.assertTrue(set(range(0,t+1,4)).issubset(s["history_end_step"].tolist()))
                    self.assertEqual(data.history_lengths[t], len(s["history_valid"]))
                last = data[64]
                torch.testing.assert_close(last["target_video"][:,:,1:],
                    last["target_video"][:,:,1:2].expand(-1,-1,9,-1,-1))
            finally:
                data.close()

    def test_off_grid_backward_and_causality_validation(self):
        m, b = model(), batch()
        m.config = FullHistoryV2Config()
        b["history_end_step"][0,-1] = 13
        b["decision_step"][0] = 13
        loss = m(b)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(m.predictor.predictor_embed.weight.grad.abs().sum().item(),0)
        b["history_end_step"][0,2] = 7
        with self.assertRaisesRegex(ValueError, "without gaps"):
            m(b)

    def test_dense_cli_and_versioned_contract(self):
        args = parse_args(["--checkpoint", "unused", "--text-cache-dir", "unused",
                           "--policy-arch", "full_history", "--history-model-version", "v2"])
        self.assertFalse(args.require_full_future_window)
        c = validate_history_args(args)
        self.assertEqual(c.execution_interval_steps, 16)
        self.assertEqual(c.observed_steps(7), (0,4,7))
        self.assertEqual(contract_from_checkpoint_metadata(c.checkpoint_metadata()), c)
        self.assertEqual(c.decision_steps(49), tuple(range(49)))

    def test_dense_cache_includes_off_grid_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_fixture(root)
            args.history_model_version = "v2"
            args.history_episode_ids = ["0"]
            args.history_max_decision_step = 5
            data = RMBenchHistoryDataset(root / "manifest.json", args)
            cached = None
            try:
                out = root / "features"
                meta = build_cache(data, ToyFrozenEncoder(), out,
                                   encoder_metadata=encoder_signature(args), device="cpu")
                self.assertEqual(set(meta["episodes"]["0"]["observations"]), set(map(str,range(6))))
                args.history_feature_cache = str(out / "manifest.json")
                cached = RMBenchHistoryDataset(root / "manifest.json", args)
                from vjepa_policy.models.vjepa_policy import encode_video_views
                for t in (1,3,5):
                    expected = encode_video_views(ToyFrozenEncoder(), data[t]["history_video"])
                    torch.testing.assert_close(expected, cached[t]["history_latents"], atol=0, rtol=0)
            finally:
                data.close()
                if cached is not None:
                    cached.close()

    def test_serving_encoder_matches_fp32_cache_under_bf16_policy(self):
        from test_memory_v0_online import IdentityNormalizer, block
        from vjepa_policy.policy_serving.memory import MemoryPolicyServing
        m = model(video=True).eval()
        m.config = FullHistoryV2Config()
        service = MemoryPolicyServing(m, IdentityNormalizer(), lambda _: None, precision="bf16")
        b = block(0)
        with torch.no_grad():
            expected = service._encode_block(b)[0]
            with torch.autocast("cpu",dtype=torch.bfloat16):
                actual = service._encode_block(b)[0]
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
