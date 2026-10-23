"""CPU checks for frozen BERT features through the full-history policy path."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm", "h5py", "lerobot", "accelerate")):
    raise unittest.SkipTest("BERT memory integration requires the installed policy environment")

import torch
from torch import nn

from scripts.train_vjepa_policy import parse_args
from scripts.cache_rmbench_history_features import build_cache
from vjepa_policy.datasets import DEFAULT_PROMPT
from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
from vjepa_policy.history_features import encoder_signature
from vjepa_policy.memory_config import FullHistoryV0Config
from vjepa_policy.memory_training import validate_history_args
from vjepa_policy.models.memory_action_expert import MemoryActionExpert
from vjepa_policy.models.memory_policy import FullHistoryVJEPAPolicy
from vjepa_policy.models.memory_predictor import MemoryWorldModelPredictor
from vjepa_policy.policy_arch import load_policy_weights, memory_checkpoint_metadata
from vjepa_policy.policy_serving.memory import MemoryPolicyServing
from vjepa_policy.policy_serving.memory_client import HistoryCollector, MemorySessionClient
from vjepa_policy.text_cache import text_cache_filename
from test_memory_v0_model import predictor_options, action_options, batch as tiny_batch, model as t5_model
from test_memory_v0_online import IdentityNormalizer
from test_memory_v0_training import write_fixture, ToyFrozenEncoder


def bert_model(*, strict=False):
    options = predictor_options()
    options["lang_dim"] = 768
    predictor = MemoryWorldModelPredictor(**options)
    options = action_options()
    options["lang_dim"] = 768
    expert = MemoryActionExpert(rope_options=predictor.rope_options, **options)
    return FullHistoryVJEPAPolicy(
        nn.Linear(1, 1), predictor, expert,
        config=FullHistoryV0Config.for_text_encoder("bert"), strict_v0_inputs=strict,
    )


class BertMemoryTests(unittest.TestCase):
    def test_cli_selects_768_without_changing_the_t5_default(self):
        required = ["--checkpoint", "unused", "--text-cache-dir", "unused", "--policy-arch", "full_history"]
        original = parse_args(required)
        self.assertEqual((original.text_encoder, original.lang_dim), ("t5", 4096))
        args = parse_args([*required, "--text-encoder", "bert"])
        self.assertEqual((args.lang_dim, args.context_len), (768, 128))
        self.assertEqual(validate_history_args(args).model_version, "v0_bert")
        args = parse_args([*required, "--text-encoder", "bert", "--lang-dim", "4096"])
        with self.assertRaisesRegex(ValueError, "lang-dim"):
            validate_history_args(args)

    def test_dataset_uses_bert_cache_and_reloads_its_visual_cache_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = write_fixture(root)
            args.text_encoder, args.lang_dim = "bert", 768
            args.history_episode_ids = ["0"]
            args.history_max_decision_step = 0
            prompt = DEFAULT_PROMPT.format(task="Move the block back.")
            cache = root / "text" / text_cache_filename(prompt, 128, "bert")
            torch.save({"context": torch.ones(128, 768), "mask": torch.ones(128, dtype=torch.bool)}, cache)
            data = cached = None
            try:
                data = RMBenchHistoryDataset(root / "manifest.json", args)
                self.assertEqual(data[0]["language"].shape, (128, 768))
                self.assertEqual(data.config.model_version, "v0_bert")
                output = root / "features"
                metadata = build_cache(data, ToyFrozenEncoder(), output,
                                       encoder_metadata=encoder_signature(args), device="cpu")
                self.assertEqual(metadata["contract"], data.config.to_dict())
                args.history_feature_cache = str(output / "manifest.json")
                cached = RMBenchHistoryDataset(root / "manifest.json", args)
                self.assertEqual(cached[0]["history_latents"].shape, (1, 588, 1024))
                self.assertEqual(cached[0]["language"].shape, (128, 768))
                cache.unlink()
                with self.assertRaises(FileNotFoundError):
                    RMBenchHistoryDataset(root / "manifest.json", args)
            finally:
                if data is not None:
                    data.close()
                if cached is not None:
                    cached.close()

    def test_bert_gradients_and_strict_weight_reload_use_768_features(self):
        torch.manual_seed(73)
        model = bert_model()
        sample = tiny_batch()
        sample["language"] = torch.randn(2, 128, 768)
        sample["language_mask"] = torch.ones(2, 128, dtype=torch.bool)
        loss, _ = model(sample)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.predictor.parameters()))
        payload = {"predictor": model.predictor.state_dict(),
                   "action_expert": model.action_expert.state_dict(),
                   **memory_checkpoint_metadata(model)}
        self.assertEqual(payload["model_version"], "v0_bert")
        restored = bert_model()
        load_policy_weights(restored, payload)
        noise = torch.randn(2, 32, 14)
        torch.testing.assert_close(model.eval().predict_action(sample, initial_noise=noise),
                                   restored.eval().predict_action(sample, initial_noise=noise), atol=0, rtol=0)
        incompatible = t5_model()
        with patch.object(incompatible.predictor, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_policy_weights(incompatible, payload)
            load.assert_not_called()

    def test_client_reset_negotiates_bert_contract_and_rejects_t5_features(self):
        model = bert_model(strict=True).eval()
        language, mask = torch.ones(1, 128, 768), torch.ones(1, 128, dtype=torch.bool)
        service = MemoryPolicyServing(model, IdentityNormalizer(), lambda _: (language, mask))
        self.assertEqual(service.metadata["model_version"], "v0_bert")
        client = MemorySessionClient(service)
        self.assertEqual(client.config, model.config)
        collector = HistoryCollector(config=client.config)
        self.assertEqual(collector.config.text_dim, 768)
        self.assertTrue(client.reset("episode", "whole task")["ok"])
        client.close()
        wrong = MemorySessionClient(service, config=FullHistoryV0Config())
        with self.assertRaises(RuntimeError):
            wrong.reset("episode", "whole task")
        language = torch.ones(1, 128, 4096)
        with self.assertRaises(RuntimeError):
            MemorySessionClient(service).reset("episode", "whole task")


if __name__ == "__main__":
    unittest.main()
