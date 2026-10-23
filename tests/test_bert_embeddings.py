"""Offline CPU token-feature generation with a small locally saved BERT."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("vjepa_policy", "lerobot", "transformers", "torch")):
    raise unittest.SkipTest("BERT embedding tests require the installed policy environment")

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from vjepa_policy.datasets.lerobot_video import TextEmbeddingCache
from vjepa_policy.text_cache import text_cache_filename
from vjepa_policy.text_embeddings import encode_prompts


class BertEmbeddings(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("JEPA_BERT_TEST_MODEL"), "Set JEPA_BERT_TEST_MODEL for the real local snapshot")
    def test_real_local_snapshot_encodes_token_features_on_cpu(self):
        with tempfile.TemporaryDirectory() as cache:
            prompts = ["put the block back", "press the button"]
            encode_prompts(prompts, cache, os.environ["JEPA_BERT_TEST_MODEL"],
                           128, device="cpu", text_encoder="bert")
            reader = TextEmbeddingCache(cache, 128, "bert")
            first, mask = reader.load(prompts[0])
            second, _ = reader.load(prompts[1])
            self.assertEqual(first.shape, (128, 768))
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(mask.any())
            self.assertFalse(first.requires_grad)
            self.assertFalse(torch.equal(first[mask], second[mask]))

    def test_frozen_token_features_roundtrip_without_loading_t5(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model_root, cache = root / "model", root / "cache"
            model_root.mkdir()
            vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                          "put", "the", "block", "back", "press", "button"]
            vocab = model_root / "vocab.txt"
            vocab.write_text("\n".join(vocabulary) + "\n")
            tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
            tokenizer.save_pretrained(model_root)
            model = BertModel(BertConfig(vocab_size=len(vocabulary), hidden_size=768,
                num_hidden_layers=1, num_attention_heads=12, intermediate_size=128,
                max_position_embeddings=128)).eval()
            model.save_pretrained(model_root)
            prompt = "put the block back"
            with patch("transformers.T5EncoderModel.from_pretrained", side_effect=AssertionError("T5 must not load")):
                encode_prompts([prompt], cache, str(model_root), 128, device="cpu", text_encoder="bert")
            context, mask = TextEmbeddingCache(cache, 128, "bert").load(prompt)
            self.assertEqual(context.shape, (128, 768))
            self.assertEqual(mask.dtype, torch.bool)
            tokens = tokenizer(prompt, max_length=128, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                expected = model(**tokens).last_hidden_state[0]
            torch.testing.assert_close(context, expected)
            self.assertTrue(torch.equal(mask, tokens.attention_mask[0].bool()))
            self.assertTrue(torch.isfinite(context).all())
            self.assertFalse(context.requires_grad)
            self.assertTrue((cache / text_cache_filename(prompt, 128, "bert")).is_file())
            with self.assertRaises(FileNotFoundError):
                TextEmbeddingCache(cache, 128).load(prompt)
            with self.assertRaisesRegex(ValueError, "avoid truncation"):
                encode_prompts([" ".join(["put"] * 130)], cache, str(model_root),
                               128, device="cpu", text_encoder="bert")
