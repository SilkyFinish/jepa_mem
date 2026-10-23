"""CPU-only asset selection and local BERT staging contracts."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.a1008.common import REPO, asset_sources, base_args, paths
from scripts.a1008.prepare_assets import prepare_text_model, verify_text_model

spec = importlib.util.spec_from_file_location("text_cache_names", REPO / "src/vjepa_policy/text_cache.py")
cache_names = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_names)


class BertAssets(unittest.TestCase):
    def test_variant_paths_and_sources_do_not_reuse_t5_artifacts(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"JEPA_SHARED_ROOT": root}):
            with patch.dict(os.environ, {"JEPA_TEXT_ENCODER": "t5"}):
                t5 = paths()
                self.assertEqual(asset_sources()["text"]["repo"], "google/t5-v1_1-xxl")
            with patch.dict(os.environ, {"JEPA_TEXT_ENCODER": "bert"}):
                bert = paths()
                self.assertEqual(asset_sources()["text"]["encoder_type"], "bert")
                args = base_args(bert)
                self.assertEqual(args[args.index("--text-encoder") + 1], "bert")
                self.assertEqual(args[args.index("--lang-dim") + 1], "768")
            for key in ("text", "text_model", "features", "ready"):
                self.assertNotEqual(bert[key], t5[key])
            for key in ("root", "data", "prepared", "encoder"):
                self.assertEqual(bert[key], t5[key])

    def test_local_bert_is_copied_and_verified_without_hub_downloads(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, destination = root / "source", root / "staged"
            source.mkdir()
            original = source / "model.safetensors"
            original.write_bytes(b"frozen local fixture")
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            spec = {"files": {original.name: digest}}
            with patch.dict(os.environ, {"JEPA_TEXT_ENCODER": "bert", "JEPA_BERT_SOURCE": str(source)}):
                prepare_text_model({"text_model": destination}, spec)
                verify_text_model({"text_model": destination}, spec)
            self.assertFalse(original.samefile(destination / original.name))
            original.write_bytes(b"changed old source")
            self.assertEqual((destination / original.name).read_bytes(), b"frozen local fixture")
            with patch.dict(os.environ, {"JEPA_TEXT_ENCODER": "bert", "JEPA_BERT_SOURCE": ""}):
                prepare_text_model({"text_model": destination}, spec)

    def test_mismatched_local_weights_do_not_become_prepared_assets(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "config.json").write_text("wrong snapshot")
            destination = root / "staged"
            with patch.dict(os.environ, {"JEPA_TEXT_ENCODER": "bert", "JEPA_BERT_SOURCE": str(root)}):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepare_text_model({"text_model": destination}, {"files": {"config.json": "0" * 64}})
            self.assertFalse((destination / "config.json").exists())

    def test_text_cache_names_keep_legacy_t5_and_bert_separate(self):
        prompt = "put the block back"
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        self.assertEqual(cache_names.text_cache_filename(prompt, 128), f"{digest}.t5_len128.pt")
        self.assertEqual(cache_names.text_cache_filename(prompt, 128, "bert"), f"{digest}.bert_len128.pt")
        with self.assertRaises(ValueError):
            cache_names.text_cache_filename(prompt, 128, "unknown")
