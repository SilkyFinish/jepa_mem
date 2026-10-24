"""Actual shell entry defaults and standalone packing/warming orchestration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.a1008 import common, job as job_module


class UpgradeTests(unittest.TestCase):
    def test_shell_plan_preserves_local_paths_and_selects_dense_full_bert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "server.local.env"
            config.write_text("".join(
                "export " + key + "=" + shlex.quote(value) + "\n"
                for key, value in dict(JEPA_SHARED_ROOT=str(root),
                                      POLICY_PYTHON=sys.executable,
                                      JEPA_TRAIN_ONLY="1", JEPA_TMPDIR=str(root / "ipc")).items()))
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("JEPA_")}
            result = subprocess.run(["bash", str(common.REPO / "scripts/a1008/node.sh"),
                                     str(config), "--plan"],
                                    env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["language"]["model_version"], "v2_full_bert")
            self.assertEqual(plan["plan"]["updates"], 27425)
            self.assertEqual(plan["plan"]["per_device_batch"], 2)
            self.assertEqual(plan["plan"]["accumulation"], 1)
            self.assertTrue(plan["packed"])
            self.assertEqual(plan["recipe"]["contract"]["history_spatial_pool"], 1)
            self.assertTrue(plan["recipe"]["shared_prefix"])
            self.assertTrue(plan["recipe"]["action_checkpointing"])
            self.assertEqual(plan["recipe"]["attention"], "flex")
            self.assertEqual(Path(plan["paths"]["root"]), root)
            self.assertEqual(Path(plan["output"]).name, "put_back_block_full50_v2_full_bert_ep25")

    def test_packing_and_optional_warmup_run_standalone_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dict(
                JEPA_SHARED_ROOT=tmp, JEPA_HISTORY_MODEL_VERSION="v2_full",
                JEPA_TEXT_ENCODER="bert", JEPA_PACK_FEATURES="1", JEPA_WARM_OBSERVATIONS="1",
                JEPA_GPUS="8", JEPA_PER_DEVICE_BATCH="2", JEPA_EPOCHS="25"), clear=True):
            job = job_module.Job(argparse.Namespace(
                train_only=True, preflight_only=False, eval_only=False, allow_other_gpus=False))
            job.attempt = Path(tmp) / "attempt"
            job.attempt.mkdir()
            lock = {"text": {"fixture": "bert"}}
            common.write_json(job.p["ready"], dict(sources=lock))
            text = job.p["text"] / "instruction.pt"
            text.parent.mkdir()
            text.write_bytes(b"mock text entry")
            common.write_json(text.parent / "provenance.json", dict(
                source=lock["text"], context_length=128, files={text.name: common.sha256(text)}))
            common.write_json(job.p["features"], {})
            dataset = Mock()
            dataset.__len__ = Mock(return_value=17562)
            dataset.history_lengths = [1, 92]
            dataset.selection = {"diagnostic_subset": False}
            dataset.provenance = {"fixture": "new dense full50"}
            entry = ModuleType("scripts.train_vjepa_policy")
            entry.parse_args = Mock(return_value=SimpleNamespace())
            data = ModuleType("vjepa_policy.datasets.rmbench_history")
            data.RMBenchHistoryDataset = Mock(return_value=dataset)
            calls = []
            def command(name, argv, **kwargs):
                calls.append((name, list(map(str, argv))))
                self.assertNotIn("torch.distributed.run", argv)
                if name == "pack_visual_cache":
                    common.write_json(job.p["packed"], dict(
                        source_manifest_sha256=common.sha256(job.p["features"])))
            with patch.dict(sys.modules, {entry.__name__: entry, data.__name__: data}), \
                    patch.object(job_module, "asset_sources", return_value=lock), \
                    patch("scripts.a1008.prepare_assets.verify_prepared"), \
                    patch.object(job, "command", side_effect=command):
                job.caches()
            self.assertEqual([name for name, _ in calls], ["pack_visual_cache", "warm_observations"])
            self.assertIn("--warm-only", calls[-1][1])
            self.assertEqual(job.plan["updates"], 27425)
            dataset.close.assert_called_once_with()

    def test_partial_packing_storage_budget_credits_committed_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "packed/manifest.json"
            shard = manifest.parent / "episode0000"
            shard.mkdir(parents=True)
            (shard / "complete.json").write_text("{}")
            (shard / "observations.npy").write_bytes(b"x" * 11)
            (shard / "targets.npy").write_bytes(b"x" * 37)
            partial = manifest.parent / "episode0001"
            partial.mkdir()
            (partial / "observations.npy").write_bytes(b"x" * 20)
            self.assertEqual(common.missing_packed_bytes(manifest, 100), 52)
            manifest.write_text("{}")
            self.assertEqual(common.missing_packed_bytes(manifest, 100), 0)

    def test_dense_full_preset_upgrades_old_local_recipe_without_changing_machine_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "server.local.env"
            config.write_text("".join(
                "export " + key + "=" + shlex.quote(value) + "\n"
                for key, value in dict(
                    JEPA_SHARED_ROOT=str(root), POLICY_PYTHON=sys.executable,
                    JEPA_TRAIN_ONLY="1", JEPA_TMPDIR=str(root / "ipc"),
                    JEPA_HISTORY_MODEL_VERSION="v0", JEPA_TEXT_ENCODER="bert",
                    JEPA_HISTORY_BACKEND="segmented", JEPA_PER_DEVICE_BATCH="1",
                    JEPA_EPOCHS="400", JEPA_RUN_NAME="old_bert_ep400").items()))
            env = {key: value for key, value in os.environ.items() if not key.startswith("JEPA_")}
            result = subprocess.run(["bash", str(common.REPO / "scripts/a1008/node.sh"),
                                     str(config), "--dense-full", "--plan"],
                                    env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(Path(plan["paths"]["root"]), root)
            self.assertEqual(plan["language"]["model_version"], "v2_full_bert")
            self.assertEqual(plan["plan"]["updates"], 27425)
            self.assertEqual(plan["plan"]["per_device_batch"], 2)
            self.assertEqual(plan["recipe"]["attention"], "flex")
            self.assertEqual(Path(plan["output"]).name, "put_back_block_full50_v2_full_bert_ep25")
            self.assertEqual(config.read_text().count("JEPA_EPOCHS=400"), 1)
