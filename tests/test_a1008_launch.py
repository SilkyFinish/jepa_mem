"""Deployment contracts that run in the source-only CI environment."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.a1008.common import schedule, serving_capacity
from scripts.a1008.prepare_assets import extract


class DeploymentTests(unittest.TestCase):
    def test_preserve_global_batch_and_drop_partial_windows(self):
        for world in (1, 2, 4, 8):
            plan = schedule(1013, world, 100)
            self.assertEqual(plan["global_batch"], world * plan["accumulation"])
            self.assertEqual(plan["updates"], 6300)
            self.assertEqual(plan["effective_samples_per_epoch"], 1008)
        with self.assertRaises(ValueError):
            schedule(1013, 3, 100)

    def test_single_gpu_selection_respects_platform_allocation(self):
        from scripts.a1008.job import one_gpu_env
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-selected-b,GPU-selected-a"}):
            self.assertEqual(one_gpu_env()["CUDA_VISIBLE_DEVICES"], "GPU-selected-b")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-selected-b,GPU-selected-a")

    def test_native_rollout_budget_covers_growth_beyond_training_prefix(self):
        blocks, budget = serving_capacity(500)
        self.assertEqual(blocks, 126)
        per_block = 588 * 1024 * 24 * 2 * 2
        # At block 81, 80 old blocks, 126 new slots and four staged blocks coexist.
        self.assertGreaterEqual(budget, (80 + 126 + 4) * per_block)
        self.assertGreater(blocks, 81)

    def test_archive_rejects_traversal_and_commits_native_payload(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / "data.zip"
            destination = root / "released"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                extract(archive, destination)
            self.assertFalse(destination.exists())
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("demo_clean/put_back_block/aloha_agilex/data/episode_0000000.hdf5", "fixture")
            extract(archive, destination)
            self.assertEqual((destination / "data/episode_0000000.hdf5").read_text(), "fixture")
