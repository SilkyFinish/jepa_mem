"""Stage boundaries and the requested continue-after-evaluation-failure policy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.train_eval_stages import Job


class StagedJobTests(unittest.TestCase):
    def job(self, root):
        return Job(dict(output_root=root, source_root=root, stage_steps=[2, 4],
                        gpu_ids=[0, 1], eval_gpu=0,
                        train_args=["--max-steps", "4", "--save-training-state"]))

    def test_evaluation_error_preserves_record_and_resumes_next_stage(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            with patch.object(job, "train_to", side_effect=lambda step, directory: Path(root)/f"checkpoint{step}.pt") as train:
                with patch.object(job, "evaluate", side_effect=[RuntimeError("simulator failure"), {"success_rate": 0., "complete": True}]):
                    job.run_stages()
            self.assertEqual([call.args[0] for call in train.call_args_list], [2, 4])
            status = json.loads((Path(root)/"status.json").read_text())
            self.assertEqual(status["phase"], "complete")
            self.assertEqual(status["evaluation_failures"], 1)
            self.assertEqual(status["stages"]["2"]["evaluation_status"], "failed")
            self.assertEqual(status["stages"]["4"]["evaluation_status"], "complete")
            self.assertTrue((Path(status["stages"]["2"]["directory"])/"evaluation_error.txt").exists())

    def test_training_failure_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            with patch.object(job, "train_to", side_effect=RuntimeError("nonfinite loss")), patch.object(job, "evaluate") as evaluation:
                with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                    job.run_stages()
                evaluation.assert_not_called()

    def test_manual_stop_does_not_continue_after_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            with patch.object(job, "train_to", return_value=Path(root)/"checkpoint.pt") as train:
                with patch.object(job, "evaluate", side_effect=InterruptedError("manual stop")):
                    with self.assertRaises(InterruptedError):
                        job.run_stages()
            self.assertEqual(train.call_count, 1)

    def test_plan_rejects_deadline_and_weight_only_training(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.job(root).config
            with self.assertRaisesRegex(ValueError, "complete optimizer"):
                Job(dict(config, train_args=["--max-steps", "4"]))
            with self.assertRaisesRegex(ValueError, "deadline"):
                Job(dict(config, train_args=config["train_args"]+["--stop-at", "2026-10-24T08:00:00+08:00"]))

    def test_cleanup_never_signals_a_finished_pid(self):
        child = Mock()
        child.poll.return_value = 0
        with patch("scripts.train_eval_stages.os.killpg") as kill:
            Job.terminate(child)
        kill.assert_not_called()
