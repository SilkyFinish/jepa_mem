import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "pipeline", Path(__file__).parents[1] / "scripts/run_training_rollout_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class TrainingRolloutPipeline(unittest.TestCase):
    def test_training_gate_and_serial_task_order(self):
        for train_exit, complete, expected in [
            (0, True, ["training", "a", "b", "c"]),
            (1, False, ["training"]),
            (0, False, ["training"]),
        ]:
            with self.subTest(train_exit=train_exit, complete=complete), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                cfg = {"pipeline_deadline": "2099-01-01T10:00:00+08:00",
                       "policy_python": sys.executable, "train_args": [],
                       "tasks": [{"key": k} for k in "abc"], "rollouts_per_task": 20}
                config = temporary / "config.json"
                config.write_text(json.dumps(cfg))
                out = temporary / "run"
                calls = []

                def stage(command, logpath, **kwargs):
                    destination = Path(command[command.index("--output-dir") + 1])
                    destination.mkdir()
                    if "train" in command:
                        calls.append("training")
                        result = {"completed": complete, "checkpoint": "checkpoint_step000010.pt", "step": 10}
                        (destination / "training_result.json").write_text(json.dumps(result))
                        return train_exit
                    task = destination.name.removeprefix("rollout_")
                    calls.append(task)
                    self.assertEqual(Path(command[command.index("--checkpoint") + 1]),
                                     out / "train/checkpoint_step000010.pt")
                    (destination / "completion.json").write_text(json.dumps(
                        {"completed": True, "episodes": 20, "successes": 3}))
                    return 0

                argv = ["pipeline", "--config", str(config), "--output-dir", str(out)]
                with patch.object(pipeline, "run_stage", stage), patch.object(sys, "argv", argv):
                    pipeline.main()
                self.assertEqual(calls, expected)
                status = json.loads((out / "status.json").read_text())
                self.assertEqual(status["completed"], train_exit == 0 and complete)
                if status["completed"]:
                    self.assertEqual(status["total_episodes"], 60)

    def test_evaluation_deadline_terminates_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            started = time.time()
            code = pipeline.run_stage([sys.executable, "-c", "import time; time.sleep(60)"],
                                      Path(directory) / "worker.log", deadline=started + 0.2)
            self.assertNotEqual(code, 0)
            self.assertLess(time.time() - started, 5)
