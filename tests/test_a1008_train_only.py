"""CPU-only launcher contracts; no training, simulator, or external commands run."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from scripts.a1008 import common, job as job_module


def option(command, flag):
    """Read a single-valued launcher flag, rejecting accidental duplicate flags."""
    if command.count(flag) != 1:
        raise AssertionError(f"Expected exactly one {flag}: {command}")
    return command[command.index(flag) + 1]


class LauncherTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)


class CommonLaunchTests(LauncherTestCase):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def test_batch_two_preserves_the_formal_training_budget(self):
        plan = common.schedule(1013, 8, 100, per_device_batch=2)
        self.assertEqual(plan, dict(
            world_size=8, per_device_batch=2, accumulation=1, global_batch=16,
            samples=1013, updates_per_epoch=63, effective_samples_per_epoch=1008,
            epochs=100, updates=6300,
        ))

    def test_valid_batches_keep_the_same_optimizer_budget(self):
        for world, batch in ((1, 1), (2, 1), (4, 1), (8, 1), (8, 2),
                             (4, 4), (2, 8), (1, 16)):
            with self.subTest(world=world, batch=batch):
                plan = common.schedule(1013, world, 100, per_device_batch=batch)
                self.assertEqual(world * batch * plan["accumulation"], 16)
                self.assertEqual(plan["per_device_batch"], batch)
                self.assertEqual(plan["updates_per_epoch"], 63)
                self.assertEqual(plan["updates"], 6300)
                self.assertEqual(plan["effective_samples_per_epoch"], 1008)

    def test_invalid_schedule_inputs_are_rejected(self):
        for samples, world, epochs, batch in (
            (1013, 0, 100, 1), (1013, -1, 100, 1),
            (1013, 8, 0, 1), (1013, 8, -1, 1),
            (1013, 8, 100, 0), (1013, 8, 100, -1),
            (1013, 3, 100, 1), (1013, 8, 100, 3),
            (1013, 16, 100, 2), (0, 8, 100, 2),
            (-1, 8, 100, 2), (15, 8, 100, 2),
        ):
            with self.subTest(samples=samples, world=world, epochs=epochs, batch=batch):
                with self.assertRaises(ValueError):
                    common.schedule(samples, world, epochs, per_device_batch=batch)

    def test_base_args_defaults_to_one_and_propagates_batch_two(self):
        paths = {name: Path("/unused") / name for name in ("encoder", "prepared", "text")}
        self.assertEqual(option(common.base_args(paths), "--batch-size"), "1")
        os.environ["JEPA_PER_DEVICE_BATCH"] = "2"
        self.assertEqual(option(common.base_args(paths), "--batch-size"), "2")

    def test_base_args_rejects_invalid_batch_environment(self):
        paths = {name: Path("/unused") / name for name in ("encoder", "prepared", "text")}
        for value in ("0", "-1", "two"):
            with self.subTest(value=value), patch.dict(os.environ, {"JEPA_PER_DEVICE_BATCH": value}):
                with self.assertRaises(ValueError):
                    common.base_args(paths)

    def test_train_only_environment_is_strict_and_opt_in(self):
        self.assertIs(common.train_only(), False)
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value), patch.dict(os.environ, {"JEPA_TRAIN_ONLY": value}):
                self.assertIs(common.train_only(), expected)
        for value in ("", "true", "false", "2", "-1"):
            with self.subTest(value=value), patch.dict(os.environ, {"JEPA_TRAIN_ONLY": value}):
                with self.assertRaises(ValueError):
                    common.train_only()


class JobExecutionTests(LauncherTestCase):
    def setUp(self):
        super().setUp()
        self.storage = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict(os.environ, {
            "JEPA_SHARED_ROOT": str(self.storage), "JEPA_RUN_NAME": "fixture",
            "JEPA_GPUS": "8", "JEPA_EPOCHS": "100",
        }, clear=True))
        # Exercise the actual GPU/NCCL orchestration against a fake torch module.
        # Importing real torch, querying a GPU, or launching a child is forbidden.
        self.cuda = SimpleNamespace(
            device_count=Mock(return_value=8),
            get_device_properties=Mock(return_value=SimpleNamespace(name="NVIDIA A100 40GB")),
            mem_get_info=Mock(return_value=(32 * 2**30, 40 * 2**30)),
        )
        torch = ModuleType("torch")
        torch.cuda = self.cuda
        torch.__version__ = "fixture"
        torch.version = SimpleNamespace(cuda="fixture")
        self.stack.enter_context(patch.dict(sys.modules, {"torch": torch}))
        self.stack.enter_context(patch.object(job_module.signal, "signal"))
        self.stack.enter_context(patch.object(job_module.subprocess, "Popen",
                                      side_effect=AssertionError("Unexpected subprocess launch")))
        self.stack.enter_context(patch.object(job_module.subprocess, "run",
                                      side_effect=AssertionError("Unexpected subprocess launch")))
        self.stack.enter_context(patch.object(job_module.urllib.request, "urlopen",
                                      side_effect=AssertionError("Unexpected network request")))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.caches = self.stack.enter_context(patch.object(job_module.Job, "caches", autospec=True))
        self.evaluate = self.stack.enter_context(patch.object(job_module.Job, "evaluate", autospec=True))
        self.commands = {}
        self.result_override = {}
        self.write_training_result = True
        self.command = self.stack.enter_context(patch.object(
            job_module.Job, "command", autospec=True, side_effect=self.fake_command))

    def fake_command(self, job, name, command, env=None, timeout=None):
        argv = list(map(str, command))
        self.commands[name] = argv
        if name == "training" and self.write_training_result:
            step = int(option(argv, "--max-steps"))
            result = dict(completed=True, step=step, checkpoint=f"checkpoint_step{step:06d}.pt")
            result.update(self.result_override)
            common.write_json(Path(option(argv, "--output-dir")) / "training_result.json", result)
        elif name == "simulator":
            common.write_json(Path(option(argv, "--output")) / "results.json", dict(
                native_expert_passed=True,
                records=[dict(mode="recorded_action_replay", controller="dense15", success=True)],
            ))

    def new_job(self, **overrides):
        args = dict(train_only=False, preflight_only=False, eval_only=False, allow_other_gpus=False)
        args.update(overrides)
        return job_module.Job(argparse.Namespace(**args))

    def enable_simulator(self):
        # These paths deliberately do not exist: simulator work must stay mocked.
        os.environ.update(RMBENCH_ROOT=str(self.storage / "absent_simulator"),
                          RMBENCH_PYTHON=str(self.storage / "absent_simulator_python"))

    def status(self, job):
        return json.loads((job.root / "status.json").read_text())

    def assert_preparation_ran(self, job):
        self.cuda.device_count.assert_called_once_with()
        self.assertEqual(self.cuda.get_device_properties.call_count, 8)
        self.assertIn("topology", self.commands)
        self.assertIn("nccl", self.commands)
        self.assertEqual(option(self.commands["nccl"], "--mode"), "collective")
        self.assertTrue((job.attempt / "hardware.json").is_file())
        self.caches.assert_called_once_with(job)

    def test_train_only_keeps_preparation_and_defers_all_evaluation(self):
        job = self.new_job(train_only=True)
        job.execute()
        self.assert_preparation_ran(job)
        self.assertIn("longest_prefix", self.commands)
        self.assertEqual(option(self.commands["longest_prefix"], "--mode"), "stress")
        self.assertIn("training", self.commands)
        self.assertNotIn("simulator", self.commands)
        self.evaluate.assert_not_called()
        status = self.status(job)
        self.assertEqual(status["phase"], "complete")
        self.assertIs(status["train_only"], True)
        self.assertEqual(status["evaluation"], "deferred_to_host")
        self.assertEqual(status["simulator_preflight"], "skipped")
        self.assertEqual(status["checkpoint"], str(job.root / "train/checkpoint_step006300.pt"))
        self.assertEqual(option(self.commands["training"], "--gradient-accumulation-steps"), "2")

    def test_batch_two_reaches_stress_and_formal_commands_without_changing_budget(self):
        os.environ["JEPA_PER_DEVICE_BATCH"] = "2"
        job = self.new_job(train_only=True)
        job.execute()
        self.assertEqual(job.plan, common.schedule(1013, 8, 100, per_device_batch=2))
        self.assertEqual(json.loads((job.root / "plan.json").read_text()), job.plan)
        self.assertEqual(self.status(job)["plan"], job.plan)
        for name in ("longest_prefix", "training"):
            self.assertEqual(option(self.commands[name], "--batch-size"), "2")
        command = self.commands["training"]
        for flag, value in (("--gradient-accumulation-steps", "1"), ("--num-epochs", "100"),
                            ("--max-steps", "6300"), ("--save-every", "315")):
            self.assertEqual(option(command, flag), value)
        index = command.index("--checkpoint-milestones")
        self.assertEqual(command[index + 1:index + 6], ["315", "630", "1260", "3150", "6300"])
        self.assertEqual(self.status(job)["phase"], "complete")
        self.evaluate.assert_not_called()

    def test_train_only_preflight_never_starts_formal_training(self):
        job = self.new_job(train_only=True, preflight_only=True)
        job.execute()
        self.assert_preparation_ran(job)
        self.assertIn("longest_prefix", self.commands)
        self.assertNotIn("training", self.commands)
        self.assertNotIn("simulator", self.commands)
        self.evaluate.assert_not_called()
        status = self.status(job)
        self.assertEqual(status["phase"], "preflight_passed")
        self.assertIs(status["train_only"], True)
        self.assertIs(status["formal_training_started"], False)
        self.assertEqual(status["simulator_preflight"], "skipped")
        self.assertFalse((job.root / "train/training_result.json").exists())

    def test_default_combined_mode_keeps_simulator_smoke_and_final_evaluation(self):
        self.enable_simulator()
        job = self.new_job()
        job.execute()
        self.assert_preparation_ran(job)
        self.assertIn("simulator", self.commands)
        self.assertIn("training", self.commands)
        self.assertEqual(self.evaluate.call_args_list, [
            call(job, job.attempt / "longest_prefix/checkpoint_step000003.pt",
                 job.attempt / "smoke_rollout", 1, True),
            call(job, job.root / "train/checkpoint_step006300.pt", job.attempt / "rollouts", 20),
        ])
        status = self.status(job)
        self.assertEqual(status["phase"], "complete")
        self.assertIs(status["train_only"], False)
        self.assertEqual(status["evaluation"], str(job.attempt / "rollouts/results.json"))
        self.assertEqual(option(self.commands["training"], "--batch-size"), "1")

    def test_default_preflight_still_requires_simulator_and_smoke_rollout(self):
        self.enable_simulator()
        job = self.new_job(preflight_only=True)
        job.execute()
        self.assert_preparation_ran(job)
        self.assertIn("longest_prefix", self.commands)
        self.assertIn("simulator", self.commands)
        self.assertNotIn("training", self.commands)
        self.evaluate.assert_called_once_with(
            job, job.attempt / "longest_prefix/checkpoint_step000003.pt",
            job.attempt / "smoke_rollout", 1, True)
        self.assertEqual(self.status(job)["phase"], "preflight_passed")
        self.assertIs(self.status(job)["train_only"], False)

    def test_default_eval_only_evaluates_existing_formal_result(self):
        self.enable_simulator()
        job = self.new_job(eval_only=True)
        common.write_json(job.root / "train/training_result.json", dict(
            completed=True, step=6300, checkpoint="checkpoint_step006300.pt"))
        job.execute()
        self.assert_preparation_ran(job)
        for name in ("longest_prefix", "simulator", "training"):
            self.assertNotIn(name, self.commands)
        self.evaluate.assert_called_once_with(
            job, job.root / "train/checkpoint_step006300.pt", job.attempt / "rollouts", 20)
        self.assertEqual(self.status(job)["phase"], "complete")
        self.assertIs(self.status(job)["train_only"], False)

    def test_batch_change_cannot_silently_reuse_an_existing_run_plan(self):
        os.environ["JEPA_PER_DEVICE_BATCH"] = "2"
        job = self.new_job(train_only=True)
        common.write_json(job.root / "plan.json", common.schedule(1013, 8, 100))
        with self.assertRaisesRegex(ValueError, "different plan"):
            job.execute()
        self.assertEqual(self.status(job)["phase"], "failed")
        self.assertNotIn("longest_prefix", self.commands)
        self.assertNotIn("training", self.commands)
        self.evaluate.assert_not_called()

    def test_train_only_rejects_a_result_with_the_wrong_update_count(self):
        self.result_override = {"step": 6299}
        job = self.new_job(train_only=True)
        with self.assertRaisesRegex(ValueError, "unexpected update count"):
            job.execute()
        self.assertEqual(self.status(job)["phase"], "failed")
        self.evaluate.assert_not_called()

    def test_train_only_does_not_mark_interrupted_training_complete(self):
        self.result_override = {"completed": False, "step": 42}
        job = self.new_job(train_only=True)
        job.execute()
        self.assertEqual(self.status(job)["phase"], "stopped")
        self.assertIs(self.status(job)["train_only"], True)
        self.evaluate.assert_not_called()

    def test_train_only_requires_a_formal_training_result(self):
        self.write_training_result = False
        job = self.new_job(train_only=True)
        with self.assertRaisesRegex(RuntimeError, "training_result.json"):
            job.execute()
        self.assertEqual(self.status(job)["phase"], "failed")
        self.evaluate.assert_not_called()


    def test_dense_full_bert_enables_optimizations_and_uses_new_cache_paths(self):
        os.environ.update(JEPA_HISTORY_MODEL_VERSION="v2_full", JEPA_TEXT_ENCODER="bert",
                          JEPA_HISTORY_BACKEND="flex", JEPA_SHARED_PREFIX="1",
                          JEPA_ACTION_CHECKPOINTING="1", JEPA_PACK_FEATURES="1",
                          JEPA_PER_DEVICE_BATCH="2", JEPA_EPOCHS="25")
        job = self.new_job(train_only=True)
        job.execute()
        self.assertIn("attention_parity", self.commands)
        self.assertEqual(option(self.commands["attention_parity"], "--mode"), "attention")
        self.assertEqual(job.plan, common.schedule(17562, 8, 25, 2))
        self.assertEqual(job.plan["updates"], 27425)
        self.assertEqual(job.plan["updates_per_epoch"], 1097)
        self.assertEqual(job.recipe["contract"]["model_version"], "v2_full_bert")
        self.assertEqual(job.recipe["contract"]["history_spatial_pool"], 1)
        self.assertIn("history_features_bert_v2", str(job.p["features"]))
        self.assertIn("history_features_bert_v2_packed", str(job.p["packed"]))
        for name in ("longest_prefix", "training"):
            command = self.commands[name]
            self.assertEqual(option(command, "--history-model-version"), "v2_full")
            self.assertEqual(option(command, "--lang-dim"), "768")
            self.assertEqual(option(command, "--history-attention-backend"), "flex")
            self.assertEqual(option(command, "--history-feature-packed"), str(job.p["packed"]))
            self.assertIn("--history-shared-prefix", command)
            self.assertIn("--history-action-checkpointing", command)
        self.assertEqual(option(self.commands["training"], "--gradient-accumulation-steps"), "1")

    def test_dense_old_run_plan_is_rejected_before_allocating_gpu_work(self):
        os.environ.update(JEPA_HISTORY_MODEL_VERSION="v2_full", JEPA_TEXT_ENCODER="bert")
        job = self.new_job(train_only=True)
        common.write_json(job.root / "plan.json", common.schedule(1013, 8, 100))
        with self.assertRaisesRegex(ValueError, "different plan"):
            job.execute()
        self.cuda.device_count.assert_not_called()
        self.caches.assert_not_called()

    def test_dense_full_state_is_forwarded_only_from_its_own_run(self):
        os.environ.update(JEPA_HISTORY_MODEL_VERSION="v2_full", JEPA_TEXT_ENCODER="bert",
                          JEPA_PER_DEVICE_BATCH="2", JEPA_EPOCHS="25")
        job = self.new_job(train_only=True)
        resume = job.root / "train/resume_state.pt"
        resume.parent.mkdir(parents=True)
        resume.write_bytes(b"mock full-state; command forwarding only")
        common.write_json(job.root / "recipe.json", job.recipe)
        job.execute()
        self.assertEqual(option(self.commands["training"], "--resume"), str(resume))


class CachePlanTests(unittest.TestCase):
    def test_cache_dataset_validation_preserves_batch_two_plan(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "JEPA_SHARED_ROOT": directory, "JEPA_GPUS": "8", "JEPA_EPOCHS": "100",
            "JEPA_PER_DEVICE_BATCH": "2",
        }, clear=True):
            job = job_module.Job(argparse.Namespace(
                train_only=True, preflight_only=False, eval_only=False, allow_other_gpus=False))
            job.attempt = Path(directory) / "attempt"
            job.attempt.mkdir()
            lock = {"text": {"fixture": "frozen_text"}}
            common.write_json(job.p["ready"], {"sources": lock})
            text_file = job.p["text"] / "instruction.pt"
            text_file.parent.mkdir(parents=True)
            text_file.write_bytes(b"fixture, not a tensor")
            common.write_json(job.p["text"] / "provenance.json", dict(
                source=lock["text"], context_length=128,
                files={text_file.name: common.sha256(text_file)}))
            common.write_json(job.p["features"], {})
            data = Mock()
            data.__len__ = Mock(return_value=1013)
            data.history_lengths = [1, 81]
            data.selection = {"diagnostic_subset": False}
            data.provenance = {"fixture": True}
            train_module = ModuleType("scripts.train_vjepa_policy")
            train_module.parse_args = Mock(return_value=SimpleNamespace())
            dataset_module = ModuleType("vjepa_policy.datasets.rmbench_history")
            dataset_module.RMBenchHistoryDataset = Mock(return_value=data)
            with patch.dict(sys.modules, {
                train_module.__name__: train_module, dataset_module.__name__: dataset_module,
            }), patch.object(job_module, "asset_sources", return_value=lock), \
                    patch("scripts.a1008.prepare_assets.verify_prepared") as verify, \
                    patch.object(job, "command", side_effect=AssertionError("Unexpected cache build")):
                job.caches()
            verify.assert_called_once_with(job.p, lock)
            data.close.assert_called_once_with()
            self.assertEqual(job.plan, common.schedule(1013, 8, 100, per_device_batch=2))
            parsed_args = train_module.parse_args.call_args.args[0]
            self.assertEqual(option(parsed_args, "--batch-size"), "2")


class JobCliTests(LauncherTestCase):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.job_class = self.stack.enter_context(patch.object(job_module, "Job", autospec=True))

    def parse(self, *arguments):
        with patch.object(sys, "argv", ["job.py", *arguments]):
            job_module.main()
        return self.job_class.call_args.args[0]

    def test_default_cli_keeps_combined_mode(self):
        args = self.parse()
        self.assertIs(args.train_only, False)
        self.job_class.return_value.execute.assert_called_once_with()

    def test_explicit_train_only_and_preflight_combination(self):
        args = self.parse("--train-only", "--preflight-only")
        self.assertIs(args.train_only, True)
        self.assertIs(args.preflight_only, True)
        self.assertIs(args.eval_only, False)
        self.job_class.return_value.execute.assert_called_once_with()

    def test_train_only_environment_sets_cli_default(self):
        os.environ["JEPA_TRAIN_ONLY"] = "1"
        self.assertIs(self.parse().train_only, True)

    def test_train_only_is_incompatible_with_eval_only(self):
        for arguments, environment in (
            (("--train-only", "--eval-only"), "0"),
            (("--eval-only",), "1"),
        ):
            with self.subTest(arguments=arguments, environment=environment), \
                    patch.dict(os.environ, {"JEPA_TRAIN_ONLY": environment}), \
                    patch.object(sys, "argv", ["job.py", *arguments]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    job_module.main()
                self.assertEqual(error.exception.code, 2)
        self.job_class.return_value.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
