"""Input/version contracts run without importing the optional ML stack."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "memory_v0_config", ROOT / "src/vjepa_policy/memory_config.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
FullHistoryV0Config = module.FullHistoryV0Config


class MemoryV0ContractTest(unittest.TestCase):
    def setUp(self):
        self.config = FullHistoryV0Config()

    def test_version_roundtrip_and_checked_in_preset(self):
        preset = json.loads((ROOT / "configs/memory_v0/input_contract.json").read_text())
        restored = FullHistoryV0Config.from_dict(preset)
        self.assertEqual(restored, self.config)
        self.assertEqual(restored.fingerprint(), self.config.fingerprint())
        self.assertEqual(module.contract_from_checkpoint_metadata(
            self.config.checkpoint_metadata()), self.config)

    def test_bert_contract_is_explicit_and_preserves_nonlanguage_fields(self):
        bert = FullHistoryV0Config.for_text_encoder("bert")
        self.assertEqual((bert.model_version, bert.text_encoder, bert.text_dim), ("v0_bert", "bert", 768))
        self.assertEqual(FullHistoryV0Config.from_dict(bert.to_dict()), bert)
        self.assertEqual(module.contract_from_checkpoint_metadata(bert.checkpoint_metadata()), bert)
        changed = {key for key, value in bert.to_dict().items() if value != self.config.to_dict()[key]}
        self.assertEqual(changed, {"model_version", "text_dim"})
        self.assertNotEqual(bert.fingerprint(), self.config.fingerprint())
        self.assertEqual(FullHistoryV0Config.for_text_encoder("t5"), self.config)

    def test_bert_width_and_version_cannot_be_selected_independently(self):
        for changes in ({"text_dim": 768}, {"model_version": "v0_bert"},
                        {"model_version": "v0_bert", "text_dim": 1024}):
            with self.assertRaises(ValueError):
                replace(self.config, **changes)
        bert = FullHistoryV0Config.for_text_encoder("bert")
        with self.assertRaises(ValueError):
            replace(bert, history_interval_steps=8)
        with self.assertRaises(ValueError):
            FullHistoryV0Config.for_text_encoder("other")
        metadata = bert.checkpoint_metadata()
        metadata["model_version"] = "v0"
        with self.assertRaises(ValueError):
            module.contract_from_checkpoint_metadata(metadata)

    def test_native_geometry_and_state_validity_width(self):
        self.assertEqual(self.config.tokens_per_block, 588)
        self.assertEqual(self.config.future_tokens, 2352)
        self.assertEqual(self.config.packed_state_dim, 96)

    def test_initial_observation_repeats_only_available_frame(self):
        sample = self.config.sample_indices("ep0", 33, 0)
        self.assertEqual(sample["history_frame_indices"], ((0, 0),))
        self.assertEqual(sample["target_frame_indices"], (0, 0, 4, 8, 12, 16, 20, 24, 28, 32))
        self.assertEqual(sample["action_indices"], tuple(range(32)))

    def test_complete_prefix_has_no_future_inputs_or_second_action_shift(self):
        sample = self.config.sample_indices("ep0", 100, 32)
        self.assertEqual(sample["history_end_step"], tuple(range(0, 33, 4)))
        self.assertEqual(sample["history_frame_indices"][-1], (28, 32))
        self.assertEqual(sample["history_state_indices"], sample["history_end_step"])
        self.assertEqual(sample["current_state_index"], 32)
        self.assertEqual(sample["action_indices"], tuple(range(32, 64)))
        self.assertEqual(sample["target_frame_indices"][-1], 64)
        for end, pair in zip(sample["history_end_step"], sample["history_frame_indices"]):
            self.assertLessEqual(max(pair), end)

    def test_tail_boundary_and_off_grid_decisions(self):
        self.assertEqual(self.config.decision_steps(32), ())
        self.assertEqual(self.config.decision_steps(33), (0,))
        self.assertEqual(self.config.decision_steps(48), (0,))
        self.assertEqual(self.config.decision_steps(49), (0, 16))
        for frames, decision in ((32, 0), (48, 16), (100, 4)):
            with self.assertRaises(ValueError):
                self.config.sample_indices("ep", frames, decision)

    def test_global_coordinates_do_not_renumber_old_blocks(self):
        first = self.config.temporal_coordinates(16)
        later = self.config.temporal_coordinates(32)
        self.assertEqual(first["history"], later["history"][:len(first["history"])])
        self.assertEqual(first["future"], (3.0, 4.0, 5.0, 6.0))
        self.assertEqual(first["action"][0], 17 / 8)
        self.assertEqual(first["action"][-1], 48 / 8)

    def test_invalid_types_unknown_versions_and_missing_fields_fail(self):
        for key, value in (("schema_version", True), ("history_interval_steps", 0),
                           ("action_label_shift", 1), ("model_version", "v1"),
                           ("cameras", ("cam_head",)), ("action_reads_future", 1)):
            with self.assertRaises(ValueError):
                replace(self.config, **{key: value})
        for value in (None, {}, {"extra": 1}):
            with self.assertRaises(ValueError):
                FullHistoryV0Config.from_dict(value)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.config.observed_steps(value)
        with self.assertRaises(ValueError):
            self.config.sample_indices("", 100, 0)

    def test_legacy_checkpoint_detection_is_backward_compatible(self):
        for old in ({}, {"predictor": {}, "action_expert": {}}, {"policy_arch": "legacy"}):
            self.assertIsNone(module.contract_from_checkpoint_metadata(old))
        for malformed in (
            {"memory_schema": self.config.to_dict()},
            {"model_version": "v0"},
            {"policy_arch": "full_history"},
            {"policy_arch": "full_history", "model_version": "v1"},
            {"policy_arch": "unknown"},
        ):
            with self.assertRaises(ValueError):
                module.contract_from_checkpoint_metadata(malformed)


if __name__ == "__main__":
    unittest.main()
