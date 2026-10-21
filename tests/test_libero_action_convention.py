import unittest
from unittest.mock import patch
import numpy as np
import torch

from vjepa_policy.policy_serving.actions import resolve_gripper_convention, to_env_action


class LiberoActionConvention(unittest.TestCase):
    def test_raw_robosuite_and_lerobot_reach_same_open_close_commands(self):
        raw = np.array([[.2, -.4, 0, .1, 0, -.2, -1],
                        [.2, -.4, 0, .1, 0, -.2, 1]], dtype=np.float32)
        lerobot = raw.copy()
        lerobot[:, -1] = (1 - raw[:, -1]) / 2
        np.testing.assert_array_equal(to_env_action(raw, gripper_action_convention="robosuite"), raw)
        np.testing.assert_array_equal(to_env_action(lerobot), raw)
        np.testing.assert_array_equal(raw[:, -1], [-1, 1])

    def test_convention_resolution_and_invalid_input(self):
        self.assertEqual(resolve_gripper_convention(), "lerobot")
        self.assertEqual(resolve_gripper_convention(metadata={"gripper_action_convention": "robosuite"}), "robosuite")
        self.assertEqual(resolve_gripper_convention("robosuite", {"gripper_action_convention": "lerobot"}), "robosuite")
        with self.assertRaises(ValueError):
            resolve_gripper_convention("ambiguous")

    def test_real_serving_path_preserves_raw_gripper_after_unnormalization(self):
        from vjepa_policy.policy_serving.libero import VJEPAPolicyServing
        from vjepa_policy.datasets.normalization import ActionStateNormalizer
        from lerobot.configs.types import FeatureType, PolicyFeature
        from types import SimpleNamespace
        raw = torch.tensor([[[.2, 0, 0, 0, 0, 0, -1], [.2, 0, 0, 0, 0, 0, 1]]])
        stats = {"action": {"q01": -np.ones(7), "q99": np.ones(7)},
                 "observation.state": {"q01": -np.ones(8), "q99": np.ones(8)}}
        normalizer = ActionStateNormalizer(
            features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
                      "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
            stats=stats, action_mode="QUANTILE", state_mode="QUANTILE")
        normalized = normalizer({"action": raw})["action"]
        policy = object.__new__(VJEPAPolicyServing)
        policy.device = torch.device("cpu")
        policy.precision = "fp32"
        policy.action_normalizer = normalizer
        policy.raw_proprio_dim = 8
        policy.max_state_dim = None
        policy.m_enc = policy.m_pred = torch.zeros((1, 1), dtype=torch.long)
        policy.context_len = 1
        policy.binarize_gripper = True
        policy.gripper_action_convention = "robosuite"
        policy.encoder = object()
        policy.predictor = lambda *args, **kwargs: (None, None)
        policy.action_expert = SimpleNamespace(predict_action=lambda *args, **kwargs: normalized)
        policy._prepare_observation = lambda obs: (
            torch.zeros(1), torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 8))
        with patch("vjepa_policy.policy_serving.libero.encode_video_views", return_value=None):
            result = policy.infer({})["actions"]
        np.testing.assert_allclose(result, raw[0].numpy(), atol=1e-6)
