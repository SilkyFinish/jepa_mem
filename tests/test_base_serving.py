"""Shared inference preserves dataset coordinates for benchmark decoders."""
import importlib.util
import unittest
from unittest.mock import Mock, patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm", "lerobot")):
    raise unittest.SkipTest("Requires the installed policy environment")

import numpy as np
import torch
from vjepa_policy.policy_serving.base import VJEPAPolicyServing

class IdentityNormalizer:
    def __call__(self, data):
        return data
    def unnormalize_action(self, actions):
        return actions

class SharedServing(unittest.TestCase):
    def test_native_joint_values_reach_benchmark_decoder_unchanged(self):
        policy = object.__new__(VJEPAPolicyServing)
        policy.device = torch.device("cpu")
        policy.precision = "fp32"
        policy.raw_proprio_dim = 14
        policy.max_state_dim = None
        policy.proprio_encoding = "identity"
        policy.context_len = 1
        policy.m_enc = torch.zeros((1, 1), dtype=torch.long)
        policy.m_pred = torch.ones((1, 1), dtype=torch.long)
        policy.action_normalizer = IdentityNormalizer()
        policy.encoder = Mock()
        policy.predictor = Mock(return_value=(None, ()))
        actions = torch.linspace(-3, 4, 56).reshape(2, 2, 14)
        policy.action_expert = Mock()
        policy.action_expert.predict_action.return_value = actions
        policy._prepare_observation = lambda obs: (
            torch.zeros(3, 2, 2, 2),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 14),
        )
        with patch("vjepa_policy.policy_serving.base.encode_video_views", return_value=None):
            result = policy.infer_batch([{}, {}])
        np.testing.assert_array_equal(np.stack([row["actions"] for row in result]), actions.numpy())
        self.assertEqual(policy.infer_batch([]), [])
