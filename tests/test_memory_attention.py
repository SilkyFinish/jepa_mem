"""Exact segmented attention: visibility, gradients, padding and model parity."""
import copy
import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "timm")):
    raise unittest.SkipTest("attention tests require the policy environment")

import torch
from torch.nn import functional as F

from vjepa_policy.models.memory_attention import history_attention_plan, future_attention_plan, validity_attention_plan
from vjepa_policy.models.memory_masks import history_attention_mask, future_attention_mask
from vjepa_policy.models.memory_layouts import visual_layout, future_layout
from test_memory_v0_model import model, batch, enable_attention_gates


def geometry():
    return dict(grid_height=2, grid_width=2, num_views=3)


def reference(q, k, v, mask, scale):
    valid = mask.any(-1, keepdim=True)
    safe = mask.clone()
    safe[..., :1] |= ~valid
    return F.scaled_dot_product_attention(q, k, v, attn_mask=safe, scale=scale).masked_fill(~valid, 0)


class SegmentedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        torch.set_num_threads(4)

    def compare(self, plan, mask):
        b, _, nq, nk = mask.shape
        inputs = [torch.randn(b, 2, n, 16, requires_grad=True) for n in (nq, nk, nk)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        a = plan.attend(*inputs, scale=.17)
        c = reference(*copies, mask, .17)
        torch.testing.assert_close(a, c, atol=2e-6, rtol=2e-5)
        a.square().sum().backward()
        c.square().sum().backward()
        for x, y in zip(inputs, copies):
            torch.testing.assert_close(x.grad, y.grad, atol=4e-6, rtol=3e-5)
        return inputs

    def test_history_block_causality_padding_and_gradients(self):
        layout = visual_layout(torch.arange(5).repeat(2, 1) * 4,
                               torch.tensor([[True]*5, [True, True, False, False, False]]), **geometry())
        layout.valid[0, 13:16] = False  # noncontiguous keys inside a block
        self.compare(history_attention_plan(layout), history_attention_mask(layout))
        q = torch.randn(2, 2, 60, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        plan = history_attention_plan(layout)
        a = plan.attend(q, k, v)
        v[:, :, 24:] += 100
        c = plan.attend(q, k, v)
        torch.testing.assert_close(a[:, :, :24], c[:, :, :24], atol=0, rtol=0)

    def test_future_filters_later_history_and_preserves_key_gradients(self):
        h = visual_layout(torch.arange(5).repeat(2, 1)*4, torch.ones(2,5,dtype=torch.bool), **geometry())
        decision = torch.tensor([8, 0])
        f = future_layout(decision, **geometry())
        f.valid[1, 4:] = False
        inputs = self.compare(future_attention_plan(h, f, decision), future_attention_mask(h, f, decision))
        self.assertEqual(inputs[1].grad[0, :, 36:60].abs().sum(), 0)
        self.assertGreater(inputs[1].grad[0, :, :36].abs().sum(), 0)

    def test_no_legal_keys_and_all_padding_have_zero_finite_gradients(self):
        qvalid = torch.tensor([[True]*4, [False]*4])
        kvalid = torch.tensor([[False]*7, [True]*7])
        plan = validity_attention_plan(qvalid, kvalid)
        inputs = self.compare(plan, qvalid[:, None, :, None] & kvalid[:, None, None, :])
        for x in inputs:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(x.grad.abs().sum(), 0)

    def test_81_block_plan_has_only_metadata_and_prefix_views(self):
        layout = visual_layout(torch.arange(81)[None]*4, torch.ones(1,81,dtype=torch.bool),
                               grid_height=14, grid_width=14, num_views=3)
        with patch("vjepa_policy.models.memory_masks.history_attention_mask", side_effect=AssertionError("dense mask")):
            plan = history_attention_plan(layout)
        self.assertEqual(len(plan.rows[0]), 81)
        self.assertEqual(plan.rows[0][-1].keys, ((0, 47628),))
        self.assertEqual(plan.query_valid.numel(), 47628)

    def test_complete_model_loss_kv_outputs_and_parameter_gradients(self):
        optimized = model(checkpointing=True).train()
        enable_attention_gates(optimized.action_expert)
        dense = copy.deepcopy(optimized)
        dense.predictor.attention_backend = dense.action_expert.attention_backend = "dense"
        b = batch()
        b["future_valid"][1, 24:] = False
        b["action_is_pad"][1, 16:] = True
        noise, time = torch.randn_like(b["action"]), torch.tensor([.3, .7])
        a = optimized(b, flow_noise=noise, flow_time=time)[0]
        c = dense(b, flow_noise=noise, flow_time=time)[0]
        torch.testing.assert_close(a, c, atol=2e-6, rtol=2e-5)
        a.backward(); c.backward()
        for (name, p), (_, r) in zip(optimized.named_parameters(), dense.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, r.grad, atol=4e-6, rtol=2e-4, msg=name)
        optimized.eval(); dense.eval()
        with torch.no_grad():
            for actual, expected in zip(optimized._encode_context(b)[:2], dense._encode_context(b)[:2]):
                for x, y in zip(actual.layers, expected.layers):
                    torch.testing.assert_close(x.key, y.key, atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(x.value, y.value, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(optimized.predict_action(b, initial_noise=noise),
                                       dense.predict_action(b, initial_noise=noise), atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
