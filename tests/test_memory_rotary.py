"""Exact rotary-factor reuse for corrected and legacy coordinates."""
import unittest
import torch
from vjepa_policy.models.backbone.rope import apply_3d_rope
from vjepa_policy.models.memory_attention import history_attention_plan
from vjepa_policy.models.memory_layouts import visual_layout
from vjepa_policy.models.memory_rotary import rotary_attention_plan

class RotaryReuseTests(unittest.TestCase):
    def test_output_and_gradients_match_with_tail_channels(self):
        torch.manual_seed(27)
        layout = visual_layout(torch.tensor([[0,400],[0,400]]),torch.ones(2,2,dtype=torch.bool),
                               grid_height=2,grid_width=2,num_views=3)
        for corrected in (True, False):
            options = dict(corrected_frequency_pairing=corrected,
                           spatial_grid=(14,14),canonical_spatial_grid=(16,16))
            plan = rotary_attention_plan(history_attention_plan(layout),layout,64,options)
            q = torch.randn(2,2,24,64,requires_grad=True)
            k = torch.randn(2,2,24,64,requires_grad=True)
            actual = plan.rotate(q,k)
            expected = (apply_3d_rope(q,layout.rope_coords,**options),
                        apply_3d_rope(k,layout.rope_coords,**options))
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            grad = torch.randn_like(q)
            a = torch.autograd.grad(sum((x*grad).sum() for x in actual),(q,k),retain_graph=True)
            b = torch.autograd.grad(sum((x*grad).sum() for x in expected),(q,k))
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
