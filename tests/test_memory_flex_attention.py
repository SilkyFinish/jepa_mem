"""CUDA checks for batched validity attention, including entirely empty rows."""
import unittest
import torch
from vjepa_policy.models.memory_attention import validity_attention_plan
from vjepa_policy.models.memory_flex_attention import flex_validity_attention_plan

@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention requires CUDA")
class FlexValidityTests(unittest.TestCase):
    def test_outputs_and_gradients_with_gaps_and_empty_rows(self):
        self._check(7,13)

    def test_bounded_mask_with_multiple_tiles(self):
        self._check(147,147)

    def test_compiled_large_mask_with_gaps_and_empty_rows(self):
        self._check(257,1025)

    def _check(self, query_length, key_length):
        for head_dim in (32, 64):
            with self.subTest(head_dim=head_dim):
                self._check_head(query_length, key_length, head_dim)

    def _check_head(self, query_length, key_length, head_dim):
        torch.manual_seed(41)
        query_valid = torch.ones(3,query_length,dtype=torch.bool,device="cuda")
        query_valid[0,-2:] = False
        key_valid = torch.ones(3,key_length,dtype=torch.bool,device="cuda")
        key_valid[0,3:7] = False
        key_valid[1] = False
        query_valid[2] = False
        reference = validity_attention_plan(query_valid,key_valid)
        candidate = flex_validity_attention_plan(query_valid,key_valid)
        for dtype in (torch.float32,torch.bfloat16):
            q = torch.randn(3,2,query_length,head_dim,device="cuda",dtype=dtype,requires_grad=True)
            k = torch.randn(3,2,key_length,head_dim,device="cuda",dtype=dtype,requires_grad=True)
            v = torch.randn_like(k,requires_grad=True)
            a, b = reference.attend(q,k,v), candidate.attend(q,k,v)
            atol, rtol = (3e-6,3e-5) if dtype==torch.float32 else (.02,.04)
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
            self.assertTrue(torch.isfinite(b).all())
            self.assertEqual(torch.count_nonzero(b[1:]).item(),0)
            weight = torch.randn_like(a)
            ga = torch.autograd.grad((a*weight).sum(),(q,k,v),retain_graph=True)
            gb = torch.autograd.grad((b*weight).sum(),(q,k,v))
            for x,y in zip(ga,gb):
                torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
                self.assertEqual(torch.count_nonzero(y[1:]).item(),0)
