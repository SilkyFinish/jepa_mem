"""CPU numerical and leakage tests for M1; skipped without policy dependencies."""

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("timm") is None:
    raise unittest.SkipTest("memory operators require the installed policy environment")

import torch

from vjepa_policy.memory_config import FullHistoryV0Config
from vjepa_policy.models.backbone.modules import RoPEAttention
from vjepa_policy.models.backbone.rope import apply_3d_rope, rope_dimensions
from vjepa_policy.models.memory_masks import (
    action_attention_mask, future_attention_mask, history_attention_mask,
)
from vjepa_policy.models.memory_types import (
    FutureEncoding, HistoryEncoding, LayerKV, TokenLayout,
)
from vjepa_policy.models.wm_predictor import WorldModelPredictorBlock

_spec = importlib.util.spec_from_file_location(
    "legacy_rope_oracle", Path(__file__).parent / "fixtures/rope_attention_e9be66b.py"
)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)


def layout(ends=(0, 4, 8), *, batch=2, per_block=4):
    end = torch.tensor(ends).long().repeat_interleave(per_block)[None].repeat(batch, 1)
    block = torch.arange(len(ends)).repeat_interleave(per_block)[None].repeat(batch, 1)
    patch = torch.arange(end.shape[1]) % 4
    coords = torch.stack(
        (end.float() / 8, (patch // 2).expand_as(end).float(), (patch % 2).expand_as(end).float()),
        dim=-1,
    )
    views = ((torch.arange(end.shape[1]) % per_block) // 4)[None].expand_as(end).clone()
    return TokenLayout(torch.ones_like(end, dtype=torch.bool), block, end, coords, views)


def take(meta, start, stop):
    return TokenLayout(**{name: getattr(meta, name)[:, start:stop] for name in meta.__dataclass_fields__})


def attention(**kwargs):
    return RoPEAttention(128, num_heads=2, qkv_bias=True, grid_size=2, **kwargs)


class MemoryTypesAndMasksTests(unittest.TestCase):
    def test_block_causal_not_token_causal_and_multiview(self):
        meta = layout(per_block=12)
        mask = history_attention_mask(meta)
        expected = torch.zeros(36, 36, dtype=torch.bool)
        for block in range(3):
            expected[12 * block:12 * (block + 1), :12 * (block + 1)] = True
        torch.testing.assert_close(mask[0, 0], expected)
        self.assertTrue(mask[0, 0, 0, 11])  # first camera reads last camera in same block
        self.assertFalse(mask[0, 0, 11, 12])

    def test_padding_and_per_sample_decision_cutoff(self):
        hist = layout()
        hist = replace(hist, valid=hist.valid.clone())
        hist.valid[1, 4:] = False
        future = layout((4,), per_block=2)
        future = replace(future, end_step=torch.tensor([[4, 4], [0, 0]]))
        future.rope_coords[..., 0] = 100  # target time must not expand history access
        future.valid[1, 1] = False
        mask = future_attention_mask(hist, future, torch.tensor([4, 0]))
        self.assertEqual(mask.shape, (2, 1, 2, 14))
        self.assertTrue(mask[0, 0, :, :8].all())
        self.assertFalse(mask[0, 0, :, 8:12].any())
        self.assertFalse(mask[1, 0, :, 4:12].any())
        self.assertFalse(mask[1, 0, 1].any())
        self.assertFalse(mask[1, 0, :, 13].any())

    def test_action_reads_future_and_bidirectional_chunk(self):
        hist, future, actions = layout(), layout((4,), per_block=2), layout((4,), per_block=3)
        mask = action_attention_mask(hist, future, actions, torch.tensor([4, 4]))
        self.assertTrue(mask[..., :8].all())
        self.assertFalse(mask[..., 8:12].any())
        self.assertTrue(mask[..., 12:].all())
        self.assertTrue(mask[0, 0, 0, -1])  # no action token triangular causal mask
        actions.valid[1, 2] = False
        mask = action_attention_mask(hist, future, actions, torch.tensor([4, 4]))
        self.assertFalse(mask[1, 0, 2].any())
        self.assertFalse(mask[1, 0, :, -1].any())

    def test_mixed_decisions_and_invalid_contracts_rejected(self):
        hist, future = layout(), layout((4,), per_block=2)
        future.end_step[0, 1] = 8
        with self.assertRaisesRegex(ValueError, "decision_step"):
            future_attention_mask(hist, future, torch.tensor([4, 4]))
        with self.assertRaisesRegex(ValueError, "decision_step"):
            action_attention_mask(hist, layout((0,)), layout((4,)), torch.tensor([4, 4]))
        with self.assertRaises(ValueError):
            replace(hist, valid=hist.valid.float())
        with self.assertRaises(ValueError):
            replace(hist, rope_coords=hist.rope_coords[..., :2])
        with self.assertRaises(ValueError):
            replace(hist, block_id=hist.block_id.float())
        with self.assertRaises(ValueError):
            replace(hist, rope_coords=torch.full_like(hist.rope_coords, float("nan")))
        with self.assertRaises(ValueError):
            history_attention_mask(hist, layout(batch=1))

    def test_dense_limit_rejects_before_pairwise_allocation(self):
        hist = layout((0,), batch=1, per_block=588 * 81)
        with self.assertRaisesRegex(ValueError, "never truncated"):
            history_attention_mask(hist)
        small = layout(batch=1)
        with self.assertRaisesRegex(ValueError, "limit"):
            future_attention_mask(small, layout((0,), batch=1), torch.tensor([0]), max_elements=1)

    def test_encoding_shapes_and_graph_are_preserved(self):
        hist = layout()
        key = torch.randn(2, 2, 12, 64, requires_grad=True)
        kv = LayerKV(key * 2, key * 3)
        encoding = HistoryEncoding((kv,), hist)
        self.assertIs(encoding.layers[0].key, kv.key)
        encoding.layers[0].key.sum().backward()
        torch.testing.assert_close(key.grad, torch.full_like(key, 2))
        future = layout((4,))
        FutureEncoding(torch.zeros(2, 4, 32), (LayerKV(kv.key[:, :, :4], kv.value[:, :, :4]),),
                       future, torch.tensor([4, 4]))
        with self.assertRaises(ValueError):
            LayerKV(kv.key, kv.value[..., :1])
        with self.assertRaises(ValueError):
            HistoryEncoding((kv,), future)
        with self.assertRaises(ValueError):
            FutureEncoding(torch.zeros(2, 3, 32), (kv,), hist, torch.tensor([0, 0]))


class LegacyAndRoPETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)

    def test_frozen_legacy_oracle_strict_load_output_and_kv(self):
        for corrected in (False, True):
            for sdpa in (False, True):
                for interpolate in (False, True):
                    for selected in (False, True):
                        for training in (False, True):
                            with self.subTest(corrected=corrected, sdpa=sdpa, interpolate=interpolate,
                                              selected=selected, training=training):
                                options = dict(dim=128, num_heads=2, qkv_bias=True, grid_size=2,
                                               corrected_frequency_pairing=corrected, use_sdpa=sdpa,
                                               interpolate_rope=interpolate, attn_drop=0.1, proj_drop=0.2)
                                old = _legacy.RoPEAttention(**options).train(training)
                                new = RoPEAttention(**options).train(training)
                                new.load_state_dict(old.state_dict(), strict=True)
                                old.load_state_dict(new.state_dict(), strict=True)
                                x = torch.randn(2, 4 if selected else 8, 128)
                                ids = torch.tensor([[0, 3, 4, 7], [2, 1, 6, 5]]) if selected else None
                                old_hook, new_hook = [], []
                                torch.manual_seed(64)
                                expected = old(x, mask=ids, T=2, H_patches=2, W_patches=2,
                                               return_kv=True, kv_out=old_hook)
                                torch.manual_seed(64)
                                actual = new(x, mask=ids, T=2, H_patches=2, W_patches=2,
                                             return_kv=True, kv_out=new_hook)
                                for a, b in zip(actual, expected):
                                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                                self.assertIs(new_hook[0][0], actual[1])
        # Preserve the legacy meaning of a float attention bias too.
        old = _legacy.RoPEAttention(128, 2, grid_size=2)
        new = RoPEAttention(128, 2, grid_size=2)
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 4, 128)
        bias = torch.zeros(4, 4)
        bias[:, -1] = float("-inf")
        torch.testing.assert_close(new(x, attn_mask=bias), old(x, attn_mask=bias), rtol=0, atol=0)

    def test_predictor_block_legacy_strict_load_and_output(self):
        new = WorldModelPredictorBlock(128, 2, grid_size=2, patch_size=16).eval()
        old = copy.deepcopy(new)
        old.self_attn = _legacy.RoPEAttention(
            128, 2, qkv_bias=True, grid_size=2, interpolate_rope=True, patch_size=16,
        )
        old.load_state_dict(new.state_dict(), strict=True)
        # The frozen attention predates the new keyword args; compare the
        # original block formula while loading exactly the same block weights.
        x, condition = torch.randn(2, 8, 128), torch.randn(2, 3, 128)
        valid, ids = torch.ones(2, 3, dtype=torch.bool), torch.arange(8).repeat(2, 1)
        a, k, v = old.self_attn(old.self_norm(x), mask=ids, T=2, H_patches=2, W_patches=2, return_kv=True)
        ref = x + old.drop_path(a)
        ref = ref + old.drop_path(old.cross_attn(old.cross_norm(ref), condition, valid))
        ref = ref + old.drop_path(old.ffn(old.ffn_norm(ref)))
        actual = new(x, condition, valid, ids, T=2, H_patches=2, W_patches=2, return_kv=True)
        for a, b in zip(actual, (ref, k, v)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_explicit_coords_match_legacy_positions_with_both_pairings(self):
        ids = torch.tensor([[0, 3, 4, 7], [7, 4, 3, 0]])
        coords = torch.stack((ids // 4, (ids % 4) // 2, ids % 2), dim=-1).float()
        for corrected in (False, True):
            model = attention(corrected_frequency_pairing=corrected, interpolate_rope=True).eval()
            x = torch.randn(2, 4, 128)
            legacy = model(x, mask=ids, T=2, H_patches=2, W_patches=2, return_kv=True)
            explicit = model(x, position_coords=coords, attn_mask=torch.ones(2, 1, 4, 4, dtype=torch.bool),
                             H_patches=2, W_patches=2, return_kv=True)
            for a, b in zip(explicit, legacy):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_shared_rope_split_remainder_and_translation(self):
        self.assertEqual(rope_dimensions(64), (20, 20, 20))
        x, y = torch.randn(2, 2, 7, 64), torch.randn(2, 2, 5, 64)
        xc, yc = torch.randn(2, 7, 3), torch.randn(2, 5, 3)
        options = dict(corrected_frequency_pairing=True, spatial_grid=(14, 14),
                       canonical_spatial_grid=(16, 16))
        xr, yr = apply_3d_rope(x, xc, **options), apply_3d_rope(y, yc, **options)
        torch.testing.assert_close(xr[..., 60:], x[..., 60:], rtol=0, atol=0)
        xc[..., 0] += 10
        yc[..., 0] += 10
        shifted_x, shifted_y = apply_3d_rope(x, xc, **options), apply_3d_rope(y, yc, **options)
        torch.testing.assert_close(xr @ yr.transpose(-2, -1),
                                   shifted_x @ shifted_y.transpose(-2, -1), rtol=3e-5, atol=3e-5)
        # Legacy frequency pairing is not an orthogonal rotation; do not
        # assert translation invariance for pretrained legacy weights.

    def test_v0_global_coordinates_and_dtype(self):
        coords = FullHistoryV0Config().temporal_coordinates(16)
        # These are control-frame target times, independent of local token order.
        self.assertEqual(coords["future"], (3.0, 4.0, 5.0, 6.0))
        self.assertEqual(coords["action"][0], 17 / 8)
        for dtype in (torch.float32, torch.bfloat16):
            for dim in (6, 12, 64):
                x, pos = torch.randn(2, 2, 4, dim).to(dtype), torch.randn(2, 4, 3)
                result = apply_3d_rope(x, pos)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            apply_3d_rope(x, pos, spatial_grid=(14, 14))


class PrefixAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def run_attention(self, model, x, meta, **kwargs):
        return model(x, position_coords=meta.rope_coords,
                     attn_mask=history_attention_mask(meta), return_kv=True, **kwargs)

    def test_causality_padding_and_batch_isolation(self):
        meta, x = layout(), torch.randn(2, 12, 128)
        meta.valid[1, 8:] = False
        for sdpa in (False, True):
            model = attention(use_sdpa=sdpa).eval()
            first = self.run_attention(model, x, meta)[0]
            changed = x.clone()
            changed[0, 8:] = torch.randn_like(changed[0, 8:]) * 100
            changed[1] = torch.randn_like(changed[1]) * 100
            second = self.run_attention(model, changed, meta)[0]
            torch.testing.assert_close(first[0, :8], second[0, :8], rtol=0, atol=0)
            self.assertFalse(torch.equal(first[0, 8:], second[0, 8:]))
            self.assertTrue(torch.isfinite(second).all())
            torch.testing.assert_close(second[1, 8:], torch.zeros_like(second[1, 8:]), rtol=0, atol=0)
            # Changing only padded keys cannot affect valid queries.
            changed = x.clone()
            changed[1, 8:] *= 100
            third = self.run_attention(model, changed, meta)[0]
            torch.testing.assert_close(first[1, :8], third[1, :8], rtol=0, atol=0)

    def test_multilayer_full_recompute_equals_incremental_and_current_only_kv(self):
        for corrected in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(corrected=corrected, dtype=dtype):
                    blocks = torch.nn.ModuleList([
                        WorldModelPredictorBlock(128, 2, grid_size=2, patch_size=16,
                                                 corrected_rope_frequency_pairing=corrected)
                        for _ in range(3)
                    ]).to(dtype).eval()
                    x, meta = torch.randn(2, 12, 128).to(dtype), layout()
                    meta.valid[1, 8:] = False
                    condition = torch.randn(2, 3, 128).to(dtype)
                    valid = torch.ones(2, 3, dtype=torch.bool)
                    full, full_kv = x, []
                    common = dict(T=None, H_patches=2, W_patches=2, return_kv=True)
                    for block in blocks:
                        full, k, v = block(full, condition, valid, None, **common,
                                          position_coords=meta.rope_coords, attn_mask=history_attention_mask(meta))
                        full_kv.append(LayerKV(k, v))
                    prefix, pieces = [None] * 3, []
                    for start in (0, 4, 8):
                        current = x[:, start:start + 4]
                        cm, km = take(meta, start, start + 4), take(meta, 0, start + 4)
                        for i, block in enumerate(blocks):
                            current, k, v = block(
                                current, condition, valid, None, **common,
                                position_coords=cm.rope_coords, prefix_kv=prefix[i],
                                attn_mask=history_attention_mask(cm, km),
                            )
                            self.assertEqual(k.shape[2], 4)
                            prefix[i] = LayerKV(k, v) if prefix[i] is None else LayerKV(
                                torch.cat((prefix[i].key, k), dim=2), torch.cat((prefix[i].value, v), dim=2)
                            )
                        pieces.append(current)
                    tol = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=0.025, rtol=0.025)
                    torch.testing.assert_close(torch.cat(pieces, dim=1), full, **tol)
                    for incremental, reference in zip(prefix, full_kv):
                        torch.testing.assert_close(incremental.key, reference.key, **tol)
                        torch.testing.assert_close(incremental.value, reference.value, **tol)

    def test_training_recompute_and_incremental_parameter_gradients_match(self):
        model = attention(corrected_frequency_pairing=True).train()
        incremental_model = copy.deepcopy(model)
        meta = layout()
        x = torch.randn(2, 12, 128, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        full = self.run_attention(model, x, meta)[0]
        full.square().sum().backward()
        prefix, pieces = None, []
        for start in (0, 4, 8):
            current, keys = take(meta, start, start + 4), take(meta, 0, start + 4)
            out, k, v = incremental_model(
                y[:, start:start + 4], position_coords=current.rope_coords,
                prefix_kv=prefix, attn_mask=history_attention_mask(current, keys), return_kv=True,
            )
            pieces.append(out)
            prefix = LayerKV(k, v) if prefix is None else LayerKV(
                torch.cat((prefix.key, k), dim=2), torch.cat((prefix.value, v), dim=2),
            )
        torch.cat(pieces, dim=1).square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)
        for reference, actual in zip(model.parameters(), incremental_model.parameters()):
            torch.testing.assert_close(reference.grad, actual.grad, atol=3e-5, rtol=3e-5)

    def test_empty_prefix_and_autocast(self):
        model, meta = attention().eval(), layout((0,))
        x = torch.randn(2, 4, 128)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            full, k, v = self.run_attention(model, x, meta)
            empty = LayerKV(k[:, :, :0], v[:, :, :0])
            actual, ak, av = model(x, position_coords=meta.rope_coords, prefix_kv=empty,
                                   attn_mask=history_attention_mask(meta), return_kv=True)
        self.assertEqual(ak.dtype, torch.bfloat16)
        self.assertEqual(av.dtype, torch.bfloat16)
        torch.testing.assert_close(full, actual, atol=0, rtol=0)
        torch.testing.assert_close(k, ak, atol=0, rtol=0)

    def test_future_reads_legal_prefix_without_mutation_and_gradients_flow(self):
        model = attention(corrected_frequency_pairing=True).eval()
        hist, future = layout(), layout((4,), per_block=4)
        future.rope_coords[..., 0] = torch.tensor([2., 3., 4., 5.])
        h = torch.randn(2, 12, 128, requires_grad=True)
        f = torch.randn(2, 4, 128, requires_grad=True)
        _, hk, hv = self.run_attention(model, h, hist)
        prefix = LayerKV(hk, hv)
        before = hk.clone(), hv.clone()
        seen = []
        mask = future_attention_mask(hist, future, torch.tensor([4, 4]))
        output, fk, fv = model(f, position_coords=future.rope_coords, prefix_kv=prefix,
                               attn_mask=mask, return_kv=True, kv_out=seen)
        self.assertEqual(fk.shape[2], 4)
        self.assertIs(seen[0][0], fk)
        self.assertIs(seen[0][1], fv)
        altered = LayerKV(hk.clone(), hv.clone())
        altered.key[:, :, 8:] = 999
        altered.value[:, :, 8:] = -999
        again = model(f, position_coords=future.rope_coords, prefix_kv=altered, attn_mask=mask)
        torch.testing.assert_close(output, again, rtol=0, atol=0)
        # Repeated temporary queries must not change an existing history prefix.
        model(f + 2, position_coords=future.rope_coords, prefix_kv=prefix, attn_mask=mask)
        torch.testing.assert_close(hk, before[0], rtol=0, atol=0)
        torch.testing.assert_close(hv, before[1], rtol=0, atol=0)
        output.square().sum().backward()
        self.assertGreater(h.grad[:, :8].abs().sum().item(), 0)
        self.assertEqual(h.grad[:, 8:].abs().sum().item(), 0)
        self.assertGreater(f.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_sdpa_matches_manual_reference_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            meta = layout()
            meta.valid[1, 8:] = False
            sdpa = attention(qk_scale=0.2).to(dtype).eval()
            manual = copy.deepcopy(sdpa)
            manual.use_sdpa = False
            x = torch.randn(2, 12, 128).to(dtype).requires_grad_()
            y = x.detach().clone().requires_grad_()
            a = self.run_attention(sdpa, x, meta)[0]
            b = self.run_attention(manual, y, meta)[0]
            tol = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=0.01, rtol=0.02)
            torch.testing.assert_close(a, b, **tol)
            a.float().square().sum().backward()
            b.float().square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad, **tol)

    def test_all_masked_rows_zero_output_and_zero_finite_gradients(self):
        for sdpa in (False, True):
            model = attention(use_sdpa=sdpa).eval()
            x, meta = torch.randn(2, 4, 128, requires_grad=True), layout((0,))
            meta.valid[:] = False
            y = self.run_attention(model, x, meta)[0]
            torch.testing.assert_close(y, torch.zeros_like(y), rtol=0, atol=0)
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.zeros_like(x.grad), rtol=0, atol=0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_eval_disables_attention_and_projection_dropout(self):
        model, meta = attention(attn_drop=0.5, proj_drop=0.5), layout((0,))
        x = torch.randn(2, 4, 128)
        model.eval()
        first = self.run_attention(model, x, meta)[0]
        second = self.run_attention(model, x, meta)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(self.run_attention(model, x, meta)[0],
                                     self.run_attention(model, x, meta)[0]))

    def test_invalid_or_ambiguous_memory_arguments_rejected(self):
        model, meta = attention().eval(), layout((0,))
        x, mask = torch.randn(2, 4, 128), history_attention_mask(meta)
        _, k, v = self.run_attention(model, x, meta)
        cases = [
            dict(mask=torch.arange(4).repeat(2, 1), position_coords=meta.rope_coords, attn_mask=mask),
            dict(prefix_kv=LayerKV(k, v), attn_mask=mask),
            dict(position_coords=meta.rope_coords),
            dict(position_coords=meta.rope_coords, attn_mask=mask.float()),
            dict(position_coords=meta.rope_coords, attn_mask=mask[:, 0]),
            dict(position_coords=meta.rope_coords, attn_mask=mask, prefix_kv=(k, v)),
            dict(position_coords=meta.rope_coords, attn_mask=mask, prefix_kv=LayerKV(k[:1], v[:1])),
            dict(position_coords=meta.rope_coords, attn_mask=mask, prefix_kv=LayerKV(k.double(), v.double())),
            dict(position_coords=meta.rope_coords, attn_mask=mask, prefix_kv=LayerKV(k, v)),
        ]
        for kwargs in cases:
            with self.subTest(keys=tuple(kwargs)), self.assertRaises(ValueError):
                model(x, **kwargs)
        model.is_causal = True
        with self.assertRaisesRegex(ValueError, "block mask"):
            self.run_attention(model, x, meta)


if __name__ == "__main__":
    unittest.main()
