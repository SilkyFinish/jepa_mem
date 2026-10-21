"""M2 CPU model tests. No dataset, pretrained checkpoint or GPU is needed."""

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("timm") is None:
    raise unittest.SkipTest("memory model tests require the installed policy environment")

import torch
from torch import nn
from torch.nn import functional as F

from vjepa_policy.encoders.vjepa2_1 import VJEPA2_1EncoderAdapter, VJEPA2_1_VITL_SPEC
from vjepa_policy.models.vjepa2_1.vision_transformer import VisionTransformer
from vjepa_policy.models.action_expert import ActionExpert, PredictorKVActionAttention
from vjepa_policy.models.backbone.rope import apply_3d_rope
from vjepa_policy.models.memory_action_expert import MemoryActionExpert
from vjepa_policy.models.memory_layouts import visual_layout, future_layout, action_layout
from vjepa_policy.models.memory_policy import FullHistoryVJEPAPolicy, build_full_history_policy, masked_sample_mean
from vjepa_policy.models.memory_predictor import MemoryWorldModelPredictor
from vjepa_policy.models.memory_types import LayerKV, HistoryEncoding
from vjepa_policy.models.vjepa_policy import encode_video_views
from vjepa_policy.models.wm_predictor import WorldModelPredictor
from vjepa_policy.state import pack_normalized_state

_spec = importlib.util.spec_from_file_location(
    "legacy_action_oracle", Path(__file__).parent / "fixtures/action_expert_1ef54c5.py"
)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)


def predictor_options(**extra):
    return dict(img_size=32, patch_size=16, num_frames=10, tubelet_size=2, embed_dim=64,
                predictor_embed_dim=128, depth=3, num_heads=2, num_views=3,
                proprio_dim=96, lang_dim=16, **extra)


def action_options():
    return dict(action_dim=14, action_chunk_size=32, num_layers=3,
                predictor_num_heads=2, predictor_head_dim=64, state_dim=96,
                hidden_size=64, lang_dim=16, condition_num_heads=2, num_inference_steps=3)


def encoder():
    # Exercise the actual V-JEPA2.1 implementation at reduced width/resolution.
    core = VisionTransformer(img_size=32, patch_size=16, num_frames=10, tubelet_size=2,
                             embed_dim=64, depth=12, num_heads=1, use_rope=True, use_sdpa=True,
                             interpolate_rope=True, img_temporal_dim_size=1,
                             modality_embedding=True, n_output_distillation=1)
    return VJEPA2_1EncoderAdapter(core, VJEPA2_1_VITL_SPEC)


def model(*, video=False, checkpointing=False):
    predictor = MemoryWorldModelPredictor(**predictor_options(use_activation_checkpointing=checkpointing))
    expert = MemoryActionExpert(rope_options=predictor.rope_options, **action_options())
    return FullHistoryVJEPAPolicy(encoder() if video else nn.Linear(1, 1), predictor, expert,
                                 encoder_batch_size=2)


def batch():
    valid = torch.tensor([[True] * 5, [True, False, False, False, False]])
    state = torch.randn(2, 5, 14)
    return dict(
        batch_schema="rmbench_history_v0", history_state=state,
        history_end_step=torch.arange(5).repeat(2, 1) * 4, history_valid=valid,
        decision_step=torch.tensor([16, 0]),
        current_state=torch.stack((state[0, 4], state[1, 0])),
        language=torch.randn(2, 4, 16), language_mask=torch.tensor([[True, True, False, False]] * 2),
        history_latents=torch.randn(2, 5, 12, 64), future_targets=torch.randn(2, 48, 64),
        future_valid=torch.ones(2, 48, dtype=torch.bool),
        action=torch.randn(2, 32, 14), action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
    )


def enable_attention_gates(expert):
    # Pretrained/updated DiT gates are nonzero; preserve zero initialization in production.
    with torch.no_grad():
        for block in expert.blocks:
            block.pre_attn_norm.modulation.bias[-64:] = 0.2


class LegacyActionCompatibilityTests(unittest.TestCase):
    def test_seeded_initialization_strict_load_forward_sampling_and_gradients(self):
        torch.manual_seed(71)
        old = _old.ActionExpert(**action_options())
        torch.manual_seed(71)
        new = ActionExpert(**action_options())
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)
        old.load_state_dict(new.state_dict(), strict=True)
        new.load_state_dict(old.state_dict(), strict=True)
        # Also cover nonzero attention gates, so compatibility is not vacuous.
        enable_attention_gates(old)
        enable_attention_gates(new)
        kv = tuple((torch.randn(2, 2, 8, 64), torch.randn(2, 2, 8, 64)) for _ in range(3))
        actions, language, state = torch.randn(2, 32, 14), torch.randn(2, 4, 16), torch.randn(2, 96)
        valid = torch.ones(2, 4, dtype=torch.bool)
        torch.manual_seed(2)
        expected = old(kv, actions, language, valid, state, context_len=4)
        torch.manual_seed(2)
        actual = new(kv, actions, language, valid, state, context_len=4)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        expected[0].square().sum().backward()
        actual[0].square().sum().backward()
        for a, b in zip(new.parameters(), old.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        torch.manual_seed(9)
        expected = old.predict_action(kv, language, valid, state)
        torch.manual_seed(9)
        actual = new.predict_action(kv, language, valid, state)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_legacy_injected_noise_and_generator(self):
        expert = ActionExpert(**action_options()).eval()
        kv = tuple((torch.randn(2, 2, 4, 64), torch.randn(2, 2, 4, 64)) for _ in range(3))
        args = (kv, torch.randn(2, 4, 16), torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 96))
        noise = torch.randn(2, 32, 14)
        a = expert.predict_action(*args, initial_noise=noise)
        b = expert.predict_action(*args, initial_noise=noise)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a = expert.predict_action(*args, generator=torch.Generator().manual_seed(3))
        b = expert.predict_action(*args, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            expert.predict_action(*args, initial_noise=noise, generator=torch.Generator())

    def test_predictor_weight_names_remain_transfer_compatible(self):
        legacy = WorldModelPredictor(**predictor_options())
        memory = MemoryWorldModelPredictor(**predictor_options())
        memory.load_state_dict(legacy.state_dict(), strict=True)
        legacy.load_state_dict(memory.state_dict(), strict=True)
        with self.assertRaisesRegex(TypeError, "forward_history"):
            memory(None, None, None)


class MemoryPredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model, self.batch = model().eval(), batch()

    def history(self, b):
        return self.model._encode_history(b)[0]

    def test_later_visual_and_state_changes_cannot_change_earlier_layers_kv(self):
        original = self.history(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["history_latents"][0, 4] *= -50
        changed["history_state"][0, 4] = torch.randn(14) * 10
        changed["current_state"][0] = changed["history_state"][0, 4]
        altered = self.history(changed)
        for a, b in zip(original.layers, altered.layers):
            torch.testing.assert_close(a.key[0, :, :48], b.key[0, :, :48], atol=0, rtol=0)
            torch.testing.assert_close(a.value[0, :, :48], b.value[0, :, :48], atol=0, rtol=0)
        self.assertFalse(torch.equal(original.layers[-1].key[0, :, 48:], altered.layers[-1].key[0, :, 48:]))

    def test_each_block_state_only_enters_subsequent_layer_kv(self):
        original = self.history(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["history_state"][0, 2] = torch.randn(14) * 5
        altered = self.history(changed)
        torch.testing.assert_close(original.layers[0].key, altered.layers[0].key, atol=0, rtol=0)
        for a, b in zip(original.layers, altered.layers):
            torch.testing.assert_close(a.key[0, :, :24], b.key[0, :, :24], atol=0, rtol=0)
        self.assertFalse(torch.equal(original.layers[1].key[0, :, 24:36], altered.layers[1].key[0, :, 24:36]))
        torch.testing.assert_close(original.layers[1].key[0, :, 36:], altered.layers[1].key[0, :, 36:], atol=0, rtol=0)

    def test_future_queries_do_not_modify_history_and_remain_local_size(self):
        history, current, language, _ = self.model._encode_history(self.batch)
        before = [(layer.key.clone(), layer.value.clone()) for layer in history.layers]
        predictor = self.model.predictor
        fl = self.model._future_layout(self.batch["decision_step"])
        first = predictor.query_future(history, current, language, self.batch["language_mask"],
                                       self.batch["decision_step"], fl)
        with torch.no_grad():
            predictor.mask_tokens[1].add_(torch.randn_like(predictor.mask_tokens[1]))
        second = predictor.query_future(history, current + 1, language, self.batch["language_mask"],
                                        self.batch["decision_step"], fl)
        self.assertEqual(first.prediction.shape, (2, 48, 64))
        self.assertEqual(predictor.num_patches, 60)
        self.assertFalse(torch.equal(first.prediction, second.prediction))
        for layer, (key, value) in zip(history.layers, before):
            torch.testing.assert_close(layer.key, key, atol=0, rtol=0)
            torch.testing.assert_close(layer.value, value, atol=0, rtol=0)

    def test_two_stage_matches_joint_masked_reference(self):
        b, predictor = self.batch, self.model.predictor
        history, future, current, language, _ = self.model._encode_context(b)
        hl, fl = history.layout, future.layout
        state = pack_normalized_state(b["history_state"], 48).masked_fill(~b["history_valid"][..., None], 0)
        hc, hv = predictor._history_condition(state, language, b["language_mask"])
        fc, fv = predictor._condition_tokens(language, b["language_mask"], current)
        x = predictor.predictor_embed(b["history_latents"]).flatten(1, 2)
        x = predictor._memory_view_embeddings(x, hl)
        f = predictor._memory_view_embeddings(predictor.mask_tokens[1].expand(2, 48, -1), fl)
        hcount = x.shape[1]
        tokens = torch.cat((x, f), 1)
        from vjepa_policy.models.memory_masks import history_attention_mask, future_attention_mask
        mask = torch.cat((
            F.pad(history_attention_mask(hl), (0, 48), value=False),
            future_attention_mask(hl, fl, b["decision_step"]),
        ), dim=2)
        coords = torch.cat((hl.rope_coords, fl.rope_coords), 1)
        valid = torch.cat((hl.valid, fl.valid), 1)
        for i, block in enumerate(predictor.predictor_blocks):
            out, key, value = block.self_attn(block.self_norm(tokens), position_coords=coords,
                                             attn_mask=mask, H_patches=2, W_patches=2, return_kv=True)
            torch.testing.assert_close(key[:, :, :hcount].masked_fill(~hl.valid[:, None, :, None], 0),
                                       history.layers[i].key, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(value[:, :, hcount:], future.layers[i].value, atol=3e-6, rtol=3e-5)
            tokens = tokens + out
            hx, fx = tokens[:, :hcount], tokens[:, hcount:]
            grouped = hx.reshape(10, 12, 128)
            hx = (grouped + block.cross_attn(block.cross_norm(grouped), hc, hv)).reshape(2, hcount, 128)
            fx = fx + block.cross_attn(block.cross_norm(fx), fc, fv)
            tokens = torch.cat((hx, fx), 1)
            tokens = (tokens + block.ffn(block.ffn_norm(tokens))).masked_fill(~valid[..., None], 0)
        ref = predictor.predictor_proj(predictor.predictor_norm(tokens[:, hcount:]))
        torch.testing.assert_close(ref, future.prediction, atol=3e-6, rtol=3e-5)

    def test_per_sample_prefix_equals_unpadded_recomputation(self):
        full = self.history(self.batch)
        b = {k: v[1:2].clone() if isinstance(v, torch.Tensor) else v for k, v in self.batch.items()}
        for name in ("history_state", "history_end_step", "history_valid", "history_latents"):
            b[name] = b[name][:, :1]
        single = self.history(b)
        for a, reference in zip(full.layers, single.layers):
            torch.testing.assert_close(a.key[1:2, :, :12], reference.key, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a.value[1:2, :, :12], reference.value, atol=2e-6, rtol=2e-5)

    def test_future_cutoff_ignores_later_history_and_state(self):
        # Lower-level query API accepts a prior decision from a complete encoding.
        history = self.history(self.batch)
        current = pack_normalized_state(self.batch["history_state"][:, 0], 48)
        decision = torch.zeros(2, dtype=torch.long)
        fl = self.model._future_layout(decision)
        kwargs = (current, self.batch["language"], self.batch["language_mask"], decision, fl)
        first = self.model.predictor.query_future(history, *kwargs)
        changed = copy.deepcopy(self.batch)
        changed["history_latents"][0, 1:] += 100
        changed["history_state"][0, 1:] *= -30
        changed["current_state"][0] = changed["history_state"][0, -1]
        second = self.model.predictor.query_future(self.history(changed), *kwargs)
        torch.testing.assert_close(first.prediction, second.prediction, atol=0, rtol=0)


class MemoryActionAndPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_action_attention_matches_explicit_rotated_reference(self):
        module = PredictorKVActionAttention(64, 2, 64).eval()
        hidden, k, v = torch.randn(2, 5, 64), torch.randn(2, 2, 7, 64), torch.randn(2, 2, 7, 64)
        cv, av = torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool)
        cv[1, -2:], av[1, -1] = False, False
        coords, options = torch.randn(2, 5, 3), dict(corrected_frequency_pairing=True)
        output = module(hidden, k, v, context_valid=cv, action_valid=av,
                        action_position_coords=coords, rope_options=options)
        clean = hidden.masked_fill(~av[..., None], 0)
        q = apply_3d_rope(module.q_proj(clean).view(2, 5, 2, 64).transpose(1, 2), coords, **options)
        ak = apply_3d_rope(module.k_proj(clean).view(2, 5, 2, 64).transpose(1, 2), coords, **options)
        value = torch.cat((v, module.v_proj(clean).view(2, 5, 2, 64).transpose(1, 2)), 2)
        allowed = torch.cat((cv, av), 1)[:, None, None, :]
        ref = F.scaled_dot_product_attention(q, torch.cat((k, ak), 2), value, attn_mask=allowed)
        ref = module.o_proj(ref.transpose(1, 2).reshape(2, 5, 128)).masked_fill(~av[..., None], 0)
        torch.testing.assert_close(output, ref, atol=0, rtol=0)
        av[:] = False
        empty = module(hidden.requires_grad_(), k, v, context_valid=cv, action_valid=av,
                       action_position_coords=coords, rope_options=options)
        torch.testing.assert_close(empty, torch.zeros_like(empty), atol=0, rtol=0)
        empty.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_full_prefix_forward_backward_optimizer_update_and_frozen_features(self):
        m, b = model().train(), batch()
        b["history_latents"].requires_grad_()
        b["future_targets"].requires_grad_()
        b["language"].requires_grad_()
        optimizer = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=1e-3)
        before = m.predictor.predictor_embed.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            loss, logs = m(b)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(logs), {"loss", "loss_world", "loss_action", "cos_sim"})
            loss.backward()
            for p in m.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
        self.assertFalse(torch.equal(before, m.predictor.predictor_embed.weight))
        self.assertIsNone(b["history_latents"].grad)
        self.assertIsNone(b["future_targets"].grad)
        self.assertIsNone(b["language"].grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.encoder.parameters()))
        self.assertFalse(m.encoder.training)

    def test_world_and_action_losses_both_reach_history_predictor(self):
        for world, action in ((1., 0.), (0., 1.)):
            m, b = model().train(), batch()
            enable_attention_gates(m.action_expert)
            m.world_loss_weight, m.action_loss_weight = world, action
            loss, _ = m(b)
            loss.backward()
            self.assertGreater(m.predictor.predictor_embed.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.predictor.proprio_projection.weight.grad.abs().sum().item(), 0)
            if action:
                self.assertGreater(m.predictor.mask_tokens[1].grad.abs().sum().item(), 0)

    def test_action_only_gradient_reaches_predictor_after_zero_gate_warmup(self):
        m, b = model().train(), batch()
        m.world_loss_weight = 0
        optimizer = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=1e-3)
        noise, time = torch.randn_like(b["action"]), torch.tensor([.3, .7])
        loss = m(b, flow_noise=noise, flow_time=time)[0]
        loss.backward()
        self.assertEqual(m.predictor.predictor_embed.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        m(b, flow_noise=noise, flow_time=time)[0].backward()
        self.assertGreater(m.predictor.predictor_embed.weight.grad.abs().sum().item(), 0)

    def test_checkpointing_matches_outputs_and_gradients(self):
        plain, checkpointed, b = model().train(), model(checkpointing=True).train(), batch()
        checkpointed.load_state_dict(plain.state_dict(), strict=True)
        enable_attention_gates(plain.action_expert)
        enable_attention_gates(checkpointed.action_expert)
        noise, time = torch.randn_like(b["action"]), torch.tensor([0.2, 0.8])
        a = plain(b, flow_noise=noise, flow_time=time)[0]
        c = checkpointed(b, flow_noise=noise, flow_time=time)[0]
        torch.testing.assert_close(a, c, atol=0, rtol=0)
        a.backward()
        c.backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)

    def test_autocast_bfloat16_forward_backward(self):
        m, b = model().train(), batch()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = m(b)
        loss.backward()
        self.assertEqual(m.forward_dtypes["predictor_k"], torch.bfloat16)
        self.assertEqual(m.forward_dtypes["action"], torch.bfloat16)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_inference_ignores_supervision_and_runs_predictor_once(self):
        m, b = model().eval(), batch()
        enable_attention_gates(m.action_expert)
        noise = torch.randn(2, 32, 14)
        original_noise = noise.clone()
        with patch.object(m.predictor, "forward_history", wraps=m.predictor.forward_history) as h, patch.object(
            m.predictor, "query_future", wraps=m.predictor.query_future
        ) as f:
            a = m.predict_action(b, initial_noise=noise)
            self.assertEqual(h.call_count, 1)
            self.assertEqual(f.call_count, 1)
        changed = copy.deepcopy(b)
        for name in ("future_targets", "action"):
            changed[name].fill_(float("nan"))
        changed["future_valid"][:] = False
        changed["action_is_pad"][:] = True
        changed["target_video"] = torch.full((1,), float("nan"))
        c = m.predict_action(changed, initial_noise=noise)
        torch.testing.assert_close(a, c, atol=0, rtol=0)
        torch.testing.assert_close(noise, original_noise, atol=0, rtol=0)
        self.assertEqual(a.shape, (2, 32, 14))
        self.assertTrue(torch.isfinite(a).all())
        with self.assertRaises(ValueError):
            m.predict_action(b, initial_noise=noise, generator=torch.Generator())

    def test_history_and_action_padding_poison_does_not_affect_valid_results(self):
        m, b = model().eval(), batch()
        enable_attention_gates(m.action_expert)
        b["future_valid"][1, 36:] = False
        b["action_is_pad"][1, 16:] = True
        noise, time = torch.randn_like(b["action"]), torch.tensor([0.3, 0.6])
        first = m(b, flow_noise=noise, flow_time=time)
        changed = copy.deepcopy(b)
        changed["history_latents"][1, 1:] = float("nan")
        changed["history_state"][1, 1:] = float("nan")
        changed["language"][:, 2:] = float("nan")
        changed["future_targets"][1, 36:] = float("nan")
        changed["action"][1, 16:] = float("nan")
        second = m(changed, flow_noise=noise, flow_time=time)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        first[0].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_appending_padding_and_changing_other_episode_does_not_change_actions(self):
        m, b = model().eval(), batch()
        enable_attention_gates(m.action_expert)
        noise = torch.randn(2, 32, 14)
        first = m.predict_action(b, initial_noise=noise)
        changed = copy.deepcopy(b)
        for name in ("history_state", "history_end_step", "history_valid", "history_latents"):
            tensor = changed[name]
            shape = (2, 2, *tensor.shape[2:])
            extra = torch.zeros(shape, dtype=tensor.dtype)
            if tensor.is_floating_point():
                extra.fill_(float("nan"))
            changed[name] = torch.cat((tensor, extra), dim=1)
        second = m.predict_action(changed, initial_noise=noise)
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
        changed["history_latents"][1, 0] *= -100
        changed["language"][1] = torch.randn_like(changed["language"][1])
        third = m.predict_action(changed, initial_noise=noise)
        torch.testing.assert_close(second[0], third[0], atol=0, rtol=0)

    def test_loss_normalization_is_per_sample(self):
        values = torch.tensor([[1., 100., 100.], [3., 3., 3.]], requires_grad=True)
        valid = torch.tensor([[True, False, False], [True, True, True]])
        result = masked_sample_mean(values, valid)
        self.assertEqual(result.item(), 2)
        result.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[.5, 0, 0], [1/6, 1/6, 1/6]]))
        self.assertEqual(masked_sample_mean(values, torch.zeros_like(valid)).item(), 0)

    def test_no_future_action_context_ablation(self):
        m, b = model().eval(), batch()
        enable_attention_gates(m.action_expert)
        h, f, state, lang, _ = m._encode_context(b)
        al = action_layout(b["decision_step"], ~b["action_is_pad"])
        noise = torch.randn_like(b["action"])
        args = (h, f, lang, b["language_mask"], state, al)
        first = m.action_expert.predict_action_memory(*args, initial_noise=noise)
        changed = replace(f, layers=tuple(LayerKV(k.key + 2, k.value + 100) for k in f.layers))
        args_changed = (h, changed, lang, b["language_mask"], state, al)
        second = m.action_expert.predict_action_memory(*args_changed, initial_noise=noise)
        self.assertFalse(torch.equal(first, second))
        m.action_expert.action_reads_future = False
        a = m.action_expert.predict_action_memory(*args, initial_noise=noise)
        c = m.action_expert.predict_action_memory(*args_changed, initial_noise=noise)
        torch.testing.assert_close(a, c, atol=0, rtol=0)

    def test_input_contract_rejects_legacy_gaps_states_and_wrong_layouts(self):
        m, good = model().eval(), batch()
        with self.assertRaises(ValueError):
            m(tuple(good.values()))
        cases = [
            ("batch_schema", "legacy"),
            ("current_state", good["current_state"] + 1),
            ("history_end_step", good["history_end_step"] + 4),
            ("decision_step", torch.tensor([20, 0])),
            ("history_valid", torch.zeros_like(good["history_valid"])),
            ("history_latents", good["history_latents"][:, :, :3]),
            ("language_mask", good["language_mask"].float()),
        ]
        for name, value in cases:
            altered = dict(good, **{name: value})
            with self.subTest(field=name), self.assertRaises(ValueError):
                m(altered)
        with self.assertRaises(ValueError):
            m(dict(good, history_video=torch.zeros(1)))
        with self.assertRaises(ValueError):
            build_full_history_policy(m.encoder, m.predictor)


class FrozenEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        self.model = model(video=True).eval()

    def test_independent_two_frame_encoding_equals_context_masked_ten_frame_encoding(self):
        e = self.model.encoder
        clips = torch.randn(2, 3, 3, 10, 32, 32)
        ids = torch.arange(12).repeat(2, 1)
        with torch.no_grad():
            independent = encode_video_views(e, clips[:, :, :, :2])
            context = encode_video_views(e, clips, ids)
            clips[:, :, :, 2:] = torch.randn_like(clips[:, :, :, 2:]) * 100
            perturbed = encode_video_views(e, clips, ids)
        torch.testing.assert_close(independent, context, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(context, perturbed, atol=0, rtol=0)

    def test_raw_video_and_frozen_feature_paths_match_without_target_norm_on_observations(self):
        m, b = self.model, batch()
        del b["history_latents"], b["future_targets"]
        b["history_video"] = torch.randn(2, 5, 3, 3, 2, 32, 32)
        b["target_video"] = torch.randn(2, 3, 3, 10, 32, 32)
        b["history_video"][1, 1:] = float("nan")  # invalid clips must not enter encoder
        z = m.encode_observed_blocks(b)
        y = m.encode_local_future_targets(b)
        with torch.no_grad():
            direct = encode_video_views(m.encoder, b["history_video"][0, 0:1])
            full_target = encode_video_views(m.encoder, b["target_video"])
            expected = F.layer_norm(full_target, (64,))[:, 12:]
        torch.testing.assert_close(z[0, 0:1], direct, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(y, expected, atol=0, rtol=0)
        cached = {k: v for k, v in b.items() if k not in ("history_video", "target_video")}
        cached.update(history_latents=z, future_targets=y)
        noise, time = torch.randn(2, 32, 14), torch.tensor([.2, .8])
        raw = m(b, flow_noise=noise, flow_time=time)[0]
        saved = m(cached, flow_noise=noise, flow_time=time)[0]
        torch.testing.assert_close(raw, saved, atol=0, rtol=0)
        raw.backward()
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
        m.train()
        self.assertFalse(m.encoder.training)
        self.assertFalse(m.encoder.encoder.training)


if __name__ == "__main__":
    unittest.main()
