"""Recompute-only full-history policy. The legacy builder remains unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from vjepa_policy.memory_config import FullHistoryV0Config
from vjepa_policy.state import pack_normalized_state
from .memory_action_expert import MemoryActionExpert
from .memory_layouts import visual_layout, future_layout, action_layout
from .memory_predictor import MemoryWorldModelPredictor
from .vjepa_policy import encode_video_views


def masked_sample_mean(values, valid):
    """Normalize each sample before averaging; empty samples contribute zero."""
    clean = values.masked_fill(~valid, 0)
    return (clean.sum(1) / valid.sum(1).clamp_min(1)).mean()


class FullHistoryVJEPAPolicy(nn.Module):
    """Model core accepting the v0 dict batch; reduced geometry supports unit tests.

    Use build_full_history_policy for production v0 geometry validation.
    H/F tensors are local to each call and are never reused across updates.
    """

    def __init__(self, encoder, predictor, action_expert, *,
                 config=None, encoder_batch_size=16, strict_v0_inputs=False,
                 world_loss_weight=1.0, action_loss_weight=1.0):
        super().__init__()
        self.config = config or FullHistoryV0Config()
        self.encoder, self.predictor, self.action_expert = encoder, predictor, action_expert
        self.encoder_batch_size = encoder_batch_size
        self.strict_v0_inputs = strict_v0_inputs
        self.world_loss_weight, self.action_loss_weight = world_loss_weight, action_loss_weight
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _validate_observations(self, batch):
        if not isinstance(batch, dict) or batch.get("batch_schema") != self.config.batch_schema:
            raise ValueError(f"expected dict batch_schema={self.config.batch_schema}; legacy tuples are not supported")
        required = ("history_state", "history_end_step", "history_valid", "decision_step",
                    "current_state", "language", "language_mask")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(f"missing observation fields: {missing}")
        state, ends, valid = batch["history_state"], batch["history_end_step"], batch["history_valid"]
        decision, current = batch["decision_step"], batch["current_state"]
        if state.ndim != 3 or state.shape[-1] != self.config.raw_state_dim:
            raise ValueError("history_state must be [B,K,14]")
        batch_size, blocks = state.shape[:2]
        if batch_size == 0 or blocks == 0:
            raise ValueError("at least one sample and history block are required")
        if ends.shape != (batch_size, blocks) or ends.dtype != torch.long:
            raise ValueError("history_end_step must be long [B,K]")
        if valid.shape != ends.shape or valid.dtype != torch.bool:
            raise ValueError("history_valid must be bool [B,K]")
        if decision.shape != (batch_size,) or decision.dtype != torch.long:
            raise ValueError("decision_step must be long [B]")
        if current.shape != (batch_size, self.config.raw_state_dim):
            raise ValueError("current_state must be [B,14]")
        for name in required:
            if batch[name].device != state.device:
                raise ValueError("observation tensors must share a device")
        lengths = valid.sum(1)
        if (lengths == 0).any():
            raise ValueError("every sample requires an initial observation")
        expected_valid = torch.arange(blocks, device=state.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected_valid):
            raise ValueError("history_valid must be a right-padded prefix")
        expected_ends = torch.arange(blocks, device=state.device)[None] * self.config.history_interval_steps
        if ((ends != expected_ends) & valid).any():
            raise ValueError("history must include every observation from episode start without gaps")
        if ((lengths - 1) * self.config.history_interval_steps != decision).any():
            raise ValueError("history must end exactly at decision_step")
        if (decision % self.config.decision_interval_steps != 0).any():
            raise ValueError("decision_step is outside the v0 decision grid")
        last = state[torch.arange(batch_size, device=state.device), lengths - 1]
        if not torch.equal(current, last):
            raise ValueError("current_state must equal the last valid historical state")
        language, language_mask = batch["language"], batch["language_mask"]
        if language.ndim != 3 or language.shape[0] != batch_size or language.shape[-1] != self.predictor.lang_dim:
            raise ValueError("language must match Predictor [B,S,D_language]")
        if self.strict_v0_inputs and language.shape[1:] != (self.config.text_length, self.config.text_dim):
            raise ValueError("production v0 requires [B,128,4096] language features")
        if language_mask.shape != language.shape[:2] or language_mask.dtype != torch.bool:
            raise ValueError("language_mask must be bool [B,S]")
        for name, tensor in (("history_state", state[valid]), ("language", language[language_mask])):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"valid {name} must be finite floating point")
        if ("history_video" in batch) == ("history_latents" in batch):
            raise ValueError("supply exactly one of history_video and history_latents")
        return state.masked_fill(~valid[..., None], 0), ends.masked_fill(~valid, 0), valid

    def _check_video_encoder(self):
        if getattr(self.encoder, "_vjepa_policy_family", None) != "vjepa2_1":
            raise ValueError("v0 video encoding requires the V-JEPA2.1 adapter")
        if self.encoder.patch_size != self.predictor.patch_size or self.encoder.tubelet_size != 2:
            raise ValueError("encoder geometry must match the v0 video tubelet path")

    @torch.no_grad()
    def encode_observed_blocks(self, batch):
        valid = batch["history_valid"]
        batch_size, blocks = valid.shape
        patches, width = self.predictor.tokens_per_history_block, self.predictor.predictor_embed.in_features
        if "history_latents" in batch:
            latents = batch["history_latents"].detach()
        else:
            self._check_video_encoder()
            video = batch["history_video"]    
            self.encoder.eval()
            selected = video[valid]
            pieces = [
                encode_video_views(self.encoder, selected[start:start + self.encoder_batch_size])
                for start in range(0, selected.shape[0], self.encoder_batch_size)
            ]
            encoded = torch.cat(pieces)
            latents = encoded.new_zeros(batch_size, blocks, patches, width)
            latents[valid] = encoded
        if latents.shape != (batch_size, blocks, patches, width):
            raise ValueError("history_latents must match [B,K,P,D_encoder]")
        if not latents.is_floating_point() or not torch.isfinite(latents[valid]).all():
            raise ValueError("Valid history features must be finite floating point")
        return latents.masked_fill(~valid[..., None, None], 0)

    @torch.no_grad()
    def encode_local_future_targets(self, batch):
        batch_size = batch["decision_step"].shape[0]
        if "future_targets" in batch:
            # Frozen feature caches already contain the extra target LayerNorm.
            target = batch["future_targets"].detach()
        else:
            self._check_video_encoder()
            video = batch["target_video"]
            expected = (batch_size, self.predictor.num_views, 3, 10,
                        self.predictor.img_height, self.predictor.img_width)
            self.encoder.eval()
            target = encode_video_views(self.encoder, video)
            target = F.layer_norm(target, (target.shape[-1],))
            target = target[:, self.predictor.tokens_per_history_block:]
        valid = batch["future_valid"]
        return target.masked_fill(~valid[..., None], 0)

    def _encode_history(self, batch):
        raw_state, ends, valid = self._validate_observations(batch)
        language = batch["language"].masked_fill(~batch["language_mask"][..., None], 0).detach()
        packed_history = pack_normalized_state(raw_state, self.config.max_state_dim)
        packed_history = packed_history.masked_fill(~valid[..., None], 0)
        packed_current = pack_normalized_state(batch["current_state"], self.config.max_state_dim)
        geometry = dict(grid_height=self.predictor.grid_height, grid_width=self.predictor.grid_width,
                        num_views=self.predictor.num_views, temporal_unit_steps=self.config.temporal_unit_steps)
        hl = visual_layout(ends, valid, **geometry)
        z = self.encode_observed_blocks(batch)
        history = self.predictor.forward_history(z, packed_history, language, batch["language_mask"], hl)
        return history, packed_current, language, z.dtype

    def _future_layout(self, decision_step, valid=None):
        return future_layout(
            decision_step, valid=valid, grid_height=self.predictor.grid_height,
            grid_width=self.predictor.grid_width, num_views=self.predictor.num_views,
            temporal_unit_steps=self.config.temporal_unit_steps,
        )

    def _encode_context(self, batch, *, future_valid=None):
        history, current, language, dtype = self._encode_history(batch)
        future = self.predictor.query_future(
            history, current, language, batch["language_mask"], batch["decision_step"],
            self._future_layout(batch["decision_step"], future_valid),
        )
        return history, future, current, language, dtype

    def forward(self, batch, *, flow_noise=None, flow_time=None):
        self._validate_observations(batch)
        actions = batch.get("action")
        action_valid = ~batch["action_is_pad"]
        history, future, current, language, encoder_dtype = self._encode_context(
            batch, future_valid=batch["future_valid"],
        )
        target = self.encode_local_future_targets(batch)
        al = action_layout(batch["decision_step"], action_valid, temporal_unit_steps=self.config.temporal_unit_steps)
        velocity, target_velocity = self.action_expert.forward_memory(
            history, future, actions, language, batch["language_mask"], current, al,
            noise=flow_noise, time=flow_time,
        )
        loss_world = masked_sample_mean(
            (future.prediction.float() - target.float()).abs().mean(-1), batch["future_valid"],
        )
        loss_action = masked_sample_mean(
            (velocity.float() - target_velocity.float()).square().mean(-1), action_valid,
        )
        loss = self.world_loss_weight * loss_world + self.action_loss_weight * loss_action
        cosine = masked_sample_mean(
            F.cosine_similarity(future.prediction.float(), target.float(), dim=-1), batch["future_valid"],
        )
        self.forward_dtypes = {
            "clips": batch.get("history_video", batch.get("history_latents")).dtype,
            "encoder": encoder_dtype, "predictor_k": history.layers[0].key.dtype,
            "predictor_v": history.layers[0].value.dtype, "action": velocity.dtype, "loss": loss.dtype,
        }
        return loss, dict(loss=loss.detach(), loss_world=loss_world.detach(),
                          loss_action=loss_action.detach(), cos_sim=cosine.detach())

    @torch.no_grad()
    def predict_action(self, batch, *, initial_noise=None, generator=None):
        # Targets, target masks, labels and action padding never enter inference.
        history, current, language, _ = self._encode_history(batch)
        return self.predict_from_history(
            history, current, language, batch["language_mask"], batch["decision_step"],
            initial_noise=initial_noise, generator=generator,
        )

    @torch.no_grad()
    def predict_from_history(self, history, packed_current_state, language, language_mask, decision_step,
                             *, initial_noise=None, generator=None):
        future = self.predictor.query_future(
            history, packed_current_state, language, language_mask, decision_step,
            self._future_layout(decision_step),
        )
        valid = torch.ones(
            future.decision_step.shape[0], self.config.action_horizon,
            device=future.decision_step.device, dtype=torch.bool,
        )
        al = action_layout(future.decision_step, valid, temporal_unit_steps=self.config.temporal_unit_steps)
        return self.action_expert.predict_action_memory(
            history, future, language, language_mask, packed_current_state, al,
            initial_noise=initial_noise, generator=generator,
        )


def build_full_history_policy(encoder, predictor, *, config=None, action_hidden_size=512,
                              condition_num_heads=8, action_num_inference_steps=10, **kwargs):
    """Build the production v0 geometry; reduced test models use the class directly."""
    config = config or FullHistoryV0Config()
    if (predictor.img_height, predictor.img_width, predictor.patch_size,
            predictor.tubelet_size, predictor.num_frames, predictor.num_views,
            predictor.grid_depth, predictor.grid_height, predictor.grid_width,
            predictor.lang_dim, predictor.proprio_dim) != (
            224, 224, 16, 2, 10, 3, 5, 14, 14, 4096, 96):
        raise ValueError("Production Full-History v0 requires the fixed M0 geometry")
    if (getattr(encoder, "_vjepa_policy_family", None) != "vjepa2_1"
            or getattr(encoder, "_vjepa_policy_model_name", None) != "vit_large"
            or getattr(encoder, "embed_dim", None) != 1024
            or predictor.predictor_embed.in_features != 1024):
        raise ValueError("Production Full-History v0 requires the V-JEPA2.1 ViT-L encoder")
    rope_options = predictor.rope_options  # also rejects empty/inconsistent layers
    attention = predictor.predictor_blocks[0].self_attn
    action_expert = MemoryActionExpert(
        action_dim=config.action_dim, action_chunk_size=config.action_horizon,
        num_layers=len(predictor.predictor_blocks), predictor_num_heads=attention.num_heads,
        predictor_head_dim=attention.head_dim, state_dim=config.packed_state_dim,
        hidden_size=action_hidden_size, lang_dim=predictor.lang_dim,
        condition_num_heads=condition_num_heads, num_inference_steps=action_num_inference_steps,
        rope_options=rope_options, action_reads_future=config.action_reads_future,
        attention_backend=predictor.attention_backend,
    )
    return FullHistoryVJEPAPolicy(
        encoder, predictor, action_expert, config=config, strict_v0_inputs=True, **kwargs,
    )


from vjepa_policy.policy_arch import HISTORY_CAPABILITIES
build_full_history_policy.training_capabilities = HISTORY_CAPABILITIES
