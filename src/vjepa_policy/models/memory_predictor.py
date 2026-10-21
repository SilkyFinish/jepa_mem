"""History/future-separated Predictor. No persistent state or KV cache."""

import torch
from torch.utils.checkpoint import checkpoint

from .memory_masks import history_attention_mask, future_attention_mask
from .memory_types import HistoryEncoding, FutureEncoding, LayerKV
from .wm_predictor import WorldModelPredictor


class MemoryWorldModelPredictor(WorldModelPredictor):
    """Reuse Predictor weights while replacing the bidirectional H/F dataflow."""

    @property
    def tokens_per_history_block(self):
        return self.num_views * self.grid_height * self.grid_width

    @property
    def future_tokens(self):
        return (self.grid_depth - 1) * self.tokens_per_history_block

    @property
    def rope_options(self):
        options = []
        for block in self.predictor_blocks:
            attn = block.self_attn
            options.append(dict(
                corrected_frequency_pairing=attn.corrected_frequency_pairing,
                spatial_grid=(self.grid_height, self.grid_width) if attn.interpolate_rope else None,
                canonical_spatial_grid=attn.canonical_spatial_grid if attn.interpolate_rope else None,
            ))
        return options[0]

    def forward(self, *args, **kwargs):
        raise TypeError("use forward_history and query_future; legacy Predictor inputs are not supported")

    def _memory_view_embeddings(self, tokens, layout):
        if self.view_embedding is not None:
            tokens = tokens + self.view_embedding(layout.view_id.clamp(0, self.num_views - 1))
        return tokens.masked_fill(~layout.valid[..., None], 0)

    def _history_condition(self, state, language, language_mask):
        batch, blocks, width = state.shape
        # Project language once, then reuse its graph for all historical blocks.
        first_condition, first_valid = self._condition_tokens(language, language_mask, state[:, 0])
        lang = first_condition[:, :-1]
        state_tokens = self.proprio_projection(self.proprio_norm(state))[:, :, None]
        condition = torch.cat((lang[:, None].expand(-1, blocks, -1, -1), state_tokens), dim=2)
        valid = first_valid[:, None].expand(-1, blocks, -1)
        return condition.flatten(0, 1), valid.flatten(0, 1)

    def _history_block(self, block, tokens, condition, condition_valid, layout, mask, blocks):
        output, key, value = block.self_attn(
            block.self_norm(tokens), position_coords=layout.rope_coords,
            attn_mask=mask, H_patches=self.grid_height, W_patches=self.grid_width,
            return_kv=True,
        )
        tokens = tokens + block.drop_path(output)
        batch, _, width = tokens.shape
        grouped = tokens.reshape(batch * blocks, self.tokens_per_history_block, width)
        grouped = grouped + block.drop_path(
            block.cross_attn(block.cross_norm(grouped), condition, condition_valid)
        )
        tokens = grouped.reshape(batch, -1, width)
        tokens = tokens + block.drop_path(block.ffn(block.ffn_norm(tokens)))
        return (
            tokens.masked_fill(~layout.valid[..., None], 0),
            key.masked_fill(~layout.valid[:, None, :, None], 0),
            value.masked_fill(~layout.valid[:, None, :, None], 0),
        )

    def forward_history(self, observed_latents, history_state, language, language_mask, history_layout):
        batch, blocks, patches, width = observed_latents.shape
        layout = history_layout
        group_valid = layout.valid.reshape(batch, blocks, patches)
        clean_latents = observed_latents.masked_fill(~group_valid[..., None], 0)
        clean_state = history_state.masked_fill(~group_valid[..., :1], 0)
        tokens = self.predictor_embed(clean_latents).flatten(1, 2)
        tokens = self._memory_view_embeddings(tokens, layout)
        condition, condition_valid = self._history_condition(clean_state, language, language_mask)
        mask = history_attention_mask(layout)
        layers = []
        for index, block in enumerate(self.predictor_blocks):
            args = (block, tokens, condition, condition_valid, layout, mask, blocks)
            if self.training and index < self.activation_checkpointing_blocks:
                tokens, key, value = checkpoint(self._history_block, *args, use_reentrant=False)
            else:
                tokens, key, value = self._history_block(*args)
            layers.append(LayerKV(key, value))
        return HistoryEncoding(tuple(layers), layout)

    def query_future(self, history, current_state, language, language_mask,
                     decision_step, future_layout, *, mask_index=1):
        batch, queries = future_layout.valid.shape
        condition, condition_valid = self._condition_tokens(language, language_mask, current_state)
        tokens = self.mask_tokens[mask_index % self.num_mask_tokens].expand(batch, queries, -1)
        tokens = self._memory_view_embeddings(tokens, future_layout)
        mask = future_attention_mask(history.layout, future_layout, decision_step)
        layers = []
        for index, block in enumerate(self.predictor_blocks):
            kwargs = dict(
                T=None, H_patches=self.grid_height, W_patches=self.grid_width, return_kv=True,
                position_coords=future_layout.rope_coords, prefix_kv=history.layers[index], attn_mask=mask,
            )
            if self.training and index < self.activation_checkpointing_blocks:
                tokens, key, value = checkpoint(
                    block, tokens, condition, condition_valid, None, use_reentrant=False, **kwargs,
                )
            else:
                tokens, key, value = block(tokens, condition, condition_valid, None, **kwargs)
            tokens = tokens.masked_fill(~future_layout.valid[..., None], 0)
            layers.append(LayerKV(
                key.masked_fill(~future_layout.valid[:, None, :, None], 0),
                value.masked_fill(~future_layout.valid[:, None, :, None], 0),
            ))
        prediction = self.predictor_proj(self.predictor_norm(tokens))
        prediction = prediction.masked_fill(~future_layout.valid[..., None], 0)
        return FutureEncoding(prediction, tuple(layers), future_layout, decision_step)
