"""History/future-separated Predictor; persistent cache ownership stays in serving."""

import torch
from torch.utils.checkpoint import checkpoint

from .memory_masks import history_attention_mask, future_attention_mask
from .memory_attention import history_attention_plan, future_attention_plan
from .memory_layouts import visual_layout
from .memory_types import HistoryEncoding, FutureEncoding, LayerKV, concatenate_layouts
from .wm_predictor import WorldModelPredictor


class MemoryWorldModelPredictor(WorldModelPredictor):
    """Reuse Predictor weights while replacing the bidirectional H/F dataflow."""

    def __init__(self, *args, attention_backend="segmented", history_spatial_pool=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_backend = attention_backend
        if (type(history_spatial_pool) is not int or history_spatial_pool not in (1, 2)
                or self.grid_height % history_spatial_pool or self.grid_width % history_spatial_pool):
            raise ValueError("History pooling must be 1 or 2 and divide the Encoder grid")
        self.history_spatial_pool = history_spatial_pool

    def _history_attention(self, query, key=None):
        builder = history_attention_plan if self.attention_backend == "segmented" else history_attention_mask
        return builder(query, key)

    def _future_attention(self, history, future, decision_step):
        builder = future_attention_plan if self.attention_backend == "segmented" else future_attention_mask
        return builder(history, future, decision_step)

    @property
    def tokens_per_observation_block(self):
        return self.num_views * self.grid_height * self.grid_width

    @property
    def tokens_per_history_block(self):
        return self.tokens_per_observation_block // self.history_spatial_pool ** 2

    @property
    def future_tokens(self):
        return (self.grid_depth - 1) * self.tokens_per_observation_block

    def history_layout(self, end_step, valid, *, temporal_unit_steps=8, block_offset=0):
        s = self.history_spatial_pool
        return visual_layout(
            end_step, valid, grid_height=self.grid_height // s, grid_width=self.grid_width // s,
            num_views=self.num_views, temporal_unit_steps=temporal_unit_steps,
            block_offset=block_offset, spatial_stride=s,
        )

    def compress_history(self, observed_latents):
        """Pool each frame/view independently before any trainable H computation.

        RoPE retains the original grid basis; each pooled cell is located at its
        source patch center. Future query/target geometry remains uncompressed.
        """
        batch, blocks, patches, width = observed_latents.shape
        if patches != self.tokens_per_observation_block:
            raise ValueError("History inputs must contain the uncompressed Encoder grid")
        s = self.history_spatial_pool
        if s == 1:
            return observed_latents
        cells = observed_latents.reshape(
            batch, blocks, self.num_views, self.grid_height // s, s, self.grid_width // s, s, width,
        )
        return cells.mean(dim=(4, 6)).reshape(batch, blocks, self.tokens_per_history_block, width)

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

    def _history_block(self, block, tokens, condition, condition_valid, layout, mask, blocks, prefix=None):
        output, key, value = block.self_attn(
            block.self_norm(tokens), position_coords=layout.rope_coords,
            attn_mask=mask, H_patches=self.grid_height, W_patches=self.grid_width,
            return_kv=True, prefix_kv=prefix,
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
        observed_latents = self.compress_history(observed_latents)
        batch, blocks, patches, width = observed_latents.shape
        layout = history_layout
        group_valid = layout.valid.reshape(batch, blocks, patches)
        clean_latents = observed_latents.masked_fill(~group_valid[..., None], 0)
        clean_state = history_state.masked_fill(~group_valid[..., :1], 0)
        tokens = self.predictor_embed(clean_latents).flatten(1, 2)
        tokens = self._memory_view_embeddings(tokens, layout)
        condition, condition_valid = self._history_condition(clean_state, language, language_mask)
        mask = self._history_attention(layout)
        layers = []
        for index, block in enumerate(self.predictor_blocks):
            args = (block, tokens, condition, condition_valid, layout, mask, blocks)
            if self.training and index < self.activation_checkpointing_blocks:
                tokens, key, value = checkpoint(self._history_block, *args, use_reentrant=False)
            else:
                tokens, key, value = self._history_block(*args)
            layers.append(LayerKV(key, value))
        return HistoryEncoding(tuple(layers), layout)

    @torch.no_grad()
    def append_history(self, prefix, observed_latents, history_state, language,
                       language_mask, history_layout):
        """Compute only new H tokens; return a staged delta, never mutate prefix.

        Each new block reads the old prefix and earlier new blocks. Its own
        state is used at every layer, exactly as in forward_history.
        """
        if self.training:
            raise ValueError("append_history requires eval mode; training must recompute history")
        observed_latents = self.compress_history(observed_latents)
        batch, blocks, patches, _ = observed_latents.shape
        layout = history_layout
        offset = 0 if prefix is None else prefix.layout.valid.shape[1] // patches
        expected = (torch.arange(blocks, device=observed_latents.device) + offset)
        expected = expected.repeat_interleave(patches)[None].expand(batch, -1)
        if not torch.equal(layout.block_id, expected):
            raise ValueError("append_history requires contiguous block IDs after the prefix")
        key_layout = layout if prefix is None else concatenate_layouts(prefix.layout, layout)
        mask = self._history_attention(layout, key_layout)
        tokens = self._memory_view_embeddings(
            self.predictor_embed(observed_latents).flatten(1, 2), layout,
        )
        condition, condition_valid = self._history_condition(history_state, language, language_mask)
        layers = []
        for index, block in enumerate(self.predictor_blocks):
            tokens, key, value = self._history_block(
                block, tokens, condition, condition_valid, layout, mask, blocks,
                None if prefix is None else prefix.layers[index],
            )
            layers.append(LayerKV(key, value))
        return HistoryEncoding(tuple(layers), layout)

    def query_future(self, history, current_state, language, language_mask,
                     decision_step, future_layout, *, mask_index=1):
        batch, queries = future_layout.valid.shape
        condition, condition_valid = self._condition_tokens(language, language_mask, current_state)
        tokens = self.mask_tokens[mask_index % self.num_mask_tokens].expand(batch, queries, -1)
        tokens = self._memory_view_embeddings(tokens, future_layout)
        mask = self._future_attention(history.layout, future_layout, decision_step)
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
