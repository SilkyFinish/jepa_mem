# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

from vjepa_policy.models.memory_types import LayerKV
from vjepa_policy.models.memory_attention import SegmentedAttention
from .rope import (
    apply_3d_rope,
    apply_rope_axes,
    rope_dimensions,
    rotate_queries_or_keys,
    validate_position_coords,
)


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    N_T = add_tokens + (H * W)
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        # -- snap spatial positions to grid size
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        # -- split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
                # --
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        interpolate_rope=False,
        canonical_spatial_grid=None,
        patch_size=16,
        corrected_frequency_pairing=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim, self.h_dim, self.w_dim = rope_dimensions(head_dim)
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.interpolate_rope = interpolate_rope
        self.corrected_frequency_pairing = corrected_frequency_pairing
        if canonical_spatial_grid is None:
            canonical_spatial_grid = {14: 18, 16: 16}.get(patch_size)
            if canonical_spatial_grid is None and interpolate_rope:
                raise ValueError(
                    "canonical_spatial_grid is required when interpolate_rope=True "
                    f"and patch_size={patch_size}"
                )
        if isinstance(canonical_spatial_grid, int):
            canonical_spatial_grid = (canonical_spatial_grid, canonical_spatial_grid)
        if canonical_spatial_grid is not None:
            if len(canonical_spatial_grid) != 2 or min(canonical_spatial_grid) <= 1:
                raise ValueError(
                    "canonical_spatial_grid must be an int or a pair of integers greater than 1"
                )
            canonical_spatial_grid = tuple(int(size) for size in canonical_spatial_grid)
        self.canonical_spatial_grid = canonical_spatial_grid

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def _scale_spatial_positions(self, height_ids, width_ids, H_patches=None, W_patches=None):
        if not self.interpolate_rope:
            return height_ids, width_ids

        if H_patches is None or W_patches is None:
            H_patches = W_patches = self.grid_size
        if H_patches <= 1 or W_patches <= 1:
            raise ValueError("RoPE interpolation requires spatial patch grids greater than 1")

        canonical_height, canonical_width = self.canonical_spatial_grid
        height_ids = height_ids * (canonical_height - 1) / (H_patches - 1)
        width_ids = width_ids * (canonical_width - 1) / (W_patches - 1)
        return height_ids, width_ids

    def forward(
        self,
        x,
        mask=None,
        attn_mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        kv_out=None,
        return_kv=False,
        *,
        position_coords=None,
        prefix_kv=None,
    ):
        # Explicit coordinates opt into memory semantics; old calls below keep
        # the original attention/dropout/position behavior and state_dict keys.
        if position_coords is not None or prefix_kv is not None:
            if mask is not None:
                raise ValueError("mask contains legacy position IDs; do not combine it with memory coordinates/KV")
            if position_coords is None:
                raise ValueError("prefix_kv requires explicit position_coords")
            return self._forward_memory(
                x, position_coords, prefix_kv, attn_mask,
                H_patches, W_patches, kv_out, return_kv,
            )
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        h_mask, w_mask = self._scale_spatial_positions(
            h_mask,
            w_mask,
            H_patches=H_patches,
            W_patches=W_patches,
        )

        positions = (d_mask, h_mask, w_mask)
        dimensions = (self.d_dim, self.h_dim, self.w_dim)
        q = apply_rope_axes(
            q, positions, dimensions,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        k = apply_rope_axes(
            k, positions, dimensions,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )

        # Opt-in hook for external modules (e.g. a MoT-style action expert) that
        # need this layer's post-RoPE keys/values as read-only context. No-op
        # unless a caller passes a list in, so existing behavior is unchanged.
        if kv_out is not None:
            kv_out.append((k, v))

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_kv:
            return x, k, v
        return x

    def _forward_memory(
        self, x, position_coords, prefix_kv, attn_mask,
        H_patches, W_patches, kv_out, return_kv,
    ):
        """Read-only [prefix, current] attention; return only current KV."""
        batch, tokens, channels = x.shape
        if tokens == 0:
            raise ValueError("memory attention requires at least one current token")
        cached_rotary = hasattr(attn_mask, "rotate")
        if not cached_rotary:
            validate_position_coords(position_coords, batch, tokens, x.device)
        if self.is_causal:
            raise ValueError("memory attention requires an explicit block mask, not is_causal=True")
        segmented = isinstance(attn_mask, SegmentedAttention)
        if not segmented and (attn_mask is None or attn_mask.dtype != torch.bool):
            raise ValueError("memory attention requires a bool attn_mask (True = allowed)")
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        rope_options = dict(corrected_frequency_pairing=self.corrected_frequency_pairing)
        if self.interpolate_rope:
            if (H_patches is None) != (W_patches is None):
                raise ValueError("supply both H_patches and W_patches")
            rope_options.update(
                spatial_grid=(self.grid_size, self.grid_size) if H_patches is None else (H_patches, W_patches),
                canonical_spatial_grid=self.canonical_spatial_grid,
            )
        if cached_rotary:
            q, k = attn_mask.rotate(q, k)
        else:
            q = apply_3d_rope(q, position_coords, **rope_options)
            k = apply_3d_rope(k, position_coords, **rope_options)
        key, value = k, v
        if prefix_kv is not None:
            if not isinstance(prefix_kv, LayerKV):
                raise ValueError("prefix_kv must be a LayerKV with already-rotated keys")
            pk, pv = prefix_kv.key, prefix_kv.value
            if (pk.shape[0], pk.shape[1], pk.shape[3]) != (batch, self.num_heads, self.head_dim):
                raise ValueError("prefix_kv batch/heads/head_dim must match current KV")
            if pk.dtype != k.dtype or pk.device != k.device:
                raise ValueError("prefix_kv dtype/device must match projected current KV")
            # No mutation, detach, re-projection or second rotation of prefix.
            key = torch.cat((pk, k), dim=2)
            value = torch.cat((pv, v), dim=2)
        dropout = self.attn_drop.p if self.training else 0.0
        if segmented:
            row_valid = attn_mask.row_valid
            output = attn_mask.attend(q, key, value, dropout_p=dropout, scale=self.scale)
        else:
            expected = (batch, 1, tokens, key.shape[2])
            if attn_mask.shape != expected or attn_mask.device != x.device:
                raise ValueError(f"memory attn_mask must have shape {expected} on the query device")
            row_valid = attn_mask.any(dim=-1, keepdim=True)
            # Make all-masked padding rows safe; remove output/gradient below.
            safe_mask = attn_mask.clone()
            safe_mask[..., :1] |= ~row_valid
            if self.use_sdpa:
                output = F.scaled_dot_product_attention(
                    q, key, value, attn_mask=safe_mask, dropout_p=dropout,
                    is_causal=False, scale=self.scale,
                )
            else:
                logits = (q @ key.transpose(-2, -1)) * self.scale
                weights = logits.masked_fill(~safe_mask, float("-inf")).softmax(dim=-1)
                output = F.dropout(weights, p=dropout, training=self.training) @ value
        output = output.masked_fill(~row_valid, 0)
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj_drop(self.proj(output))
        # Clear projection bias too. Padding residuals/FFNs belong to the caller.
        output = output.masked_fill(~row_valid[:, 0], 0)
        if kv_out is not None:
            kv_out.append((k, v))
        if return_kv:
            return output, k, v
        return output


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        interpolate_rope=False,
        canonical_spatial_grid=None,
        patch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                interpolate_rope=interpolate_rope,
                canonical_spatial_grid=canonical_spatial_grid,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, kv_out=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(
                self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches,
                kv_out=kv_out,
            )
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
