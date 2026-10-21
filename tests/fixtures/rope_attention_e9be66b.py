# Copyright (c) Meta Platforms, Inc. and affiliates. MIT licensed.
# Frozen numerical oracle copied verbatim from backbone/modules.py at e9be66b.
# Do not refactor this alongside the production implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_queries_or_keys(x, pos, corrected_frequency_pairing=False):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # Preserve the published V-JEPA2 expansion by default for checkpoint compatibility.
    if corrected_frequency_pairing:
        emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
        emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)
    else:
        # NOTE: The legacy expansion does not duplicate each frequency within its
        # rotated channel pair. V-JEPA2 pretrained weights depend on this behavior.
        # Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
        emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
        emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


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
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
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
    ):
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

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(
            q[..., s : s + self.d_dim],
            pos=d_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        kd = rotate_queries_or_keys(
            k[..., s : s + self.d_dim],
            pos=d_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(
            q[..., s : s + self.h_dim],
            pos=h_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        kh = rotate_queries_or_keys(
            k[..., s : s + self.h_dim],
            pos=h_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(
            q[..., s : s + self.w_dim],
            pos=w_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
        kw = rotate_queries_or_keys(
            k[..., s : s + self.w_dim],
            pos=w_mask,
            corrected_frequency_pairing=self.corrected_frequency_pairing,
        )
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

