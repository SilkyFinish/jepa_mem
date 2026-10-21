# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository root.
"""Shared 3D RoPE for projected visual and action Q/K; no trainable weights."""

import torch


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



def rope_dimensions(head_dim):
    """V-JEPA split: 64 channels -> (20, 20, 20), with 4 untouched."""
    axis_dim = 2 * ((head_dim // 3) // 2)
    return axis_dim, axis_dim, axis_dim


def apply_rope_axes(x, positions, dimensions, *, corrected_frequency_pairing=False):
    """Apply the original arithmetic to three already-broadcastable axes."""
    chunks = []
    offset = 0
    for pos, dim in zip(positions, dimensions):
        part = x[..., offset : offset + dim]
        if dim:
            part = rotate_queries_or_keys(
                part, pos, corrected_frequency_pairing=corrected_frequency_pairing
            )
        chunks.append(part)
        offset += dim
    if offset < x.shape[-1]:
        chunks.append(x[..., offset:])
    return torch.cat(chunks, dim=-1)


def validate_position_coords(coords, batch, tokens, device):
    if coords.shape != (batch, tokens, 3) or not coords.is_floating_point():
        raise ValueError("position_coords must be floating point [B, N, 3]")
    if coords.device != device or not torch.isfinite(coords).all():
        raise ValueError("position_coords must be finite and on the query device")


def apply_3d_rope(
    x,
    position_coords,
    *,
    corrected_frequency_pairing=False,
    spatial_grid=None,
    canonical_spatial_grid=None,
):
    """Rotate [B, heads, N, D] using explicit time/raw-patch coordinates.

    Time is already in the caller's common temporal unit (v0: steps / 8).
    When interpolating, both grids are (height, width); only spatial axes
    are scaled. The caller must use the same options for visual/action Q/K.
    Cached keys are already rotated and must never pass through this function.
    Angles use at least float32; the result keeps x.dtype for mixed precision.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("RoPE input must be floating point [B, heads, N, D]")
    validate_position_coords(position_coords, x.shape[0], x.shape[2], x.device)
    angle_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    coords = position_coords.to(dtype=angle_dtype)
    time, height, width = coords.unbind(-1)
    if (spatial_grid is None) != (canonical_spatial_grid is None):
        raise ValueError("supply both spatial_grid and canonical_spatial_grid, or neither")
    if spatial_grid is not None:
        if len(spatial_grid) != 2 or len(canonical_spatial_grid) != 2:
            raise ValueError("spatial grids must be (height, width) pairs")
        if min(*spatial_grid, *canonical_spatial_grid) <= 1:
            raise ValueError("spatial grids must be greater than 1")
        height = height * (canonical_spatial_grid[0] - 1) / (spatial_grid[0] - 1)
        width = width * (canonical_spatial_grid[1] - 1) / (spatial_grid[1] - 1)
    dimensions = rope_dimensions(x.shape[-1])
    # With a one-frequency axis both pairings are identical. Avoid the old
    # squeeze(-1) expansion's shape ambiguity for tiny (6..11 channel) heads.
    pairing = corrected_frequency_pairing or dimensions[0] == 2
    result = apply_rope_axes(
        x.to(dtype=angle_dtype),
        (time[:, None], height[:, None], width[:, None]),
        dimensions,
        corrected_frequency_pairing=pairing,
    )
    return result.to(dtype=x.dtype)
