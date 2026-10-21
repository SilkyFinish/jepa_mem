"""Canonical time-view-height-width layouts for the recompute memory model."""

import torch

from .memory_types import TokenLayout


def visual_layout(end_step, valid, *, grid_height, grid_width, num_views,
                  temporal_unit_steps=8, target_step=None):
    """Expand [B,K] cutoff/validity to [B,K*V*H*W]; target_step sets RoPE time."""
    if end_step.ndim != 2 or end_step.dtype != torch.long:
        raise ValueError("end_step must be long [B,K]")
    if valid.shape != end_step.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,K]")
    if min(grid_height, grid_width, num_views, temporal_unit_steps) <= 0:
        raise ValueError("layout dimensions and temporal unit must be positive")
    target_step = end_step if target_step is None else target_step
    if target_step.shape != end_step.shape:
        raise ValueError("target_step must match end_step")
    batch, groups = end_step.shape
    patches = grid_height * grid_width
    per_block = num_views * patches
    spatial = torch.arange(per_block, device=end_step.device)
    view = spatial // patches
    height = (spatial % patches) // grid_width
    width = spatial % grid_width
    shape = (batch, groups, per_block)
    coords = torch.stack((
        (target_step.float() / temporal_unit_steps)[:, :, None].expand(shape),
        height.expand(shape).float(), width.expand(shape).float(),
    ), dim=-1).reshape(batch, -1, 3)
    return TokenLayout(
        valid=valid[:, :, None].expand(shape).reshape(batch, -1),
        block_id=torch.arange(groups, device=end_step.device)[None, :, None].expand(shape).reshape(batch, -1),
        end_step=end_step[:, :, None].expand(shape).reshape(batch, -1),
        rope_coords=coords,
        view_id=view.expand(shape).reshape(batch, -1),
    )


def future_layout(decision_step, *, grid_height, grid_width, num_views,
                  target_offsets=(8, 16, 24, 32), temporal_unit_steps=8, valid=None):
    offsets = torch.tensor(target_offsets, device=decision_step.device, dtype=torch.long)
    targets = decision_step[:, None] + offsets[None]
    layout = visual_layout(
        decision_step[:, None].expand_as(targets), torch.ones_like(targets, dtype=torch.bool),
        grid_height=grid_height, grid_width=grid_width, num_views=num_views,
        temporal_unit_steps=temporal_unit_steps, target_step=targets,
    )
    if valid is None:
        return layout
    return TokenLayout(valid, layout.block_id, layout.end_step, layout.rope_coords, layout.view_id)


def action_layout(decision_step, valid, *, temporal_unit_steps=8):
    if decision_step.ndim != 1 or decision_step.dtype != torch.long:
        raise ValueError("decision_step must be long [B]")
    if valid.ndim != 2 or valid.shape[0] != decision_step.shape[0]:
        raise ValueError("action valid must be [B,A]")
    batch, steps = valid.shape
    target = decision_step[:, None] + torch.arange(1, steps + 1, device=decision_step.device)
    coords = torch.zeros(batch, steps, 3, device=decision_step.device, dtype=torch.float32)
    coords[..., 0] = target.float() / temporal_unit_steps
    return TokenLayout(
        valid, torch.zeros_like(target), decision_step[:, None].expand_as(target),
        coords, torch.full_like(target, -1),
    )
