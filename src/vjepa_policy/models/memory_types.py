"""Tensor contracts for full-history v0; constructing these never detaches KV."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LayerKV:
    """Post-RoPE keys and unrotated values, both [B, heads, N, head_dim].

    The container is frozen, not the tensors. Readers must not mutate them.
    Training prefixes retain their computation graph.
    """

    key: Tensor
    value: Tensor

    def __post_init__(self):
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError("KV must have identical [B, heads, N, head_dim] shapes")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("KV must share device and dtype")
        if not self.key.is_floating_point():
            raise ValueError("KV must be floating point")


    @property
    def shape(self):
        return self.key.shape

    @property
    def dtype(self):
        return self.key.dtype

    @property
    def device(self):
        return self.key.device


class JoinedLayerKV(LayerKV):
    """Differentiable KV pieces concatenated only inside a consuming layer.

    Shared regular-history pieces can remain expanded views of one batch row.
    Materialized keys/values are deliberately never cached on this object.
    """

    def __init__(self, parts):
        parts = tuple(parts)
        if not parts or any(not isinstance(part, LayerKV) for part in parts):
            raise ValueError("Joined KV requires nonempty LayerKV pieces")
        first = parts[0]
        for part in parts[1:]:
            if ((part.shape[0], part.shape[1], part.shape[3]) !=
                    (first.shape[0], first.shape[1], first.shape[3])
                    or part.dtype != first.dtype or part.device != first.device):
                raise ValueError("Joined KV pieces must share batch/heads/width/dtype/device")
        object.__setattr__(self, "parts", parts)

    @property
    def key(self):
        return torch.cat([part.key for part in self.parts], dim=2)

    @property
    def value(self):
        return torch.cat([part.value for part in self.parts], dim=2)

    @property
    def shape(self):
        first = self.parts[0].shape
        return (first[0], first[1], sum(part.shape[2] for part in self.parts), first[3])

    @property
    def dtype(self):
        return self.parts[0].dtype

    @property
    def device(self):
        return self.parts[0].device


@dataclass(frozen=True)
class TokenLayout:
    """One episode per batch row; never pack different episodes in a row.

    All metadata is [B, N], except rope_coords [B, N, 3] (time/height/width).
    end_step is the observation cutoff, including for future/action queries;
    their prediction target time belongs in rope_coords, not end_step.
    Spatial coordinates are raw patch indices; attention applies interpolation.
    view_id=-1 denotes a nonvisual token. Padding has valid=False.
    """

    valid: Tensor
    block_id: Tensor
    end_step: Tensor
    rope_coords: Tensor
    view_id: Tensor

    def __post_init__(self):
        if self.valid.ndim != 2 or self.valid.dtype != torch.bool:
            raise ValueError("valid must be bool [B, N]")
        for name in ("block_id", "end_step", "view_id"):
            tensor = getattr(self, name)
            if tensor.shape != self.valid.shape or tensor.dtype != torch.long:
                raise ValueError(f"{name} must be long [B, N]")
        if self.rope_coords.shape != (*self.valid.shape, 3) or not self.rope_coords.is_floating_point():
            raise ValueError("rope_coords must be floating point [B, N, 3]")
        for name in ("block_id", "end_step", "rope_coords", "view_id"):
            if getattr(self, name).device != self.valid.device:
                raise ValueError("layout tensors must share a device")
        if not torch.isfinite(self.rope_coords).all():
            raise ValueError("rope_coords must be finite, including padding")
        if (self.block_id[self.valid] < 0).any() or (self.end_step[self.valid] < 0).any():
            raise ValueError("valid tokens require nonnegative block_id and end_step")
        if (self.view_id[self.valid] < -1).any():
            raise ValueError("view_id must be -1 or a nonnegative camera index")


def validate_decision_step(decision_step: Tensor, layout: TokenLayout):
    if decision_step.shape != (layout.valid.shape[0],) or decision_step.dtype != torch.long:
        raise ValueError("decision_step must be long [B]")
    if decision_step.device != layout.valid.device or (decision_step < 0).any():
        raise ValueError("decision_step must be nonnegative and on the layout device")
    if ((layout.end_step != decision_step[:, None]) & layout.valid).any():
        raise ValueError("each query row must belong to exactly its supplied decision_step")


def _validate_layers(layers, layout):
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a nonempty tuple of LayerKV")
    for layer in layers:
        if not isinstance(layer, LayerKV):
            raise ValueError("layers must contain LayerKV")
        if (layer.shape[0], layer.shape[2]) != layout.valid.shape:
            raise ValueError("KV batch/token dimensions must match the layout")
        if layer.device != layout.valid.device:
            raise ValueError("KV and layout must share a device")


@dataclass(frozen=True)
class HistoryEncoding:
    layers: tuple[LayerKV, ...]
    layout: TokenLayout

    def __post_init__(self):
        _validate_layers(self.layers, self.layout)


@dataclass(frozen=True)
class FutureEncoding:
    prediction: Tensor
    layers: tuple[LayerKV, ...]
    layout: TokenLayout
    decision_step: Tensor

    def __post_init__(self):
        _validate_layers(self.layers, self.layout)
        validate_decision_step(self.decision_step, self.layout)
        if self.prediction.ndim != 3 or self.prediction.shape[:2] != self.layout.valid.shape:
            raise ValueError("prediction must be [B, Q, D] matching the future layout")
        if self.prediction.device != self.layout.valid.device or not self.prediction.is_floating_point():
            raise ValueError("prediction must be floating point on the layout device")


def concatenate_layouts(*layouts):
    """Join token layouts without renumbering their global block/time coordinates."""
    return TokenLayout(**{
        name: torch.cat([getattr(layout, name) for layout in layouts], dim=1)
        for name in ("valid", "block_id", "end_step", "rope_coords", "view_id")
    })
