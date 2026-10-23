"""Per-forward rotary factors shared across layers; no persistent tensor cache."""
from dataclasses import dataclass
from functools import lru_cache
import torch
from .backbone.rope import rope_dimensions, validate_position_coords
from .memory_attention import SegmentedAttention

def _rotate(x, cosine, sine):
    dtype = x.dtype
    x = x.to(cosine.dtype)
    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(-1)
    rotated = torch.stack((-second, first), -1).flatten(-2)
    return (x * cosine + rotated * sine).to(dtype)

def _rotate_pair(q, k, cosine, sine):
    return _rotate(q, cosine, sine), _rotate(k, cosine, sine)

@lru_cache(maxsize=1)
def _kernel():
    return torch.compile(_rotate_pair, dynamic=True)

@dataclass(frozen=True)
class RotaryAttentionPlan(SegmentedAttention):
    attention: object = None
    cosine: object = None
    sine: object = None

    @property
    def row_valid(self):
        return self.attention.row_valid

    def attend(self, *args, **kwargs):
        return self.attention.attend(*args, **kwargs)

    def rotate(self, q, k):
        fn = _kernel() if q.device.type == "cuda" else _rotate_pair
        return fn(q, k, self.cosine, self.sine)

def rotary_attention_plan(attention, layout, head_dim, options):
    if head_dim % 2:
        raise ValueError("Rotary attention requires an even head dimension")
    coords = layout.rope_coords
    validate_position_coords(coords, *layout.valid.shape, coords.device)
    coords = coords.float()
    time, height, width = coords.unbind(-1)
    spatial = options.get("spatial_grid")
    canonical = options.get("canonical_spatial_grid")
    if spatial is not None:
        height = height * (canonical[0] - 1) / (spatial[0] - 1)
        width = width * (canonical[1] - 1) / (spatial[1] - 1)
    dimensions = rope_dimensions(head_dim)
    corrected = options.get("corrected_frequency_pairing", False) or dimensions[0] == 2
    cosines, sines = [], []
    for position, dim in zip((time, height, width), dimensions):
        if not dim:
            continue
        omega = torch.arange(dim // 2, dtype=torch.float32, device=coords.device)
        omega = 1.0 / 10000 ** (omega / (dim / 2.0))
        angle = torch.einsum("..., f -> ... f", position[:, None], omega)
        c, s = angle.cos(), angle.sin()
        if corrected:
            c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
        else:
            c, s = torch.cat((c, c), -1), torch.cat((s, s), -1)
        cosines.append(c); sines.append(s)
    tail = head_dim - sum(dimensions)
    if tail:
        shape = (coords.shape[0], 1, coords.shape[1], tail)
        cosines.append(torch.ones(shape, device=coords.device))
        sines.append(torch.zeros(shape, device=coords.device))
    return RotaryAttentionPlan(attention.query_valid, attention.key_length, attention.rows,
                               attention, torch.cat(cosines, -1), torch.cat(sines, -1))
