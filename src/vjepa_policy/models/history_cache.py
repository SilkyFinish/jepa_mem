"""Inference-only observed-history KV storage with atomic publication.

Allocation budgets cover KV buffers and staged KV, not encoder/attention workspaces.
Training HistoryEncoding remains separate and differentiable.
"""
from dataclasses import dataclass

import torch

from .memory_types import HistoryEncoding, LayerKV, concatenate_layouts


class HistoryCacheError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class HistoryBlock:
    block_id: int
    end_step: int
    digest: str
    state: torch.Tensor  # packed, normalized [1,96]


@dataclass(frozen=True)
class _Committed:
    buffers: tuple
    history: HistoryEncoding | None
    records: tuple
    capacity_blocks: int
    revision: int


class HistoryKVCache:
    """One serialized session, batch size one. No nn.Module/state_dict ownership."""

    def __init__(self, *, depth, heads, head_dim, tokens_per_block, device, dtype,
                 max_blocks=5, max_bytes=1024**3):
        if min(depth, heads, head_dim, tokens_per_block, max_blocks, max_bytes) <= 0:
            raise ValueError("cache geometry and limits must be positive")
        self.depth, self.heads, self.head_dim = depth, heads, head_dim
        self.tokens_per_block = tokens_per_block
        self.device, self.dtype = torch.device(device), dtype
        self.max_blocks, self.max_bytes = max_blocks, max_bytes
        self.bytes_per_block = depth * 2 * heads * head_dim * tokens_per_block * torch.empty((), dtype=dtype).element_size()
        self._committed = _Committed((), None, (), 0, 0)

    @property
    def records(self):
        return self._committed.records

    @property
    def revision(self):
        return self._committed.revision

    @property
    def valid_length(self):
        return len(self.records) * self.tokens_per_block

    @property
    def capacity_blocks(self):
        return self._committed.capacity_blocks

    def snapshot(self):
        """Read-only views; callers must hold the session lock through inference."""
        return self._committed.history

    def preflight(self, new_blocks, *, replace=False):
        total = len(self.records) + new_blocks
        if new_blocks <= 0 or total > self.max_blocks:
            raise HistoryCacheError("history_capacity_exceeded", f"need {total} blocks; limit {self.max_blocks}")
        capacity = self.capacity_blocks
        if total > capacity:
            capacity = min(self.max_blocks, max(total, max(1, capacity * 2)))
        # Growth keeps the old buffers live until every layer has been copied.
        old = self.capacity_blocks
        allocated = capacity if replace or capacity != old else 0
        staged = total if replace else new_blocks
        if (old + allocated + staged) * self.bytes_per_block > self.max_bytes:
            raise HistoryCacheError("history_capacity_exceeded", "KV allocation/staging budget exceeded")
        return capacity

    def _allocate(self, capacity):
        shape = (1, self.heads, capacity * self.tokens_per_block, self.head_dim)
        return LayerKV(torch.empty(shape, device=self.device, dtype=self.dtype),
                       torch.empty(shape, device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def commit(self, delta, records, *, replace=False):
        """Publish all layers, records and revision together; failed copies leave old views valid."""
        records = tuple(records)
        capacity = self.preflight(len(records), replace=replace)
        old = self._committed
        all_records = old.records + records
        for expected, record in enumerate(records, start=len(old.records)):
            if record.block_id != expected or record.state.requires_grad:
                raise ValueError("cache records must be contiguous and detached")
            if record.state.shape != (1, 96) or not torch.isfinite(record.state).all():
                raise ValueError("invalid cached block state")
            previous_end = all_records[expected - 1].end_step if expected else -1
            if record.end_step <= previous_end:
                raise ValueError("cache block times must increase")
        expected_records = all_records if replace else records
        length = len(expected_records) * self.tokens_per_block
        expected_ids = torch.tensor([r.block_id for r in expected_records], device=self.device).repeat_interleave(self.tokens_per_block)[None]
        expected_ends = torch.tensor([r.end_step for r in expected_records], device=self.device).repeat_interleave(self.tokens_per_block)[None]
        if (delta.layout.valid.shape != (1, length) or not delta.layout.valid.all()
                or not torch.equal(delta.layout.block_id, expected_ids)
                or not torch.equal(delta.layout.end_step, expected_ends)):
            raise ValueError("staged KV layout disagrees with records")
        if len(delta.layers) != self.depth:
            raise ValueError("staged KV depth mismatch")
        for layer in delta.layers:
            for tensor in (layer.key, layer.value):
                if (tensor.shape != (1, self.heads, length, self.head_dim)
                        or tensor.dtype != self.dtype or tensor.device != self.device
                        or tensor.requires_grad or not torch.isfinite(tensor).all()):
                    raise ValueError("invalid staged KV shape/device/dtype/values")
        layout = delta.layout if replace or old.history is None else concatenate_layouts(old.history.layout, delta.layout)
        grow = replace or capacity != old.capacity_blocks
        buffers = tuple(self._allocate(capacity) for _ in range(self.depth)) if grow else old.buffers
        start = 0 if replace else self.valid_length
        end = start + length
        for index, (buffer, source) in enumerate(zip(buffers, delta.layers)):
            if grow and not replace and old.history is not None:
                buffer.key[:, :, :start].copy_(old.history.layers[index].key)
                buffer.value[:, :, :start].copy_(old.history.layers[index].value)
            buffer.key[:, :, start:end].copy_(source.key)
            buffer.value[:, :, start:end].copy_(source.value)
        history = HistoryEncoding(tuple(LayerKV(b.key[:, :, :end], b.value[:, :, :end]) for b in buffers), layout)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # do not acknowledge asynchronous copy failure
        self._committed = _Committed(buffers, history, all_records, capacity, old.revision + 1)
