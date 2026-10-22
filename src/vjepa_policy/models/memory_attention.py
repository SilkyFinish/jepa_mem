"""Exact full-history attention without token-pair masks.

Queries with identical visibility share one SDPA call over ALL their legal
keys. Only queries are partitioned: softmax is never split across key chunks.
Canonical history uses prefix views, so no per-block copies of old KV are made.
Plans contain metadata only and can be reused across layers/checkpoint replay.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .memory_types import validate_decision_step


def _spans(valid):
    """CPU bool vector -> contiguous allowed intervals (including empty)."""
    edges = torch.diff(F.pad(valid.to(torch.int8), (1, 1))).nonzero().flatten().tolist()
    return tuple(zip(edges[::2], edges[1::2]))


@dataclass(frozen=True)
class AttentionSegment:
    start: int
    stop: int
    keys: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SegmentedAttention:
    query_valid: torch.Tensor
    key_length: int
    rows: tuple[tuple[AttentionSegment, ...], ...]

    @property
    def row_valid(self):
        valid = self.query_valid.clone()
        for b, segments in enumerate(self.rows):
            for segment in segments:
                if not segment.keys:
                    valid[b, segment.start:segment.stop] = False
        return valid[:, None, :, None]

    def attend(self, q, k, v, *, dropout_p=0.0, scale=None):
        if (q.shape[0], q.shape[2]) != self.query_valid.shape or k.shape[2] != self.key_length:
            raise ValueError("segmented attention plan does not match Q/K lengths")
        if self.query_valid.device != q.device or k.shape != v.shape:
            raise ValueError("segmented attention plan and Q/K/V must share device and shape")
        batches = []
        for b, segments in enumerate(self.rows):
            outputs = []
            for segment in segments:
                # Safe dummy key for empty rows; output AND gradients are zeroed.
                spans = segment.keys or ((0, 1),)
                keys = [k[b:b + 1, :, start:stop] for start, stop in spans]
                values = [v[b:b + 1, :, start:stop] for start, stop in spans]
                key = keys[0] if len(keys) == 1 else torch.cat(keys, dim=2)
                value = values[0] if len(values) == 1 else torch.cat(values, dim=2)
                output = F.scaled_dot_product_attention(
                    q[b:b + 1, :, segment.start:segment.stop], key, value,
                    dropout_p=dropout_p, is_causal=False, scale=scale,
                )
                valid = self.query_valid[b:b + 1, None, segment.start:segment.stop, None]
                if not segment.keys:
                    valid = torch.zeros_like(valid)
                outputs.append(output.masked_fill(~valid, 0))
            batches.append(torch.cat(outputs, dim=2))
        return torch.cat(batches, dim=0)


def history_attention_plan(query, key=None):
    """Same block-id/time/validity predicate as history_attention_mask.

    Construct one key-validity vector at a time on CPU, never [Nq,Nk].
    Metadata transfers happen once per history call, outside layer checkpointing.
    """
    key = query if key is None else key
    if query.valid.shape[0] != key.valid.shape[0] or query.valid.device != key.valid.device:
        raise ValueError("query and key layouts must share batch size and device")
    qb, qt = query.block_id.cpu(), query.end_step.cpu()
    kb, kt, kv = key.block_id.cpu(), key.end_step.cpu(), key.valid.cpu()
    rows = []
    for b in range(qb.shape[0]):
        changes = ((qb[b, 1:] != qb[b, :-1]) | (qt[b, 1:] != qt[b, :-1])).nonzero().flatten() + 1
        boundaries = [0, *changes.tolist(), qb.shape[1]]
        segments = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            allowed = kv[b] & (kb[b] <= qb[b, start]) & (kt[b] <= qt[b, start])
            segments.append(AttentionSegment(start, stop, _spans(allowed)))
        rows.append(tuple(segments))
    return SegmentedAttention(query.valid, key.valid.shape[1], tuple(rows))


def validity_attention_plan(query_valid, key_valid):
    """F/A queries share key visibility; padding need not be contiguous."""
    if query_valid.dtype != torch.bool or key_valid.dtype != torch.bool:
        raise ValueError("attention validity must be bool")
    if query_valid.shape[0] != key_valid.shape[0] or query_valid.device != key_valid.device:
        raise ValueError("query and key validity must share batch size and device")
    rows = tuple((AttentionSegment(0, query_valid.shape[1], _spans(row)),)
                 for row in key_valid.cpu())
    return SegmentedAttention(query_valid, key_valid.shape[1], rows)


def future_attention_plan(history, future, decision_step):
    validate_decision_step(decision_step, future)
    legal_history = history.valid & (history.end_step <= decision_step[:, None])
    return validity_attention_plan(future.valid, torch.cat((legal_history, future.valid), dim=1))
