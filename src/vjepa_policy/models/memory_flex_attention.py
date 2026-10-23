"""Compiled block-sparse history attention with the exact reference visibility."""
from dataclasses import dataclass
from functools import lru_cache
import torch
from .memory_attention import SegmentedAttention

@lru_cache(maxsize=1)
def _kernels():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    def mask_kernel(mask_mod, B, H, Q_LEN, KV_LEN, device):
        # Keep tile sizes literal. Symbolic BLOCK_SIZE in Torch 2.7 can trigger
        # division-by-zero codegen for small query/key grids. Sequence lengths
        # remain dynamic; the actual mask and 128-token tiles are unchanged.
        return create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE=128)
    return (torch.compile(flex_attention, dynamic=True, fullgraph=True),
            torch.compile(mask_kernel, dynamic=True, fullgraph=True))


def _compiled_call(function, *args, **kwargs):
    # H/F/A, batch-one shared histories and mixed-episode recomputation need
    # multiple guarded layouts. Eager Flex backward materializes dense scores,
    # so exceeding the bounded compiler cache must fail instead of falling back.
    with torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True):
        return function(*args, **kwargs)

def _block_mask(mask, *, B, H, Q_LEN, KV_LEN, device):
    # Torch 2.7 dynamic reduction codegen is fragile for one-tile tiny grids.
    # Only mask construction uses this explicitly bounded eager path; attention
    # and its backward always remain compiled. The temporary mask is <= 64 KiB.
    if B * (H or 1) * Q_LEN * KV_LEN <= 65536:
        from torch.nn.attention.flex_attention import create_block_mask
        return create_block_mask(mask, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE=128)
    return _compiled_call(_kernels()[1], mask, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)


@dataclass(frozen=True)
class FlexHistoryAttention(SegmentedAttention):
    block_mask: object = None

    @property
    def row_valid(self):
        return self.query_valid[:, None, :, None]

    def attend(self, q, k, v, *, dropout_p=0.0, scale=None):
        if dropout_p:
            raise ValueError("Flex history attention requires zero attention dropout")
        if q.device.type != "cuda":
            raise ValueError("Flex history attention requires CUDA")
        if q.shape[2] != self.query_valid.shape[1] or k.shape[2] != self.key_length:
            raise ValueError("Flex history plan does not match Q/K lengths")
        out = _compiled_call(_kernels()[0], q, k, v, block_mask=self.block_mask, scale=scale,
                            kernel_options={"BLOCK_M": 64, "BLOCK_N": 64,
                                            "BLOCK_M1": 32, "BLOCK_N1": 32,
                                            "BLOCK_M2": 32, "BLOCK_N2": 32,
                                            "num_stages": 1, "num_warps": 4})
        return out.masked_fill(~self.query_valid[:, None, :, None], 0)

def flex_validity_attention_plan(query_valid, key_valid):
    """One batched softmax over all legal keys, without per-row KV copies."""
    if query_valid.dtype != torch.bool or key_valid.dtype != torch.bool:
        raise ValueError("attention validity must be bool")
    if query_valid.device.type != "cuda" or query_valid.device != key_valid.device:
        raise ValueError("Flex validity attention requires a shared CUDA device")
    if query_valid.shape[0] != key_valid.shape[0]:
        raise ValueError("query and key validity must share batch size")
    row_valid = query_valid & key_valid.any(dim=1)[:, None]
    nq, nk = query_valid.shape[1], key_valid.shape[1]
    def mask(batch, head, q, k):
        qi, ki = q.clamp(max=nq-1), k.clamp(max=nk-1)
        valid = row_valid[batch, qi]
        return (q < nq) & (k < nk) & ((valid & key_valid[batch, ki]) | (~valid & (k == 0)))
    block_mask = _block_mask(mask, B=query_valid.shape[0], H=None, Q_LEN=nq, KV_LEN=nk,
                               device=query_valid.device)
    return FlexHistoryAttention(row_valid, nk, (), block_mask)


def flex_future_attention_plan(history, future, decision_step):
    from .memory_types import validate_decision_step
    validate_decision_step(decision_step, future)
    legal_history = history.valid & (history.end_step <= decision_step[:, None])
    return flex_validity_attention_plan(future.valid, torch.cat((legal_history, future.valid), dim=1))


def flex_history_attention_plan(query, key=None):
    key = query if key is None else key
    if query.valid.device.type != "cuda":
        raise ValueError("Flex history attention requires CUDA")
    qb, qt, qv = query.block_id, query.end_step, query.valid
    kb, kt, kv = key.block_id, key.end_step, key.valid
    nq, nk = qb.shape[1], kb.shape[1]
    def mask(batch, head, q, k):
        # BlockMask may inspect a padded tile beyond the physical Q/K length.
        qi, ki = q.clamp(max=nq-1), k.clamp(max=nk-1)
        allowed = kv[batch,ki] & (kb[batch,ki] <= qb[batch,qi]) & (kt[batch,ki] <= qt[batch,qi])
        return (q < nq) & (k < nk) & ((qv[batch,qi] & allowed) | (~qv[batch,qi] & (k == 0)))
    block_mask = _block_mask(mask, B=qb.shape[0], H=None, Q_LEN=nq, KV_LEN=nk, device=query.valid.device)
    return FlexHistoryAttention(query.valid, nk, (), block_mask)
