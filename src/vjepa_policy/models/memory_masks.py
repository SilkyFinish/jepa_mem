"""Small-sequence reference visibility for full-history v0.

Masks are bool [B, 1, N_query, N_key], with True meaning allowed. Each row
contains one episode and, for F/A, one decision. Callers must align episodes
across layouts. This is a dense correctness backend, not long-history serving.
"""

import torch

from .memory_types import TokenLayout, validate_decision_step

# Bound the result before allocating pairwise intermediates. Never truncate H.
DEFAULT_MAX_MASK_ELEMENTS = 16 * 1024 * 1024


def _check_size(query, keys, max_elements):
    if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements <= 0:
        raise ValueError("max_elements must be a positive integer")
    batch, queries = query.valid.shape
    for key in keys:
        if key.valid.shape[0] != batch or key.valid.device != query.valid.device:
            raise ValueError("query and key layouts must share batch size and device")
    elements = batch * queries * sum(key.valid.shape[1] for key in keys)
    if elements > max_elements:
        raise ValueError(
            f"dense reference mask needs {elements} elements, limit is {max_elements}; "
            "use a block attention backend for long history (history is never truncated)"
        )


def history_attention_mask(
    query: TokenLayout,
    key: TokenLayout | None = None,
    *,
    max_elements: int = DEFAULT_MAX_MASK_ELEMENTS,
):
    """H_i reads H_j with j <= i, bidirectionally within a whole visual block.

    key defaults to query for recomputation. For appending, supply the combined
    prefix/current key layout. Both block order and actual cutoff must agree.
    """
    key = query if key is None else key
    _check_size(query, (key,), max_elements)
    allowed = key.block_id[:, None, :] <= query.block_id[:, :, None]
    allowed &= key.end_step[:, None, :] <= query.end_step[:, :, None]
    allowed &= query.valid[:, :, None] & key.valid[:, None, :]
    return allowed[:, None]


def future_attention_mask(
    history: TokenLayout,
    future: TokenLayout,
    decision_step: torch.Tensor,
    *,
    max_elements: int = DEFAULT_MAX_MASK_ELEMENTS,
):
    """F_t reads H ending <= t and every valid token in its own F_t.

    Keys must be concatenated as [H, F]. No action keys are accepted.
    """
    _check_size(future, (history, future), max_elements)
    validate_decision_step(decision_step, future)
    history_valid = history.valid & (history.end_step <= decision_step[:, None])
    key_valid = torch.cat((history_valid, future.valid), dim=1)
    return (future.valid[:, :, None] & key_valid[:, None, :])[:, None]


def action_attention_mask(
    history: TokenLayout,
    future: TokenLayout,
    action: TokenLayout,
    decision_step: torch.Tensor,
    *,
    max_elements: int = DEFAULT_MAX_MASK_ELEMENTS,
):
    """A_t reads legal H, this F_t, and its whole bidirectional action chunk.

    Keys must be concatenated as [H, F, A]. Future target times do not determine
    historical visibility. Different decisions in one row are rejected.
    """
    _check_size(action, (history, future, action), max_elements)
    validate_decision_step(decision_step, future)
    validate_decision_step(decision_step, action)
    history_valid = history.valid & (history.end_step <= decision_step[:, None])
    key_valid = torch.cat((history_valid, future.valid, action.valid), dim=1)
    return (action.valid[:, :, None] & key_valid[:, None, :])[:, None]
