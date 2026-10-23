"""Reuse one causal regular prefix for multiple decisions in the SAME update."""
import torch
from torch.utils.checkpoint import checkpoint
from .memory_types import HistoryEncoding, LayerKV, JoinedLayerKV, TokenLayout, concatenate_layouts

def checkpoint_independent_samples(policy, batch, noise, time):
    """Bound memory for rare mixed-episode batches without dropping any row.

    Keep the original batch mean and flow draws. Each independent sample is
    recomputed during backward, so full H/F activations for all rows need not
    coexist. Non-reentrant checkpointing preserves ordinary DDP gradient hooks.
    """
    actions = batch["action"]
    if noise is None:
        noise = torch.randn_like(actions)
    elif noise.shape != actions.shape or noise.dtype != actions.dtype or noise.device != actions.device:
        raise ValueError("training noise must match actions")
    if time is None:
        time = policy.action_expert.sample_time(actions.shape[0], actions.device, actions.dtype)
    results = []
    for row in range(actions.shape[0]):
        sample = {name: value[row:row+1] if torch.is_tensor(value) or name in ("episode_id", "sample_key")
                  else value for name, value in batch.items()}
        results.append(checkpoint(policy._forward_batch, sample, flow_noise=noise[row:row+1],
                                  flow_time=time[row:row+1], use_reentrant=False))
    loss = torch.stack([result[0] for result in results]).mean()
    metrics = {key: torch.stack([result[1][key] for result in results]).mean()
               for key in results[0][1]}
    return loss, metrics


def expand_layout(layout, batch):
    return TokenLayout(**{name: getattr(layout, name).expand(batch, *getattr(layout, name).shape[1:])
                          for name in ("valid", "block_id", "end_step", "rope_coords", "view_id")})

def encode_shared_history(predictor, z, states, language, language_mask, ends, valid, decisions, config):
    # This path is selected only for rows from the same recorded episode.
    # Pick the longest row to obtain every regular block needed by all queries.
    source = int(decisions.argmax())
    regular = valid[source] & (ends[source] % config.history_interval_steps == 0)
    regular_ends = ends[source:source+1, regular]
    regular_valid = torch.ones_like(regular_ends,dtype=torch.bool)
    layout = predictor.history_layout(regular_ends, regular_valid, temporal_unit_steps=config.temporal_unit_steps)
    common = predictor.forward_history(z[source:source+1,regular], states[source:source+1,regular],
                                       language[source:source+1],language_mask[source:source+1],layout)
    batch = z.shape[0]
    shared = HistoryEncoding(tuple(LayerKV(kv.key.expand(batch,-1,-1,-1),
                                          kv.value.expand(batch,-1,-1,-1)) for kv in common.layers),
                             expand_layout(common.layout,batch))
    # Off-grid observations are separate leaves: each reads only its own legal
    # regular prefix. A leaf never becomes another decision's history.
    rows = torch.arange(batch,device=z.device)
    last = valid.sum(1)-1
    leaf_valid = (decisions % config.history_interval_steps != 0)[:,None]
    leaf_layout = predictor.history_layout(decisions[:,None],leaf_valid,
                    temporal_unit_steps=config.temporal_unit_steps,block_offset=regular_ends.shape[1])
    leaf = predictor.forward_history_append(shared,z[rows,last][:,None],states[rows,last][:,None],
                                            language,language_mask,leaf_layout)
    joined = concatenate_layouts(shared.layout,leaf.layout)
    legal = joined.valid & (joined.end_step <= decisions[:,None])
    joined = TokenLayout(legal,joined.block_id,joined.end_step,joined.rope_coords,joined.view_id)
    return HistoryEncoding(tuple(JoinedLayerKV((h, l)) for h, l in zip(shared.layers, leaf.layers)), joined)
