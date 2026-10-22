---
name: vjepa-policy-inference
description: Serve a checkpoint and verify native RMBench or GR-1 actions using its saved encoder, geometry, state and text-cache contract.
---

# Policy inference

Read [Setup](../../../docs/setup.md), [RMBench](../../../examples/rmbench/README.md) and the [GR-1 skill](../vjepa-policy-gr1/SKILL.md) as applicable. Run from the repository root.

Resolve checkpoint metadata with `PolicyServingConfig.from_checkpoint` in `policy_serving/base.py`. Inspect camera order, encoder registry/key, resize mode, latent geometry, state packing, text length, action shape and RoPE settings. Preserve the matching encoder, statistics and instruction cache.

The shared loader returns unnormalized dataset coordinates. Native RMBench retains its 14D absolute-joint decoder; GR-1 retains its own 29D decoder. Validate normalization and expert-action replay before interpreting rollout scores.

Install the serving extra. Inspect `examples/rmbench/serve_policy.py --help` and the selected client with their own interpreters. Use the selected GPU, a free port and fresh logs; verify readiness, real inference, action shape and finite values separately.

Short-context serving requires both past and current images. RMBench uses three native unrotated RGB views and a fixed whole-task instruction. `policy_serving/observations.py` handles canonical and OpenPI aliases. A real action request is separate from a simulator success-rate result.


## Full-History v0 sessions

For RMBench memory checkpoints, read [M4 implementation/status](../../../docs/full_history_v0_m4.md). The shared RMBench server dispatches by checkpoint architecture; the legacy adapter remains separate. The memory loader checks saved encoder identity, statistics and text-cache provenance. Use the `rmbench_memory_v0` reset/append/infer/close protocol, with raw two-frame RGB blocks, complete episode prefixes and fixed whole-task instructions. Keep session KV out of training checkpoints.

Validate retries against content and revision, and require finite actions after real observation append; health alone is insufficient. A disconnect frees its sessions, so reconnect requires reset and rebuilding history from episode start. Capacity failures must be explicit, never silent history eviction. Full-history models default to exact segmented attention; the default five-block session limit and KV allocation budget remain explicit serving constraints. See [attention capacity results](../../../docs/full_history_attention.md); cached-feature operator tests do not establish full websocket/simulator rollout performance.

The M4 record separates the previously passed tests from later untested boundary fixes and lists checks deferred by the user. Do not present those earlier results as full validation of the current implementation.
