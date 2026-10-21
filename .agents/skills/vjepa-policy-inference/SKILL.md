---
name: vjepa-policy-inference
description: Serve a VJEPA-Policy checkpoint and verify one action request using its saved encoder, geometry, state, and text-cache contract. Use for LIBERO websocket inference or checkpoint serving diagnostics, not benchmark scoring.
---

# V-JEPA Policy inference

Run from the repository root, three levels above this skill directory. Read [Setup](../../../docs/setup.md) for dependencies and [LIBERO evaluation](../../../examples/libero/README.md) for the server/client commands. For GR-1 use the [GR-1 skill](../vjepa-policy-gr1/SKILL.md).

## Resolve the serving contract

Use the checkpoint, matching statistics, visual encoder, and cache supplied by the user. Default to checkpoint metadata, not settings inferred from the run name. `examples/libero/serve_policy.py` passes explicit CLI overrides to `PolicyServingConfig.from_checkpoint` in `src/vjepa_policy/policy_serving/libero.py`. That method resolves `policy_serving.config`, `model_topology`, and legacy fallbacks.

- Inspect camera order, encoder registry ID/key, resize mode, latent grid, state packing, text length, action shape, and RoPE settings before adding overrides.
- The single serving entry point also dispatches registered DINO/WAN encoders. Discover IDs through `vjepa_policy.encoders.encoder_names()`; do not reconstruct an alternate server from a historical experiment.
- Legacy checkpoints may need explicit `--max-state-dim 0`, `--t5-len`, views, and encoder/RoPE arguments from the original run. The current LIBERO recipe uses packed state 48 and text length 128, but those are not universal defaults.
- Cache payloads use `context` and `mask`; prompt text and filename hashing come from `src/vjepa_policy/datasets/prompts.py` and `text_embeddings.py`.

## Execute and verify

Install the `serving` extra when needed. Inspect server `--help` in the selected policy environment and client `--help` in the simulator environment. Server flags are flat; the LIBERO client's flags use `--args.*`.

Use a free port and the user's selected GPU. Keep logs and commands in a fresh artifact directory. Wait for `/healthz`, then make a real inference request with the existing client (`--args.max-tasks 1 --args.num-trials-per-task 1` for a rollout check). Health alone proves neither inference nor task success. Check the returned action shape and finite values against the checkpoint contract when inspecting a request directly.

The client supplies current and past images, rotates both views by 180 degrees, and maintains frame-stride history. Do not replace this with a single-frame request when checking the trained policy. Read `run_libero_client.py` and the serving adapter when adapting an observation source.

Report which stages succeeded: imports, loading, readiness, action request, or rollout. Preserve errors and stop only processes created for this run. Use the [LIBERO skill](../vjepa-policy-libero-eval/SKILL.md) for a success-rate evaluation.

The serving boundary accepts canonical LeRobot keys and OpenPI wire aliases `observation/<camera>`, `observation/<camera>_past`, `observation/state`, and `prompt`. Wire images are uint8 HWC; canonical normalized CHW tensors also work. Both past images remain required, and image rotation stays in the simulator client.

Verify the gripper convention separately from image/state shapes: original HDF5 targets use robosuite -1=open/+1=closed, while reference LeRobot targets use 0=closed/1=open. The output decoder resolves explicit override, checkpoint/statistics metadata, then legacy LeRobot default. For older raw-HDF5 weights, explicitly select `--gripper-action-convention robosuite`. Validate normalization plus output decoding against recorded expert actions before claiming rollout scores.

For the native RMBench ALOHA baseline, use the [RMBench skill](../vjepa-policy-rmbench/SKILL.md) and its separate 14D joint-action adapter.

## Full-History v0 sessions

For RMBench memory checkpoints, read [M4 implementation/status](../../../docs/full_history_v0_m4.md). The shared RMBench server dispatches by checkpoint architecture; the legacy adapter remains separate. The memory loader checks saved encoder identity, statistics and text-cache provenance. Use the `rmbench_memory_v0` reset/append/infer/close protocol, with raw two-frame RGB blocks, complete episode prefixes and fixed whole-task instructions. Keep session KV out of training checkpoints.

Validate retries against content and revision, and require finite actions after real observation append; health alone is insufficient. A disconnect frees its sessions, so reconnect requires reset and rebuilding history from episode start. Capacity failures must be explicit, never silent history eviction. The current dense backend and default five-block limit support short diagnostics; they do not establish long-history serving performance.

The M4 record separates the previously passed tests from later untested boundary fixes and lists checks deferred by the user. Do not present those earlier results as full validation of the current implementation.
