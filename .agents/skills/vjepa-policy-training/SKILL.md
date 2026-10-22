---
name: vjepa-policy-training
description: Prepare data and text caches, run VJEPA-Policy predictor-only pretraining or downstream joint training, and resume checkpoints. Use for From Scratch and Pretrained Predictor workflows, with native RMBench as the reference example and support for custom datasets.
---

# V-JEPA Policy training

Run from the repository root. Read [Setup](../../../docs/setup.md) and [Training](../../../docs/training.md); use the [GR-1 skill](../vjepa-policy-gr1/SKILL.md) for its separate data/state preparation.

## Select the current entry point

| Task | Source |
| --- | --- |
| Predictor-only pretraining on DROID | `scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh` |
| From Scratch (RMBench) | `examples/rmbench/train.sh` |
| Pretrained Predictor | `scripts/train_vjepa_policy.py --predictor-init PATH` |
| Custom predictor dataset | `scripts/pretrain_predictor.py` |
| Custom policy dataset / LeRobot | `scripts/train_vjepa_policy.py` |

Copy the relevant `configs/*.env` template to an ignored `.local.env`, resolve artifact paths, and source it explicitly. Check the launcher and Python parser when changing flags; not every shell script forwards arguments. Prefer the user's requested recipe and hardware; label adaptations to the reference protocol.

## Non-obvious contracts

- Use **From Scratch** and **Pretrained Predictor** in user-facing descriptions. Predictor-only pretraining uses DROID videos, instructions, and observed state without action labels or an action expert. Downstream training jointly optimizes both modules, with the action expert initialized from scratch in both variants. The visual and text encoders remain frozen.

- DROID uses a 129-token memmap cache built by `scripts/cache_droid_t5.py`; RMBench uses 128-token prompt files built by `vjepa_policy.text_embeddings`. Its default is 128; pass other recipes' lengths explicitly.
- The DROID reference shell launcher enforces eight GPUs. Use its Python entry point with Accelerate for another topology, not an ineffective environment override.
- Predictor transfer needs an explicit path and compatible state dimensions. Only predictor weights transfer. `--resume` reloads policy weights/progress and advances the schedule; AdamW moments are not checkpointed.
- Global batch is GPUs × per-device batch × accumulation. Keep it fixed for a reproduction, or record the changed value for an experiment.
- DDP shares gradient storage with communication buckets (`gradient_as_bucket_view=True`). Preserve the Trainer's `zero_grad(set_to_none=True)` behavior and avoid in-place `detach_()` on gradient views. Single-GPU capacity measurements do not include DDP bucket overhead; stress the longest history after the first optimizer update before estimating multi-GPU capacity. Record process-local allocator overrides alongside NCCL settings.
- Templates and launchers share `configs/recipes/*.env`. Existing environment variables override defaults; source the local template for artifact paths. RMBench uses V-JEPA 2.1 ViT-L, packed state width 48, and text length 128.
- New checkpoints serialize serving metadata. Preserve the generated dataset statistics and cache/encoder provenance for later inference.

Use a fresh output directory for a smoke run. Report whether cache construction, data loading, forward/backward, optimizer updates, and checkpoint saving were actually tested. Do not infer training success from imports or CLI help alone. For custom data/encoders follow [Extension contracts](../../../docs/extensions.md).

Require successful launcher exit after checkpoint saving and record any NCCL transport overrides. A logged optimizer step alone does not establish distributed teardown.

For the native RMBench ALOHA baseline, use the [RMBench skill](../vjepa-policy-rmbench/SKILL.md) and its separate 14D joint-action adapter.

For Full-History v0, read [M3 training integration](../../../docs/full_history_v0_m3.md) and [attention capacity measurements](../../../docs/full_history_attention.md). Select `--policy-arch full_history` explicitly; preserve the frozen M0 contract and complete observed prefixes. Record any early-decision/episode diagnostic selection. Feature caches contain independent observed Encoder features and separately normalized future targets, never trainable Predictor K/V. Resume and reconstruction must validate architecture/schema, rotary settings and frozen/data provenance before strict load; legacy loaders reject memory checkpoints. The default `segmented` attention backend partitions queries while preserving one softmax over every legal key. `--history-attention-backend dense` retains the bounded correctness reference and its preflight limit. Do not infer 81-block trainability from bypassing the dense limit: measure forward/backward and AdamW updates with the intended model, batch, precision and checkpointing. Use fresh cache directories when encoding source or weights change.
