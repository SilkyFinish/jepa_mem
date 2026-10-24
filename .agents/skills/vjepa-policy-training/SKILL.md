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
- Predictor transfer needs an explicit path and compatible state dimensions. Only predictor weights transfer. `--save-training-state` writes atomic `resume_state.pt` with AdamW, scheduler, per-rank RNG and data cursor; `--resume` uses this for full continuation with matching data, topology, batch and schedule. Ordinary `checkpoint_step*.pt` remain weight/progress exports and do not restore AdamW moments.
- Global batch is GPUs × per-device batch × accumulation. Keep it fixed for a reproduction, or record the changed value for an experiment.
- DDP shares gradient storage with communication buckets (`gradient_as_bucket_view=True`). Preserve the Trainer's `zero_grad(set_to_none=True)` behavior and avoid in-place `detach_()` on gradient views. Single-GPU capacity measurements do not include DDP bucket overhead; stress the longest history after the first optimizer update before estimating multi-GPU capacity. Record process-local allocator overrides alongside NCCL settings.
- Templates and launchers share `configs/recipes/*.env`. Existing environment variables override defaults; source the local template for artifact paths. RMBench uses V-JEPA 2.1 ViT-L, packed state width 48, and text length 128.
- New checkpoints serialize serving metadata. Preserve the generated dataset statistics and cache/encoder provenance for later inference.

Use a fresh output directory for a smoke run. Report whether cache construction, data loading, forward/backward, optimizer updates, and checkpoint saving were actually tested. Do not infer training success from imports or CLI help alone. For custom data/encoders follow [Extension contracts](../../../docs/extensions.md).

Require successful launcher exit after checkpoint saving and record any NCCL transport overrides. A logged optimizer step alone does not establish distributed teardown.

For staged local training/evaluation, read [the serial job guide](../../../docs/full_history_staged_training.md). Keep `--max-steps` fixed for the complete learning-rate schedule and use `--stop-after-step` only for absolute optimizer boundaries. `scripts/train_eval_stages.py` resumes full state between stages, releases training processes before evaluation, records evaluation exceptions and continues training, but stops on training errors or an explicit manual STOP. Never implement stages as repeated short learning-rate schedules or weight-only restarts. Preserve milestone weight exports separately from the latest rolling resume state.

For optional spatial compression, use `--policy-arch full_history --history-model-version v1` and read [the v1 contract](../../../docs/full_history_v1_spatial.md). v0 remains the default. v1 pools each view's observed Encoder features by 2x2 before the Predictor, preserves full temporal prefixes and full-resolution future targets, and uses original-grid pooled-center RoPE coordinates. Frozen cache tensors and RGB sampling stay uncompressed; saved model-version/schema select the compression at reconstruction. Reject v0/v1 cross-version policy resume even though weight shapes match. Online append must apply the same per-block pooling and layout as training. Benchmark with `scripts/benchmark_history_training.py`; keep longest-history stress updates separate from representative throughput and record real dataset coverage. Timing diagnostics do not establish rollout success.

For the native RMBench ALOHA baseline, use the [RMBench skill](../vjepa-policy-rmbench/SKILL.md) and its separate 14D joint-action adapter.

For Full-History v0, read [M3 training integration](../../../docs/full_history_v0_m3.md) and [attention capacity measurements](../../../docs/full_history_attention.md). Select `--policy-arch full_history` explicitly; preserve the frozen M0 contract and complete observed prefixes. Record any early-decision/episode diagnostic selection. Feature caches contain independent observed Encoder features and separately normalized future targets, never trainable Predictor K/V. Resume and reconstruction must validate architecture/schema, rotary settings and frozen/data provenance before strict load; legacy loaders reject memory checkpoints. The default `segmented` attention backend partitions queries while preserving one softmax over every legal key. `--history-attention-backend dense` retains the bounded correctness reference and its preflight limit. Do not infer 81-block trainability from bypassing the dense limit: measure forward/backward and AdamW updates with the intended model, batch, precision and checkpointing. Use fresh cache directories when encoding source or weights change.

For dense decision training, read [the v2 guide](../../../docs/full_history_v2_dense.md). v2 includes every recorded anchor and clips future RGB at episode tails while masking padded actions. Preserve the 16-step execution cadence separately from the 1-step training grid. `--history-shared-prefix` shares differentiable regular histories only within one microbatch/update; off-grid current leaves must remain isolated. Record grouped sampling in resume provenance, stress leftover mixed-episode batches, and compare loss/gradients with independent history computation. Mixed episodes checkpoint each sample loss separately with non-reentrant recomputation; preserve flow draws and the original sample-mean reduction, including padded labels. `--history-attention-backend flex` and `--history-action-checkpointing` add no trainable parameters. Flex H/F/A kernels must remain compiled: use the bounded variant cache and explicit exhaustion error rather than eager dense fallback. Keep joined shared KV lazy through checkpoint boundaries, and stress adjacent longest decisions with different visibility gaps as well as mixed episodes. Frozen v2 features and the serving visual encoder both compute in FP32; Predictor/action BF16 is separate. Quote measured full50 throughput, never infer a 24-hour completion guarantee from a short or single-episode benchmark.

For HDD feature storage, the v2 guide documents optional lossless packing with `scripts/pack_rmbench_history_features.py` and `--history-feature-packed`. Retain the original `--history-feature-cache` manifest as logical data identity: packing must verify original hashes and exact FP32 tensor bytes, and the reader must bind every packed row to that identity. Compact history inputs contain one observation bank and a per-sample index; mixed-episode checkpointing must retain the whole bank while slicing indices. Validate old-reader to packed-reader full-state continuation, not a weight-only restart. HDD mmap observation pages share physical RAM but remain reclaimable; bound loader prefetch and leave an explicit host-memory reserve. Before torchrun, complete standalone observation warmup with the packer --warm-only and verify its successful exit; omit --history-feature-warm-observations from distributed training. Rank-zero-only warmup before DDP initialization can time out other ranks, so the in-training warm flag rejects distributed use. Confirm strict full-state restoration and new optimizer updates before reporting a background resume as training.
