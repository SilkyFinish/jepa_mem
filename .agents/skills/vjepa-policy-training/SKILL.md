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
- Native full-history supports the explicit frozen BERT-base language variant through `--text-encoder bert`: `v0_bert` uses 128×768 token features; original T5 `v0` remains 128×4096. Keep encoder-specific cache filenames, contracts and run directories separate. Reuse the existing trainable language input layers at the selected width; do not pool tokens or pad features to mimic T5. In the shared A100 workflow, set `JEPA_TEXT_ENCODER=bert` and `JEPA_BERT_SOURCE` in the local env; host preparation stages the pinned local model and generates the text cache on CPU without downloading T5. Checkpoints and online session metadata must retain the selected variant.
- The DROID reference shell launcher enforces eight GPUs. Use its Python entry point with Accelerate for another topology, not an ineffective environment override.
- Predictor transfer needs an explicit path and compatible state dimensions. Only predictor weights transfer. Weight exports restore weights/progress only. With `--save-training-state`, `resume_state.pt` additionally stores AdamW, scheduler, per-rank RNG and consumed data cursor; full resume requires identical data/world/batch/seeds/schedule. Do not describe a weight-only restart as exact optimizer continuation.
- Global batch is GPUs × per-device batch × accumulation. Keep it fixed for a reproduction, or record the changed value for an experiment.
- DDP shares gradient storage with communication buckets (`gradient_as_bucket_view=True`). Preserve the Trainer's `zero_grad(set_to_none=True)` behavior and avoid in-place `detach_()` on gradient views. Single-GPU capacity measurements do not include DDP bucket overhead; stress the longest history after the first optimizer update before estimating multi-GPU capacity. Record process-local allocator overrides alongside NCCL settings.
- Templates and launchers share `configs/recipes/*.env`. Existing environment variables override defaults; source the local template for artifact paths. RMBench uses V-JEPA 2.1 ViT-L, packed state width 48, and text length 128.
- New checkpoints serialize serving metadata. Preserve the generated dataset statistics and cache/encoder provenance for later inference.

Use a fresh output directory for a smoke run. Report whether cache construction, data loading, forward/backward, optimizer updates, and checkpoint saving were actually tested. Do not infer training success from imports or CLI help alone. For custom data/encoders follow [Extension contracts](../../../docs/extensions.md).

Require successful launcher exit after checkpoint saving and record any NCCL transport overrides. A logged optimizer step alone does not establish distributed teardown.

For ephemeral eight-A100 nodes mounting persistent shared storage, use [the deployment guide](../../../docs/a1008_shared_training.md) and `scripts/a1008/host_setup.sh` / `node.sh`. Keep the node entry process in the foreground; preserve platform GPU allocation. The formal plan is full50 / 100 epochs / global batch 16. Default combined mode requires real NCCL, longest-history AdamW, rendering/replay and websocket rollout gates before training. Opt-in `JEPA_TRAIN_ONLY=1` or `--train-only` preserves GPU/NCCL, cache/provenance and longest-history two-update checks but skips all simulator checks, smoke rollouts and post-training evaluation; it does not require simulator paths or interpreter on the node. Report simulation as skipped and evaluation as deferred, never passed. `--preflight-only` is compatible; `--eval-only` is not. Set `JEPA_PER_DEVICE_BATCH=2` for eight GPUs × batch 2 × accumulation 1, preserving global16 and 6,300 updates; default remains batch 1 × accumulation 2. Recheck the intended batch on the target GPUs and measure throughput rather than promising a speedup. Batch/accumulation changes require a fresh run name, not reuse of full resume state. Local RTX/CPU validation cannot certify remote GPU capacity or simulation. Host evaluation is a separate later operation using matching formal checkpoint/cache/stats, not automatically launched by train-only completion.

For the native RMBench ALOHA baseline, use the [RMBench skill](../vjepa-policy-rmbench/SKILL.md) and its separate 14D joint-action adapter.

For Full-History v0, read [M3 training integration](../../../docs/full_history_v0_m3.md) and [attention capacity measurements](../../../docs/full_history_attention.md). Select `--policy-arch full_history` explicitly; preserve the frozen M0 contract and complete observed prefixes. Record any early-decision/episode diagnostic selection. Feature caches contain independent observed Encoder features and separately normalized future targets, never trainable Predictor K/V. Resume and reconstruction must validate architecture/schema, rotary settings and frozen/data provenance before strict load; legacy loaders reject memory checkpoints. The default `segmented` attention backend partitions queries while preserving one softmax over every legal key. `--history-attention-backend dense` retains the bounded correctness reference and its preflight limit. Do not infer 81-block trainability from bypassing the dense limit: measure forward/backward and AdamW updates with the intended model, batch, precision and checkpointing. Use fresh cache directories when encoding source or weights change.
