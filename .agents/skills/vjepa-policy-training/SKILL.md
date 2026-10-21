---
name: vjepa-policy-training
description: Prepare data and text caches, run VJEPA-Policy predictor-only pretraining or downstream joint training, and resume checkpoints. Use for From Scratch and Pretrained Predictor workflows, with LIBERO as the reference example and support for custom datasets.
---

# V-JEPA Policy training

Run from the repository root. Read [Setup](../../../docs/setup.md) and [Training](../../../docs/training.md); use the [GR-1 skill](../vjepa-policy-gr1/SKILL.md) for its separate data/state preparation.

## Select the current entry point

| Task | Source |
| --- | --- |
| Predictor-only pretraining on DROID | `scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh` |
| From Scratch (LIBERO example) | `scripts/train_vjepa_policy_fresh_packed48.sh` |
| Pretrained Predictor (LIBERO example) | `scripts/train_vjepa_policy_droid_init.sh` |
| Custom predictor dataset | `scripts/pretrain_predictor.py` |
| Custom policy dataset / LeRobot | `scripts/train_vjepa_policy.py` |

Copy the relevant `configs/*.env` template to an ignored `.local.env`, resolve artifact paths, and source it explicitly. Check the launcher and Python parser when changing flags; not every shell script forwards arguments. Prefer the user's requested recipe and hardware; label adaptations to the reference protocol.

## Non-obvious contracts

- Use **From Scratch** and **Pretrained Predictor** in user-facing descriptions. Predictor-only pretraining uses DROID videos, instructions, and observed state without action labels or an action expert. Downstream training jointly optimizes both modules, with the action expert initialized from scratch in both variants. The visual and text encoders remain frozen.

- DROID uses a 129-token memmap cache built by `scripts/cache_droid_t5.py`; LIBERO uses 128-token prompt files built by `vjepa_policy.text_embeddings`. Its default is 128; pass other recipes' lengths explicitly.
- The DROID reference shell launcher enforces eight GPUs. Use its Python entry point with Accelerate for another topology, not an ineffective environment override.
- The fresh wrapper clears inherited `PREDICTOR_INIT`; predictor transfer needs the explicit source path and packed state 48. Only predictor weights transfer. `--resume` reloads policy weights/progress and advances the schedule; AdamW moments are not checkpointed.
- Global batch is GPUs × per-device batch × accumulation. Keep it fixed for a reproduction, or record the changed value for an experiment.
- Templates and launchers share `configs/recipes/*.env`. Existing environment variables override defaults; source the local template for artifact paths. LIBERO defaults to V-JEPA 2.1 ViT-L, state width 48, and text length 128.
- New checkpoints serialize serving metadata. Preserve the generated dataset statistics and cache/encoder provenance for later inference.

Use a fresh output directory for a smoke run. Report whether cache construction, data loading, forward/backward, optimizer updates, and checkpoint saving were actually tested. Do not infer training success from imports or CLI help alone. For custom data/encoders follow [Extension contracts](../../../docs/extensions.md).

For a single original LIBERO task, `scripts/prepare_libero_single_task.py` creates disjoint train/heldout roots with training-only statistics and a source manifest. It preserves no-ops and uses native 128px images rotated 180 degrees. `--save-initial-checkpoint` saves an untrained comparison policy; use separate fresh directories for smoke and formal training. `scripts/run_single_task_experiment.py` requires an explicit artifact/config JSON; label its smaller task/batch/text-encoder choices as adaptations.

For single-host diagnostics with several network interfaces, record any NCCL transport overrides in the explicit JSON `extra_env`. Require the launcher to exit successfully after checkpoint saving; a logged optimizer step is not evidence that decoder workers and distributed teardown completed.

Selected multi-task diagnostics use `scripts/prepare_libero_multi_task.py` with exact pooled training-only normalization and one shared prompt cache. `--stop-at` requires a timezone; `--stop-file` is checked collectively at optimizer boundaries. Inspect `training_result.json` for actual progress. `scripts/run_training_rollout_pipeline.py` gates serial rollouts on completed training and enforces a process-group deadline; reserve its two-minute forced-stop allowance before GPU handover.

Original-HDF5 conversion preserves robosuite gripper signs and records `gripper_action_convention=robosuite`. Preserve this metadata through pooled normalization and checkpoint export; test expert-action normalization/decoding round trips.

For the native RMBench ALOHA baseline, use the [RMBench skill](../vjepa-policy-rmbench/SKILL.md) and its separate 14D joint-action adapter.

For Full-History v0, read [M3 training integration](../../../docs/full_history_v0_m3.md). Select `--policy-arch full_history` explicitly; preserve the frozen M0 contract and complete observed prefixes. Record any early-decision/episode diagnostic selection. Feature caches contain independent observed Encoder features and separately normalized future targets, never trainable Predictor K/V. Resume and reconstruction must validate architecture/schema, rotary settings and frozen/data provenance before strict load; legacy loaders reject memory checkpoints. The dense reference backend has a preflight capacity limit and cannot yet train all 81-block prefixes. Use fresh cache directories when encoding source or weights change.
