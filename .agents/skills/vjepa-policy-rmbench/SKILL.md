---
name: vjepa-policy-rmbench
description: Prepare native RMBench ALOHA data, train the original short-context policy, and verify real simulator/action contracts.
---

# RMBench short-context baseline

Read [the native example](../../../examples/rmbench/README.md), and use the training/inference skills for shared model and checkpoint contracts.

Use a separate pinned RMBench simulator environment. Preserve its XPolicyLab and CuRobo revisions, asset provenance, CUDA extension build configuration, and any author-machine path corrections. Keep artifacts in the user's selected storage location.

The native v1.0 HDF5 already shifts actions one saved frame ahead of state. Do not shift labels again. Use the official RGB codec and exactly the three head/left-wrist/right-wrist views; do not reuse LIBERO's image rotation. The native 14D absolute joint output has grippers at 6 and 13, with 0=closed/1=open. Do not route it through LIBERO's seven-dimensional clipping/sign conversion.

Retain the original short-context architecture for the raw baseline. Call the random predictor/action-expert variant From Scratch, with the visual/text encoders still pretrained and frozen. Preserve the whole-task instruction; simulator target-pad identity and stage/subtask labels are not policy observations.

Validate every released frame, action/state alignment and training-only statistics. Full50 uses all released training demonstrations; the 40/10 split is a separate diagnostic and is not a formal holdout for full50. Test seeds begin at 100000 and use the native expert-solvable filter.

Require real rendering, native expert success, recorded expert-action replay, finite optimizer updates, checkpoint saving and strict reload, and a real finite 32 by 14 action request before declaring formal-training readiness. Check train/serve image equality. A smoke checkpoint is not a performance result.

Record the controller explicitly: native_qpos is TOPP-based benchmark execution, while dense15 is the dataset-cadence diagnostic. Do not combine scores across them or claim paper comparability for a changed controller. Preserve failed attempts and partial outputs and report any readiness check that remains unresolved.

Use a short HDD TMPDIR for DataLoader/planner IPC sockets and close HDF5 handles explicitly. Verify the CUDA_HOME include/lib layout when building against a conda CUDA toolkit.

SAPIEN 3.0.0b1 on RTX 5090 needs the verified OIDN 2.3.3 CUDA/core-library and loader fix. Preserve native ray-tracing settings, record loaded library hashes, and compare real RGB with the released dataset. Clean tasks also import the global object metadata JSON files. Use the native v1.0 ZIP rather than mixing it with legacy individual HDF5 files.
