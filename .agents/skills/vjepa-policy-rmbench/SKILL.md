---
name: vjepa-policy-rmbench
description: Prepare native RMBench ALOHA data, train the original short-context policy, and verify real simulator/action contracts.
---

# RMBench short-context baseline

Read [the native example](../../../examples/rmbench/README.md), and use the training/inference skills for shared model and checkpoint contracts.

Use a separate pinned RMBench simulator environment. Preserve its XPolicyLab and CuRobo revisions, asset provenance, CUDA extension build configuration, and any author-machine path corrections. Keep artifacts in the user's selected storage location.

The native v1.0 HDF5 already shifts actions one saved frame ahead of state. Do not shift labels again. Use the official RGB codec and exactly the three head/left-wrist/right-wrist views; preserve native image orientation. The native 14D absolute joint output has grippers at 6 and 13, with 0=closed/1=open. Preserve all fourteen native joint commands.

Retain the original short-context architecture for the raw baseline. Call the random predictor/action-expert variant From Scratch, with the visual/text encoders still pretrained and frozen. Preserve the whole-task instruction; simulator target-pad identity and stage/subtask labels are not policy observations.

Validate every released frame, action/state alignment and training-only statistics. Full50 uses all released training demonstrations; the 40/10 split is a separate diagnostic and is not a formal holdout for full50. Test seeds begin at 100000 and use the native expert-solvable filter.

Require real rendering, native expert success, recorded expert-action replay, finite optimizer updates, checkpoint saving and strict reload, and a real finite 32 by 14 action request before declaring formal-training readiness. Check train/serve image equality. A smoke checkpoint is not a performance result.

Record the controller explicitly: native_qpos is TOPP-based benchmark execution, while dense15 is the dataset-cadence diagnostic. Do not combine scores across them or claim paper comparability for a changed controller. Preserve failed attempts and partial outputs and report any readiness check that remains unresolved.

Use a short HDD TMPDIR for DataLoader/planner IPC sockets and close HDF5 handles explicitly. Verify the CUDA_HOME include/lib layout when building against a conda CUDA toolkit.

SAPIEN 3.0.0b1 on RTX 5090 needs the verified OIDN 2.3.3 CUDA/core-library and loader fix. Preserve native ray-tracing settings, record loaded library hashes, and compare real RGB with the released dataset. Clean tasks also import the global object metadata JSON files. Use the native v1.0 ZIP rather than mixing it with legacy individual HDF5 files.

## Full-History v0 input checks

For v0 development, read [the M0 contract](../../../docs/full_history_v0_m0.md). Keep the model version v0 separate from schema version 1; preserve the legacy default entry points. Use scripts/check_memory_v0.py with a fresh output directory to verify the canonical input_contract.json and native data. M0 adds input contracts only; consult the later M3/M4 records for implemented training/serving paths and their validation limits. Legacy evidence alone does not establish v0 readiness. The memory contract fixes complete episode prefixes, history interval 4, decision interval 16, no additional action shift, and complete future windows.

## Full-History v0 online serving

Use the shared architecture-dispatching server and the separate `run_memory_client.py` entry point described in [M4](../../../docs/full_history_v0_m4.md). The memory collector starts with [0,0], samples every four executed control steps, retains unacknowledged blocks, and replans every 16 steps. A server acknowledgement must match the session, episode, revision and submitted prefix. Close sessions on episode completion and error; reconnect does not preserve history.

The new client defaults to one trial and 32 control steps as a diagnostic, retaining the native expert-solvable filter and explicit controller label. Native recorded RGB plus a real action request is not a simulator rollout score. Current M4 evidence covers short-prefix FP32 requests before the last boundary fixes; final checks and actual new-client rollouts remain deferred as recorded in the M4 document.

For complete episodes, `run_memory_client.py --max-steps 0` uses the simulator's native limit (500 for put_back_block). Pass the training release's fixed whole-task instruction with `--instruction` so the text-cache provenance matches. The client uses deterministic per-seed/trial session IDs to keep serving noise comparable across checkpoints, and writes atomic `progress.json` for liveness monitoring. For the optional v1 2x2 spatial pooling model, the local serial job configures 126 history blocks and a 4 GiB KV allocation/staging budget. See [staged training](../../../docs/full_history_staged_training.md); rollout process errors are recorded separately from ordinary task failures and do not stop the next training stage.
