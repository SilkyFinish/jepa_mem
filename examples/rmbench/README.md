# RMBench short-context baseline

This example adapts the unmodified V-JEPA Policy model architecture to the native RMBench ALOHA joint-action dataset. The frozen V-JEPA 2.1 ViT-L and T5-XXL encoders are reused. **From Scratch** initializes the predictor and action expert randomly; training jointly optimizes these modules. The input remains the original short past/current context, rather than an episode memory.

## Environment and assets

Use a dedicated Python 3.10 conda environment for the pinned [RMBench checkout](https://github.com/RoboTwin-Platform/RMBench), its pinned XPolicyLab submodule, SAPIEN, MPLib, CuRobo and their assets. Keep the policy training environment separate. On Blackwell, select a compatible CUDA/PyTorch build and compile the CUDA extensions for the actual GPU. Preserve the exact dependency report and any simulator compatibility changes.

Download the official `put_back_block/demo_clean.zip` through RMBench's data downloader. The native collection layout is `data/demo_clean/put_back_block/aloha_agilex/`. The ALOHA embodiment and `objects/005_button` are required by this task; arbitrary benchmark assets are not training data. Some released CuRobo YAML files contain the author's absolute asset paths: retain the original files and record the local path correction.

## Prepare data and text

Run from the V-JEPA Policy repository root:

```bash
"$POLICY_PYTHON" examples/rmbench/prepare.py \
  --source "$RMBENCH_DATA" --rmbench-root "$RMBENCH_ROOT" \
  --output "$PREPARED_ROOT"

"$POLICY_PYTHON" -m vjepa_policy.text_embeddings \
  --dataset-dirs "$PREPARED_ROOT/full50" \
  --cache-dir "$TEXT_EMBEDDING_CACHE" --model-name "$T5_MODEL" \
  --context-length 128 --device cuda
```

Preparation validates every RGB frame using the official codec, verifies the next-frame action alignment, records source hashes/seeds, and computes quantiles. `full50` uses all 50 released training demonstrations. The separate `diagnostic_train40` and `diagnostic_heldout10` manifests partition whole episodes; the heldout diagnostic uses statistics from its 40 training episodes. Those ten episodes are part of full50 and therefore do not constitute heldout validation for a formal full50 run. Formal closed-loop evaluation uses fresh test seeds starting at 100000.

## Model and data contract

- Three independent RGB views, in order: `cam_head`, `cam_left_wrist`, `cam_right_wrist`. Native 320 by 240 images are stretched to 224 by 224 with the shared video transform. Use the official decoder; no image rotation or additional channel conversion is applied.
- State and action each have 14 values: left six arm joints, left gripper, right six arm joints, right gripper. Arm joints are absolute radians. Both continuous grippers use 0=closed and 1=open; only their two output dimensions are clipped to [0,1]. Preserve arm commands outside [-1,1].
- Native v1.0 files already pair observation `state[t]` with the next saved joint state in `action[t]`. Consume these labels directly. Check `action[:-1] == state[1:]`; applying another temporal shift changes the control target.
- The clip uses offsets -4, 0, 4, ..., 32, with a 32-step action chunk. The leading past/current tubelet is context, and the other tubelets are future supervision. At inference the adapter only receives past/current images. Episode starts repeat their first image, episode tails repeat their final values and mask padded actions.
- T5-XXL context is 128 by 4096. Task input is the fixed whole-task instruction plus measured joints and RGB. Stage labels, original target-pad identity, and future images are not policy inputs.
- State is normalized then packed into the original width-48 state/validity representation. Store the corresponding statistics with each checkpoint.

## Training readiness and launch

First run contract tests, a small optimizer-update job, and `check_policy.py` with its saved checkpoint. The check performs real GPU inference, validates all 14 outputs, and compares the training and serving context images. A passed smoke job establishes pipeline readiness and does not establish benchmark performance.

```bash
source configs/rmbench_put_back_block.local.env
export RMBENCH_PREPARED_DATA="$PREPARED_ROOT/full50"
export OUTDIR=/path/to/fresh/formal_run
export MAX_STEPS=21360
bash examples/rmbench/train.sh
```

The wrapper uses the original model widths/depths and frozen encoders. Its default single-GPU batch is 2 with accumulation 8, for global batch 16. These are local experimental settings rather than a published RMBench V-JEPA Policy recipe; choose the training budget explicitly and record it. A different GPU topology or global batch requires a separate documented configuration. The wrapper starts From Scratch unless explicit predictor-initialization arguments are supplied.

## Simulator and serial rollout checks

Run `python -m examples.rmbench.check_simulator` in the simulator environment to test real rendering, native expert planning and recorded expert-action replay. Save the results for both controllers when diagnosing control cadence. `native_qpos` calls the benchmark's TOPP-based `take_action`; `dense15` interpolates over 15 physics substeps at the dataset's nominal cadence. Their results are separate protocols and must be labelled by controller.

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$POLICY_PYTHON" examples/rmbench/serve_policy.py \
  --ckpt "$CHECKPOINT" --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" --port 10010

CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$RMBENCH_PYTHON" -m examples.rmbench.run_client \
  --rmbench-root "$RMBENCH_ROOT" --port 10010 --trials 100 \
  --seed-start 100000 --controller native_qpos --output "$EVAL_OUTPUT"
```

The client runs one environment serially, uses the native expert-solvable seed filter, and records exact accepted seeds and controller. It collects an observation after every action and replans every 16 actions using the image four executed steps earlier. Unexpected simulator/inference errors abort the run; an incomplete directory is not a benchmark score. Training and serving use the same source checkout and artifact contract.

Keep temporary-directory paths short on HDD: Python DataLoader workers create AF_UNIX socket paths with a platform length limit. Close HDF5 handles before removing temporary data on FUSE/NTFS filesystems. Conda CUDA toolkits may store headers under `targets/x86_64-linux/include`; extension builds need a CUDA_HOME layout that exposes these headers and the matching CUDA libraries.

On RTX 5090, the bundled SAPIEN 3.0.0b1 OIDN 2.0.1 libraries can report CUDA-device errors. The verified local preparation uses the [upstream SAPIEN fix](https://github.com/haosulab/SAPIEN/issues/243): official OIDN 2.3.3 CUDA/core libraries and the matching loader version, retaining native ray tracing and OIDN denoising. Record library hashes and verify the actually loaded versions and RGB output. The clean task still imports the global object metadata files `objects/objaverse/list.json`, `objects/same.json`, and `objects/points_info.json`; download these alongside task assets.

Use the native v1.0 task ZIP for this adapter. The same Hugging Face repository also exposes older individual HDF5 files with a different layout; keep those formats separate.

## Full-History v0 development

The new version starts with the separate [M0 input contract](../../docs/full_history_v0_m0.md). Its check tool validates complete episode-prefix sampling and native data without changing this short-context baseline. M0 does not yet provide a memory model, training launcher, or stateful server.

## Full-History v0 training

The existing RMBench launchers remain the legacy baseline. The explicit full_history Python route adds complete observed prefixes, frozen feature caching and versioned checkpoints; see [M3 integration](../../docs/full_history_v0_m3.md). Its current dense attention backend supports short-prefix diagnostics only; it does not establish a full-history rollout score or replace the native baseline serving protocol.
