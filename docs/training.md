# Training

V-JEPA Policy jointly trains an instruction-conditioned future-latent predictor and a flow-matching action expert on downstream demonstrations, with the visual and text encoders frozen. This guide uses **LIBERO as the worked example** for data preparation, caching, and training. The generic entry points also support other LeRobot datasets and [custom dataset adapters](extensions.md).

The paper compares two variants:

| Variant | Future predictor | Action expert |
| --- | --- | --- |
| From Scratch | Random initialization | Random initialization |
| Pretrained Predictor | Predictor-only pretraining on DROID without action labels | Random initialization |

Both modules are jointly optimized during downstream training. For each task, the variants use the same downstream data, training protocol, and budget. Predictor-only pretraining adds an upstream stage; it uses videos, instructions, and proprioceptive states with no action expert or action-label supervision.

Run from the repository root after [Setup](setup.md). Copy, edit, and source the relevant environment template. Shell variables override launcher defaults; extra CLI arguments are forwarded only by launchers that explicitly pass `"$@"`. The LIBERO launcher requires batch overrides through `BATCH_SIZE` / `GRAD_ACCUM` so they remain covered by its global-batch validation.

## Recipe settings

| Recipe | Template | GPU × batch × accumulation | Updates | Text length | Output variable |
| --- | --- | --- | ---: | ---: | --- |
| Predictor-only pretraining on DROID | `configs/droid_vjepa21.env` | 8 × 24 × 1 = 192 | 100,000 | 129 | `OUTPUT_DIR` |
| Downstream joint training on LIBERO (both variants) | `configs/libero_vjepa21.env` | 8 × 16 × 1 = 128 | 21,360 | 128 | `OUTDIR` |
| Downstream joint training on GR-1 | `configs/gr1_vjepa21.env` | 4 × 64 × 1 = 256 | 50,000 | 48 | `OUTDIR` |

Templates and launchers load the same defaults from `configs/recipes/`. GR-1 uses global batch 256 without gradient accumulation in both cases. Existing environment values take precedence; start a clean shell when switching recipes. Launchers check the expected global batch before starting training. `MAX_STEPS=20` selects a pipeline check, not a paper result. See [configuration precedence](development.md#recipe-precedence).

## LIBERO

### Prepare the data and configuration

`LIBERO_DATA_ROOT` must contain these four LeRobot datasets:

```text
libero_object_no_noops_1.0.0_lerobot/
libero_goal_no_noops_1.0.0_lerobot/
libero_spatial_no_noops_1.0.0_lerobot/
libero_10_no_noops_1.0.0_lerobot/
```

Copy the configuration template, fill in the dataset/encoder/cache paths, and source it:

```bash
cp configs/libero_vjepa21.env configs/libero_vjepa21.local.env
# Edit the three artifact paths in configs/libero_vjepa21.local.env.
source configs/libero_vjepa21.local.env
```

### Build the instruction cache

```bash
python -m vjepa_policy.text_embeddings \
  --dataset-dirs \
    "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot" \
    "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot" \
    "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot" \
    "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot" \
  --cache-dir "$TEXT_EMBEDDING_CACHE" \
  --model-name google/t5-v1_1-xxl --context-length 128 --device cuda
```

The cache builder's default is 128, so pass `--context-length 128` explicitly. Each file is `<sha256(prompt)>.t5_len128.pt`, containing `context` and `mask`. The prompt template is defined in `src/vjepa_policy/datasets/prompts.py`.

### Train a policy

For **From Scratch**, jointly train the predictor and action expert from their random initialization:

```bash
bash scripts/train_vjepa_policy_fresh_packed48.sh
```

For **Pretrained Predictor**, complete the [predictor-only pretraining stage](#predictor-only-pretraining) and use its checkpoint with the same downstream configuration:

```bash
export PREDICTOR_INIT=/path/to/droid/checkpoint_step100000.pt
bash scripts/train_vjepa_policy_droid_init.sh
```

Only predictor weights transfer; the action expert starts from scratch. Both wrappers use packed state (`max_state_dim=48`: 48 values plus 48 validity bits). The fresh wrapper clears inherited `PREDICTOR_INIT`. When the source predictor has four view-embedding rows, transfer maps the first two rows by default; other predictor parameters load strictly. `--predictor-view-map` overrides this mapping for a different camera order.

### Outputs and hardware changes

Keep the checkpoint, generated `dataset_stats.json`, encoder identity, text-cache provenance, resolved arguments, and training revision together. Continue with the [LIBERO evaluation example](../examples/libero/README.md), or evaluate the same policy on [LIBERO-Plus](../examples/libero_plus/README.md) without further fine-tuning.

For fewer GPUs, preserve `NUM_GPUS × BATCH_SIZE × GRAD_ACCUM=128` if reproducing the reference batch. For example, four GPUs, batch 16, accumulation 2. For a small smoke run a different batch is fine if labelled as such. Set a unique `OUTDIR`; this launcher appends to `train.log`.

## Predictor-only pretraining

DROID supplies the video–instruction pairs for predictor-only pretraining. The predictor also receives observed proprioceptive state. The visual and text encoders remain frozen; no action expert or action labels are used at this stage.

Copy `configs/droid_vjepa21.env` to `configs/droid_vjepa21.local.env`, edit its paths, then build the cache and launch:

```bash
source configs/droid_vjepa21.local.env
python scripts/cache_droid_t5.py --dataset-root "$DATASET_ROOT" \
  --cache-dir "$TEXT_CACHE_DIR" --model-name google/t5-v1_1-xxl \
  --context-length 129 --device cuda --batch-size 1
bash scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh
```

The recipe uses two camera views. `--max-views 4` sets view-embedding table capacity; it does not add two more input cameras. The DROID cache uses `embeddings.bin`, `index.json`, and `manifest.json`, rather than LIBERO's prompt-indexed files. The reference shell launcher enforces eight GPUs and global batch 192. For another GPU topology, inspect `scripts/pretrain_droid_predictor.py --help` and launch it directly with Accelerate. Select `PYTHON` and `ACCELERATE` from the same environment.

## Checkpoint resumption and other datasets

`--predictor-init` imports only predictor weights; `--resume` reloads trainable weights and progress, then fast-forwards the learning-rate schedule; AdamW moments are not saved or restored. Pass these to a supported Python entry point or a wrapper that forwards arguments. `scripts/gr1/train.sh` only accepts the encoder name as its positional argument.

The generic entry points are `scripts/pretrain_predictor.py` for predictor-only pretraining and `scripts/train_vjepa_policy.py` for downstream joint training. See [Extension contracts](extensions.md) for `--dataset-factory`, camera order, tensor shapes, and encoder registration. For RoboCasa-GR1, use its [preparation/training guide](../scripts/gr1/README.md) and [evaluation example](../examples/gr1/README.md).

## Single-task preparation and diagnostics

For a small experiment on an existing original LIBERO HDF5, use `scripts/prepare_libero_single_task.py --source SOURCE.hdf5 --output runs/task_data`. It preserves native 20 Hz frames/actions, rotates both cameras 180 degrees to match the client, splits whole demonstrations deterministically (40 train, remainder heldout by default), and writes training-only quantile statistics plus a source-hashed manifest. The source must provide `ee_pos`, axis-angle `ee_ori`, `gripper_states`, `agentview_rgb`, and `eye_in_hand_rgb`. This adapter currently expects native 128px images. Build the text cache for that instruction before training.

The generic trainer accepts `--save-initial-checkpoint` to save a step-zero policy for a matched untrained baseline. This flag does not resume or initialize from another policy. A frozen visual encoder remains required.

`scripts/run_single_task_experiment.py {smoke,train,eval} --config CONFIG.json --output-dir FRESH_DIR` records exact commands and launches an explicitly configured diagnostic. The JSON contains interpreter/artifact paths, exact `train_args`, GPU IDs, smoke/training step budgets, and fixed evaluation task/state IDs. `eval` additionally requires `--checkpoint`, uses statistics beside that checkpoint, and can reduce coverage using `--trials`. Keep each phase in a fresh output directory. See [single-task usage](single-task.md). These experiments are reduced diagnostics, not the four-suite paper protocol.

### Selected multi-task runs with a time budget

For a selected LIBERO subset, `scripts/prepare_libero_multi_task.py --request REQUEST_JSON` converts each task separately and pools only training action/state samples to compute shared quantiles. Its request records task names, source HDF5 files, prepared-data directories, seed, training demonstrations per task, epochs, global batch, and rollout budget; generated manifests preserve the split and source hashes. Build the text cache across all training roots.

The Python training entry points accept `--stop-at ISO8601_WITH_TIMEZONE` and `--stop-file PATH`. Each rank checks the request at optimizer boundaries and collectively agrees to save and exit. `training_result.json` records actual steps, completion, and the final checkpoint. Checkpoints preserve model weights and progress; AdamW moments are not saved. A stop request does not count as completed training.

`scripts/run_training_rollout_pipeline.py --config EXPERIMENT_JSON --output-dir FRESH_DIR` runs the explicit training configuration, then evaluates each configured task serially using the completed final checkpoint. Configuration includes `tasks`, `rollouts_per_task`, and a timezone-qualified `pipeline_deadline`, in addition to the diagnostic launcher fields. It skips rollouts after failed or incomplete training, writes `status.json`, and owns a separate process group for each stage. At the deadline it requests a training save, allows 90 seconds, then terminates remaining workers and forcibly releases its process group after 120 seconds; schedule the deadline early enough for this allowance. Rollouts terminate at the deadline, with incomplete coverage preserved as partial results.

The original-HDF5 converters retain raw robosuite gripper targets (-1=open, +1=closed) and record `gripper_action_convention=robosuite` in statistics/manifests. New training checkpoints copy this convention into serving metadata. For existing weights trained from raw targets with older metadata, use the explicit serving override; correcting this output conversion does not require retraining.
