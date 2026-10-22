# Training

V-JEPA Policy jointly trains a future predictor and flow-matching action expert with frozen visual/text encoders. From Scratch initializes both trainable modules randomly; Pretrained Predictor imports predictor-only weights and initializes a fresh action expert.

## Native RMBench

Follow [the native guide](../examples/rmbench/README.md) for dataset, cache, frame/action alignment and training-only statistics. From the repository root:

```bash
source configs/rmbench_put_back_block.local.env
export RMBENCH_PREPARED_DATA="$PREPARED_ROOT/full50"
export OUTDIR=/path/to/fresh/formal_run
export MAX_STEPS=21360
bash examples/rmbench/train.sh
```

These are local single-task settings. Preserve resolved arguments, checkpoint, statistics, encoder identity, text-cache provenance and source revision. Label optimizer diagnostics separately from formal training and task-success evaluation.

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

The recipe uses two camera views. `--max-views 4` sets view-embedding table capacity; it does not add two more input cameras. The DROID cache uses `embeddings.bin`, `index.json`, and `manifest.json`, rather than prompt-indexed files. The reference shell launcher enforces eight GPUs and global batch 192. For another GPU topology, inspect `scripts/pretrain_droid_predictor.py --help` and launch it directly with Accelerate. Select `PYTHON` and `ACCELERATE` from the same environment.

## Checkpoint resumption and other datasets

`--predictor-init` imports only predictor weights; `--resume` reloads trainable weights and progress, then fast-forwards the learning-rate schedule; AdamW moments are not saved or restored. Pass these to a supported Python entry point or a wrapper that forwards arguments. `scripts/gr1/train.sh` only accepts the encoder name as its positional argument.

The generic entry points are `scripts/pretrain_predictor.py` for predictor-only pretraining and `scripts/train_vjepa_policy.py` for downstream joint training. See [Extension contracts](extensions.md) for `--dataset-factory`, camera order, tensor shapes, and encoder registration. For RoboCasa-GR1, use its [preparation/training guide](../scripts/gr1/README.md) and [evaluation example](../examples/gr1/README.md).


The Python entry points accept `--stop-at ISO8601_WITH_TIMEZONE` and `--stop-file PATH` at optimizer boundaries. Check `training_result.json` for actual completion. `--save-initial-checkpoint` saves an untrained comparison. DDP uses gradient bucket views; preserve `zero_grad(set_to_none=True)` and measure capacity with the selected accumulation and longest history.


## Full-History v0

The default policy architecture remains legacy. Use the explicit full_history route for native RMBench complete-prefix training, frozen observation/target feature caches, versioned checkpoints and strict reconstruction; see [M3 training integration](full_history_v0_m3.md) for commands, diagnostic selection and resume semantics. Full-history defaults to exact segmented SDPA; `--history-attention-backend dense` selects the bounded reference. See [attention optimization and capacity measurements](full_history_attention.md) before choosing a training batch/history length.
