#!/usr/bin/env bash
set -euo pipefail
: "${POLICY_PYTHON:?Set POLICY_PYTHON from the local environment file}"
: "${RMBENCH_PREPARED_DATA:?Set RMBENCH_PREPARED_DATA}"
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE}"
: "${OUTDIR:?Set a fresh OUTDIR}"
if [[ -e "$OUTDIR" ]]; then
  echo "Use a fresh OUTDIR: $OUTDIR" >&2
  exit 1
fi
mkdir -p "$OUTDIR"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
"$POLICY_PYTHON" scripts/train_vjepa_policy.py \
  --dataset-factory vjepa_policy.datasets.rmbench:build_dataset \
  --dataset-dirs "$RMBENCH_PREPARED_DATA" \
  --dataset-stats "$RMBENCH_PREPARED_DATA/dataset_stats.json" \
  --views observation.images.cam_head observation.images.cam_left_wrist observation.images.cam_right_wrist \
  --view-layout independent --action-dim 14 --proprio-dim 14 --max-state-dim 48 \
  --checkpoint "$VJEPA2_ENCODER_CHECKPOINT" --encoder vjepa2_1_vitl \
  --encoder-family vjepa2_1 --model-name vit_large --encoder-checkpoint-key ema_encoder \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" --lang-dim 4096 --context-len 128 \
  --num-frames 33 --past-frames 4 --video-frame-stride 4 --action-chunk-size 32 \
  --pred-depth 24 --pred-embed-dim 1024 --pred-num-heads 16 \
  --action-hidden-size 512 --action-num-layers 24 --condition-num-heads 8 \
  --action-normalization QUANTILE --state-normalization QUANTILE \
  --batch-size "${BATCH_SIZE:-2}" --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-4}" --prefetch-factor 2 \
  --use-activation-checkpointing --activation-checkpointing-blocks 8 \
  --lr 1e-4 --weight-decay 0.01 --mixed-precision bf16 --seed 7 \
  --save-every "${SAVE_EVERY:-1000}" --log-every 10 --save-initial-checkpoint \
  --max-steps "${MAX_STEPS:-21360}" --output-dir "$OUTDIR" "$@" \
  2>&1 | tee "$OUTDIR/train.log"
