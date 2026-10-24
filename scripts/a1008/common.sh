#!/usr/bin/env bash
set -euo pipefail
JEPA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_CONFIG="${1:-$JEPA_REPO/configs/a1008.local.env}"
if [[ ! -f "$JEPA_CONFIG" ]]; then
  echo "Missing $JEPA_CONFIG; follow docs/a1008_shared_training.md to configure shared paths." >&2
  exit 2
fi
source "$JEPA_CONFIG"
for option in "${@:2}"; do
  if [[ "$option" == --train-only ]]; then export JEPA_TRAIN_ONLY=1; fi
  if [[ "$option" == --dense-full ]]; then
    # Upgrade the experiment recipe after sourcing machine paths. Old v0 local
    # files often pin 100/400 epochs and an existing run name.
    export JEPA_TEXT_ENCODER=bert
    export JEPA_HISTORY_MODEL_VERSION=v2_full
    export JEPA_HISTORY_BACKEND=flex
    export JEPA_SHARED_PREFIX=1
    export JEPA_ACTION_CHECKPOINTING=1
    export JEPA_PER_DEVICE_BATCH=2
    export JEPA_EPOCHS=25
    export JEPA_RUN_NAME="${JEPA_DENSE_FULL_RUN_NAME:-put_back_block_full50_v2_full_bert_ep25}"
  fi
done
source "$JEPA_REPO/configs/full_history_a1008.env"
if [[ ! -d "$JEPA_SHARED_ROOT" ]]; then
  echo 'Shared storage is not mounted.' >&2
  exit 2
fi
if [[ "$JEPA_TRAIN_ONLY" != 1 ]]; then
  if [[ ! -d "$RMBENCH_ROOT" ]]; then
    echo 'Existing RMBench checkout is not mounted.' >&2
    exit 2
  fi
fi
mkdir -p "$TMPDIR"
export PYTHONPATH="$JEPA_REPO/src:$JEPA_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$JEPA_REPO"
