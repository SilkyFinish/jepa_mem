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
