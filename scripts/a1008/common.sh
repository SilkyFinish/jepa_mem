#!/usr/bin/env bash
set -euo pipefail
JEPA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_CONFIG="${1:-$JEPA_REPO/configs/a1008.local.env}"
if [[ ! -f "$JEPA_CONFIG" ]]; then
  echo "Missing $JEPA_CONFIG; follow docs/a1008_shared_training.md to configure shared paths." >&2
  exit 2
fi
source "$JEPA_CONFIG"
source "$JEPA_REPO/configs/full_history_a1008.env"
if [[ ! -d "$JEPA_SHARED_ROOT" || ! -d "$RMBENCH_ROOT" ]]; then
  echo 'Shared storage or existing RMBench checkout is not mounted.' >&2
  exit 2
fi
mkdir -p "$TMPDIR"
export PYTHONPATH="$JEPA_REPO/src:$JEPA_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$JEPA_REPO"
