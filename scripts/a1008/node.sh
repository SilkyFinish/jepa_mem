#!/usr/bin/env bash
# Use as the platform's foreground entry command. Logs/checkpoints live on shared storage.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! -x "$POLICY_PYTHON" || ! -x "$RMBENCH_PYTHON" ]]; then
  echo 'Policy/simulator interpreter is missing on this mount; run host_setup.sh first.' >&2
  exit 2
fi
exec "$POLICY_PYTHON" -u -m scripts.a1008.job "$@"
