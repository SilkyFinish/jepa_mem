#!/usr/bin/env bash
# Set shared machine paths once; the generated file is ignored by Git.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash scripts/a1008/configure.sh SHARED_ARTIFACT_ROOT RMBENCH_ROOT RMBENCH_PYTHON' >&2
  exit 2
fi
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$project/configs/a1008.local.env"
if [[ -e "$target" ]]; then
  echo "Preserving existing $target; edit it directly to change paths." >&2
  exit 1
fi
for path in "$@"; do
  [[ "$path" = /* ]] || { echo 'Use absolute shared paths.' >&2; exit 2; }
done
[[ -d "$2" && -x "$3" ]] || { echo 'Existing simulator checkout/interpreter not found.' >&2; exit 2; }
mkdir -p "$1"
printf 'export JEPA_SHARED_ROOT=%q\nexport RMBENCH_ROOT=%q\nexport RMBENCH_PYTHON=%q\n' "$1" "$2" "$3" > "$target"
echo "Configured $target"
