#!/usr/bin/env bash
# Run once on the persistent host. No GPU is required; never edits the simulator environment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "${1:-}"
exec 9>"$JEPA_SHARED_ROOT/.host_setup.lock"
flock -n 9 || { echo 'Another host preparation is running.' >&2; exit 1; }
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$CONDA_PKGS_DIRS"
# Wheel/build staging can be large; only runtime IPC uses the short node-local TMPDIR.
export TMPDIR="$JEPA_SHARED_ROOT/tmp/install"
mkdir -p "$TMPDIR"
if [[ ! -x "$POLICY_PYTHON" ]]; then
  "${CONDA_EXE:-conda}" create -y -p "$POLICY_ENV" python=3.10 pip
fi
"$POLICY_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$JEPA_TORCH_INDEX"
"$POLICY_PYTHON" -m pip install -c configs/a1008-requirements.txt -e '.[serving]' h5py opencv-python-headless
"$POLICY_PYTHON" -m pip check
"$POLICY_PYTHON" -m pip freeze > "$JEPA_SHARED_ROOT/policy-environment.txt"
# Transport-only overlay: leave the previously verified simulator environment intact.
"$POLICY_PYTHON" -m pip --python "$RMBENCH_PYTHON" install --target "$JEPA_SHARED_ROOT/envs/sim_client" --upgrade --no-deps \
  openpi-client==0.1.2 websockets==16.1.1 msgpack==1.1.2 typing-extensions==4.15.0
"$POLICY_PYTHON" -m scripts.a1008.prepare_assets
