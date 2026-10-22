# Setup

Use Python 3.10+ and a dedicated conda policy environment. Install a PyTorch build compatible with the GPU driver, then install from the repository root:

```bash
python -m pip install -e '.[serving]'
python -c 'from vjepa_policy.encoders import encoder_names; print(encoder_names())'
```

Keep policy and simulator environments separate. Native RMBench requires its pinned SAPIEN, planning stack, robot assets and released data; follow [its guide](../examples/rmbench/README.md). RoboCasa-GR1 uses [its dedicated environment and guide](../examples/gr1/README.md).

## Artifacts

Supply matching frozen visual weights, the exact T5 snapshot or prepared instruction cache, training data, normalization statistics and policy checkpoint. RMBench uses 128-token prompt files; DROID uses a 129-token memmap; GR-1 defaults to 48-token prompt files. Keep provenance with every run.

Local paths belong in ignored `configs/*.local.env` files. Source them explicitly; launchers do not discover local files automatically. Shared visual/T5 weights belong in the HDD project's `weights/` directory, separate from task-specific experiments.

Use `configs/gr1_vjepa21.env` as the portable GR-1 template. Native RMBench preparation and local settings are documented in its example; Full-History settings are recorded in [M3](full_history_v0_m3.md) and [M4](full_history_v0_m4.md).

The websocket transport uses `openpi-client` message serialization; the server itself is in this repository.

For a persistent host sharing storage with an ephemeral eight-A100 node, use [the dedicated deployment guide](a1008_shared_training.md). It provides a separate conda environment, pinned official downloads, a foreground platform entry command, resumable training and serial native-limit RMBench evaluation.

## Checks

Check the actual interpreter, package versions and CUDA availability independently for policy and simulator. Compare camera order, resize mode, state/action contract, encoder identity and cache length against checkpoint metadata. Require real action requests and simulator rollouts before reporting performance. Preserve failed renderer/planner diagnostics.
