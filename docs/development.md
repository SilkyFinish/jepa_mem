# Development checks

Run from the repository root. The baseline suite uses only the Python standard library and Bash; it does not install the training package or download weights.

```bash
python scripts/check_environment.py --profile source
python -m unittest discover -s tests -v
```

The serving-observation, RMBench, full-history operator, full-history model, full-history training, and online-cache test modules require the installed policy environment; RMBench also requires h5py, and the memory tests require PyTorch and timm. Standard-library-only environments report these dependency-requiring modules as skipped before importing optional dependencies. In a complete policy environment, the same discovery command runs all tests; unexpected import failures and failed assertions still fail the suite. Training-stop tests load their standard-library helper directly, without importing the ML package initializer.

The lightweight checks do not validate model numerics. In the policy environment, the [full-history M1 tests](full_history_v0_m1.md) additionally check CPU attention/RoPE numerics, prefix gradients, and legacy compatibility. The [M2 model tests](full_history_v0_m2.md) add two-stage H/F equivalence, per-block state causality, frozen V-JEPA2.1 encoding, joint-loss gradients, optimizer updates, and deterministic action sampling. Neither suite validates GPU memory requirements, simulator rendering, or published success rates. A passing workflow means the lightweight contracts passed; it does not establish benchmark reproduction.

The [segmented attention checks](full_history_attention.md) compare outputs and gradients against the bounded dense reference. `scripts/check_memory_attention_cuda.py` verifies the CUDA fused backend; `scripts/benchmark_memory_attention.py` measures actual model updates at a selected complete-prefix length. Run GPU diagnostics explicitly on the user's selected device and keep their artifacts separate from formal training. Capacity, checkpoint reload and online KV equivalence do not establish closed-loop success rates.

## Inspect an execution environment

Use the interpreter that will run each part of the experiment:

## Recipe precedence

The [M3 training tests](full_history_v0_m3.md) additionally require LeRobot and Accelerate. They exercise native history data/collation, frozen feature-cache provenance, explicit architecture dispatch and actual Trainer update/save/reload/resume, while a separate real-weight GPU diagnostic checks feature parity and small-sample fitting.

The [M4 online-cache record](full_history_v0_m4.md) documents incremental/full-recompute equivalence, atomic cache publication, request/session checks and native-RGB WebSocket diagnostics. `tests/test_memory_v0_online.py` adds CPU tests; `scripts/check_memory_online.py` uses real checkpoint/data assets and the serving dependencies. The recorded 106-test pass predates the last boundary fixes. Remaining tests, CLI checks and final review are explicitly deferred; consult the M4 status before treating the current tree as validated.


Shared serving is implemented in `policy_serving/base.py`. Native RMBench and GR-1 keep separate action decoders. The GR-1 launcher regression remains independent of simulator assets.
