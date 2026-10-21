# Single-task diagnostic

Run from the repository root in the policy environment. Local preparation writes `runs/drawer_prepare/experiment.json`; it records exact artifacts, training arguments, split provenance, fixed evaluation initial states, and deviations from the paper protocol. The ignored `configs/libero_drawer.local.env` exports its absolute path. Source it explicitly.

```bash
conda activate vjepa-policy
source configs/libero_drawer.local.env
python scripts/run_single_task_experiment.py smoke --config "$DRAWER_EXPERIMENT_CONFIG" --output-dir runs/drawer_smoke_new
python scripts/run_single_task_experiment.py train --config "$DRAWER_EXPERIMENT_CONFIG" --output-dir runs/drawer_train
```

The smoke phase is separate from formal training. Do not resume the formal run from the smoke checkpoint: they have different schedules. The formal run saves step zero and every 500 updates through step 2000. Both the predictor and action expert start randomly, while the official V-JEPA 2.1 visual encoder and T5-base text encoder remain frozen. This single-task, smaller-text-encoder, global-batch-16 run is a functional diagnostic.

Evaluate the formal run's step-zero and final checkpoints on the same saved 20 initial states:

```bash
python scripts/run_single_task_experiment.py eval --config "$DRAWER_EXPERIMENT_CONFIG" --output-dir runs/drawer_eval_initial --checkpoint runs/drawer_train/checkpoint_step000000.pt
python scripts/run_single_task_experiment.py eval --config "$DRAWER_EXPERIMENT_CONFIG" --output-dir runs/drawer_eval_trained --checkpoint runs/drawer_train/checkpoint_step002000.pt
```

Each evaluation starts its own server, waits for readiness, launches the separate simulator environment, checks task/state coverage, saves videos and a completion report, and stops that server. Output directories must be fresh. A proposed engineering acceptance line is 12/20 successes and a clear gain over the step-zero policy; this is not a published benchmark threshold or evidence that the world-model component beats an ablation. Reserved demonstration episodes are separate from the fixed simulator evaluation states; the stock trainer does not implement heldout loss evaluation.

Training data is split by whole demonstrations (seed 7), with statistics computed only on the training split. Native images are 128px, both cameras are rotated 180 degrees, and all raw no-op frames are preserved at 20 Hz. Simulator diagnostics render at 128px to match these captures. Encoder inputs are resized to 224px. Exact source episode IDs and hashes live beside the dataset in its manifest.

The policy and simulator use separate Conda environments. The simulator environment uses its own copied LIBERO checkout/assets under the project's ignored third_party directory. A local path file handles LIBERO's implicit namespace package, which its upstream setuptools package discovery omits. The existing source project is preserved. Normal training uses local weight files and a cached instruction; it does not require online model downloads.

The JSON may provide `extra_env` for machine-specific launch settings and `train_script` for an instrumented diagnostic entry point. This machine's local configuration constrains NCCL to single-host loopback/shared-memory transport. Training closes decoder workers, synchronizes ranks after checkpoint/log writes, and explicitly ends the distributed process group.
