# LIBERO evaluation

Run commands from the repository root after [Setup](../../docs/setup.md). For policy training, see the [training tutorial](../../docs/training.md#libero). Use a checkpoint with its matching normalization statistics, frozen encoder weights, and instruction cache. The same procedure applies to the From Scratch and Pretrained Predictor variants.

## Setup and launch

Copy and edit [configs/libero_eval.env](../../configs/libero_eval.env) as `configs/libero_eval.local.env`. Install the LIBERO simulator and assets in the selected client environment; install the serving extra in the policy environment. The shell launcher starts one server/client pair per selected suite, with a GPU and port for each pair. Four parallel suites need four GPU entries; set `EVAL_SERIAL=1 EVAL_GPUS=0` to run all suites sequentially on one GPU. The launcher has no per-suite sharding or `EVAL_SIM_GPUS` option. The launcher also supports `MAX_TASKS=1 ALLOW_INCOMPLETE=1` for a reduced diagnostic and `RECORD_VIDEO=0` to disable videos. `FRAME_STRIDE` defaults to 4. Native [LIBERO-Plus](../libero_plus/README.md) reuses this launcher/client with a separate simulator environment and task manifest.

```bash
cp configs/libero_eval.env configs/libero_eval.local.env
# Edit the artifact and interpreter paths before sourcing.
source configs/libero_eval.local.env
export EVAL_DIR="$RUN_DIR/libero_eval_$(date -u +%Y%m%dT%H%M%SZ)"
bash examples/libero/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
```

## Evaluation protocol

The paper protocol is `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`: 10 tasks per suite × 50 trials = **2,000 episodes**, seed 7, `replan_steps=16`, `frame_stride=4`, 256px simulator images. The reference policy uses two independent 224px inputs and 32-step action chunks. The server resolves model geometry from checkpoint metadata; set overrides only for a documented legacy checkpoint or an intentional experiment.

## Diagnostic runs

For a **10-episode smoke run** (one trial on each task of one suite):

```bash
EVAL_SUITES=libero_spatial EVAL_GPUS=0 NUM_TRIALS_PER_TASK=1 \
  EVAL_DIR="$RUN_DIR/libero_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash examples/libero/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
```

A single-task diagnostic is available through the direct client:

```bash
# Start this in the policy environment, then run the client in a second terminal.
"$POLICY_PYTHON" examples/libero/serve_policy.py \
  --ckpt "$CHECKPOINT" --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
  --dataset-stats "$DATASET_STATS" --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
  --host 127.0.0.1 --port 14001
# Wait for http://127.0.0.1:14001/healthz before the client.
MUJOCO_GL=egl "$LIBERO_PYTHON" examples/libero/run_libero_client.py \
  --args.host 127.0.0.1 --args.port 14001 \
  --args.task-suite-name libero_spatial --args.max-tasks 1 \
  --args.num-trials-per-task 1 --args.replan-steps 16 --args.frame-stride 4 \
  --args.seed 7 --args.video-out-path "$RUN_DIR/inference_smoke"
```

The client flags require `--args.*`; the server flags do not. The client rotates both images by 180 degrees and sends the past frame explicitly. Preserve this preprocessing when writing another client.

## Results

Results include `summary.txt`, server/client/runner logs, status files, per-task counts, and rollout videos. Accept a full result only when all four suite workers exit successfully, each covers IDs 0–9 with 50 episodes per task, and totals agree with the per-task counts. The launcher validates these counts and refuses to reuse a nonempty output directory. It has no automatic resume mode.

Record the checkpoint and encoder identities, cache provenance, repository revision, interpreter versions, and selected protocol alongside the results.

## Legacy checkpoints

For older checkpoints without complete metadata, recover the original run configuration before launching. `POLICY_MAX_STATE_DIM=0` explicitly selects variable-width state. Set `POLICY_T5_LEN`, encoder identity, views, and RoPE settings to that run's actual values; “scratch” does not imply a particular text length or state contract. Keep training-template variables out of an unrelated evaluation environment.

## Fixed task and state diagnostics

The direct client supports `--args.task-ids 7` to select an actual suite ID rather than the first task, and `--args.initial-state-ids 24 3` with `--args.num-trials-per-task 2` to use fixed saved initial states. Both lists must be nonempty, unique, and in range; `task-ids` cannot be combined with `max-tasks`. Every episode records its `initial_state_id`. Reuse exactly the same selected states when comparing untrained and trained checkpoints. This direct selection is a diagnostic, not a complete suite score; the standard shell validator still expects full task IDs.

`--args.render-resolution 128 --args.resize-size 128` matches a policy trained on original native 128px HDF5 images; the server then applies the checkpoint's 224px transform. The default reference render resolution remains 256. Official legacy LIBERO `.pruned_init` files contain NumPy pickles; with PyTorch 2.6+ a dedicated simulator launcher can set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` when loading these trusted local benchmark assets.

The serving boundary accepts canonical LeRobot keys and OpenPI wire aliases `observation/<camera>`, `observation/<camera>_past`, `observation/state`, and `prompt`. Wire images are uint8 HWC; canonical normalized CHW tensors also work. Both past images remain required, and image rotation stays in the simulator client.

Gripper action conventions must match the training data. The reference LeRobot encoding is 0=closed and 1=open; original LIBERO HDF5 actions already use robosuite -1=open and +1=closed. Serving reads `gripper_action_convention` from checkpoint/statistics metadata. For older raw-HDF5 checkpoints, pass `--gripper-action-convention robosuite`; the diagnostic JSON launcher accepts the same `gripper_action_convention` field. Do not apply the LeRobot inversion to raw robosuite targets. Legacy checkpoints without this metadata retain the LeRobot default.
