"""Serial full-state training stages and RMBench evaluations on shared GPUs.

The JSON config supplies machine paths and normal training arguments. Training
errors stop the job. Evaluation errors are recorded and the next stage resumes
the same optimizer, RNG, data cursor and full-run learning-rate schedule.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Job:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output_root"])
        self.source = Path(config["source_root"])
        self.train = self.root / "train"
        self.stages = config["stage_steps"]
        args = config["train_args"]
        if (not self.stages or any(type(s) is not int or s <= 0 for s in self.stages)
                or self.stages != sorted(set(self.stages))
                or self.stages[-1] != int(args[args.index("--max-steps")+1])):
            raise ValueError("Stage steps must increase to the full-run max_steps")
        if "--save-training-state" not in args:
            raise ValueError("Stage training requires complete optimizer state")
        if any(flag in args for flag in ("--stop-at", "--stop-after-step", "--stop-file", "--resume", "--output-dir")):
            raise ValueError("The scheduler owns stage stops, resume and output paths; no time deadline is supported")
        if not config["gpu_ids"] or config["eval_gpu"] not in config["gpu_ids"]:
            raise ValueError("Evaluation GPU must be one of the allocated training GPUs")
        self.children, self.stopping = [], False
        self.state = dict(phase="starting", pid=os.getpid(), started_at=time.time(), stages={},
                          config=config, no_time_deadline=True, evaluation_failure_policy="record_and_continue")
        if (self.root / "status.json").exists():
            previous = json.loads((self.root / "status.json").read_text())
            if previous["config"] != config:
                raise ValueError("Existing run config differs; use a fresh output directory")
            self.state["stages"] = previous["stages"]
            self.state["original_started_at"] = previous.get("original_started_at", previous["started_at"])

    def status(self, phase, **extra):
        self.state.update(phase=phase, updated_at=time.time(), **extra)
        write_json(self.root / "status.json", self.state)

    def stop(self, signum, frame):
        self.stopping = True
        (self.root / "STOP").touch()
        print("Stop requested: save training at the next optimizer boundary.", flush=True)
        if self.state["phase"] != "training":
            for child in self.children:
                self.terminate(child)

    @staticmethod
    def terminate(child):
        # A finished PID/group may be reused during a multi-day job.
        if child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=20)

    def environment(self, evaluation=False, simulator=False):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config["eval_gpu"]) if evaluation else ",".join(map(str, self.config["gpu_ids"]))
        # Keep the recorded source first even if the parent inherited another checkout.
        source_paths = [str(self.source / "src"), str(self.source)]
        if simulator and self.config.get("simulator_overlay"):
            source_paths.insert(0, self.config["simulator_overlay"])
        env["PYTHONPATH"] = os.pathsep.join(source_paths + [env.get("PYTHONPATH", "")])
        return env

    def command(self, command, directory, name, *, evaluation=False, simulator=False, progress=None):
        if self.stopping or (self.root / "STOP").exists():
            raise InterruptedError("Manual job stop requested")
        write_json(directory / f"{name}.command.json", command)
        log = directory / f"{name}.log"
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=self.source,
                                     env=self.environment(evaluation, simulator), stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(child)
            self.status(self.state["phase"], child_pid=child.pid, child_log=str(log))
            last_progress = time.time()
            try:
                while child.poll() is None:
                    time.sleep(5)
                    if evaluation:
                        if self.stopping or (self.root / "STOP").exists():
                            raise InterruptedError("Manual stop during evaluation")
                        for path in (log, progress):
                            if path is not None and path.exists():
                                last_progress = max(last_progress, path.stat().st_mtime)
                        if time.time() - last_progress > self.config.get("eval_idle_seconds", 1200):
                            raise TimeoutError("Evaluation produced no progress within the idle watchdog interval")
                    elif (self.train / "progress.json").exists():
                        try:
                            self.state["training_progress"] = json.loads((self.train / "progress.json").read_text())
                        except json.JSONDecodeError:
                            pass
                    self.status(self.state["phase"])
                if child.returncode:
                    raise RuntimeError(f"{name} exited {child.returncode}; see {log}")
            finally:
                self.terminate(child)

    @staticmethod
    def resume_step(path):
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not state.get("training_state", {}).get("optimizer"):
            raise ValueError("Refusing a weight-only restart between training stages")
        return int(state["step"])

    def train_to(self, target, directory):
        resume = self.train / "resume_state.pt"
        current = self.resume_step(resume) if resume.exists() else 0
        checkpoint = self.train / f"checkpoint_step{target:06d}.pt"
        if current >= target:
            if not checkpoint.exists():
                raise ValueError("Resume state passed a milestone without its protected checkpoint")
            return checkpoint
        if current == 0 and self.train.exists() and any(self.train.glob("checkpoint_step*.pt")):
            raise ValueError("Existing weights without full resume state; inspect the previous attempt")
        args = self.config["train_args"] + ["--output-dir", str(self.train), "--stop-after-step", str(target),
                                            "--stop-file", str(self.root / "STOP")]
        if resume.exists():
            args += ["--resume", str(resume)]
        command = [self.config["policy_python"], "-u", "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={len(self.config['gpu_ids'])}", "--module", "scripts.train_vjepa_policy", *args]
        self.status("training", target_step=target, resumed_from_step=current)
        self.command(command, directory, "training")
        if self.stopping or (self.root / "STOP").exists():
            raise InterruptedError("Training saved after a manual stop")
        result = json.loads((self.train / "training_result.json").read_text())
        if (result["step"] != target or not result["optimizer_state_saved"] or not checkpoint.exists()
                or self.resume_step(resume) != target):
            raise RuntimeError("Training did not save the requested stage with complete resume state")
        write_json(directory / "training_result.json", result)
        return checkpoint

    def evaluate(self, checkpoint, directory, trials=None, max_steps=None):
        manifest = json.loads(Path(self.config["manifest"]).read_text())
        instructions = {row["instruction"] for row in manifest["episodes"]}
        if len(instructions) != 1:
            raise ValueError("Expected one fixed whole-task training instruction")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        trials = self.config["rollouts"] if trials is None else trials
        max_steps = self.config.get("eval_max_steps", 0) if max_steps is None else max_steps
        rollout = directory / "rollout"
        command = [self.config["policy_python"], "-u", "examples/rmbench/serve_policy.py", "--ckpt", str(checkpoint),
                   "--pretrained-encoder", self.config["encoder"], "--text-cache-dir", self.config["text_cache"],
                   "--dataset-stats", str(Path(self.config["manifest"]).parent / "dataset_stats.json"),
                   "--host", "127.0.0.1", "--port", str(port), "--memory-precision", "bf16",
                   "--max-history-blocks", str(self.config["eval_history_blocks"]),
                   "--max-cache-bytes", str(self.config["eval_cache_bytes"]), "--seed", str(self.config.get("policy_seed", 7))]
        write_json(directory / "server.command.json", command)
        self.status("evaluating", checkpoint=str(checkpoint), eval_output=str(rollout))
        with (directory / "server.log").open("w") as stream:
            server = subprocess.Popen(command, cwd=self.source, env=self.environment(True),
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(server)
            try:
                started = time.monotonic()
                while True:
                    if self.stopping or (self.root / "STOP").exists():
                        raise InterruptedError("Manual stop during server startup")
                    if server.poll() is not None:
                        raise RuntimeError("Policy server exited before readiness")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                            if response.read() == b"ready\n":
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic()-started > 600:
                        raise TimeoutError("Policy server did not become ready")
                    time.sleep(1)
                self.command([self.config["simulator_python"], "-u", "-m", "examples.rmbench.run_memory_client",
                    "--rmbench-root", self.config["simulator_root"], "--host", "127.0.0.1", "--port", str(port),
                    "--trials", str(trials), "--seed-start", str(self.config.get("seed_start", 100000)),
                    "--controller", "native_qpos", "--instruction", next(iter(instructions)),
                    "--max-steps", str(max_steps), "--output", str(rollout)], directory, "rollout",
                    evaluation=True, simulator=True, progress=rollout / "progress.json")
                result = json.loads((rollout / "results.json").read_text())
                if (not result["complete"] or result["episodes"] != trials or result.get("error")):
                    raise RuntimeError("Rollout result is incomplete")
                episodes = [json.loads(line) for line in (rollout / "episodes.jsonl").read_text().splitlines() if line]
                if len(episodes) != trials or any(row.get("error") for row in episodes):
                    raise RuntimeError("Rollout episode records are incomplete or contain errors")
                result["accepted_seeds"] = [row["seed"] for row in episodes]
                return result
            finally:
                self.terminate(server)

    def run_stages(self):
        for target in self.stages:
            previous = self.state["stages"].get(str(target), {})
            if previous.get("evaluation_status") in ("complete", "failed"):
                continue
            if self.stopping or (self.root / "STOP").exists():
                raise InterruptedError("Manual stop requested")
            directory = self.root / f"step{target:06d}-attempt-{time.time_ns()}"
            directory.mkdir()
            checkpoint = self.train_to(target, directory)
            stage = dict(step=target, checkpoint=str(checkpoint), directory=str(directory), training_status="complete")
            self.state["stages"][str(target)] = stage
            try:
                result = self.evaluate(checkpoint, directory)
                stage.update(evaluation_status="complete", result=result)
            except InterruptedError:
                raise
            except Exception as error:
                # A low success rate is a complete evaluation, not an exception.
                # Simulator/server failures are isolated from the next training stage.
                stage.update(evaluation_status="failed", error=str(error))
                (directory / "evaluation_error.txt").write_text(traceback.format_exc())
                print(f"Evaluation at step {target} failed; recording and continuing: {error}", flush=True)
            self.status("stage_complete", last_completed_stage=target)
        failures = sum(s.get("evaluation_status") == "failed" for s in self.state["stages"].values())
        self.status("complete", evaluation_failures=failures, finished_at=time.time(), child_pid=None)

    def execute(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".pipeline.lock").open("w") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, self.stop)
            write_json(self.root / "plan.json", self.config)
            try:
                self.status("starting")
                self.run_stages()
            except InterruptedError as error:
                self.status("stopped", error=str(error))
            except Exception as error:
                self.status("failed", error=str(error))
                raise
            finally:
                for child in self.children:
                    self.terminate(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-checkpoint", type=Path, help="Standalone simulator preflight; does not start training")
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=0)
    args = parser.parse_args()
    job = Job(json.loads(args.config.read_text()))
    if args.eval_checkpoint:
        if args.eval_output is None:
            parser.error("--eval-checkpoint requires a fresh --eval-output")
        args.eval_output.mkdir(parents=True, exist_ok=False)
        # Keep preflight status separate from the formal job's status.
        job.root = args.eval_output
        result = job.evaluate(args.eval_checkpoint, args.eval_output, args.trials, args.eval_max_steps)
        write_json(args.eval_output / "preflight_result.json", result)
    else:
        job.execute()


if __name__ == "__main__":
    main()
