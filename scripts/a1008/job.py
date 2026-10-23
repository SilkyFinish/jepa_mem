"""Foreground job for a single ephemeral node with persistent shared storage."""
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
import urllib.request

from scripts.a1008.common import (
    REPO, asset_sources, base_args, paths, per_device_batch, schedule, serving_capacity, sha256,
    text_encoder, train_only, write_json,
)


def one_gpu_env():
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0] if visible else "0"
    return env


class Job:
    def __init__(self, args):
        self.args, self.p = args, paths()
        name = os.environ.get("JEPA_RUN_NAME", "put_back_block_full50_ep100")
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("JEPA_RUN_NAME must be one directory name")
        self.root = self.p["root"] / "runs" / name
        self.world = int(os.environ.get("JEPA_GPUS", "8"))
        self.plan = schedule(1013, self.world, int(os.environ.get("JEPA_EPOCHS", "100")), per_device_batch())
        self.children = []
        self.stopping = False
        self.phase = "starting"

    def status(self, phase, **extra):
        self.phase = phase
        write_json(self.root / "status.json", dict(phase=phase, updated_at=time.time(),
                   hostname=socket.gethostname(), pid=os.getpid(), plan=self.plan,
                   train_only=self.args.train_only, **extra))
        print(f"[stage] {phase}", flush=True)

    def stop(self, signum, frame):
        self.stopping = True
        (self.root / "STOP").touch()
        print("[stop] requested; training will save at the next optimizer boundary", flush=True)
        if self.phase != "training":
            for child in self.children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)

    def command(self, name, command, env=None, timeout=None):
        if self.stopping or (self.root / "STOP").exists():
            self.stopping = True
            raise InterruptedError("Job stop requested")
        log = self.attempt / f"{name}.log"
        write_json(self.attempt / f"{name}.command.json", list(map(str, command)))
        print(f"[run] {name}; log={log}", flush=True)
        with log.open("w") as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(child)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.terminate(child)
                raise
            if code:
                raise RuntimeError(f"{name} exited {code}; see {log}")

    @staticmethod
    def terminate(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def distributed(self):
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                f"--nproc-per-node={self.world}"]

    def gpu_check(self):
        import torch
        if torch.cuda.device_count() != self.world:
            raise RuntimeError(f"Expected {self.world} visible GPUs, found {torch.cuda.device_count()}; check allocation")
        hardware = []
        for i in range(self.world):
            prop = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            if not self.args.allow_other_gpus and ("A100" not in prop.name or total < 39 * 2**30):
                raise RuntimeError(f"Expected A100 40/80 GB, got {prop.name}, {total / 2**30:.1f} GiB")
            if free < 28 * 2**30:
                raise RuntimeError(f"GPU {i} has only {free / 2**30:.1f} GiB free")
            hardware.append(dict(index=i, name=prop.name, total_gib=total / 2**30, free_gib=free / 2**30))
        report = dict(gpus=hardware, torch=torch.__version__, cuda=torch.version.cuda,
                      environment={k: v for k, v in os.environ.items() if k.startswith("NCCL_")
                                   or k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")})
        write_json(self.attempt / "hardware.json", report)
        self.command("topology", ["nvidia-smi", "topo", "-m"])
        self.command("nccl", [*self.distributed(), "-m", "scripts.a1008.probe", "--mode", "collective",
                              "--output", self.attempt / "nccl.json"], timeout=300)

    def caches(self):
        from scripts.a1008.prepare_assets import verify_prepared
        from scripts.train_vjepa_policy import parse_args
        from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
        lock = asset_sources()
        ready = json.loads(self.p["ready"].read_text())
        if ready["sources"] != lock:
            raise ValueError("Run host_setup.sh with this branch's pinned assets first")
        verify_prepared(self.p, lock)
        # One writer across jobs sharing frozen assets. Readers validate committed manifests.
        with (self.p["root"] / ".cache_build.lock").open("w") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            text_marker = self.p["text"] / "provenance.json"
            if not text_marker.exists():
                self.command("text_cache", [sys.executable, "-m", "vjepa_policy.text_embeddings",
                    "--dataset-dirs", self.p["prepared"], "--cache-dir", self.p["text"],
                    "--model-name", self.p["text_model"], "--text-encoder", text_encoder(),
                    "--context-length", "128"], env=one_gpu_env())
                write_json(text_marker, dict(source=lock["text"], context_length=128,
                    files={f.name: sha256(f) for f in sorted(self.p["text"].glob("*.pt"))}))
            text = json.loads(text_marker.read_text())
            if text["source"] != lock["text"] or text["context_length"] != 128 or not text["files"]:
                raise ValueError("Text cache provenance mismatch")
            for name, digest in text["files"].items():
                if sha256(self.p["text"] / name) != digest:
                    raise ValueError("Text cache hash mismatch")
            if not self.p["features"].exists():
                if self.p["features"].parent.exists():
                    self.p["features"].parent.rename(self.p["root"] / f"history_features.incomplete.{time.time_ns()}")
                self.command("visual_cache", [sys.executable, "-m", "scripts.cache_rmbench_history_features",
                    "--cache-output", self.p["features"].parent, *base_args(self.p)], env=one_gpu_env())
        args = parse_args([*base_args(self.p), "--history-feature-cache", str(self.p["features"]),
                           "--output-dir", str(self.attempt / "data_check")])
        data = RMBenchHistoryDataset(self.p["prepared"] / "manifest.json", args)
        try:
            if len(data) != 1013 or max(data.history_lengths) != 81 or data.selection["diagnostic_subset"]:
                raise ValueError("Full50 history contract differs")
            self.plan = schedule(len(data), self.world, self.plan["epochs"], self.plan["per_device_batch"])
            write_json(self.attempt / "dataset.json", dict(selection=data.selection, provenance=data.provenance))
        finally:
            data.close()

    def sim_env(self):
        env = one_gpu_env()
        overlay = self.p["root"] / "envs/sim_client"
        env["PYTHONPATH"] = str(overlay) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def evaluate(self, checkpoint, directory, trials, diagnostic=False):
        import yaml
        manifest = json.loads((self.p["prepared"] / "manifest.json").read_text())
        instructions = {row["instruction"] for row in manifest["episodes"]}
        if len(instructions) != 1:
            raise ValueError("Expected one fixed whole-task instruction")
        instruction = next(iter(instructions))
        limits = yaml.safe_load((Path(os.environ["RMBENCH_ROOT"]) /
                                 "env_cfg/task_config/_eval_step_limit.yml").read_text())
        limit = int(limits.get("put_back_block", 1000))
        blocks, cache_budget = serving_capacity(limit)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server_command = [sys.executable, "examples/rmbench/serve_policy.py", "--ckpt", str(checkpoint),
            "--pretrained-encoder", str(self.p["encoder"]), "--text-cache-dir", str(self.p["text"]),
            "--dataset-stats", str(self.p["prepared"] / "dataset_stats.json"), "--port", str(port),
            "--memory-precision", "bf16", "--max-history-blocks", str(blocks),
            "--max-cache-bytes", str(cache_budget)]
        write_json(self.attempt / f"{directory.name}.server.command.json", server_command)
        with (self.attempt / f"{directory.name}.server.log").open("w") as stream:
            server = subprocess.Popen(server_command, cwd=REPO, env=one_gpu_env(),
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(server)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if self.stopping or server.poll() is not None:
                        raise RuntimeError("Policy server stopped before readiness")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server did not become ready in 600 seconds")
                    time.sleep(1)
                self.command(directory.name, [os.environ["RMBENCH_PYTHON"], "examples/rmbench/run_memory_client.py",
                    "--rmbench-root", os.environ["RMBENCH_ROOT"], "--output", directory, "--port", str(port),
                    "--trials", str(trials), "--seed-start", "100000", "--controller", "native_qpos",
                    "--instruction", instruction,
                    "--max-steps", "32" if diagnostic else "0"], env=self.sim_env())
                result = json.loads((directory / "results.json").read_text())
                if not result["complete"] or result["episodes"] != trials or result["error"]:
                    raise ValueError("Rollout job incomplete")
            finally:
                self.terminate(server)

    def execute(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".job.lock").open("w") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.root / "STOP").exists():
                raise RuntimeError(f"Remove {self.root / 'STOP'} when you intend to resume")
            self.attempt = self.root / f"attempt-{time.time_ns()}"
            self.attempt.mkdir()
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, self.stop)
            try:
                self.status("preflight")
                self.command("revision", ["git", "rev-parse", "HEAD"])
                self.command("working_tree", ["git", "diff", "--stat"])
                self.command("versions", [sys.executable, "-m", "pip", "freeze"])
                self.gpu_check()
                self.status("preparing_caches")
                self.caches()
                plan_file = self.root / "plan.json"
                if plan_file.exists() and json.loads(plan_file.read_text()) != self.plan:
                    raise ValueError("Existing run uses a different plan; choose a new JEPA_RUN_NAME")
                write_json(plan_file, self.plan)
                train = self.root / "train"
                final = train / "training_result.json"
                already_complete = final.exists() and json.loads(final.read_text())["completed"]
                if not already_complete and not self.args.eval_only:
                    self.status("training_preflight")
                    stress = self.attempt / "longest_prefix"
                    self.command("longest_prefix", [*self.distributed(), "-m", "scripts.a1008.probe", "--mode", "stress",
                        *base_args(self.p), "--history-feature-cache", self.p["features"], "--output-dir", stress])
                    if self.args.train_only:
                        self.status("simulator_deferred", reason="Train-only: simulation checks and rollouts run on the host")
                    else:
                        self.status("simulator_preflight")
                        self.command("simulator", [os.environ["RMBENCH_PYTHON"], "examples/rmbench/check_simulator.py",
                            "--rmbench-root", os.environ["RMBENCH_ROOT"], "--dataset-root", self.p["data"],
                            "--output", self.attempt / "simulator", "--seeds", "0", "100000",
                            "--replay-episodes", "0", "--controllers", "dense15", "native_qpos"], env=self.sim_env())
                        sim = json.loads((self.attempt / "simulator/results.json").read_text())
                        if not sim["native_expert_passed"] or not any(r["mode"] == "recorded_action_replay"
                                and r["controller"] == "dense15" and r["success"] for r in sim["records"]):
                            raise ValueError("Native expert or recorded-action replay failed")
                        self.evaluate(stress / "checkpoint_step000002.pt", self.attempt / "smoke_rollout", 1, True)
                    if self.args.preflight_only:
                        self.status("preflight_passed", formal_training_started=False,
                                    simulator_preflight="skipped" if self.args.train_only else "passed")
                        return
                    self.status("training")
                    command = [*self.distributed(), "scripts/train_vjepa_policy.py", *base_args(self.p),
                        "--history-feature-cache", self.p["features"], "--output-dir", train,
                        "--num-epochs", str(self.plan["epochs"]), "--max-steps", str(self.plan["updates"]),
                        "--gradient-accumulation-steps", str(self.plan["accumulation"]),
                        "--save-every", str(5 * self.plan["updates_per_epoch"]), "--save-training-state",
                        "--keep-last-checkpoints", "2", "--stop-file", self.root / "STOP", "--checkpoint-milestones",
                        *[str(e * self.plan["updates_per_epoch"]) for e in (20, 50, self.plan["epochs"])]]
                    resume = train / "resume_state.pt"
                    if resume.exists():
                        command += ["--resume", resume]
                    elif train.exists() and any(train.glob("checkpoint_step*.pt")):
                        raise RuntimeError("Run has weights but no full resume state; choose a fresh run or inspect the interruption")
                    self.command("training", command)
                if self.args.preflight_only:
                    self.status("already_trained")
                    return
                if not final.exists():
                    raise RuntimeError("No completed training_result.json to evaluate")
                result = json.loads(final.read_text())
                if not result["completed"] or self.stopping or (self.root / "STOP").exists():
                    self.status("stopped", training=result)
                    return
                if result["step"] != self.plan["updates"]:
                    raise ValueError("Training ended at an unexpected update count")
                if self.args.train_only:
                    self.status("complete", checkpoint=str(train / result["checkpoint"]),
                                evaluation="deferred_to_host", simulator_preflight="skipped")
                    return
                self.status("evaluating")
                # Each attempt is fresh: interrupted/failed trials are never silently merged into a score.
                evaluation = self.attempt / "rollouts"
                self.evaluate(train / result["checkpoint"], evaluation, int(os.environ.get("JEPA_ROLLOUTS", "20")))
                self.status("complete", checkpoint=str(train / result["checkpoint"]),
                            evaluation=str(evaluation / "results.json"))
            except Exception as error:
                self.status("stopped" if self.stopping else "failed", error=str(error), attempt=str(self.attempt))
                raise
            finally:
                for child in self.children:
                    self.terminate(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Print resolved plan without running GPUs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", default=train_only(),
                        help="Keep GPU diagnostics; defer all simulation checks and rollouts to the host")
    parser.add_argument("--allow-other-gpus", action="store_true", help="Explicit local diagnostic; record hardware")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        parser.error("--eval-only is incompatible with train-only mode; unset JEPA_TRAIN_ONLY for host evaluation")
    job = Job(args)
    if args.plan:
        print(json.dumps(dict(plan=job.plan, output=str(job.root), train_only=args.train_only,
            evaluation="deferred_to_host" if args.train_only else "after_training",
            language=dict(encoder=text_encoder(), context_length=128,
                          hidden_size=768 if text_encoder() == "bert" else 4096,
                          model_version="v0_bert" if text_encoder() == "bert" else "v0"),
            paths={k: str(v) for k, v in job.p.items()}), indent=2))
    else:
        job.execute()


if __name__ == "__main__":
    main()
