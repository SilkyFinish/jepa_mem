"""Run training and selected task evaluations serially, with a hard time budget."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def kill_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def run_stage(command, logpath, *, deadline, stop_file=None, on_start=None):
    """Each stage owns a process group, including its GPU workers and servers."""
    with logpath.open("x") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if on_start:
                on_start(process.pid)
            while process.poll() is None:
                overdue = time.time() - deadline
                if overdue >= 0:
                    if stop_file is not None:
                        stop_file.touch(exist_ok=True)
                    if stop_file is None or overdue >= 90:
                        kill_group(process.pid, signal.SIGTERM)
                    if overdue >= 120:
                        kill_group(process.pid, signal.SIGKILL)
                time.sleep(0.5)
            return process.returncode
        finally:
            # A failed launcher can leave a simulator or distributed worker alive.
            kill_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_group(process.pid, signal.SIGKILL)
                process.wait()
            kill_group(process.pid, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh pipeline output: {out}")
    deadline = datetime.fromisoformat(cfg["pipeline_deadline"])
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        raise ValueError("Pipeline deadline must be in the future, with timezone")
    deadline = deadline.timestamp()
    write_json(out / "config.json", cfg)
    status = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
              "deadline": cfg["pipeline_deadline"], "completed": False,
              "evaluations": []}
    stop_file = out / "STOP_TRAINING"
    status_path = out / "status.json"

    def started(pid):
        status["child_pid"] = pid
        write_json(status_path, status)

    try:
        train_cfg = dict(cfg)
        train_cfg["train_args"] = [*cfg["train_args"], "--stop-at", cfg["pipeline_deadline"],
                                   "--stop-file", str(stop_file)]
        train_config = out / "train_config.json"
        write_json(train_config, train_cfg)
        command_base = [cfg["policy_python"], "-u", "scripts/run_single_task_experiment.py"]
        status["phase"] = "training"
        write_json(status_path, status)
        rc = run_stage([*command_base, "train", "--config", str(train_config),
                        "--output-dir", str(out / "train")], out / "training_launcher.log",
                       deadline=deadline, stop_file=stop_file, on_start=started)
        status["training_exit_code"] = rc
        result_path = out / "train/training_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        status["training"] = result
        write_json(status_path, status)
        if rc or not result.get("completed"):
            status["phase"] = "training_failed" if rc else "stopped_at_deadline"
            return
        checkpoint = out / "train" / result["checkpoint"]
        for task in cfg["tasks"]:
            if time.time() >= deadline:
                status["phase"] = "stopped_at_deadline"
                return
            eval_cfg = {**cfg, **task}
            eval_path = out / f"eval_config_{task['key']}.json"
            write_json(eval_path, eval_cfg)
            status["phase"] = f"rollout:{task['key']}"
            write_json(status_path, status)
            destination = out / f"rollout_{task['key']}"
            rc = run_stage([*command_base, "eval", "--config", str(eval_path),
                            "--output-dir", str(destination), "--checkpoint", str(checkpoint),
                            "--trials", str(cfg["rollouts_per_task"])],
                           out / f"rollout_launcher_{task['key']}.log",
                           deadline=deadline, on_start=started)
            completion = destination / "completion.json"
            record = {"task": task["key"], "exit_code": rc,
                      "result": json.loads(completion.read_text()) if completion.exists() else {}}
            status["evaluations"].append(record)
            write_json(status_path, status)
            if time.time() >= deadline:
                status["phase"] = "stopped_at_deadline"
                return
            if rc or not record["result"].get("completed"):
                status["phase"] = f"rollout_failed:{task['key']}"
                return
        status["completed"] = True
        status["phase"] = "complete"
        status["total_episodes"] = sum(e["result"]["episodes"] for e in status["evaluations"])
        status["total_successes"] = sum(e["result"]["successes"] for e in status["evaluations"])
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc))
        raise
    finally:
        status["finished_at"] = datetime.now().astimezone().isoformat()
        status.pop("child_pid", None)
        write_json(status_path, status)


if __name__ == "__main__":
    main()
