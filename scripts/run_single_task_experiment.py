"""Run an explicitly configured single-task diagnostic; always use a fresh output."""
import argparse, json, os, pathlib, subprocess, sys, time, urllib.request

def run(cmd, env, logpath):
    with logpath.open("x") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
            rc = p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
    if rc: raise subprocess.CalledProcessError(rc, cmd)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["smoke", "train", "eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--trials", type=int)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    out = pathlib.Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Use a fresh directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out/"experiment.json").write_text(json.dumps(cfg, indent=2)+"\n")
    env = os.environ.copy()
    cache = pathlib.Path(cfg.get("cache_root", out.parent/"cache"))
    for key, subdir in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HF_DATASETS_CACHE": "huggingface/datasets", "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp", "PIP_CACHE_DIR": "pip",
        "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib",
    }.items():
        directory = cache/subdir
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    env["LIBERO_CONFIG_PATH"] = cfg["libero_config"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.update(cfg.get("extra_env", {}))
    if args.phase != "eval":
        if args.checkpoint or args.trials: ap.error("--checkpoint/--trials require eval")
        steps = cfg["smoke_steps"] if args.phase == "smoke" else cfg["train_steps"]
        cmd = [cfg["policy_python"], "-u", "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", f"--nproc_per_node={len(cfg['train_gpus'])}",
               cfg.get("train_script", "scripts/train_vjepa_policy.py"), *cfg["train_args"], "--output-dir", str(out),
               "--max-steps", str(steps), "--save-every",
               str(steps if args.phase == "smoke" else cfg["save_every"]),
               "--log-every", "1" if args.phase == "smoke" else "10", "--save-initial-checkpoint"]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,cfg["train_gpus"]))
        (out/"commands.json").write_text(json.dumps({"phase":args.phase,"command":cmd,
            "CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"],"revision":
            subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},indent=2)+"\n")
        start = time.monotonic()
        run(cmd,env,out/"train.log")
        result = json.loads((out/"training_result.json").read_text())
        (out/"completion.json").write_text(json.dumps({**result,"phase":args.phase,
            "steps":result["step"],"elapsed_seconds":time.monotonic()-start},indent=2)+"\n")
        return
    if not args.checkpoint: ap.error("eval requires --checkpoint")
    ckpt = pathlib.Path(args.checkpoint).resolve()
    stats = ckpt.parent/"dataset_stats.json"
    if not ckpt.is_file() or not stats.is_file(): raise FileNotFoundError("Keep checkpoint and dataset_stats.json together")
    trials = args.trials if args.trials is not None else len(cfg["evaluation_initial_state_ids"])
    if not 0 < trials <= len(cfg["evaluation_initial_state_ids"]): ap.error("Invalid --trials")
    ids = cfg["evaluation_initial_state_ids"][:trials]
    port = cfg["eval_port"]
    server = [cfg["policy_python"],"-u","examples/libero/serve_policy.py","--ckpt",str(ckpt),
              "--pretrained-encoder",cfg["encoder_checkpoint"],"--dataset-stats",str(stats),
              "--text-cache-dir",cfg["text_cache"],"--host","127.0.0.1","--port",str(port)]
    if cfg.get("gripper_action_convention"):
        server += ["--gripper-action-convention", cfg["gripper_action_convention"]]
    client = [cfg["libero_python"],"-u","examples/libero/run_libero_client.py","--args.host",
              "127.0.0.1","--args.port",str(port),"--args.task-suite-name",cfg["suite"],
              "--args.task-ids",str(cfg["task_id"]),"--args.num-trials-per-task",str(trials),
              "--args.initial-state-ids",*map(str,ids),"--args.render-resolution",str(cfg["native_image_size"]),
              "--args.resize-size",str(cfg["native_image_size"]),"--args.replan-steps","16",
              "--args.frame-stride","4","--args.seed",str(cfg["seed"]),"--args.video-out-path",str(out/"rollouts")]
    env.update(CUDA_VISIBLE_DEVICES=str(cfg["eval_gpu"]), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    (out/"commands.json").write_text(json.dumps({"server":server,"client":client,"initial_state_ids":ids,
        "checkpoint":str(ckpt),"diagnostic_only":True},indent=2)+"\n")
    import socket
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1",port))
    with (out/"server.log").open("x") as log:
        service = subprocess.Popen(server,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+300
            while time.monotonic()<deadline:
                if service.poll() is not None: raise RuntimeError(f"Server exited; inspect {out/'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",timeout=2) as response:
                        if response.status==200: break
                except OSError: time.sleep(.5)
            else: raise TimeoutError("Server readiness timed out")
            run(client,env,out/"client.log")
        finally:
            service.terminate()
            try: service.wait(timeout=20)
            except subprocess.TimeoutExpired: service.kill(); service.wait()
    rows=[json.loads(line) for line in (out/"rollouts/episodes.jsonl").read_text().splitlines()]
    if len(rows)!=trials or [r["initial_state_id"] for r in rows]!=ids or any(
            r["task_id"]!=cfg["task_id"] or r["suite"]!=cfg["suite"] for r in rows):
        raise RuntimeError("Unexpected evaluation coverage")
    (out/"completion.json").write_text(json.dumps({"diagnostic_only":True,"completed":True,
        "episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "success_rate":sum(r["success"] for r in rows)/len(rows)},indent=2)+"\n")

if __name__ == "__main__":
    main()
