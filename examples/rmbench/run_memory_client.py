"""Full-history RMBench client; use --max-steps 0 for native episode limits."""
import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from examples.rmbench.environment import CAMERA_MAP, make_environment, dense_joint_step
from vjepa_policy.policy_serving.memory_client import HistoryCollector, MemorySessionClient


def collect(collector, env):
    native = env.get_obs()
    collector.add(int(env.take_action_cnt),
                  {key: native["observation"][source]["rgb"] for key, source in CAMERA_MAP.items()},
                  np.asarray(native["joint_action"]["vector"], dtype=np.float32))


def evaluate(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(args.rmbench_root).resolve()
    episodes, attempts, candidate = [], 0, args.seed_start
    env, transport, session = None, None, None
    complete, error = False, None
    diagnostic = args.max_steps != 0 or args.controller != "native_qpos"
    def progress(phase, **values):
        path = output / "progress.json"
        path.with_suffix(".tmp").write_text(json.dumps(dict(phase=phase, updated_at=time.time(), **values)) + "\n")
        path.with_suffix(".tmp").replace(path)
    try:
        transport = WebsocketClientPolicy(host=args.host, port=args.port)
        if transport.get_server_metadata().get("protocol") != "rmbench_memory_v0":
            raise ValueError("server is not a full-history v0 endpoint")
        for trial in range(args.trials):
            while True:
                if attempts >= args.max_seed_attempts:
                    raise RuntimeError("Expert seed search exhausted")
                attempts += 1
                seed, candidate = candidate, candidate + 1
                progress("expert_filter", trial=trial, seed=seed)
                env, _, instruction = make_environment(root, seed, existing=env)
                try:
                    env.play_once()
                    expert_ok = bool(env.plan_success and env.check_success())
                finally:
                    env.close_env()
                if expert_ok:
                    break
                with (output / "rejected_seeds.jsonl").open("a") as stream:
                    stream.write(json.dumps({"seed": seed, "reason": "expert_unsolved"}) + "\n")
            env, _, instruction = make_environment(root, seed, existing=env)
            if args.instruction is not None:
                instruction = args.instruction
                env.set_instruction(instruction)
            # Serving noise includes session ID; keep it stable across checkpoints.
            session = MemorySessionClient(transport, session_id=f"put_back_block-{seed}-{trial}")
            collector = HistoryCollector(config=session.config)
            start, queries, episode_error = time.time(), 0, None
            step_limit = min(env.step_lim, args.max_steps) if args.max_steps else env.step_lim
            progress("policy_rollout", trial=trial, seed=seed, steps=0, step_limit=step_limit)
            try:
                session.reset(f"put_back_block:{seed}:{trial}", instruction)
                collect(collector, env)
                while env.take_action_cnt < step_limit and not env.eval_success:
                    session.append(collector)
                    actions = session.infer(int(env.take_action_cnt))
                    queries += 1
                    progress("policy_rollout", trial=trial, seed=seed, steps=int(env.take_action_cnt), queries=queries)
                    for action in actions[:16]:
                        if args.controller == "native_qpos":
                            env.take_action(action, action_type="qpos")
                        else:
                            dense_joint_step(env, action, substeps=15)
                        collect(collector, env)
                        if env.eval_success or env.take_action_cnt >= step_limit:
                            break
            except Exception as exc:
                episode_error = str(exc)
                raise
            finally:
                cleanup_error = None
                final_revision = session.revision
                try:
                    session.close()
                except Exception as exc:
                    cleanup_error = exc
                    logging.exception("Failed to close memory session")
                finally:
                    session = None
                    try:
                        env.close_env()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
                        logging.exception("Failed to close simulator episode")
                row = {"task": "put_back_block", "trial": trial, "seed": seed,
                       "success": bool(env.eval_success), "steps": int(env.take_action_cnt),
                       "queries": queries, "seconds": time.time() - start,
                       "controller": args.controller, "error": episode_error or (str(cleanup_error) if cleanup_error else None),
                       "diagnostic": diagnostic, "step_limit": step_limit, "cache_revision": final_revision}
                episodes.append(row)
                with (output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                progress("episode_complete", trial=trial, seed=seed, episodes=len(episodes))
                if cleanup_error is not None and episode_error is None:
                    raise cleanup_error
        complete = True
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logging.exception("Failed to close memory session")
        if transport is not None:
            # Pinned openpi-client exposes no public close; disconnect also frees sessions.
            try:
                transport._ws.close()
            except Exception as exc:
                complete = False
                error = error or str(exc)
                logging.exception("Failed to close policy transport")
        result = {"complete": complete, "diagnostic": diagnostic, "error": error,
                  "episodes": len(episodes), "successes": sum(x["success"] for x in episodes),
                  "success_rate": sum(x["success"] for x in episodes)/len(episodes) if episodes else None,
                  "arguments": vars(args)}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10010)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--max-seed-attempts", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=32,
                        help="0 uses the simulator's native episode limit; default is a 32-step diagnostic.")
    parser.add_argument("--controller", choices=("native_qpos", "dense15"), default="native_qpos")
    parser.add_argument("--instruction", help="Fixed whole-task instruction from the training release (never stage/target labels)")
    args = parser.parse_args(argv)
    if min(args.trials, args.max_seed_attempts) <= 0 or args.max_steps < 0:
        parser.error("trials/attempts must be positive and max-steps must be nonnegative")
    evaluate(args)


if __name__ == "__main__":
    main()
