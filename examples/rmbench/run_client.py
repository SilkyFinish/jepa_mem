"""Serial RMBench rollouts using short past/current observations and native qpos execution."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from examples.rmbench.environment import environment_config, make_environment, policy_observation, dense_joint_step

def evaluate(args):
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError("Use a fresh rollout output directory")
    output.mkdir(parents=True)
    client=WebsocketClientPolicy(host=args.host,port=args.port)
    root=Path(args.rmbench_root).resolve()
    episodes=[]
    candidate=args.seed_start
    attempts=0
    env=None
    for trial in range(args.trials):
        # Match upstream: select the next expert-solvable test seed.
        while True:
            if attempts>=args.max_seed_attempts:raise RuntimeError("Expert seed search exhausted")
            attempts+=1;seed=candidate;candidate+=1
            env,config,instruction=make_environment(root,seed,existing=env)
            try:
                info=env.play_once()
                expert_ok=bool(env.plan_success and env.check_success())
            finally:env.close_env()
            if expert_ok:break
            with (output/"rejected_seeds.jsonl").open("a") as f:f.write(json.dumps({"seed":seed,"reason":"expert_unsolved"})+"\n")
        env,config,instruction=make_environment(root,seed,existing=env)
        history=deque(maxlen=args.frame_stride+1)
        history.append(env.get_obs())
        start=time.time();queries=0
        try:
            while env.take_action_cnt < min(env.step_lim,args.max_steps) and not env.eval_success:
                native=history[-1]
                past=history[0]
                obs=policy_observation(native,past,instruction)
                actions=np.asarray(client.infer(obs)["actions"])
                if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError(f"Invalid policy output {actions.shape}")
                queries+=1
                for action in actions[:args.replan_steps]:
                    if args.controller=="native_qpos":env.take_action(action,action_type="qpos")
                    else:dense_joint_step(env,action,substeps=15)
                    history.append(env.get_obs())
                    if env.eval_success or env.take_action_cnt>=min(env.step_lim,args.max_steps):break
            row={"task":"put_back_block","trial":trial,"seed":seed,"success":bool(env.eval_success),
                 "steps":int(env.take_action_cnt),"queries":queries,"seconds":time.time()-start,"controller":args.controller}
            episodes.append(row)
            with (output/"episodes.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        finally:env.close_env()
    result={"complete":True,"episodes":len(episodes),"successes":sum(x["success"] for x in episodes),
            "success_rate":sum(x["success"] for x in episodes)/len(episodes),"arguments":vars(args)}
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rmbench-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=10010)
    p.add_argument("--trials",type=int,default=100);p.add_argument("--seed-start",type=int,default=100000)
    p.add_argument("--max-seed-attempts",type=int,default=1000);p.add_argument("--max-steps",type=int,default=1000)
    p.add_argument("--replan-steps",type=int,default=16);p.add_argument("--frame-stride",type=int,default=4)
    p.add_argument("--controller",choices=("native_qpos","dense15"),default="native_qpos")
    a=p.parse_args()
    if not 0<a.replan_steps<=32 or a.trials<=0 or a.frame_stride<=0:p.error("Invalid rollout budget")
    evaluate(a)
