"""Check real rendering, native expert planning and recorded joint-action replay."""
import argparse
import json
from pathlib import Path
import time

import h5py
import importlib.util
from PIL import Image
import numpy as np
from examples.rmbench.environment import make_environment, dense_joint_step

def joint_vector(f, group):
    fields=("left_arm_joint_states","left_ee_joint_states","right_arm_joint_states","right_ee_joint_states")
    return np.concatenate([f[group+"/"+key][:] for key in fields],axis=-1).astype(np.float32)

def run(args):
    root=Path(args.rmbench_root).resolve();data=Path(args.dataset_root).resolve();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    records=[];env=None
    for seed in args.seeds:
        start=time.time()
        env,config,instruction=make_environment(root,seed,existing=env)
        observation=env.get_obs()
        shapes={k:list(observation["observation"][k]["rgb"].shape) for k in ("head_camera","left_camera","right_camera")}
        if any(shape!=[240,320,3] for shape in shapes.values()):raise ValueError("Render geometry differs")
        if np.asarray(observation["joint_action"]["vector"]).shape!=(14,):raise ValueError("State shape differs")
        image_stats={}
        for camera in ("head_camera","left_camera","right_camera"):
            image=observation["observation"][camera]["rgb"]
            if image.dtype!=np.uint8 or float(image.std())<5:
                raise ValueError("Blank or invalid rendered RGB: "+camera)
            image_stats[camera]={"mean":float(image.mean()),"std":float(image.std()),"min":int(image.min()),"max":int(image.max())}
            Image.fromarray(image).save(out/f"seed{seed}_{camera}.png")
        if seed==0:
            spec=importlib.util.spec_from_file_location("native_rgb_codec",root/"data/decode_image_bit.py")
            codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
            with h5py.File(data/"data/episode_0000000.hdf5","r") as handle:
                for camera,source_camera in (("head_camera","cam_head"),("left_camera","cam_left_wrist"),("right_camera","cam_right_wrist")):
                    source_rgb=codec.decode_image_bit(handle["vision/"+source_camera+"/colors"][0])
                    Image.fromarray(source_rgb).save(out/f"dataset_seed0_{camera}.png")
                    image_stats[camera]["source_frame0_mae"]=float(np.abs(source_rgb.astype(np.float32)-observation["observation"][camera]["rgb"].astype(np.float32)).mean())
        maps=Path("/proc/self/maps")
        oidn_libraries=sorted({line.split()[-1] for line in maps.read_text().splitlines() if "OpenImageDenoise" in line}) if maps.exists() else []
        info=env.play_once()
        success=bool(env.plan_success and env.check_success())
        row={"mode":"native_expert","seed":seed,"success":success,"rgb_shapes":shapes,"image_statistics":image_stats,"oidn_libraries":oidn_libraries,"seconds":time.time()-start}
        records.append(row);print(json.dumps(row),flush=True)
        env.close_env()
        (out/"progress.json").write_text(json.dumps(records,indent=2)+"\n")
        if not success:raise RuntimeError("Native expert failed; investigate before training")
    seed_values=[int(x) for x in (data/"seed.txt").read_text().split()]
    for episode in args.replay_episodes:
        path=data/"data"/f"episode_{episode:07d}.hdf5"
        with h5py.File(path,"r") as f:
            targets=joint_vector(f,"action")
            initial=joint_vector(f,"state")[0]
        for controller in args.controllers:
            start=time.time();env,_,_=make_environment(root,seed_values[episode],existing=env)
            current=np.asarray(env.get_obs()["joint_action"]["vector"])
            initial_error=float(np.abs(current-initial).max())
            executed=0
            for action in targets:
                if controller=="native_qpos":env.take_action(action,action_type="qpos")
                else:dense_joint_step(env,action,15)
                executed+=1
                if env.eval_success:break
            success=bool(env.eval_success or env.check_success())
            row={"mode":"recorded_action_replay","episode":episode,"seed":seed_values[episode],"controller":controller,
                 "success":success,"frames":len(targets),"executed":executed,
                 "initial_state_max_error":initial_error,"seconds":time.time()-start}
            records.append(row);print(json.dumps(row),flush=True)
            env.close_env()
            (out/"progress.json").write_text(json.dumps(records,indent=2)+"\n")
    result={"native_expert_passed":all(x["success"] for x in records if x["mode"]=="native_expert"),"records":records}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rmbench-root",required=True);p.add_argument("--dataset-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[0,100000])
    p.add_argument("--replay-episodes",nargs="+",type=int,default=[0, 1, 2, 3])
    p.add_argument("--controllers",nargs="+",choices=("native_qpos","dense15"),default=["native_qpos","dense15"])
    run(p.parse_args())
