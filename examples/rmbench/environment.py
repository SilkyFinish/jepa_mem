"""Thin configuration/observation bridge to the pinned RMBench simulator."""
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import yaml

CAMERA_MAP = {"cam_head":"head_camera", "cam_left_wrist":"left_camera", "cam_right_wrist":"right_camera"}

def environment_config(rmbench_root, task="put_back_block"):
    root=Path(rmbench_root).resolve()
    sys.path.insert(0,str(root))
    os.chdir(root)  # Upstream URDF and object loaders use repository-relative assets.
    config=yaml.safe_load((root/"env_cfg/task_config/demo_clean.yml").read_text())
    robots=yaml.safe_load((root/"env_cfg/task_config/_embodiment_config.yml").read_text())
    path=root/robots[config["embodiment"][0]]["file_path"]
    robot=yaml.safe_load((path/"config.yml").read_text())
    config.update(task_name=task, left_robot_file=str(path),right_robot_file=str(path),
                  left_embodiment_config=robot,right_embodiment_config=robot,dual_arm_embodied=True,
                  dual_arm=True,save_data=False,collect_data=False,eval_video_log=False,render_freq=0,eval_mode=True)
    instruction=json.loads((root/"description/task_instruction"/(task+".json")).read_text())["seen"][0]
    return config,instruction

def make_environment(rmbench_root, seed, task="put_back_block", existing=None):
    config,instruction=environment_config(rmbench_root,task)
    cls=getattr(importlib.import_module("envs."+task),task)
    env=existing if existing is not None else cls()
    env.setup_demo(seed=seed,now_ep_num=0,is_test=True,**config)
    env.set_instruction(instruction)
    return env,config,instruction

def policy_observation(native, past, instruction):
    result={"observation.state":np.asarray(native["joint_action"]["vector"],dtype=np.float32),"prompt":instruction}
    for camera,source in CAMERA_MAP.items():
        key="observation.images."+camera
        result[key]=np.asarray(native["observation"][source]["rgb"],dtype=np.uint8)
        result[key+"_past"]=np.asarray(past["observation"][source]["rgb"],dtype=np.uint8)
    return result

def dense_joint_step(env, action, substeps=15):
    """Dataset-cadence executor, for explicit controller diagnostics only."""
    action=np.asarray(action,dtype=np.float32)
    if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError("Expected 14 finite joints")
    current=np.asarray(env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState())
    dt=env.scene.get_timestep()
    for j in range(1,substeps+1):
        fraction=j/substeps
        target=current+(action-current)*fraction
        for side,offset in (("left",0),("right",7)):
            velocity=(action[offset:offset+6]-current[offset:offset+6])/(dt*substeps)
            env.robot.set_arm_joints(target[offset:offset+6],velocity,side)
            env.robot.set_gripper(float(np.clip(target[offset+6],0,1)),side)
        env.scene.step()
        if env.check_success():env.eval_success=True
    env.take_action_cnt+=1
    env._update_render()
