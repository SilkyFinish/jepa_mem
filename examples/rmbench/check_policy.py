"""Validate dataset normalization and real checkpoint inference before formal training."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from vjepa_policy.datasets.rmbench import CAMERAS, build_dataset, image_decoder, read_joint_vector
from vjepa_policy.policy_serving.rmbench import RMBenchPolicyServing

def check(args):
    start=time.time()
    args.dataset_dirs=[args.prepared_data]
    args.views=["observation.images."+x for x in CAMERAS]
    args.view_layout="independent";args.action_dim=14;args.proprio_dim=14
    args.num_frames=33;args.past_frames=4;args.video_frame_stride=4
    args.crop_size=224;args.context_len=128;args.dataset_stats=None
    args.require_full_future_window=False
    dataset=build_dataset(args)
    first=dataset[0];last=dataset[len(dataset)-1]
    if first["video"].shape != (3,3,10,224,224):raise ValueError("Video geometry differs")
    if first["context"].shape != (128,4096):raise ValueError("Text geometry differs")
    if not last["action_is_pad"][1:].all():raise ValueError("Tail action padding differs")
    rows=dataset.episodes
    action_errors=[];state_errors=[]
    for row in rows:
        with h5py.File(row["path"],"r") as f:
            values={"action":torch.from_numpy(read_joint_vector(f,"action")),
                    "observation.state":torch.from_numpy(read_joint_vector(f,"state"))}
        restored=dataset.normalizer.unnormalize(dataset.normalizer(values))
        action_errors.append(float((restored["action"]-values["action"]).abs().max()))
        state_errors.append(float((restored["observation.state"]-values["observation.state"]).abs().max()))
    report={"dataset_samples":len(dataset),"episodes":len(rows),"video_shape":list(first["video"].shape),
            "language_shape":list(first["context"].shape),"action_shape":list(first["action"].shape),
            "action_roundtrip_max_error":max(action_errors),"state_roundtrip_max_error":max(state_errors)}
    if max(action_errors+state_errors)>5e-4:raise ValueError("Action/state normalization roundtrip mismatch")
    if args.checkpoint:
        torch.manual_seed(7);np.random.seed(7)
        policy=RMBenchPolicyServing(args.checkpoint,args.encoder_checkpoint,args.text_cache_dir,device=args.device)
        row=rows[0];t=8;past=t-4
        with h5py.File(row["path"],"r") as f:
            state=read_joint_vector(f,"state")
            obs={"observation.state":state[t],"prompt":row["instruction"]}
            decode=image_decoder(dataset.codec_path)
            for camera in CAMERAS:
                key="observation.images."+camera
                obs[key]=decode(f["vision/"+camera+"/colors"][t])
                obs[key+"_past"]=decode(f["vision/"+camera+"/colors"][past])
        clip=policy._prepare_observation(obs)[0]
        if not torch.equal(clip[:,:,:2],dataset[t]["video"][:,:,:2]):
            raise ValueError("Train/serve context images differ")
        result=policy.infer(obs)["actions"]
        if result.shape!=(32,14) or not np.isfinite(result).all() or np.any(result[:,[6,13]]<0) or np.any(result[:,[6,13]]>1):
            raise ValueError("Invalid real inference output")
        report.update(inference_action_shape=list(result.shape),inference_finite=True,train_serve_context_equal=True,
                      cuda_max_memory_GiB=torch.cuda.max_memory_allocated()/2**30)
    report.update(ok=True,seconds=time.time()-start)
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.output_dir)/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-data",required=True);p.add_argument("--text-cache-dir",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--checkpoint");p.add_argument("--encoder-checkpoint");p.add_argument("--device",default="cuda")
    check(p.parse_args())
