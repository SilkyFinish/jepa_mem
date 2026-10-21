"""Validate native RMBench HDF5 and prepare manifests/statistics without re-encoding images."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.rmbench import CAMERAS, image_decoder, read_joint_vector

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def prepare(source, rmbench_root, output, seed=7):
    source, rmbench_root, output = map(lambda p: Path(p).resolve(), (source, rmbench_root, output))
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True)
    codec = rmbench_root / "data/decode_image_bit.py"
    decode = image_decoder(str(codec))
    seeds = [int(x) for x in (source / "seed.txt").read_text().split()]
    files = sorted((source / "data").glob("*.hdf5"))
    if len(files) != 50 or len(seeds) != len(files):
        raise ValueError("Expected the official 50-demo put_back_block release")
    rows, values = [], {}
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            state, action = read_joint_vector(f, "state"), read_joint_vector(f, "action")
            if state.shape != action.shape or state.shape[1] != 14 or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Invalid vectors: {path}")
            if not (np.all(action[:, [6,13]] >= -1e-5) and np.all(action[:, [6,13]] <= 1+1e-5)):
                raise ValueError("Gripper convention differs")
            # The release shifts action one saved frame ahead of observed state.
            alignment = float(np.max(np.abs(action[:-1] - state[1:])))
            if alignment > 1e-5:
                raise ValueError(f"Next-frame action/state alignment changed: {alignment}")
            shape = {}
            for cam in CAMERAS:
                images = f[f"vision/{cam}/colors"]
                if len(images) != len(state): raise ValueError("Image/state lengths differ")
                for t in range(len(state)):
                    rgb = decode(images[t])
                    if rgb.dtype != np.uint8 or rgb.shape != (240,320,3):
                        raise ValueError(f"Unexpected image geometry {rgb.shape}")
                shape[cam] = list(rgb.shape)
            instruction = json.loads((source / "instruction" / (path.stem + ".json")).read_text())["seen"]
            if len(instruction) != 1: raise ValueError("Expected one fixed task instruction")
            frequency = int(f["additional_info/frequency"][()])
            version = f["data_format_version"].asstr()[()]
            rows.append({"path":str(path), "source_episode":i, "seed":seeds[i], "frames":len(state),
                         "instruction":instruction[0], "sha256":sha256(path), "rgb_shapes":shape,
                         "data_format_version":version, "physics_steps_per_frame":frequency,
                         "next_state_action_max_error":alignment})
            values[i] = {"action":action, "observation.state":state}
        print(f"validated {i+1}/{len(files)} frames={len(state)}", flush=True)
    order = np.random.default_rng(seed).permutation(len(rows)).tolist()
    groups = {"full50":list(range(50)), "diagnostic_train40":order[:40], "diagnostic_heldout10":order[40:]}
    for name, indices in groups.items():
        target=output/name;target.mkdir()
        training_indices=groups["diagnostic_train40"] if name=="diagnostic_heldout10" else indices
        stats={}
        for key in ("action","observation.state"):
            array=np.concatenate([values[i][key] for i in training_indices])
            stats[key]={"mean":array.mean(0),"std":array.std(0),"min":array.min(0),"max":array.max(0),
                        "q01":np.quantile(array,.01,axis=0),"q99":np.quantile(array,.99,axis=0),"count":np.array([len(array)])}
        normalizer=ActionStateNormalizer(
            features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(14,)),
                      "observation.state":PolicyFeature(type=FeatureType.STATE,shape=(14,))},
            stats=stats,action_mode="QUANTILE",state_mode="QUANTILE",
            metadata={"benchmark":"RMBench","action_contract":"aloha_absolute_joint_14","statistics_source_episodes":training_indices,
                      "joint_units":"radians","gripper_indices":[6,13],"gripper_range":[0,1],"gripper_open":1,
                      "camera_order":list(CAMERAS),"rgb_decoder_sha256":sha256(codec),"image_transform":"stretch224_rgb",
                      "physics_hz":250,"physics_steps_per_frame":15,"target_alignment":"action[t] = state[t+1]"})
        stat_path=target/"dataset_stats.json";normalizer.save(stat_path)
        manifest={"action_contract":"aloha_absolute_joint_14","task":"put_back_block","split":name,"seed":seed,
                  "codec_path":str(codec),"codec_sha256":sha256(codec),"statistics":str(stat_path),"episodes":[rows[i] for i in indices]}
        (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (target/"meta").mkdir()
        tasks=sorted({rows[i]["instruction"] for i in indices})
        (target/"meta/tasks.jsonl").write_text("".join(json.dumps({"task_index":j,"task":text})+"\n" for j,text in enumerate(tasks)))
    summary={"source":str(source),"episodes":len(rows),"frames":sum(r["frames"] for r in rows),
             "diagnostic_train_episodes":groups["diagnostic_train40"],"diagnostic_heldout_episodes":groups["diagnostic_heldout10"],
             "formal_training_episodes":list(range(50)),"formal_rollout_seeds_start":100000,
             "note":"Full50 uses all official training demonstrations; diagnostic heldout is disjoint only from diagnostic_train40.",
             "codec_sha256":sha256(codec)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True);p.add_argument("--rmbench-root",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=7)
    a=p.parse_args();prepare(a.source,a.rmbench_root,a.output,a.seed)
