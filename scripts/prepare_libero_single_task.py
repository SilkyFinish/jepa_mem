"""Convert one original LIBERO HDF5 into disjoint LeRobot train/heldout roots.

Images are rotated 180 degrees to match run_libero_client.py. Native 20 Hz
actions and all frames are preserved, including no-op actions. Normalization
statistics use the training episodes only. No training is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature
from vjepa_policy.datasets.normalization import ActionStateNormalizer


def convert(source, output, seed=7, train_episodes=40):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already prepared: {manifest_path}")
    with h5py.File(source, "r") as handle:
        data = handle["data"]
        names = sorted(data, key=lambda name: int(name.rsplit("_", 1)[1]))
        if not 0 < train_episodes < len(names):
            raise ValueError("Require nonempty train and heldout splits")
        instruction = json.loads(data.attrs["problem_info"])["language_instruction"]
        fps = int(json.loads(data.attrs["env_args"])["env_kwargs"]["control_freq"])
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(names)).tolist()
        splits = {"train": order[:train_episodes], "heldout": order[train_episodes:]}
        features = {
            "observation.images.image": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist_image": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (8,), "names": ["eef_x", "eef_y", "eef_z", "axis_x", "axis_y", "axis_z", "gripper_left", "gripper_right"]},
            "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]},
        }
        train_values = {"action": [], "observation.state": []}
        records = {}
        for split, indices in splits.items():
            root = output / split
            if root.exists():
                raise FileExistsError(f"Use a fresh output directory: {root}")
            dataset = LeRobotDataset.create(repo_id=f"local/drawer_{split}", root=root,
                fps=fps, robot_type="panda", features=features, use_videos=True,
                image_writer_threads=4, video_backend="pyav")
            rows = []
            for episode_index, index in enumerate(indices):
                name = names[index]
                demo = data[name]
                obs = demo["obs"]
                action = np.asarray(demo["actions"], dtype=np.float32)
                state = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], axis=-1).astype(np.float32)
                assert action.shape == (len(state), 7)
                assert np.isin(action[:, -1], [-1.0, 1.0]).all(), "Expected raw robosuite gripper actions"
                assert np.isfinite(action).all() and np.isfinite(state).all()
                if split == "train":
                    train_values["action"].append(action)
                    train_values["observation.state"].append(state)
                for t in range(len(action)):
                    dataset.add_frame({
                        "observation.images.image": np.ascontiguousarray(obs["agentview_rgb"][t][::-1, ::-1]),
                        "observation.images.wrist_image": np.ascontiguousarray(obs["eye_in_hand_rgb"][t][::-1, ::-1]),
                        "observation.state": state[t],
                        "action": action[t],
                    }, task=instruction)
                dataset.save_episode()
                rows.append({"episode_index": episode_index, "source_demo": name, "frames": len(action)})
                print(f"{split} {episode_index+1}/{len(indices)} {name} frames={len(action)}", flush=True)
            dataset.stop_image_writer()
            records[split] = {"root": str(root.resolve()), "episodes": rows, "frames": sum(r["frames"] for r in rows)}
        values = {key: np.concatenate(arrays) for key, arrays in train_values.items()}
        stats = {key: {
            "mean": value.mean(0), "std": value.std(0), "min": value.min(0), "max": value.max(0),
            "q01": np.quantile(value, .01, axis=0), "q99": np.quantile(value, .99, axis=0),
            "count": np.array([len(value)]),
        } for key, value in values.items()}
        normalizer = ActionStateNormalizer(
            features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
                      "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
            stats=stats, action_mode="QUANTILE", state_mode="QUANTILE",
            metadata={"source": str(source.resolve()), "statistics_split": "train",
                      "train_source_demos": [names[i] for i in splits["train"]],
                      "state_layout": "eef_pos, axis_angle, gripper_qpos",
                      "image_transform": "rotate180", "fps": fps, "gripper_action_convention": "robosuite"})
        normalizer.save(output / "dataset_stats.json")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(chunk)
        manifest = {"source": str(source.resolve()), "source_sha256": digest.hexdigest(),
            "seed": seed, "instruction": instruction, "fps": fps,
            "image_transform": "rotate180", "native_image_size": 128,
            "preserve_noops": True, "gripper_action_convention": "robosuite", "splits": records,
            "stats": str((output/"dataset_stats.json").resolve())}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print("Prepared", manifest_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=40)
    args = parser.parse_args()
    convert(args.source, args.output, args.seed, args.train_episodes)
