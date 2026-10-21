"""Prepare disjoint original-LIBERO task roots and exact pooled training statistics."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from lerobot.configs.types import FeatureType, PolicyFeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_libero_single_task import convert
from vjepa_policy.datasets.normalization import ActionStateNormalizer


def prepare(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    output = request_path.parent
    arrays = {"action": [], "observation.state": []}
    tasks = []
    for task in request["tasks"]:
        task = dict(task)
        source = Path(task["source"])
        # Keep new training inputs on the requested data disk.
        if task.get("copy_source", False):
            raw = output / "raw" / source.name
            raw.parent.mkdir(exist_ok=True)
            if not raw.exists():
                shutil.copy2(source, raw)
            task["source_original"] = str(source)
            task["source"] = str(raw)
            source = raw
        target = Path(task["prepared_data"])
        if not (target / "manifest.json").exists():
            convert(source, target, request["seed"], request["train_episodes_per_task"])
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["seed"] == request["seed"]
        assert manifest["fps"] == 20 and manifest["native_image_size"] == 128
        train = manifest["splits"]["train"]
        heldout = manifest["splits"]["heldout"]
        assert len(train["episodes"]) == request["train_episodes_per_task"]
        assert not ({r["source_demo"] for r in train["episodes"]} &
                    {r["source_demo"] for r in heldout["episodes"]})
        for row in train["episodes"]:
            parquet = Path(train["root"]) / ("data/chunk-000/episode_%06d.parquet" % row["episode_index"])
            frame = pd.read_parquet(parquet, columns=["action", "observation.state"])
            assert len(frame) == row["frames"]
            for key, width in (("action", 7), ("observation.state", 8)):
                value = np.stack(frame[key]).astype(np.float32)
                assert value.shape == (len(frame), width) and np.isfinite(value).all()
                arrays[key].append(value)
        task.update(instruction=manifest["instruction"], manifest=str(target/"manifest.json"),
                    train_root=train["root"], heldout_root=heldout["root"],
                    train_frames=train["frames"], heldout_frames=heldout["frames"],
                    train_episodes=train["episodes"], heldout_episodes=heldout["episodes"],
                    source_sha256=manifest["source_sha256"])
        tasks.append(task)
        print("Task prepared:", task["key"], task["train_frames"], flush=True)
    values = {key: np.concatenate(items) for key, items in arrays.items()}
    stats = {}
    for key, value in values.items():
        stats[key] = {"mean":value.mean(0), "std":value.std(0), "min":value.min(0), "max":value.max(0),
                      "q01":np.quantile(value,.01,axis=0), "q99":np.quantile(value,.99,axis=0),
                      "count":np.array([len(value)])}
    normalizer = ActionStateNormalizer(
        features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(7,)),
                  "observation.state":PolicyFeature(type=FeatureType.STATE,shape=(8,))},
        stats=stats, action_mode="QUANTILE", state_mode="QUANTILE",
        metadata={"statistics_split":"train", "task_sources":[
            {"key":t["key"],"source":t["source"],"source_sha256":t["source_sha256"],
             "source_demos":[r["source_demo"] for r in t["train_episodes"]]} for t in tasks],
            "state_layout":"eef_pos, axis_angle, gripper_qpos","fps":20,"image_transform":"rotate180",
            "gripper_action_convention":"robosuite"})
    normalizer.save(output/"dataset_stats.json")
    total = sum(t["train_frames"] for t in tasks)
    steps = total // request["global_batch"]
    manifest = dict(request, tasks=tasks, train_frames=total,
        heldout_frames=sum(t["heldout_frames"] for t in tasks),
        optimizer_steps_per_epoch=steps, train_steps=steps*request["epochs"],
        dataset_stats=str(output/"dataset_stats.json"))
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Combined train frames:",total,"optimizer steps:",manifest["train_steps"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",required=True)
    prepare(parser.parse_args().request)
