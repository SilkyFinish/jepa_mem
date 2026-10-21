"""RMBench XPolicyLab HDF5 adapter; the policy architecture is unchanged."""
from __future__ import annotations
import bisect
import functools
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from vjepa_policy.datasets import DEFAULT_PROMPT, VideoClipTransform, combine_video_clips
from vjepa_policy.datasets.lerobot_video import TextEmbeddingCache, make_video_offsets
from vjepa_policy.datasets.normalization import ActionStateNormalizer

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_FIELDS = ("left_arm_joint_states", "left_ee_joint_states",
                "right_arm_joint_states", "right_ee_joint_states")

def read_joint_vector(handle, group):
    """Left six joints, left gripper, right six joints, right gripper."""
    return np.concatenate([handle[f"{group}/{key}"][:] for key in JOINT_FIELDS], axis=-1).astype(np.float32)

@functools.lru_cache(maxsize=4)
def image_decoder(codec_path):
    spec = importlib.util.spec_from_file_location("_rmbench_rgb_codec", codec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit

def decode_joint_actions(actions):
    """Native absolute joint radians; continuous grippers 0=closed, 1=open."""
    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim < 2 or result.shape[-1] != 14 or not np.isfinite(result).all():
        raise ValueError("RMBench actions must be finite [..., horizon, 14]")
    result[..., [6, 13]] = np.clip(result[..., [6, 13]], 0.0, 1.0)
    return result

class RMBenchDataset(Dataset):
    def __init__(self, manifest, args, include_action=True):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["action_contract"] != "aloha_absolute_joint_14":
            raise ValueError("Unsupported RMBench action contract")
        self.episodes = self.manifest["episodes"]
        self.codec_path = self.manifest["codec_path"]
        self.video_keys = list(args.views)
        expected = ["observation.images." + key for key in CAMERAS]
        if self.video_keys != expected or args.view_layout != "independent":
            raise ValueError(f"Use independent views in order {expected}")
        if args.action_dim != 14 or args.proprio_dim != 14:
            raise ValueError("RMBench requires action_dim=proprio_dim=14")
        self.offsets = np.asarray(make_video_offsets(args.num_frames, args.past_frames, args.video_frame_stride))
        self.horizon = args.num_frames - 1
        self.include_action = include_action
        self.transform = VideoClipTransform((args.crop_size, args.crop_size))
        self.cache = TextEmbeddingCache(args.text_cache_dir, args.context_len)
        self.normalizer = ActionStateNormalizer.load(args.dataset_stats or self.manifest["statistics"])
        self.normalizer.save(Path(args.output_dir) / "dataset_stats.json")
        self._numeric = []
        self._ends = []
        count = 0
        for row in self.episodes:
            with h5py.File(row["path"], "r") as f:
                state, action = read_joint_vector(f, "state"), read_joint_vector(f, "action")
            if state.shape != action.shape or state.shape != (row["frames"], 14):
                raise ValueError("Episode vector shape changed")
            self._numeric.append((torch.from_numpy(state), torch.from_numpy(action)))
            count += len(state) - (max(0, int(self.offsets.max())) if args.require_full_future_window else 0)
            self._ends.append(count)
        self._handles = OrderedDict()

    def __len__(self):
        return self._ends[-1]

    def _handle(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
            if len(self._handles) > 8:
                self._handles.popitem(last=False)[1].close()
        self._handles.move_to_end(path)
        return self._handles[path]

    def __getstate__(self):
        result = self.__dict__.copy()
        result["_handles"] = OrderedDict()
        return result

    def close(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = OrderedDict()

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self._ends, index)
        t = index - (self._ends[episode - 1] if episode else 0)
        row = self.episodes[episode]
        state, action = self._numeric[episode]
        times = np.clip(t + self.offsets, 0, len(state) - 1)
        f = self._handle(row["path"])
        decode = image_decoder(self.codec_path)
        clips = []
        for camera in CAMERAS:
            images = np.stack([decode(f[f"vision/{camera}/colors"][int(i)]) for i in times])
            clip = torch.from_numpy(images).permute(0, 3, 1, 2)
            clips.append(self.transform(clip))
        language, mask = self.cache.load(DEFAULT_PROMPT.format(task=row["instruction"]))
        current = self.normalizer({"observation.state": state[t]})["observation.state"]
        result = {"video": combine_video_clips(clips, "independent"),
                  "context": language, "context_mask": mask, "proprio": current}
        if self.include_action:
            indices = torch.arange(t, t + self.horizon)
            result.update(action=self.normalizer({"action": action[indices.clamp(max=len(action)-1)]})["action"],
                          action_is_pad=indices >= len(action))
        return result

def build_dataset(args, include_action=True):
    if not args.dataset_dirs or len(args.dataset_dirs) != 1:
        raise ValueError("Pass one prepared RMBench manifest directory")
    return RMBenchDataset(Path(args.dataset_dirs[0]) / "manifest.json", args, include_action)
