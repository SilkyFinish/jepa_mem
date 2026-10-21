"""Native RMBench decisions with complete observed prefixes (Full-History v0)."""
from collections import OrderedDict
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from vjepa_policy.memory_config import FullHistoryV0Config
from vjepa_policy.datasets import DEFAULT_PROMPT, VideoClipTransform
from vjepa_policy.datasets.lerobot_video import TextEmbeddingCache
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.rmbench import image_decoder, read_joint_vector


class RMBenchHistoryDataset(Dataset):
    """One episode/decision per row; diagnostic selection never removes old blocks."""

    batch_schema = FullHistoryV0Config().batch_schema

    def __init__(self, manifest, args, include_action=True):
        if not include_action:
            raise ValueError("History v0 currently supports joint policy training")
        self.config = FullHistoryV0Config()
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["action_contract"] != self.config.action_contract:
            raise ValueError("RMBench history requires native absolute 14D actions")
        self.codec_path = self.manifest["codec_path"]
        self.transform = VideoClipTransform((self.config.image_size,) * 2)
        self.text_cache = TextEmbeddingCache(args.text_cache_dir, self.config.text_length)
        self.normalizer = ActionStateNormalizer.load(args.dataset_stats or self.manifest["statistics"])
        if (self.normalizer.action_mode, self.normalizer.state_mode, self.normalizer.clip_values) != (
                "QUANTILE", "QUANTILE", False):
            raise ValueError("v0 requires unclipped QUANTILE action/state normalization")
        for key in ("action", "observation.state"):
            if tuple(self.normalizer.features[key].shape) != (14,):
                raise ValueError(f"{key} statistics must have width 14")
        self.normalizer.save(Path(args.output_dir) / "dataset_stats.json")
        requested = getattr(args, "history_episode_ids", None)
        requested = None if requested is None else set(map(str, requested))
        limit = getattr(args, "history_max_decision_step", None)
        if limit is not None and (limit < 0 or limit % self.config.decision_interval_steps):
            raise ValueError("history-max-decision-step must be a nonnegative multiple of 16")
        self.episodes, self.index, self._numeric = [], [], []
        self._handles = OrderedDict()
        found = set()
        for row in self.manifest["episodes"]:
            episode_id = str(row["source_episode"])
            if requested is not None and episode_id not in requested:
                continue
            if episode_id in found:
                raise ValueError(f"Duplicate episode id {episode_id}")
            found.add(episode_id)
            row = dict(row, episode_id=episode_id)
            with h5py.File(row["path"], "r") as handle:
                state, action = read_joint_vector(handle, "state"), read_joint_vector(handle, "action")
            if state.shape != action.shape or state.shape != (row["frames"], 14):
                raise ValueError(f"Episode {episode_id} vector shape changed")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Episode {episode_id} has nonfinite state/action")
            # Released native labels already command the next observed state.
            if not np.allclose(action[:-1], state[1:], rtol=0, atol=1e-6):
                raise ValueError(f"Episode {episode_id} violates native next-state action alignment")
            episode_index = len(self.episodes)
            self.episodes.append(row)
            self._numeric.append(self.normalizer({
                "observation.state": torch.from_numpy(state), "action": torch.from_numpy(action),
            }))
            for decision in self.config.decision_steps(row["frames"]):
                if limit is None or decision <= limit:
                    self.index.append((episode_index, decision))
        if requested is not None and requested != found:
            raise ValueError(f"Unknown history episode ids: {sorted(requested - found)}")
        if not self.index:
            raise ValueError("No legal history decisions selected")
        self.history_lengths = [t // self.config.history_interval_steps + 1 for _, t in self.index]
        capacity = getattr(args, "max_history_blocks", None)
        if capacity is not None and max(self.history_lengths) > capacity:
            raise ValueError("Selected complete prefixes exceed --max-history-blocks; no truncation is allowed")
        self.selection = {
            "episode_ids": sorted(found),
            "max_decision_step": limit,
            "diagnostic_subset": requested is not None or limit is not None,
            "samples": len(self.index),
            "max_history_blocks": max(self.history_lengths),
        }
        self.feature_cache = None
        cache_path = getattr(args, "history_feature_cache", None)
        if cache_path:
            from vjepa_policy.history_features import FrozenHistoryFeatureCache, encoder_signature
            self.feature_cache = FrozenHistoryFeatureCache(
                cache_path, self, encoder_signature(args),
            )
        self.provenance = self._provenance(args)

    def _provenance(self, args):
        from vjepa_policy.history_features import sha256_file
        prompts = sorted({DEFAULT_PROMPT.format(task=row["instruction"]) for row in self.episodes})
        import hashlib
        text_files = {
            hashlib.sha256(prompt.encode()).hexdigest(): sha256_file(
                Path(args.text_cache_dir) /
                f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len{self.config.text_length}.pt"
            ) for prompt in prompts
        }
        return {
            "manifest_sha256": sha256_file(self.manifest_path),
            "statistics_sha256": sha256_file(Path(args.output_dir) / "dataset_stats.json"),
            "text_cache_sha256": text_files,
            "selection": self.selection,
            "feature_cache_fingerprint": self.feature_cache.fingerprint if self.feature_cache else None,
        }

    def __len__(self):
        return len(self.index)

    def _handle(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
            if len(self._handles) > 4:
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
        # h5py globals may already be cleared during interpreter shutdown.
        try:
            self.close()
        except (TypeError, AttributeError):
            pass

    def decode_frames(self, episode_index, frame_indices):
        """Decode each unique frame/camera once; return [V,C,T,H,W] RGB."""
        row = self.episodes[episode_index]
        handle = self._handle(row["path"])
        decode = image_decoder(self.codec_path)
        unique = sorted(set(map(int, frame_indices)))
        lookup = {t: i for i, t in enumerate(unique)}
        order = torch.tensor([lookup[int(t)] for t in frame_indices])
        clips = []
        for camera in self.config.cameras:
            images = np.stack([decode(handle[f"vision/{camera}/colors"][t]) for t in unique])
            if images.shape[1:] != self.config.source_image_shape or images.dtype != np.uint8:
                raise ValueError("Expected native uint8 RGB [240,320,3]")
            transformed = self.transform(torch.from_numpy(images).permute(0, 3, 1, 2)).permute(1, 0, 2, 3)
            clips.append(transformed.index_select(1, order))
        return torch.stack(clips)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index, decision = self.index[index]
        row = self.episodes[episode_index]
        config = self.config
        ends = config.observed_steps(decision)
        state = self._numeric[episode_index]["observation.state"][list(ends)]
        language, mask = self.text_cache.load(DEFAULT_PROMPT.format(task=row["instruction"]))
        if language.shape != (config.text_length, config.text_dim):
            raise ValueError("v0 requires cached T5 [128,4096]")
        sample = {
            "batch_schema": config.batch_schema,
            "episode_id": row["episode_id"],
            "sample_key": f'{row["episode_id"]}:{decision}',
            "history_state": state,
            "history_end_step": torch.tensor(ends, dtype=torch.long),
            "history_valid": torch.ones(len(ends), dtype=torch.bool),
            "decision_step": torch.tensor(decision, dtype=torch.long),
            "current_state": state[-1].clone(),
            "language": language.float(),
            "language_mask": mask,
            "future_valid": torch.ones(config.future_tokens, dtype=torch.bool),
            # No second action shift; all selected decisions have full targets.
            "action": self._numeric[episode_index]["action"][decision:decision + config.action_horizon],
            "action_is_pad": torch.zeros(config.action_horizon, dtype=torch.bool),
        }
        if self.feature_cache:
            sample.update(self.feature_cache.load(row["episode_id"], decision, ends))
        else:
            history_frames = [t for end in ends for t in config.observation_frames(end)]
            target_frames = [max(0, decision + t) for t in config.target_offsets]
            decoded = self.decode_frames(episode_index, history_frames + target_frames)
            split = len(history_frames)
            sample["history_video"] = decoded[:, :, :split].reshape(
                3, 3, len(ends), 2, config.image_size, config.image_size,
            ).permute(2, 0, 1, 3, 4, 5).contiguous()
            sample["target_video"] = decoded[:, :, split:].contiguous()
        return sample


class HistoryBatchCollator:
    """Right-pad K; keep each episode in its own batch row."""

    batch_schema = FullHistoryV0Config().batch_schema

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty history batch")
        if any(s.get("batch_schema") != self.batch_schema for s in samples):
            raise ValueError("History collator requires rmbench_history_v0 samples")
        fields = set(samples[0])
        if any(set(s) != fields for s in samples):
            raise ValueError("Cannot mix video and cached history samples")
        max_blocks = max(len(s["history_valid"]) for s in samples)
        result = {"batch_schema": self.batch_schema}
        for name in fields - {"batch_schema"}:
            values = [sample[name] for sample in samples]
            if name in ("episode_id", "sample_key"):
                result[name] = values
            elif name.startswith("history_"):
                result[name] = torch.stack([
                    torch.cat((value, value.new_zeros((max_blocks - len(value), *value.shape[1:]))))
                    for value in values
                ])
            else:
                result[name] = torch.stack(values)
        return result


def build_dataset(args, include_action=True):
    if not args.dataset_dirs or len(args.dataset_dirs) != 1:
        raise ValueError("Pass one prepared native RMBench manifest directory")
    dataset = RMBenchHistoryDataset(Path(args.dataset_dirs[0]) / "manifest.json", args, include_action)
    return dataset, HistoryBatchCollator()
