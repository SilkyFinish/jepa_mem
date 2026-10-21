"""Versioned, frozen visual-feature cache; never stores trainable Predictor K/V."""
import hashlib
import json
from pathlib import Path

import torch

from vjepa_policy.memory_config import FullHistoryV0Config


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encoding_code_signature():
    root = Path(__file__).parent
    files = ["history_features.py", "models/memory_policy.py", "models/vjepa_policy.py",
             "datasets/transforms.py", "datasets/rmbench_history.py"]
    files += [str(p.relative_to(root)) for folder in ("encoders", "models/vjepa2_1", "models/backbone")
              for p in sorted((root / folder).rglob("*.py"))]
    result = {name: sha256_file(root / name) for name in files}
    result["scripts/cache_rmbench_history_features.py"] = sha256_file(
        root.parents[1] / "scripts/cache_rmbench_history_features.py"
    )
    return result


def encoder_signature(args):
    config = FullHistoryV0Config()
    return {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "registry": config.encoder, "checkpoint_key": args.encoder_checkpoint_key,
        "interpolate_rope": args.encoder_interpolate_rope,
        "output_branch": "vjepa2_1_video_training_false_distillation",
        "image_size": config.image_size, "patch_size": config.patch_size,
        "tubelet_size": config.tubelet_size, "embed_dim": 1024,
    }


class FrozenHistoryFeatureCache:
    def __init__(self, manifest, dataset, expected_encoder):
        self.path = Path(manifest)
        self.metadata = json.loads(self.path.read_text())
        metadata = self.metadata
        self.fingerprint = fingerprint(metadata)
        config = FullHistoryV0Config()
        if metadata.get("format") != "rmbench_frozen_history_v0":
            raise ValueError("Unknown history feature cache format")
        if metadata.get("contract") != config.to_dict():
            raise ValueError("Feature cache input contract mismatch")
        if metadata.get("encoder") != expected_encoder:
            raise ValueError("Feature cache encoder checkpoint/config mismatch")
        if metadata.get("encoding_code") != encoding_code_signature():
            raise ValueError("Feature cache encoding code changed; rebuild the frozen cache")
        if metadata.get("observation_layer_norm") is not False or metadata.get("target_layer_norm") is not True:
            raise ValueError("Feature cache must preserve observation/target LayerNorm asymmetry")
        if metadata.get("compute_dtype") != "float32":
            raise ValueError("History feature cache must declare float32 encoding")
        if metadata.get("dtype") not in ("float32", "bfloat16"):
            raise ValueError("Unsupported feature cache storage dtype")
        if metadata.get("codec_sha256") != sha256_file(dataset.codec_path):
            raise ValueError("Feature cache RGB codec mismatch")
        for row in dataset.episodes:
            cached = metadata["episodes"].get(row["episode_id"])
            if not cached or cached["source_sha256"] != sha256_file(row["path"]):
                raise ValueError(f'Feature cache source mismatch for episode {row["episode_id"]}')
            if cached["frames"] != row["frames"]:
                raise ValueError("Feature cache episode length mismatch")
        for episode, decision in dataset.index:
            episode_id = dataset.episodes[episode]["episode_id"]
            cached = metadata["episodes"][episode_id]
            for end in config.observed_steps(decision):
                if str(end) not in cached["observations"]:
                    raise ValueError(f"Missing observed feature {episode_id}:{end}")
            if str(decision) not in cached["targets"]:
                raise ValueError(f"Missing future target {episode_id}:{decision}")
        self._checked = set()

    def _tensor(self, entry, shape, frames):
        if entry["frame_indices"] != list(frames):
            raise ValueError("Feature cache input frame indices mismatch")
        path = self.path.parent / entry["file"]
        if path not in self._checked:
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Corrupt frozen feature: {path}")
            self._checked.add(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        dtype = getattr(torch, self.metadata["dtype"])
        if not torch.is_tensor(value) or value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError(f"Invalid frozen feature tensor: {path}")
        # Keep the existing Trainer's strict FP32 input contract; autocast is separate.
        return value.float()

    def load(self, episode_id, decision, ends):
        config = FullHistoryV0Config()
        row = self.metadata["episodes"][episode_id]
        history = torch.stack([
            self._tensor(row["observations"][str(end)], (config.tokens_per_block, 1024),
                         config.observation_frames(end)) for end in ends
        ])
        target = self._tensor(row["targets"][str(decision)], (config.future_tokens, 1024),
                              [max(0, decision + offset) for offset in config.target_offsets])
        return {"history_latents": history, "future_targets": target}
