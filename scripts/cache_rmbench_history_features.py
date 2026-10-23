"""Cache frozen observations and supervised future targets for Full-History v0.

Remaining arguments use train_vjepa_policy.py's full_history CLI contract.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.train_vjepa_policy import parse_args, build_frozen_encoder
from vjepa_policy.memory_training import validate_history_args
from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
from vjepa_policy.history_features import sha256_file, encoder_signature, encoding_code_signature
from vjepa_policy.models.vjepa_policy import encode_video_views


@torch.no_grad()
def build_cache(dataset, encoder, output_dir, *, encoder_metadata, storage_dtype=torch.float32,
                device="cuda", batch_size=1, decode_episode=False):
    """Commit manifest only after every independent observation/target is complete."""
    if batch_size < 1:
        raise ValueError("cache batch_size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "features").exists():
        raise FileExistsError("Use a fresh feature-cache directory; existing features are preserved")
    (output / "features").mkdir()
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dataset.config
    metadata = {
        "format": "rmbench_frozen_history_v0",
        "contract": config.to_dict(),
        "encoder": encoder_metadata,
        "encoding_code": encoding_code_signature(),
        "codec_sha256": sha256_file(dataset.codec_path),
        "dtype": str(storage_dtype).removeprefix("torch."),
        "compute_dtype": "float32",
        "encoder_batch_size": batch_size,
        "torch_version": torch.__version__,
        "observation_layer_norm": False,
        "target_layer_norm": True,
        "source_manifest_sha256": sha256_file(dataset.manifest_path),
        "selection": dataset.selection,
        "episodes": {},
    }

    def save_feature(value, name, frames, expected_tokens):
        if value.shape != (expected_tokens, 1024) or not torch.isfinite(value).all():
            raise ValueError(f"Unexpected encoder features: {value.shape}")
        path = output / "features" / f"{name}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(value.to(dtype=storage_dtype, device="cpu").contiguous(), temporary)
        os.replace(temporary, path)
        return {"file": str(path.relative_to(output)), "sha256": sha256_file(path),
                "frame_indices": list(frames)}

    for episode_index, row in enumerate(dataset.episodes):
        decisions = [t for i, t in dataset.index if i == episode_index]
        if not decisions:
            continue
        source_sha = sha256_file(row["path"])
        if row.get("sha256") and row["sha256"] != source_sha:
            raise ValueError(f'Source HDF5 hash changed: {row["episode_id"]}')
        entry = {
            "source_sha256": source_sha, "frames": row["frames"],
            "observations": {}, "targets": {},
        }
        metadata["episodes"][row["episode_id"]] = entry
        # Transform each real frame once when building dense caches. Clips remain
        # independent Encoder examples; no temporal context leaks across clips.
        decoded = dataset.decode_frames(episode_index, range(row["frames"])) if decode_episode else None
        def video_for(frames):
            return decoded[:, :, list(frames)] if decoded is not None else dataset.decode_frames(episode_index, frames)
        observation_ends = sorted({end for t in decisions for end in config.observed_steps(t)})
        for start in range(0, len(observation_ends), batch_size):
            selected = observation_ends[start:start + batch_size]
            frames = [config.observation_frames(end) for end in selected]
            videos = torch.stack([video_for(f) for f in frames]).to(device)
            features = encode_video_views(encoder, videos)
            for end, f, z in zip(selected, frames, features):
                entry["observations"][str(end)] = save_feature(
                    z, f"episode{episode_index}_obs{end}", f, config.tokens_per_block)
        # y_future sees each local ten-frame clip, then extra LN and selection.
        for start in range(0, len(decisions), batch_size):
            selected = decisions[start:start + batch_size]
            frames = [[min(row["frames"] - 1, max(0, t + offset)) for offset in config.target_offsets]
                      for t in selected]
            videos = torch.stack([video_for(f) for f in frames]).to(device)
            features = encode_video_views(encoder, videos)
            features = F.layer_norm(features, (features.shape[-1],))[:, config.tokens_per_block:]
            for decision, f, target in zip(selected, frames, features):
                entry["targets"][str(decision)] = save_feature(
                    target, f"episode{episode_index}_future{decision}", f, config.future_tokens)
        del decoded
        print(f'[cache] episode={row["episode_id"]} observations={len(entry["observations"])} '
              f'targets={len(entry["targets"])}', flush=True)
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Pass native dataset/checkpoint/text-cache and history selection flags as in training.",
    )
    parser.add_argument("--cache-output", required=True)
    parser.add_argument("--storage-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    parser.add_argument("--decode-episode", action="store_true")
    cache_args, training_argv = parser.parse_known_args(argv)
    args = parse_args(["--policy-arch", "full_history", "--output-dir",
                       str(Path(cache_args.cache_output) / "preparation"), *training_argv])
    validate_history_args(args)
    if args.history_feature_cache:
        raise ValueError("Build features from native RGB, not from an existing feature cache")
    dataset = RMBenchHistoryDataset(Path(args.dataset_dirs[0]) / "manifest.json", args)
    try:
        build_cache(dataset, build_frozen_encoder(args), cache_args.cache_output,
                    encoder_metadata=encoder_signature(args),
                    storage_dtype=getattr(torch, cache_args.storage_dtype), device=cache_args.device,
                    batch_size=cache_args.encoder_batch_size, decode_episode=cache_args.decode_episode)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
