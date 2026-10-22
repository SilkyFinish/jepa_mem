"""Small standard-library helpers shared by preparation and node jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def paths():
    root = Path(os.environ["JEPA_SHARED_ROOT"]).resolve(strict=True)
    return dict(root=root, data=root / "dataset/demo_clean/put_back_block/aloha_agilex",
                prepared=root / "prepared/full50", text=root / "text_cache",
                t5=root / "weights/t5-v1_1-xxl",
                encoder=root / "weights/vjepa2_1_vitl_dist_vitG_384.pt",
                features=root / "history_features/manifest.json")


def base_args(p):
    return ["--policy-arch", "full_history", "--checkpoint", str(p["encoder"]),
            "--dataset-dirs", str(p["prepared"]), "--dataset-stats", str(p["prepared"] / "dataset_stats.json"),
            "--text-cache-dir", str(p["text"]), "--action-normalization", "QUANTILE",
            "--state-normalization", "QUANTILE", "--pred-depth", "24", "--pred-embed-dim", "1024",
            "--pred-num-heads", "16", "--action-hidden-size", "512", "--action-num-layers", "24",
            "--condition-num-heads", "8", "--use-activation-checkpointing",
            "--activation-checkpointing-blocks", os.environ.get("JEPA_CHECKPOINT_BLOCKS", "24"),
            "--history-attention-backend", "segmented", "--batch-size", "1", "--mixed-precision", "bf16",
            "--lr", "1e-4", "--weight-decay", "0.01", "--seed", "7", "--log-every", "1",
            "--num-workers", os.environ.get("JEPA_NUM_WORKERS", "2"), "--prefetch-factor", "2"]


def schedule(samples, world, epochs):
    if world < 1 or 16 % world or epochs < 1:
        raise ValueError("World size must divide global batch 16; epochs must be positive")
    accumulation = 16 // world
    updates = (samples // world) // accumulation
    if updates < 1:
        raise ValueError("Insufficient data for one global batch")
    return dict(world_size=world, per_device_batch=1, accumulation=accumulation,
                global_batch=16, samples=samples, updates_per_epoch=updates,
                effective_samples_per_epoch=updates * 16, epochs=epochs, updates=updates * epochs)


def serving_capacity(episode_steps):
    """Include old + grown buffers + a four-block delta during atomic KV growth."""
    if episode_steps < 1:
        raise ValueError("Native episode limit must be positive")
    blocks = episode_steps // 4 + 1
    bytes_per_block = 588 * 1024 * 24 * 2 * 2  # K/V, BF16
    return blocks, (2 * blocks + 4) * bytes_per_block + 512 * 2**20


def run(command, **kwargs):
    print("[command]", __import__("shlex").join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, cwd=REPO, **kwargs)
