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


def text_encoder():
    name = os.environ.get("JEPA_TEXT_ENCODER", "t5")
    if name not in ("t5", "bert"):
        raise ValueError("JEPA_TEXT_ENCODER must be t5 or bert")
    return name


def train_only():
    value = os.environ.get("JEPA_TRAIN_ONLY", "0")
    if value not in ("0", "1"):
        raise ValueError("JEPA_TRAIN_ONLY must be 0 or 1")
    return value == "1"


def per_device_batch():
    value = int(os.environ.get("JEPA_PER_DEVICE_BATCH", "1"))
    if value < 1:
        raise ValueError("JEPA_PER_DEVICE_BATCH must be positive")
    return value


def asset_sources():
    lock = json.loads((REPO / "configs/a1008-assets.json").read_text())
    bert = lock.pop("bert")
    if text_encoder() == "bert":
        lock["text"] = bert
    return lock


def paths():
    root = Path(os.environ["JEPA_SHARED_ROOT"]).resolve(strict=True)
    bert = text_encoder() == "bert"
    return dict(root=root, data=root / "dataset/demo_clean/put_back_block/aloha_agilex",
                prepared=root / "prepared/full50", text=root / ("text_cache_bert" if bert else "text_cache"),
                text_model=root / "weights" / ("bert-base-uncased" if bert else "t5-v1_1-xxl"),
                ready=root / ("assets_ready_bert.json" if bert else "assets_ready.json"),
                encoder=root / "weights/vjepa2_1_vitl_dist_vitG_384.pt",
                features=root / ("history_features_bert" if bert else "history_features") / "manifest.json")


def base_args(p):
    return ["--policy-arch", "full_history", "--text-encoder", text_encoder(),
            "--lang-dim", "768" if text_encoder() == "bert" else "4096", "--checkpoint", str(p["encoder"]),
            "--dataset-dirs", str(p["prepared"]), "--dataset-stats", str(p["prepared"] / "dataset_stats.json"),
            "--text-cache-dir", str(p["text"]), "--action-normalization", "QUANTILE",
            "--state-normalization", "QUANTILE", "--pred-depth", "24", "--pred-embed-dim", "1024",
            "--pred-num-heads", "16", "--action-hidden-size", "512", "--action-num-layers", "24",
            "--condition-num-heads", "8", "--use-activation-checkpointing",
            "--activation-checkpointing-blocks", os.environ.get("JEPA_CHECKPOINT_BLOCKS", "24"),
            "--history-attention-backend", "segmented", "--batch-size", str(per_device_batch()),
            "--mixed-precision", "bf16",
            "--lr", "1e-4", "--weight-decay", "0.01", "--seed", "7", "--log-every", "1",
            "--num-workers", os.environ.get("JEPA_NUM_WORKERS", "2"), "--prefetch-factor", "2"]


def schedule(samples, world, epochs, per_device_batch=1):
    if world < 1 or per_device_batch < 1 or 16 % (world * per_device_batch) or epochs < 1:
        raise ValueError("World size times per-device batch must divide global batch 16; epochs must be positive")
    accumulation = 16 // (world * per_device_batch)
    updates = (samples // (world * per_device_batch)) // accumulation
    if updates < 1:
        raise ValueError("Insufficient data for one global batch")
    return dict(world_size=world, per_device_batch=per_device_batch, accumulation=accumulation,
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
