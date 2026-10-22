"""CPU host preparation: pinned official V1 archive, codec, encoder and T5 snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import zipfile

from scripts.a1008.common import REPO, paths, run, sha256, write_json


def verified(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return Path(path)


def download(url, destination, digest):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"[download] {url}", flush=True)
        # curl resumes interrupted downloads and handles large public weight files.
        run(["curl", "--fail", "--location", "--retry", "5", "--continue-at", "-",
             "--output", str(partial), url])
        verified(partial, digest).replace(destination)
    return verified(destination, digest)


def extract(archive, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=destination.parent) as tmp:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                path = Path(info.filename)
                if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsafe archive member: {info.filename}")
            source.extractall(tmp)
        candidates = [p for p in Path(tmp).rglob("aloha_agilex")
                      if p.parent.name == "put_back_block" and (p / "data").is_dir()]
        if len(candidates) != 1:
            raise ValueError("Official V1 archive must contain exactly one task payload")
        candidates[0].rename(destination)


def verify_prepared(p, lock):
    manifest = json.loads((p["prepared"] / "manifest.json").read_text())
    episodes = manifest["episodes"]
    lengths = [row["frames"] for row in episodes]
    decisions = [t for frames in lengths for t in range(0, frames - 32, 16)]
    actual = dict(episodes=len(episodes), frames=sum(lengths), decisions=len(decisions),
                  max_history_blocks=max(decisions) // 4 + 1)
    if actual != lock["expected"]:
        raise ValueError(f"Unexpected release: {actual}, expected {lock['expected']}")
    canonical = [{key: row[key] for key in ("source_episode", "frames", "sha256", "instruction")}
                 for row in episodes]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != lock["episode_manifest_sha256"]:
        raise ValueError("Official dataset content differs from the validated 50-demo release")
    verified(manifest["codec_path"], lock["codec"]["sha256"])
    for row in episodes:
        verified(row["path"], row["sha256"])
    return actual


def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    p = paths()
    lock = json.loads((REPO / "configs/a1008-assets.json").read_text())
    marker = p["root"] / "assets_ready.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["sources"] != lock:
            raise ValueError("Asset lock changed; use a fresh JEPA_SHARED_ROOT")
        verify_prepared(p, lock)
        verified(p["encoder"], lock["encoder"]["sha256"])
        verified(p["t5"] / "pytorch_model.bin", lock["text"]["weights_sha256"])
        print("[ready] pinned assets already prepared", flush=True)
        return
    if shutil.disk_usage(p["root"]).free < 120 * 1024**3:
        raise RuntimeError("Allow at least 120 GiB free shared storage for downloads, environment, caches and checkpoints")
    spec = lock["dataset"]
    archive = hf_hub_download(repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
                              filename=spec["filename"], local_dir=p["root"] / "downloads")
    extract(verified(archive, spec["sha256"]), p["data"])
    codec_root = p["root"] / "official_codec"
    download(lock["codec"]["url"], codec_root / "data/decode_image_bit.py", lock["codec"]["sha256"])
    download(lock["encoder"]["url"], p["encoder"], lock["encoder"]["sha256"])
    snapshot_download(repo_id=lock["text"]["repo"], revision=lock["text"]["revision"],
                      local_dir=p["t5"], allow_patterns=["config.json", "pytorch_model.bin", "spiece.model",
                                                       "special_tokens_map.json", "tokenizer_config.json"])
    verified(p["t5"] / "pytorch_model.bin", lock["text"]["weights_sha256"])
    prepared_root = p["prepared"].parent
    if not (prepared_root / "summary.json").is_file():
        if prepared_root.exists():
            prepared_root.rename(prepared_root.with_name(f"prepared.incomplete.{time.time_ns()}"))
        run([sys.executable, "examples/rmbench/prepare.py", "--source", p["data"],
             "--rmbench-root", codec_root, "--output", prepared_root])
    actual = verify_prepared(p, lock)
    write_json(marker, dict(sources=lock, actual=actual, prepared_at=time.time()))
    print("[ready] CPU assets prepared; node.sh will build T5 and visual feature caches on one GPU", flush=True)


if __name__ == "__main__":
    main()
