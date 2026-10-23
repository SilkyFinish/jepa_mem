"""CPU host preparation: pinned official data, vision weights and frozen text encoder."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import zipfile

from scripts.a1008.common import asset_sources, paths, run, sha256, text_encoder, write_json


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


def prepare_text_model(p, spec):
    destination = p["text_model"]
    if text_encoder() == "bert":
        source = os.environ.get("JEPA_BERT_SOURCE")
        destination.mkdir(parents=True, exist_ok=True)
        for name, digest in spec["files"].items():
            target = destination / name
            if not target.exists():
                if not source:
                    raise ValueError("Set JEPA_BERT_SOURCE to the existing local bert-base-uncased directory")
                original = verified(Path(source) / name, digest)
                partial = target.with_suffix(target.suffix + ".partial")
                shutil.copy2(original, partial)
                verified(partial, digest).replace(target)
            verified(target, digest)
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=destination,
                      allow_patterns=["config.json", "pytorch_model.bin", "spiece.model",
                                      "special_tokens_map.json", "tokenizer_config.json"])
    verified(destination / "pytorch_model.bin", spec["weights_sha256"])


def verify_text_model(p, spec):
    if text_encoder() == "bert":
        for name, digest in spec["files"].items():
            verified(p["text_model"] / name, digest)
    else:
        verified(p["text_model"] / "pytorch_model.bin", spec["weights_sha256"])


def prepare_bert_cache(p, spec):
    from vjepa_policy.text_embeddings import precompute_text_embeddings
    marker = p["text"] / "provenance.json"
    with (p["root"] / ".cache_build.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if marker.exists():
            cached = json.loads(marker.read_text())
            if cached["source"] != spec or cached["context_length"] != 128 or not cached["files"]:
                raise ValueError("BERT cache provenance mismatch")
            for name, digest in cached["files"].items():
                verified(p["text"] / name, digest)
            return
        precompute_text_embeddings([p["prepared"]], p["text"], str(p["text_model"]),
                                   128, device="cpu", text_encoder="bert")
        files = {f.name: sha256(f) for f in sorted(p["text"].glob("*.bert_len128.pt"))}
        if not files:
            raise ValueError("No BERT instruction embeddings were generated")
        write_json(marker, dict(source=spec, context_length=128, files=files))


def main():
    from huggingface_hub import hf_hub_download
    p = paths()
    lock = asset_sources()
    marker = p["ready"]
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["sources"] != lock:
            raise ValueError("Asset lock changed; use a fresh JEPA_SHARED_ROOT")
        verify_prepared(p, lock)
        verified(p["encoder"], lock["encoder"]["sha256"])
        verify_text_model(p, lock["text"])
        if text_encoder() == "bert":
            prepare_bert_cache(p, lock["text"])
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
    prepare_text_model(p, lock["text"])
    prepared_root = p["prepared"].parent
    if not (prepared_root / "summary.json").is_file():
        if prepared_root.exists():
            prepared_root.rename(prepared_root.with_name(f"prepared.incomplete.{time.time_ns()}"))
        run([sys.executable, "examples/rmbench/prepare.py", "--source", p["data"],
             "--rmbench-root", codec_root, "--output", prepared_root])
    actual = verify_prepared(p, lock)
    if text_encoder() == "bert":
        prepare_bert_cache(p, lock["text"])
    write_json(marker, dict(sources=lock, actual=actual, prepared_at=time.time()))
    print(f"[ready] CPU assets prepared; text_encoder={text_encoder()}; "
          "node.sh will build frozen text and visual feature caches", flush=True)


if __name__ == "__main__":
    main()
