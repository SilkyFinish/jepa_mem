"""Lossless packed storage; retains the original frozen cache's logical identity."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from vjepa_policy.history_features import fingerprint, sha256_file

FORMAT = "rmbench_frozen_history_packed_v1"



def warm_packed_observations(manifest, reserve_bytes=16 * 2**30):
    """Warm shared file pages before launching distributed training processes."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Observation prewarming must finish before torchrun; run "
            "scripts.pack_rmbench_history_features --warm-only in one process, "
            "then omit --history-feature-warm-observations from distributed training"
        )
    manifest = Path(manifest)
    storage = json.loads(manifest.read_text())
    if storage.get("format") != FORMAT or storage.get("dtype") != "float32":
        raise ValueError("Unknown packed history format")
    files = []
    for row in storage["episodes"].values():
        record = row["observations"]
        path = (manifest.parent / record["file"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("Packed path escapes cache")
        if path.stat().st_size != record["file_bytes"]:
            raise ValueError("Packed file size changed")
        files.append((path, record["file_bytes"]))
    total = sum(size for _, size in files)
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    if available < total + reserve_bytes:
        raise MemoryError("Not enough RAM to warm observations and retain training reserve")
    start, warmed = time.monotonic(), 0
    for index, (path, size) in enumerate(files, 1):
        with path.open("rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
        warmed += size
        print(f"[packed-cache] warmed {index}/{len(files)} observations "
              f"{warmed / 2**30:.2f}/{total / 2**30:.2f} GiB "
              f"elapsed={time.monotonic() - start:.1f}s", flush=True)
    return {"bytes": total, "elapsed_seconds": time.monotonic() - start,
            "storage": "shared kernel file cache; no duplicate worker observation LRU"}


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else value
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def pack_frozen_cache(manifest, output):
    """Copy checked FP32 bytes into episode arrays; resume completed shards."""
    manifest, output = Path(manifest), Path(output)
    source = json.loads(manifest.read_text())
    if source.get("format") != "rmbench_frozen_history_v0" or source.get("dtype") != "float32":
        raise ValueError("Lossless packing requires an FP32 frozen history cache")
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(source)
    header = {"format": FORMAT, "source_fingerprint": identity,
              "source_manifest_sha256": sha256_file(manifest), "dtype": "float32", "episodes": {}}
    start = time.monotonic()
    for ordinal, (episode, row) in enumerate(source["episodes"].items()):
        shard = output / f"episode{ordinal:04d}"
        shard.mkdir(exist_ok=True)
        completed = shard / "complete.json"
        if completed.exists():
            entry = json.loads(completed.read_text())
            if entry["source_fingerprint"] != identity or entry["episode_id"] != episode:
                raise ValueError("Packed shard belongs to another source")
            for kind in ("observations", "targets"):
                record = entry[kind]
                array = np.load(output / record["file"], mmap_mode="r", allow_pickle=False)
                if list(array.shape) != record["shape"] or array.dtype != np.float32:
                    raise ValueError("Completed shard geometry changed")
            header["episodes"][episode] = entry
            continue
        entry = {"source_fingerprint": identity, "episode_id": episode}
        for kind in ("observations", "targets"):
            ends = sorted(map(int, row[kind]))
            first = row[kind][str(ends[0])]
            first_path = manifest.parent / first["file"]
            if sha256_file(first_path) != first["sha256"]:
                raise ValueError(f"Corrupt source tensor: {first_path}")
            first_value = torch.load(first_path, map_location="cpu", weights_only=True)
            if first_value.dtype != torch.float32 or first_value.ndim != 2:
                raise ValueError("Expected FP32 [tokens,width] features")
            shape = (len(ends), *first_value.shape)
            destination = shard / f"{kind}.npy"
            temporary = destination.with_suffix(".npy.tmp")
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            hashes, source_hashes = [], []
            for index, end in enumerate(ends):
                original = row[kind][str(end)]
                path = manifest.parent / original["file"]
                if index == 0:
                    value = first_value
                else:
                    if sha256_file(path) != original["sha256"]:
                        raise ValueError(f"Corrupt source tensor: {path}")
                    value = torch.load(path, map_location="cpu", weights_only=True)
                if value.dtype != torch.float32 or tuple(value.shape) != shape[1:] or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid frozen tensor: {path}")
                digest = tensor_digest(value)
                array[index] = value.numpy()
                if tensor_digest(array[index]) != digest:
                    raise ValueError(f"Packed tensor differs: {path}")
                hashes.append(digest)
                source_hashes.append(original["sha256"])
            array.flush()
            del array
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            entry[kind] = {"file": str(destination.relative_to(output)), "shape": list(shape),
                           "ends": ends, "tensor_sha256": hashes, "source_entry_sha256": source_hashes,
                           "file_bytes": destination.stat().st_size}
        write_json(completed, entry)
        header["episodes"][episode] = entry
        write_json(output / "progress.json", {"episodes": len(header["episodes"]),
                   "total_episodes": len(source["episodes"]), "elapsed_seconds": time.monotonic() - start,
                   "last_episode": episode, "updated_at": time.time()})
        print(f"[pack] {len(header['episodes'])}/{len(source['episodes'])} episode={episode}", flush=True)
    write_json(output / "manifest.json", header)
    return header


class PackedHistoryFeatureCache:
    """Clean mmap pages share physical RAM across ranks and workers.

    Observation pages use the kernel file cache rather than duplicate worker LRUs.
    Loader lookahead overlaps contiguous target reads with GPU computation.
    Every packed row is bound to the original entry and checked on first use.
    """
    def __init__(self, logical_cache, packed):
        self.path = Path(packed)
        self.metadata = logical_cache.metadata
        self.fingerprint = logical_cache.fingerprint
        self.config = logical_cache.config
        self.storage = json.loads(self.path.read_text())
        if self.storage.get("format") != FORMAT or self.storage.get("dtype") != "float32":
            raise ValueError("Unknown packed history format")
        if self.storage.get("source_fingerprint") != self.fingerprint:
            raise ValueError("Packed cache logical feature identity differs")
        if self.storage.get("source_manifest_sha256") != sha256_file(logical_cache.path):
            raise ValueError("Packed source manifest bytes changed")
        if set(self.storage["episodes"]) != set(self.metadata["episodes"]):
            raise ValueError("Packed episode coverage differs")
        self._lookups = {}
        for episode, row in self.metadata["episodes"].items():
            entry = self.storage["episodes"][episode]
            if entry.get("source_fingerprint") != self.fingerprint or entry.get("episode_id") != episode:
                raise ValueError("Packed episode identity differs")
            for kind, tokens in (("observations", self.config.tokens_per_block), ("targets", self.config.future_tokens)):
                record = entry[kind]
                ends = sorted(map(int, row[kind]))
                if (record["ends"] != ends or record["shape"] != [len(ends), tokens, 1024]
                        or len(record["tensor_sha256"]) != len(ends)
                        or record["source_entry_sha256"] != [row[kind][str(e)]["sha256"] for e in ends]):
                    raise ValueError("Packed rows differ from original manifest")
                for end in ends:
                    expected_frames = (list(self.config.observation_frames(end)) if kind == "observations"
                                       else [min(row["frames"] - 1, max(0, end + offset))
                                             for offset in self.config.target_offsets])
                    if row[kind][str(end)]["frame_indices"] != expected_frames:
                        raise ValueError("Packed source feature frame indices mismatch")
                self._lookups[episode, kind] = {end: index for index, end in enumerate(ends)}
                if self._path(record).stat().st_size != record["file_bytes"]:
                    raise ValueError("Packed file size changed")
        self._arrays, self._checked = OrderedDict(), set()

    def _path(self, record):
        path = (self.path.parent / record["file"]).resolve()
        if not path.is_relative_to(self.path.parent.resolve()):
            raise ValueError("Packed path escapes cache")
        return path

    def __getstate__(self):
        result = self.__dict__.copy()
        result["_arrays"], result["_checked"] = OrderedDict(), set()
        return result

    def _array(self, episode, kind):
        key = episode, kind
        if key not in self._arrays:
            record = self.storage["episodes"][episode][kind]
            # No writes occur: clean private pages remain shared, HDD-backed pages.
            array = np.load(self._path(record), mmap_mode="c", allow_pickle=False)
            if list(array.shape) != record["shape"] or array.dtype != np.float32 or not array.flags.c_contiguous:
                raise ValueError("Packed header changed")
            self._arrays[key] = array
            if len(self._arrays) > 104:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(key)
        return self._arrays[key]

    def rows(self, episode, kind, ends):
        record = self.storage["episodes"][episode][kind]
        indices = [self._lookups[episode, kind][int(end)] for end in ends]
        array = self._array(episode, kind)
        # One read range for adjacent anchors; the collator preserves sample order.
        if indices and indices == list(range(indices[0], indices[0] + len(indices))):
            values = array[indices[0]:indices[0] + len(indices)]
        else:
            values = array[indices]
        for position, index in enumerate(indices):
            key = episode, kind, index
            if key not in self._checked:
                if tensor_digest(values[position]) != record["tensor_sha256"][index]:
                    raise ValueError(f"Corrupt packed tensor: {episode}:{kind}:{index}")
                self._checked.add(key)
        return torch.from_numpy(values)

    def load(self, episode_id, decision, ends):
        return {"history_feature_ref": (episode_id, decision, tuple(ends))}

    def warm_observations(self, reserve_bytes=16 * 2**30):
        """Warm observations with bounded reads; no full private RAM allocation."""
        return warm_packed_observations(self.path, reserve_bytes)


class PackedHistoryBatchCollator:
    """Transfer unique observations once and retain the complete history by index."""
    def __init__(self, cache, base_collator):
        self.cache, self.base_collator = cache, base_collator

    def __call__(self, samples):
        references = [sample["history_feature_ref"] for sample in samples]
        clean = [{k: v for k, v in sample.items() if k != "history_feature_ref"} for sample in samples]
        result = self.base_collator(clean)
        grouped = OrderedDict()
        for episode, decision, ends in references:
            item = grouped.setdefault(episode, {"ends": set(), "decisions": []})
            item["ends"].update(ends)
            item["decisions"].append(decision)
        pieces, lookup, targets, offset = [], {}, {}, 0
        for episode, item in grouped.items():
            ends = sorted(item["ends"])
            pieces.append(self.cache.rows(episode, "observations", ends))
            lookup.update({(episode, end): offset + i for i, end in enumerate(ends)})
            offset += len(ends)
            decisions = sorted(set(item["decisions"]))
            target = self.cache.rows(episode, "targets", decisions)
            targets.update({(episode, decision): target[i] for i, decision in enumerate(decisions)})
        indices = torch.zeros_like(result["history_end_step"])
        for row, (episode, decision, ends) in enumerate(references):
            indices[row, :len(ends)] = torch.tensor([lookup[episode, end] for end in ends])
        result["history_latents"] = torch.cat(pieces, dim=0)
        result["history_latent_index"] = indices
        result["future_targets"] = torch.stack([targets[e, d] for e, d, _ in references])
        return result


def prepare_packed_history(dataset, collator, args):
    from vjepa_policy.datasets.rmbench_history import RMBenchHistoryDataset
    if not isinstance(dataset, RMBenchHistoryDataset) or dataset.feature_cache is None:
        raise ValueError("Packed storage requires native history data and its original cache manifest")
    cache = PackedHistoryFeatureCache(dataset.feature_cache, args.history_feature_packed)
    dataset.feature_cache = cache
    if getattr(args, "history_feature_warm_observations", False):
        print(f"[packed-cache] warm observations: {cache.warm_observations()}", flush=True)
    return dataset, PackedHistoryBatchCollator(cache, collator)
