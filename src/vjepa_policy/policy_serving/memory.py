"""Serialized full-history v0 sessions over the existing OpenPI transport."""
from contextlib import nullcontext
import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import threading

import numpy as np
import torch

from vjepa_policy.datasets import ActionStateNormalizer, DEFAULT_PROMPT, VideoClipTransform
from vjepa_policy.datasets.rmbench import decode_joint_actions
from vjepa_policy.history_features import sha256_file
from vjepa_policy.memory_config import FullHistoryV0Config, contract_from_checkpoint_metadata
from vjepa_policy.memory_checkpoint import load_full_history_checkpoint
from vjepa_policy.models.history_cache import HistoryKVCache, HistoryBlock, HistoryCacheError
from vjepa_policy.models.memory_layouts import visual_layout
from vjepa_policy.models.memory_masks import DEFAULT_MAX_MASK_ELEMENTS
from vjepa_policy.models.vjepa_policy import encode_video_views
from vjepa_policy.state import pack_normalized_state
from vjepa_policy.text_cache import text_cache_filename

PROTOCOL = "rmbench_memory_v0"


def content_digest(value):
    """Canonical wire-content digest, including array shape, dtype and bytes."""
    def canonical(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("object arrays are not protocol values")
            return {"array": item.dtype.str, "shape": list(item.shape),
                    "sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest()}
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("protocol keys must be strings")
            return {key: canonical(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(val) for val in item]
        if item is None or type(item) in (str, int, float, bool):
            return item
        raise ValueError(f"unsupported protocol value: {type(item).__name__}")
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


@dataclass
class _Session:
    episode_id: str
    owner: str | None
    instruction_digest: str
    config_signature: str
    language: torch.Tensor
    language_mask: torch.Tensor
    cache: HistoryKVCache
    replies: dict = field(default_factory=dict)
    features: tuple = ()  # recompute diagnostic backend only
    needs_reset: bool = False


class MemoryPolicyServing:
    """Immutable eval model; a lock covers each session transaction and inference."""

    def __init__(self, model, normalizer, text_loader, *, backend="incremental",
                 precision="fp32", max_history_blocks=5, max_cache_bytes=1024**3,
                 max_sessions=1, max_requests=4096, seed=7, model_signature="local"):
        if model.training:
            raise ValueError("memory serving requires an eval model")
        if backend not in ("incremental", "recompute") or precision not in ("fp32", "bf16"):
            raise ValueError("unsupported memory backend or precision")
        if min(max_sessions, max_requests, max_history_blocks, max_cache_bytes) <= 0:
            raise ValueError("serving limits must be positive")
        self.model, self.normalizer, self.text_loader = model, normalizer, text_loader
        self.config = model.config
        self.device = next(model.parameters()).device
        self.dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        self.precision, self.backend, self.seed = precision, backend, seed
        self.max_history_blocks, self.max_cache_bytes = max_history_blocks, max_cache_bytes
        self.max_sessions, self.max_requests = max_sessions, max_requests
        self.signature = content_digest([model_signature, self.config.to_dict(), precision])
        self.sessions, self.lock = {}, threading.RLock()
        self.transform = VideoClipTransform((self.config.image_size,) * 2,
                                            mean=self.config.image_mean, std=self.config.image_std,
                                            resize_mode=self.config.resize_mode)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, pretrained_encoder, text_cache_dir,
                        dataset_stats=None, device="cuda", **options):
        model = load_full_history_checkpoint(checkpoint_path, pretrained_encoder, device=device)
        provenance = model.training_data_provenance or {}
        stats_path = Path(dataset_stats or Path(checkpoint_path).parent / "dataset_stats.json")
        if sha256_file(stats_path) != provenance.get("statistics_sha256"):
            raise ValueError("serving statistics differ from checkpoint training statistics")
        normalizer = ActionStateNormalizer.load(stats_path)
        if (normalizer.metadata.get("action_contract") != model.config.action_contract
                or normalizer.action_mode != "QUANTILE" or normalizer.state_mode != "QUANTILE"
                or normalizer.clip_values):
            raise ValueError("serving requires the v0 unclipped joint/state normalization")
        text_root = Path(text_cache_dir)

        def text_loader(instruction):
            prompt_digest = hashlib.sha256(DEFAULT_PROMPT.format(task=instruction).encode()).hexdigest()
            path = text_root / text_cache_filename(
                DEFAULT_PROMPT.format(task=instruction), model.config.text_length, model.config.text_encoder,
            )
            if sha256_file(path) != provenance.get("text_cache_sha256", {}).get(prompt_digest):
                raise ValueError("instruction/text cache differs from checkpoint provenance")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            return payload["context"][None], payload["mask"][None]
        return cls(model, normalizer, text_loader,
                   model_signature=sha256_file(checkpoint_path), **options)

    @property
    def metadata(self):
        return {"policy_arch": "full_history", "model_version": self.config.model_version, "protocol": PROTOCOL,
                "action_dim": 14, "action_contract": self.config.action_contract,
                "cache_backend": self.backend, "memory_schema": self.config.to_dict(),
                "max_history_blocks": self.max_history_blocks}

    def _precision(self):
        return nullcontext() if self.precision == "fp32" else torch.autocast(self.device.type, dtype=self.dtype)

    def _reply(self, sid, session=None, **extra):
        return {"ok": True, "session_id": sid,
                "episode_id": session.episode_id if session else None,
                "cache_revision": session.cache.revision if session else None,
                "last_block_id": len(session.cache.records) - 1 if session else -1, **extra}

    def infer(self, request):
        return self.infer_connection(request, None)

    @torch.no_grad()
    def infer_connection(self, request, connection_id):
        with self.lock:
            sid, session, request_id, digest, reserved = None, None, None, None, False
            try:
                if not isinstance(request, dict):
                    raise ValueError("request must be a mapping")
                sid = _identifier(request.get("session_id"), "session_id")
                op = request.get("op")
                required = {
                    "reset": {"op", "session_id", "request_id", "episode_id", "instruction", "protocol", "sampling_config"},
                    "append": {"op", "session_id", "request_id", "base_revision", "blocks"},
                    "infer": {"op", "session_id", "request_id", "expected_revision", "decision_step"},
                    "close": {"op", "session_id"},
                }
                if op not in required or set(request) != required[op]:
                    raise ValueError("unknown operation or mismatched operation fields")
                session = self.sessions.get(sid)
                if session is not None and session.owner != connection_id:
                    # Do not disclose another connection's session metadata.
                    session = None
                    raise HistoryCacheError("session_owner_mismatch", "session belongs to another connection")
                if op == "close":
                    self.sessions.pop(sid, None)
                    return self._reply(sid, session, closed=True)
                request_id = _identifier(request["request_id"], "request_id")
                digest = content_digest(request)
                if session is not None:
                    if session.needs_reset:
                        raise HistoryCacheError("session_needs_reset", "device failure invalidated this session")
                    if request_id in session.replies:
                        previous_digest, response = session.replies[request_id]
                        if previous_digest != digest:
                            raise HistoryCacheError("request_conflict", "request_id was bound to different content/revision")
                        return copy.deepcopy(response)
                    if len(session.replies) >= self.max_requests:
                        raise HistoryCacheError("request_capacity_exceeded", "request ledger is full; close/rebuild session")
                if op == "reset":
                    if session is not None:
                        raise HistoryCacheError("session_exists", "close existing session before a new reset")
                    response, session = self._reset(sid, request, connection_id)
                    self.sessions[sid] = session
                    reserved = True
                else:
                    if session is None:
                        raise HistoryCacheError("unknown_session", "reset/rebuild the session first")
                    reserved = True
                    if self.model.training:
                        raise ValueError("model was switched out of eval mode")
                    with self._precision():
                        response = (self._append(sid, session, request) if op == "append"
                                    else self._infer_actions(sid, session, request))
                session.replies[request_id] = (digest, copy.deepcopy(response))
                return response
            except Exception as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    code = "device_out_of_memory"
                elif isinstance(error, HistoryCacheError):
                    code = error.code
                elif isinstance(error, (ValueError, TypeError, KeyError, FileNotFoundError)):
                    code = "invalid_request"
                else:
                    code = "internal_error"
                    logging.exception("memory request failed")
                # CUDA runtime errors other than allocation failure can poison the context.
                if session is not None and isinstance(error, RuntimeError) and not isinstance(error, torch.cuda.OutOfMemoryError):
                    session.needs_reset = True
                response = self._reply(sid, session, ok=False, error={
                    "code": code, "message": str(error),
                    "needs_reset": bool(session and session.needs_reset),
                })
                if reserved:
                    session.replies[request_id] = (digest, copy.deepcopy(response))
                return response

    def _reset(self, sid, request, owner):
        if len(self.sessions) >= self.max_sessions:
            raise HistoryCacheError("session_capacity_exceeded", "maximum active sessions reached")
        # The wire sampling protocol is unchanged by history feature compression.
        schema = request["sampling_config"]
        if not isinstance(schema, dict):
            raise ValueError("sampling_config must be an object")
        requested = contract_from_checkpoint_metadata({
            "policy_arch": "full_history", "model_version": schema.get("model_version"),
            "memory_schema": schema,
        })
        # Legacy v0 clients use the same online 4-step observation grid and
        # 16-step execution cadence; require matching frozen text semantics.
        wire_v0 = FullHistoryV0Config.for_text_encoder(self.config.text_encoder)
        if request["protocol"] != PROTOCOL or requested not in (self.config, wire_v0):
            raise ValueError("memory protocol/sampling configuration mismatch")
        episode_id = _identifier(request["episode_id"], "episode_id")
        instruction = request["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("whole-task instruction is required")
        language, mask = self.text_loader(instruction)
        expected_width = self.model.predictor.lang_dim
        if (language.ndim != 3 or language.shape[0] != 1 or language.shape[-1] != expected_width
                or mask.shape != language.shape[:2] or mask.dtype != torch.bool or not mask.any()
                or not torch.isfinite(language).all()):
            raise ValueError("invalid text cache tensors")
        if self.model.strict_v0_inputs and language.shape[1:] != (self.config.text_length, self.config.text_dim):
            raise ValueError(
                f"{self.config.model_version} requires "
                f"{self.config.text_length}x{self.config.text_dim} text features"
            )
        language = language.masked_fill(~mask[..., None], 0).detach().to(self.device)
        mask = mask.detach().to(self.device)
        predictor = self.model.predictor
        attn = predictor.predictor_blocks[0].self_attn
        cache = HistoryKVCache(depth=len(predictor.predictor_blocks), heads=attn.num_heads,
                               head_dim=attn.head_dim, tokens_per_block=predictor.tokens_per_history_block,
                               device=self.device, dtype=self.dtype, max_blocks=self.max_history_blocks,
                               max_bytes=self.max_cache_bytes)
        session = _Session(episode_id, owner, hashlib.sha256(instruction.encode()).hexdigest(),
                           self.signature, language, mask, cache)
        return self._reply(sid, session, protocol=PROTOCOL, memory_schema=self.config.to_dict()), session

    def _validate_block(self, block):
        if not isinstance(block, dict) or set(block) != {"block_id", "end_step", "frame_indices", "images", "state"}:
            raise ValueError("block must contain IDs, frame_indices, three-camera images and raw state only")
        block_id = _integer(block["block_id"], "block_id")
        end = _integer(block["end_step"], "end_step")
        if end != block_id * self.config.history_interval_steps:
            raise ValueError("block_id/end_step are off the fixed history grid")
        if type(block["frame_indices"]) not in (list, tuple) or any(type(i) is not int for i in block["frame_indices"]):
            raise ValueError("frame_indices must be integer indices")
        if tuple(block["frame_indices"]) != self.config.observation_frames(end):
            raise ValueError("incorrect observation frame indices")
        if not isinstance(block["images"], dict) or set(block["images"]) != set(self.config.cameras):
            raise ValueError("exactly the three v0 cameras are required")
        for pixels in block["images"].values():
            if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != (2, *self.config.source_image_shape):
                raise ValueError("each camera must be uint8 [2,240,320,3] RGB")
            if end == 0 and not np.array_equal(pixels[0], pixels[1]):
                raise ValueError("initial [0,0] block must repeat the same RGB frame")
        state = block["state"]
        if not isinstance(state, np.ndarray) or state.dtype != np.float32 or state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("raw state must be finite float32 [14]")
        return content_digest(block)

    def _encode_block(self, block):
        clips = [self.transform(torch.from_numpy(block["images"][camera].copy()).permute(0, 3, 1, 2))
                 for camera in self.config.cameras]
        video = torch.stack(clips).permute(0, 2, 1, 3, 4)[None].to(self.device)
        # One block per encoder call, matching the M3 reproducibility reference.
        if self.config.dense_decisions:
            # Frozen v2 training features are encoded in FP32, independently of
            # Predictor/action autocast. Preserve that precision online too.
            with torch.autocast(self.device.type, enabled=False):
                latent = encode_video_views(self.model.encoder, video.float())
        else:
            latent = encode_video_views(self.model.encoder, video)
        state = torch.from_numpy(block["state"].copy())[None]
        state = self.normalizer({"observation.state": state})["observation.state"]
        state = pack_normalized_state(state, self.config.max_state_dim).to(self.device)
        return latent, state

    def _layout(self, count, offset=0):
        ends = torch.arange(offset, offset + count, device=self.device)[None] * self.config.history_interval_steps
        p = self.model.predictor
        return p.history_layout(ends, torch.ones_like(ends, dtype=torch.bool),
                                temporal_unit_steps=self.config.temporal_unit_steps, block_offset=offset)

    def _append(self, sid, session, request):
        cache = session.cache
        if _integer(request["base_revision"], "base_revision") != cache.revision:
            raise HistoryCacheError("revision_conflict", "append base_revision is stale")
        blocks = request["blocks"]
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ValueError("append requires a nonempty ordered list of blocks")
        if len(blocks) > self.max_history_blocks:
            raise HistoryCacheError("history_capacity_exceeded", "append batch exceeds block budget")
        new = []
        previous = None
        for block in blocks:
            digest = self._validate_block(block)
            index = block["block_id"]
            if previous is not None and index != previous + 1:
                raise HistoryCacheError("block_order_conflict", "blocks must be contiguous and ordered")
            previous = index
            if index < len(cache.records):
                if cache.records[index].digest != digest:
                    raise HistoryCacheError("block_conflict", "same block_id has different images/state")
            elif index == len(cache.records) + len(new):
                new.append((block, digest))
            else:
                raise HistoryCacheError("block_order_conflict", "history contains a gap")
        if not new:
            return self._reply(sid, session, appended_blocks=0)
        replace = self.backend == "recompute"
        cache.preflight(len(new), replace=replace)
        p = self.model.predictor
        total_tokens = (len(cache.records) + len(new)) * p.tokens_per_history_block
        query_tokens = total_tokens if replace else len(new) * p.tokens_per_history_block
        if (p.attention_backend == "dense" and
                max(query_tokens * total_tokens, p.future_tokens * (total_tokens + p.future_tokens)) > DEFAULT_MAX_MASK_ELEMENTS):
            raise HistoryCacheError("history_capacity_exceeded", "dense reference mask budget exceeded; M5 backend required")
        encoded = [self._encode_block(block) for block, _ in new]
        records = tuple(HistoryBlock(block["block_id"], block["end_step"], digest, item[1])
                        for (block, digest), item in zip(new, encoded))
        new_features = tuple(item[0] for item in encoded)
        features = session.features + new_features if replace else new_features
        all_records = cache.records + records if replace else records
        z = torch.stack(features, dim=1)
        state = torch.stack([r.state for r in all_records], dim=1)
        layout = self._layout(len(all_records), 0 if replace else len(cache.records))
        args = (z, state, session.language, session.language_mask, layout)
        delta = (p.forward_history(*args) if replace else p.append_history(cache.snapshot(), *args))
        cache.commit(delta, records, replace=replace)
        if replace:
            session.features = features
        return self._reply(sid, session, appended_blocks=len(new))

    def _infer_actions(self, sid, session, request):
        cache = session.cache
        if _integer(request["expected_revision"], "expected_revision") != cache.revision:
            raise HistoryCacheError("revision_conflict", "infer expected_revision is stale")
        step = _integer(request["decision_step"], "decision_step")
        if not cache.records or step != cache.records[-1].end_step or step % getattr(self.config, "execution_interval_steps", self.config.decision_interval_steps):
            raise ValueError("infer must use the latest committed observation on the decision grid")
        noise_seed = int(content_digest([self.seed, sid, session.episode_id, request["request_id"], cache.revision])[:16], 16) % (2**63)
        generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        actions = self.model.predict_from_history(
            cache.snapshot(), cache.records[-1].state, session.language, session.language_mask,
            torch.tensor([step], device=self.device), generator=generator,
        )
        actions = self.normalizer.unnormalize_action(actions.float().cpu()).numpy()[0]
        actions = decode_joint_actions(actions)
        if actions.shape != (32, 14) or not np.isfinite(actions).all():
            raise ValueError("model returned invalid actions")
        return self._reply(sid, session, actions=actions, decision_step=step)

    def on_disconnect(self, connection_id):
        with self.lock:
            for sid in [sid for sid, session in self.sessions.items() if session.owner == connection_id]:
                del self.sessions[sid]
