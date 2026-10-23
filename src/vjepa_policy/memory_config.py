"""Versioned full-history contracts; v0 input sampling and defaults are stable."""
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from typing import Mapping


@dataclass(frozen=True)
class FullHistoryV0Config:
    policy_arch: str = "full_history"
    model_version: str = "v0"
    schema_version: int = 1
    batch_schema: str = "rmbench_history_v0"
    encoder: str = "vjepa2_1_vitl"
    cameras: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    view_layout: str = "independent"
    image_size: int = 224
    source_image_shape: tuple[int, ...] = (240, 320, 3)
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    resize_mode: str = "stretch"
    action_normalization: str = "QUANTILE"
    state_normalization: str = "QUANTILE"
    patch_size: int = 16
    tubelet_size: int = 2
    observation_offsets: tuple[int, ...] = (-4, 0)
    history_interval_steps: int = 4
    decision_interval_steps: int = 16
    target_offsets: tuple[int, ...] = (-4, 0, 4, 8, 12, 16, 20, 24, 28, 32)
    action_horizon: int = 32
    action_dim: int = 14
    raw_state_dim: int = 14
    max_state_dim: int = 48
    text_length: int = 128
    text_dim: int = 4096
    temporal_unit_steps: int = 8
    time_basis: str = "frame_or_control_index"
    visual_position_rule: str = "tubelet_right_edge"
    action_position_rule: str = "target_state_time"
    state_conditioning: str = "per_history_block"
    history_scope: str = "episode_start_to_decision"
    tail_policy: str = "require_full_future_window"
    action_contract: str = "aloha_absolute_joint_14"
    action_label_shift: int = 0
    gripper_indices: tuple[int, ...] = (6, 13)
    gripper_open_value: int = 1
    instruction_scope: str = "whole_task"
    action_reads_future: bool = True

    def __post_init__(self):
        # M0 freezes one recipe. A changed recipe needs an explicit contract
        # revision rather than silently changing an existing v0 experiment.
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not type(field.default) or value != field.default:
                raise ValueError(
                    f"Full-History v0 requires {field.name}={field.default!r}, got {value!r}"
                )
            if isinstance(value, tuple) and any(
                type(item) is not type(expected)
                for item, expected in zip(value, field.default)
            ):
                raise ValueError(f"Invalid element types in {field.name}")

    @property
    def tokens_per_block(self):
        """Uncompressed frozen Encoder tokens (also the cache format)."""
        return len(self.cameras) * (self.image_size // self.patch_size) ** 2

    @property
    def history_pool_size(self):
        return getattr(self, "history_spatial_pool", 1)

    @property
    def history_tokens_per_block(self):
        return self.tokens_per_block // self.history_pool_size ** 2

    @property
    def future_tokens(self):
        return (len(self.target_offsets) // self.tubelet_size - 1) * self.tokens_per_block

    @property
    def packed_state_dim(self):
        return 2 * self.max_state_dim

    def to_dict(self):
        # Canonical JSON-compatible arrays, including tuple fields.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("memory_schema must be an object")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError(
                f"memory_schema fields differ: missing={sorted(expected-set(value))}, "
                f"unknown={sorted(set(value)-expected)}"
            )
        values = dict(value)
        for field in fields(cls):
            if isinstance(field.default, tuple):
                if not isinstance(values[field.name], (list, tuple)):
                    raise ValueError(f"{field.name} must be an array")
                values[field.name] = tuple(values[field.name])
        return cls(**values)

    def fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def checkpoint_metadata(self):
        """Future checkpoint envelope; M0 does not create memory weights."""
        return {
            "policy_arch": self.policy_arch,
            "model_version": self.model_version,
            "memory_schema": self.to_dict(),
        }

    def decision_steps(self, episode_frames):
        _integer("episode_frames", episode_frames, minimum=1)
        return tuple(range(0, max(0, episode_frames - self.action_horizon),
                           self.decision_interval_steps))

    def observed_steps(self, end_step):
        _integer("end_step", end_step)
        if end_step % self.history_interval_steps:
            raise ValueError("History endpoint must lie on the v0 observation grid")
        return tuple(range(0, end_step + 1, self.history_interval_steps))

    def observation_frames(self, end_step):
        self.observed_steps(end_step)
        return tuple(max(0, end_step + offset) for offset in self.observation_offsets)

    def sample_indices(self, episode_id, episode_frames, decision_step):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        _integer("decision_step", decision_step)
        if decision_step not in self.decision_steps(episode_frames):
            raise ValueError("Decision is off-grid or lacks a complete future window")
        ends = self.observed_steps(decision_step)
        return {
            "episode_id": episode_id,
            "decision_step": decision_step,
            "history_end_step": ends,
            "history_frame_indices": tuple(self.observation_frames(i) for i in ends),
            "history_state_indices": ends,
            "target_frame_indices": tuple(
                min(episode_frames - 1, max(0, decision_step + offset)) for offset in self.target_offsets
            ),
            # The dataset has ALREADY shifted actions to next-state targets.
            "action_indices": tuple(min(i, episode_frames - 1) for i in range(decision_step, decision_step + self.action_horizon)),
            "current_state_index": decision_step,
        }

    def temporal_coordinates(self, decision_step):
        _integer("decision_step", decision_step)
        if decision_step % self.decision_interval_steps:
            raise ValueError("Decision must lie on the v0 decision grid")
        unit = self.temporal_unit_steps
        return {
            "history": tuple(i / unit for i in self.observed_steps(decision_step)),
            "future": tuple(
                (decision_step + offset) / unit for offset in self.target_offsets[3::2]
            ),
            "action": tuple(
                (decision_step + j + 1) / unit for j in range(self.action_horizon)
            ),
        }


@dataclass(frozen=True)
class FullHistoryV1Config(FullHistoryV0Config):
    """Same full temporal prefix/targets, with per-view 2x2 average pooling of H."""

    model_version: str = "v1"
    history_spatial_pool: int = 2
    history_compression: str = "per_view_avg_pool_before_predictor"
    history_spatial_position: str = "original_patch_centers"


@dataclass(frozen=True)
class FullHistoryV2Config(FullHistoryV1Config):
    """v1 architecture with dense training decisions and a current off-grid block."""

    model_version: str = "v2"
    schema_version: int = 2
    decision_interval_steps: int = 1
    execution_interval_steps: int = 16
    history_endpoint_rule: str = "regular_grid_plus_current"
    tail_policy: str = "clip_future_frames_mask_padded_actions"

    def decision_steps(self, episode_frames):
        _integer("episode_frames", episode_frames, minimum=1)
        return tuple(range(episode_frames))

    def observed_steps(self, end_step):
        _integer("end_step", end_step)
        regular = tuple(range(0, end_step + 1, self.history_interval_steps))
        return regular if regular[-1] == end_step else (*regular, end_step)


def full_history_config(version="v0"):
    if version == "v0":
        return FullHistoryV0Config()
    if version == "v1":
        return FullHistoryV1Config()
    if version == "v2":
        return FullHistoryV2Config()
    raise ValueError(f"Unsupported full-history model version: {version!r}")


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def contract_from_checkpoint_metadata(metadata):
    """Resolve a future envelope without changing existing checkpoint loaders.

    Metadata-free historical checkpoints remain legacy. Partial/unknown memory
    markers must not silently fall back to the original architecture.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be an object")
    arch = metadata.get("policy_arch", "legacy")
    if arch == "legacy":
        if "memory_schema" in metadata or "model_version" in metadata:
            raise ValueError("Legacy checkpoint contains incompatible memory version markers")
        return None
    if arch != "full_history" or metadata.get("model_version") not in ("v0", "v1", "v2"):
        raise ValueError("Unsupported policy architecture/model version")
    cls = type(full_history_config(metadata["model_version"]))
    config = cls.from_dict(metadata.get("memory_schema"))
    if config.policy_arch != arch or config.model_version != metadata["model_version"]:
        raise ValueError("Checkpoint architecture and memory schema disagree")
    return config
