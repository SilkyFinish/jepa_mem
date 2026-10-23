"""M0 contracts for Full-History v0. No model imports or legacy defaults change.

Model version (v0) and serialization schema version (1) are independent.
These declarations do not implement a trainable memory policy.
"""
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
        if self.model_version not in ("v0", "v0_bert"):
            raise ValueError(f"Unsupported Full-History model version: {self.model_version!r}")
        language = {"model_version": self.model_version,
                    "text_dim": 768 if self.model_version == "v0_bert" else 4096}
        for field in fields(self):
            value = getattr(self, field.name)
            expected = language.get(field.name, field.default)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"Full-History {self.model_version} requires {field.name}={expected!r}, got {value!r}"
                )
            if isinstance(value, tuple) and any(
                type(item) is not type(expected_item)
                for item, expected_item in zip(value, field.default)
            ):
                raise ValueError(f"Invalid element types in {field.name}")

    @classmethod
    def for_text_encoder(cls, text_encoder):
        if text_encoder == "t5":
            return cls()
        if text_encoder == "bert":
            return cls(model_version="v0_bert", text_dim=768)
        raise ValueError(f"Unsupported text encoder: {text_encoder!r}")

    @property
    def text_encoder(self):
        return "bert" if self.model_version == "v0_bert" else "t5"

    @property
    def tokens_per_block(self):
        return len(self.cameras) * (self.image_size // self.patch_size) ** 2

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
                max(0, decision_step + offset) for offset in self.target_offsets
            ),
            # The dataset has ALREADY shifted actions to next-state targets.
            "action_indices": tuple(range(decision_step, decision_step + self.action_horizon)),
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
    if arch != "full_history" or metadata.get("model_version") not in ("v0", "v0_bert"):
        raise ValueError("Unsupported policy architecture/model version")
    config = FullHistoryV0Config.from_dict(metadata.get("memory_schema"))
    if config.policy_arch != arch or config.model_version != metadata["model_version"]:
        raise ValueError("Checkpoint architecture and memory schema disagree")
    return config
