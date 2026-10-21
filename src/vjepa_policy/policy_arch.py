"""Explicit training capabilities and architecture-aware checkpoint loading."""
from dataclasses import dataclass

from vjepa_policy.memory_config import FullHistoryV0Config, contract_from_checkpoint_metadata


@dataclass(frozen=True)
class TrainingCapabilities:
    supports_packed_state: bool = False
    supports_local_layout: bool = False
    supports_predictor_init: bool = False
    batch_schema: str = "legacy"


LEGACY_CAPABILITIES = TrainingCapabilities(True, True, True)
HISTORY_CAPABILITIES = TrainingCapabilities(True, True, True, "rmbench_history_v0")


def capabilities(builder):
    return getattr(builder, "training_capabilities", TrainingCapabilities())


def require_checkpoint_arch(checkpoint, expected_arch):
    contract = contract_from_checkpoint_metadata(checkpoint)
    actual = "legacy" if contract is None else contract.policy_arch
    if actual != expected_arch:
        raise ValueError(
            f"Checkpoint policy_arch={actual} cannot be loaded as {expected_arch}; "
            "use the matching architecture (predictor-only transfer is a separate operation)"
        )
    return contract


def memory_model_semantics(model):
    predictor = model.predictor
    return {
        "rope_options": dict(predictor.rope_options),
        "latent_grid": [predictor.grid_depth, predictor.grid_height, predictor.grid_width],
        "num_views": predictor.num_views,
        "encoder_embed_dim": predictor.predictor_embed.in_features,
        "predictor_depth": len(predictor.predictor_blocks),
        "predictor_heads": predictor.predictor_blocks[0].self_attn.num_heads,
        "action_condition_heads": model.action_expert.blocks[0].condition_attention.num_heads,
        "action_inference_steps": model.action_expert.num_inference_steps,
        "flow_time_buckets": model.action_expert.num_timestep_buckets,
        "flow_beta": [model.action_expert._beta_alpha, model.action_expert._beta_beta],
        "action_reads_future": model.action_expert.action_reads_future,
        "world_loss_weight": model.world_loss_weight,
        "action_loss_weight": model.action_loss_weight,
    }


def validate_policy_checkpoint(checkpoint, model):
    config = getattr(model, "config", None)
    expected_arch = config.policy_arch if isinstance(config, FullHistoryV0Config) else "legacy"
    contract = require_checkpoint_arch(checkpoint, expected_arch)
    if contract is None:
        return
    if contract != config:
        raise ValueError("Memory checkpoint contract differs from the model")
    saved = checkpoint.get("memory_training")
    if not isinstance(saved, dict) or saved.get("model_semantics") != memory_model_semantics(model):
        raise ValueError("Memory checkpoint model/RoPE semantics mismatch")
    expected_encoder = getattr(model, "frozen_encoder_signature", None)
    if expected_encoder is not None and saved.get("frozen_encoder") != expected_encoder:
        raise ValueError("Memory checkpoint frozen encoder mismatch")
    expected_data = getattr(model, "training_data_provenance", None)
    if expected_data:
        for key in ("statistics_sha256", "text_cache_sha256"):
            if saved.get("data", {}).get(key) != expected_data.get(key):
                raise ValueError(f"Memory checkpoint {key} mismatch")
    if "action_expert" not in checkpoint:
        raise ValueError("Full-history resume requires Action Expert weights")


def load_policy_weights(model, checkpoint):
    """Validate architecture/semantics before any parameter is mutated."""
    validate_policy_checkpoint(checkpoint, model)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    if hasattr(model, "action_expert") and "action_expert" in checkpoint:
        model.action_expert.load_state_dict(checkpoint["action_expert"], strict=True)
    if hasattr(model, "proprio_encoder") and "proprio_encoder" in checkpoint:
        model.proprio_encoder.load_state_dict(checkpoint["proprio_encoder"], strict=True)


def memory_checkpoint_metadata(model):
    return {
        **model.config.checkpoint_metadata(),
        "memory_training": {
            "model_semantics": memory_model_semantics(model),
            "frozen_encoder": getattr(model, "frozen_encoder_signature", None),
            "data": getattr(model, "training_data_provenance", None),
            "cache_backend": "recompute",
            "cache_contents": "observed_history_only",
        },
    }
