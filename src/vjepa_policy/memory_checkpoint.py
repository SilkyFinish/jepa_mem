"""Rebuild and strictly load an M3 recompute policy; no serving/session cache."""
from types import SimpleNamespace

import torch

from vjepa_policy.encoders import build_encoder
from vjepa_policy.history_features import encoder_signature
from vjepa_policy.models.memory_policy import build_full_history_policy
from vjepa_policy.models.memory_predictor import MemoryWorldModelPredictor
from vjepa_policy.policy_arch import require_checkpoint_arch, load_policy_weights


def load_full_history_checkpoint(checkpoint_path, encoder_checkpoint, *, device="cpu", attention_backend="segmented"):
    """Restore normalized-action inference from checkpoint metadata and frozen weights.

    Input normalization still uses the saved dataset_stats.json. This loader
    consumes complete v0 observation batches; M4 will add an online session API.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    contract = require_checkpoint_arch(checkpoint, "full_history")
    saved = checkpoint.get("policy_serving", {}).get("config")
    if not isinstance(saved, dict):
        raise ValueError("Full-history checkpoint is missing reconstruction metadata")
    signature = encoder_signature(SimpleNamespace(
        checkpoint=encoder_checkpoint, encoder_checkpoint_key=saved["encoder_checkpoint_key"],
        encoder_interpolate_rope=saved["encoder_interpolate_rope"],
    ))
    if checkpoint.get("memory_training", {}).get("frozen_encoder") != signature:
        raise ValueError("Frozen encoder checkpoint/config differs from training")
    encoder = build_encoder(
        contract.encoder, checkpoint=encoder_checkpoint, image_size=contract.image_size,
        video_frames=len(contract.target_offsets), tubelet_size=contract.tubelet_size,
        interpolate_rope=saved["encoder_interpolate_rope"],
        checkpoint_key=saved["encoder_checkpoint_key"],
    )
    predictor = MemoryWorldModelPredictor(
        img_size=contract.image_size, patch_size=contract.patch_size,
        num_frames=len(contract.target_offsets), tubelet_size=contract.tubelet_size,
        embed_dim=encoder.embed_dim, predictor_embed_dim=saved["pred_embed_dim"],
        depth=saved["pred_depth"], num_heads=saved["pred_num_heads"],
        num_mask_tokens=saved["num_mask_tokens"], lang_dim=contract.text_dim,
        proprio_dim=contract.packed_state_dim, num_views=len(contract.cameras),
        max_views=len(contract.cameras), interpolate_rope=True,
        canonical_spatial_grid=encoder.spec.canonical_spatial_grid,
        corrected_rope_frequency_pairing=saved["corrected_predictor_rope"],
        activation_checkpointing_blocks=0,
        attention_backend=attention_backend,
        history_spatial_pool=contract.history_pool_size,
    )
    semantics = checkpoint["memory_training"]["model_semantics"]
    model = build_full_history_policy(
        encoder, predictor, config=contract, action_hidden_size=saved["action_hidden_size"],
        condition_num_heads=saved["condition_num_heads"],
        action_num_inference_steps=saved["action_num_inference_steps"],
        world_loss_weight=semantics["world_loss_weight"],
        action_loss_weight=semantics["action_loss_weight"],
    )
    model.frozen_encoder_signature = signature
    model.policy_serving = checkpoint["policy_serving"]
    model.training_data_provenance = checkpoint["memory_training"].get("data")
    load_policy_weights(model, checkpoint)
    return model.to(device).eval()
