"""Short-context V-JEPA Policy serving for native RMBench joint commands."""
from contextlib import nullcontext
import numpy as np
import torch
from vjepa_policy.datasets.rmbench import CAMERAS, decode_joint_actions
from vjepa_policy.models.vjepa_policy import encode_video_views
from vjepa_policy.policy_serving.libero import PolicyServingConfig, VJEPAPolicyServing
from vjepa_policy.state import prepare_proprio_state

class RMBenchPolicyServing(VJEPAPolicyServing):
    def __init__(self, ckpt_path, pretrained_encoder, text_cache_dir, dataset_stats=None, device="cuda"):
        config = PolicyServingConfig.from_checkpoint(ckpt_path, overrides={"dataset_stats":dataset_stats})
        expected = tuple("observation.images." + c for c in CAMERAS)
        if config.action_dim != 14 or config.proprio_dim != 14 or config.camera_keys != expected:
            raise ValueError("Checkpoint is not the three-camera RMBench joint policy")
        super().__init__(ckpt_path, pretrained_encoder, text_cache_dir, config=config, device=device)
        if self.action_normalizer.metadata.get("action_contract") != "aloha_absolute_joint_14":
            raise ValueError("RMBench normalization contract missing")

    @torch.no_grad()
    def infer_batch(self, observations):
        if not observations:
            return []
        prepared = [self._prepare_observation(obs) for obs in observations]
        clips = torch.stack([x[0] for x in prepared]).to(self.device)
        language = torch.cat([x[1] for x in prepared])
        language_mask = torch.cat([x[2] for x in prepared])
        state = torch.cat([x[3] for x in prepared])
        state = self.action_normalizer({"observation.state":state})["observation.state"].to(self.device)
        state = prepare_proprio_state(state, raw_state_dim=self.raw_proprio_dim,
                                      encoding=self.proprio_encoding,max_state_dim=self.max_state_dim)
        m_enc = self.m_enc.expand(len(observations),-1)
        m_pred = self.m_pred.expand(len(observations),-1)
        precision = torch.autocast(device_type=self.device.type,dtype=self.dtype) if self.precision != "fp32" else nullcontext()
        with precision:
            latent = encode_video_views(self.encoder,clips,m_enc)
            _, kv = self.predictor(latent,m_enc,m_pred,language=language,language_mask=language_mask,proprio=state)
            actions = self.action_expert.predict_action(kv,language,language_mask,state,context_len=self.context_len)
        actions = self.action_normalizer.unnormalize_action(actions.float().cpu()).numpy()
        return [{"actions":decode_joint_actions(action)} for action in actions]
