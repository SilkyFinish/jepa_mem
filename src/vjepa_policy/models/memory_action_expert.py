"""Flow matching over complete history and this decision's temporary future KV."""

import torch

from .action_expert import ActionExpert
from .memory_types import validate_decision_step
from .memory_attention import validity_attention_plan


class _HistoryFutureLayers:
    """Assemble H/F only for the current action layer, retaining autograd.

    Eagerly concatenating all layers holds a second full history copy alive
    while action attention saves its own [H,F,A] tensors for backward.
    """

    def __init__(self, history, future):
        self.history, self.future = history, future

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        h, f = self.history[index], self.future[index]
        return torch.cat((h.key, f.key), dim=2), torch.cat((h.value, f.value), dim=2)


class MemoryActionExpert(ActionExpert):
    def __init__(self, *, rope_options, action_reads_future=True, attention_backend="segmented", **kwargs):
        super().__init__(**kwargs)
        self.rope_options = dict(rope_options)
        self.action_reads_future = bool(action_reads_future)
        self.read_full_predictor_kv = True
        self.attention_backend = attention_backend

    def _memory_context(self, history, future, action_layout):
        validate_decision_step(future.decision_step, action_layout)
        legal_history = history.layout.valid & (
            history.layout.end_step <= future.decision_step[:, None]
        )
        if self.action_reads_future:
            context_valid = torch.cat((legal_history, future.layout.valid), dim=1)
            layers = _HistoryFutureLayers(history.layers, future.layers) if self.attention_backend == "segmented" else tuple(
                (torch.cat((h.key, f.key), dim=2), torch.cat((h.value, f.value), dim=2))
                for h, f in zip(history.layers, future.layers)
            )
        else:
            context_valid = legal_history
            layers = tuple((h.key, h.value) for h in history.layers)
        return layers, dict(
            context_valid=context_valid, action_valid=action_layout.valid,
            action_position_coords=action_layout.rope_coords, rope_options=self.rope_options,
            memory_attention_plan=validity_attention_plan(
                action_layout.valid, torch.cat((context_valid, action_layout.valid), dim=1),
            ) if self.attention_backend == "segmented" else None,
        )

    def forward_memory(self, history, future, actions, language, language_mask, state,
                       action_layout, *, noise=None, time=None):
        layers, options = self._memory_context(history, future, action_layout)
        valid = action_layout.valid[..., None]
        actions = actions.masked_fill(~valid, 0)
        if noise is None:
            noise = torch.randn_like(actions)
        elif noise.shape != actions.shape or noise.dtype != actions.dtype or noise.device != actions.device:
            raise ValueError("training noise must match actions")
        noise = noise.masked_fill(~valid, 0)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        condition, condition_valid = self.conditioner(language, language_mask, state)
        velocity = self._run_blocks(
            self._encode_actions(noisy, valid.expand_as(actions)),
            layers, None, self._time_cond((time * self.num_timestep_buckets).long()),
            condition, condition_valid, **options,
        )
        return velocity, (actions - noise).masked_fill(~valid, 0)

    @torch.no_grad()
    def predict_action_memory(self, history, future, language, language_mask, state,
                              action_layout, *, initial_noise=None, generator=None):
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        layers, options = self._memory_context(history, future, action_layout)
        actions = self._initial_actions(layers[0][0], initial_noise=initial_noise, generator=generator)
        valid = action_layout.valid[..., None]
        actions = actions.masked_fill(~valid, 0)
        condition, condition_valid = self.conditioner(language, language_mask, state)
        step_size = 1.0 / self.num_inference_steps
        # H/F were computed once before this loop. Only the action stream evolves.
        for step in range(self.num_inference_steps):
            timestep = torch.full(
                (actions.shape[0],), int(step / self.num_inference_steps * self.num_timestep_buckets),
                device=actions.device, dtype=torch.long,
            )
            velocity = self._run_blocks(
                self._encode_actions(actions, valid.expand_as(actions)),
                layers, None, self._time_cond(timestep), condition, condition_valid, **options,
            )
            actions = (actions + step_size * velocity).masked_fill(~valid, 0)
        return actions
