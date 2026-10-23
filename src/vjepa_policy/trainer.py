# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Architecture-agnostic training loop (accelerate + a model that exposes a loss).
# The public training path uses vjepa_policy.models.vjepa_policy.VJEPAPolicy.
# The action-expert branch is enabled purely by `hasattr(model, "action_expert")`
# / the presence of "loss_action" in the returned metrics, so the loop never
# needs to know which architecture it is driving.

import gc
import json
import os
import time
import warnings
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler

from vjepa_policy.stopping import deadline_timestamp, stop_requested
from vjepa_policy.training_state import atomic_torch_save, capture_rng, restore_rng, resume_contract
from vjepa_policy.utils.logging import get_logger

logger = get_logger(__name__, force=True)

try:
    import psutil
except ImportError:  # psutil is optional; memory logging degrades gracefully without it
    psutil = None


def _cgroup_mem():
    """(used_bytes, limit_bytes) for this process's cgroup, or (None, None).

    The training box confines the job to a cgroup memory limit well below total
    system RAM, so the OOM killer fires against THIS number -- track it, not
    `free`. Supports cgroup v2 (memory.current/max) and v1 (usage/limit).
    """
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            used = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        limit = None if raw == "max" else int(raw)
        return used, limit
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            used = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
        return used, (None if limit > (1 << 62) else limit)
    except (OSError, ValueError):
        return None, None


def _tree_rss_bytes(proc):
    """RSS of `proc` plus all of its (data-loader worker) children, in bytes."""
    if psutil is None or proc is None:
        return None
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _full_accumulation_microbatches(microbatches, accumulation_steps):
    if microbatches <= 0 or accumulation_steps <= 0:
        raise ValueError("microbatches and accumulation_steps must be positive")
    return (microbatches // accumulation_steps) * accumulation_steps


class _EpochRandomSampler(Sampler):
    """Randomize once per epoch while allowing a loader iterator to restart.

    Video workers are periodically respawned to reclaim decoder memory.  A
    normal ``shuffle=True`` DataLoader creates a new permutation every time
    its iterator is rebuilt, which would repeat and omit samples inside one
    epoch.  Keeping the permutation and an explicit sample cursor makes the
    worker restart transparent to epoch coverage.
    """

    def __init__(self, data_source, *, seed: int, bucket_size=0, batch_size=1):
        self.data_source = data_source
        self.bucket_size, self.batch_size = bucket_size, batch_size
        self.seed = int(seed)
        self.epoch = -1
        self.position = 0
        self._order = None
        self.start_epoch(0)

    def start_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        self._order = torch.randperm(len(self.data_source), generator=generator)
        lengths = getattr(self.data_source, "history_lengths", None)
        if getattr(self.data_source, "share_history_prefixes", False) and self.batch_size > 1:
            # Every anchor appears exactly once. Random cyclic offsets change
            # adjacent-anchor grouping each epoch; shuffle groups, not rows.
            episodes = {}
            for i, (episode, decision) in enumerate(self.data_source.index):
                episodes.setdefault(episode, []).append(i)
            batches, leftovers = [], []
            for indices in episodes.values():
                shift = int(torch.randint(len(indices), (1,), generator=generator))
                indices = indices[shift:] + indices[:shift]
                stop = len(indices) // self.batch_size * self.batch_size
                batches.extend(indices[i:i+self.batch_size] for i in range(0,stop,self.batch_size))
                leftovers.extend(indices[stop:])
            if leftovers:
                order = torch.randperm(len(leftovers), generator=generator).tolist()
                leftovers = [leftovers[i] for i in order]
                stop = len(leftovers) // self.batch_size * self.batch_size
                batches.extend(leftovers[i:i+self.batch_size] for i in range(0,stop,self.batch_size))
                leftovers = leftovers[stop:]
            order = torch.randperm(len(batches), generator=generator).tolist()
            self._order = torch.tensor([i for b in order for i in batches[b]] + leftovers, dtype=torch.long)
        elif lengths is not None and self.bucket_size:
            # Random pools, locally sorted lengths, then shuffled whole batches.
            # One index per sample; worker recycling retains this same order.
            pool = max(self.batch_size, self.bucket_size // self.batch_size * self.batch_size)
            order = self._order.tolist()
            order = [i for start in range(0, len(order), pool)
                     for i in sorted(order[start:start + pool], key=lambda i: lengths[i])]
            complete = len(order) // self.batch_size * self.batch_size
            batches = [order[i:i + self.batch_size] for i in range(0, complete, self.batch_size)]
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            self._order = torch.tensor(
                [i for b in permutation for i in batches[b]] + order[complete:], dtype=torch.long,
            )
        self.position = 0

    def set_position(self, position: int) -> None:
        position = int(position)
        if not 0 <= position <= len(self.data_source):
            raise ValueError(
                f"sampler position {position} is outside [0, {len(self.data_source)}]"
            )
        self.position = position

    def __iter__(self):
        # Do not advance ``position`` while yielding: DataLoader prefetch can
        # ask for indices that have not reached the training loop yet.  The
        # trainer sets the consumed cursor explicitly before a worker cycle.
        for index in self._order[self.position :]:
            yield int(index)

    def __len__(self):
        # The batch sampler needs the full epoch length when it is prepared by
        # Accelerate, even when the next iterator starts at a non-zero cursor.
        return len(self.data_source)


class Trainer:
    def __init__(self, model, dataset, collator, *, cfg):
        self.cfg = cfg
        self.stop_at = deadline_timestamp(cfg.get("stop_at"))
        self.stop_file = cfg.get("stop_file")
        self.stop_after_step = cfg.get("stop_after_step")
        if self.stop_after_step is not None and (type(self.stop_after_step) is not int or self.stop_after_step <= 0):
            raise ValueError("stop_after_step must be a positive absolute optimizer step")
        self.output_dir = cfg["output_dir"]
        self.max_grad_norm = cfg["max_grad_norm"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            mixed_precision=cfg["mixed_precision"],
            # Predictor keeps all pretrained mask tokens but forward uses only
            # mask_index=1; the others get no gradient, so DDP must allow it.
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    # Share gradients with the communication buckets instead of
                    # keeping a second model-sized copy on each device. Our loop
                    # uses zero_grad(set_to_none=True), never in-place detach_.
                    gradient_as_bucket_view=True,
                )
            ],
        )

        # ---- OOM fix: torchcodec's video decode leaks ~0.4 MB per decoded clip
        # that glibc only returns to the OS when the worker PROCESS exits (verified:
        # worker-tree RSS climbs ~6 MB/step, then drops straight back to baseline
        # the instant the workers are torn down). Left unchecked this fills the
        # ~800 GiB cgroup and can eventually OOM-kill a long run.
        #
        # Fix: persistent_workers=False. The worker pool is torn down and
        # respawned at each epoch boundary, which reclaims the leak. This is the
        # same throughput here as persistent workers
        # (~2 step/s -- the pipeline is decode-bound, not warmth-bound, on this
        # box), so there's no reason to keep persistent workers alive and leaking.
        # (An earlier "keep persistent + periodically rebuild the loader" scheme
        # was abandoned: accelerator.prepare() retains every prepared loader in
        # accelerator._dataloaders, so rebuilding ADDS workers instead of
        # replacing them -- memory climbed instead of resetting.)
        self._dataset = dataset
        data_seed = cfg.get("data_seed")
        data_generator = None
        if data_seed is not None:
            data_generator = torch.Generator().manual_seed(int(data_seed))
        # Keep one random permutation per epoch so a worker-cycle iterator
        # restart resumes the same sample stream instead of silently starting
        # a fresh shuffle permutation.
        sampler_seed = int(data_seed if data_seed is not None else cfg.get("train_seed", 0))
        self._epoch_sampler = _EpochRandomSampler(dataset, seed=sampler_seed,
            bucket_size=cfg.get("history_bucket_size", 0), batch_size=cfg["batch_size"])
        self._loader_kwargs = dict(
            batch_size=cfg["batch_size"],
            shuffle=False,
            sampler=self._epoch_sampler,
            generator=data_generator,
            num_workers=cfg["num_workers"],
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=cfg.get("prefetch_factor", 4) if cfg["num_workers"] > 0 else None,
        )
        self.recycle_every = cfg.get("recycle_workers_every", 2000)
        base_loader = DataLoader(dataset, **self._loader_kwargs)
        self.dataset_size = len(dataset)
        self.gradient_accumulation_steps = cfg["gradient_accumulation_steps"]

        # Encoder is frozen; the predictor (+ FiLM) and, if present, the action
        # expert train together (one optimizer, one combined loss).
        trainable_params = cfg.get("optimizer_param_groups")
        if trainable_params is None:
            trainable_params = list(model.predictor.parameters())
            if hasattr(model, "action_expert"):
                trainable_params += list(model.action_expert.parameters())
            if hasattr(model, "proprio_encoder"):
                trainable_params += list(model.proprio_encoder.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
        )

        estimated_microbatches_per_rank = len(base_loader) // self.accelerator.num_processes
        estimated_usable_microbatches = _full_accumulation_microbatches(
            estimated_microbatches_per_rank, self.gradient_accumulation_steps
        )
        base_steps_per_epoch = max(
            estimated_usable_microbatches // self.gradient_accumulation_steps, 1
        )
        self.max_steps = cfg["max_steps"] or base_steps_per_epoch * cfg["num_epochs"]
        # accelerate's AcceleratedScheduler (built by accelerator.prepare below)
        # steps the wrapped scheduler `num_processes` times per external
        # `.step()` call (it's calibrated for a schedule length measured against
        # the full un-sharded dataset, not against `self.global_step`, which
        # only advances once per synchronized multi-process step). To make the
        # schedule complete exactly once over `self.max_steps` real steps (not
        # `num_processes`x too fast, wrapping into repeated bogus cosine
        # cycles), the constructor args must be scaled UP by num_processes so
        # that total INTERNAL steps (= real_steps * num_processes) match.
        sched_steps = self.max_steps * self.accelerator.num_processes
        sched_warmup = int(0.05 * self.max_steps) * self.accelerator.num_processes
        scheduler = self._build_scheduler(optimizer, sched_steps, warmup=sched_warmup)

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model, optimizer, scheduler
        )
        self.loader = self.accelerator.prepare(base_loader)
        self.prepared_microbatches_per_epoch = len(self.loader)
        self.usable_microbatches_per_epoch = _full_accumulation_microbatches(
            self.prepared_microbatches_per_epoch, self.gradient_accumulation_steps
        )
        if self.usable_microbatches_per_epoch <= 0:
            raise ValueError(
                "gradient_accumulation_steps exceeds prepared microbatches per epoch"
            )
        self.dropped_microbatches_per_epoch = (
            self.prepared_microbatches_per_epoch - self.usable_microbatches_per_epoch
        )
        prepared_steps_per_epoch = (
            self.usable_microbatches_per_epoch // self.gradient_accumulation_steps
        )
        self.optimizer_steps_per_epoch = prepared_steps_per_epoch
        self.effective_samples_per_epoch = (
            prepared_steps_per_epoch
            * cfg["batch_size"]
            * self.accelerator.num_processes
            * self.gradient_accumulation_steps
        )

        self.log_every = cfg["log_every"]
        self.save_every = cfg["save_every"]
        self.global_step = 0
        self._microbatches_in_epoch = 0
        self._resume_rng = None
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        self._peak_tree_rss = 0
        self._maybe_resume(cfg.get("resume_from"))
        self.wandb_run = self._init_wandb(cfg, prepared_steps_per_epoch)
        logger.info(
            "Trainer ready: max_steps=%d optimizer_steps_per_epoch=%d "
            "prepared_microbatches_per_epoch=%d dropped_microbatches_per_epoch=%d "
            "effective_samples_per_epoch=%d base_batches_per_epoch=%d start_step=%d "
            "train_seed=%s data_seed=%s",
            self.max_steps,
            prepared_steps_per_epoch,
            self.prepared_microbatches_per_epoch,
            self.dropped_microbatches_per_epoch,
            self.effective_samples_per_epoch,
            len(base_loader),
            self.global_step,
            cfg.get("train_seed"),
            cfg.get("data_seed"),
        )

    def _maybe_resume(self, resume_from):
        """Restore full optimizer-boundary state, or load a legacy weight export.

        Weight-only exports retain the old schedule-fast-forward behavior; they
        are warm restarts, not exact continuation of the original optimizer.
        """
        if not resume_from:
            return
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False, mmap=True)
        state = ckpt.get("training_state")
        if state is not None and (state.get("version") != 1 or state["contract"] != resume_contract(self)):
            raise ValueError("Full training resume requires the same data, world size, batch, seeds and schedule")
        if self.cfg["mixed_precision"] == "no":
            if ckpt.get("train_precision") != "no" or ckpt.get("allow_tf32") is not False:
                raise ValueError(
                    "Strict FP32 training can only resume a strict FP32 checkpoint "
                    f"(train_precision={ckpt.get('train_precision')!r}, "
                    f"allow_tf32={ckpt.get('allow_tf32')!r}): {resume_from}"
                )
        core = self.accelerator.unwrap_model(self.model)
        from vjepa_policy.policy_arch import load_policy_weights
        load_policy_weights(core, ckpt)
        logger.info("[resume] architecture validated; trainable weights loaded strictly")
        self.global_step = int(ckpt.get("step", 0))
        if state is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if self.accelerator.scaler is not None:
                self.accelerator.scaler.load_state_dict(state["scaler"])
            self._epoch_sampler.start_epoch(state["epoch"])
            self._microbatches_in_epoch = state["microbatches_in_epoch"]
            if self._microbatches_in_epoch >= self.usable_microbatches_per_epoch:
                self._epoch_sampler.start_epoch(state["epoch"] + 1)
                self._microbatches_in_epoch = 0
            self._epoch_sampler.set_position(self._microbatches_in_epoch *
                                            self.accelerator.num_processes * self.cfg["batch_size"])
            self._resume_rng = state["rank_rng"][self.accelerator.process_index]
            logger.info("[resume] full state step=%d epoch=%d microbatch=%d",
                        self.global_step, self._epoch_sampler.epoch, self._microbatches_in_epoch)
            return
        # Replay the scheduler's external .step() calls so the LR resumes at the
        # right point (AcceleratedScheduler handles the internal num_processes
        # scaling per external step, matching the live training loop).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()" is expected here
            for _ in range(self.global_step):
                self.scheduler.step()
        logger.info("[resume] resumed from %s at step=%d, lr=%.2e",
                    resume_from, self.global_step, self.optimizer.param_groups[0]["lr"])

    def _mem_report(self):
        """One-line memory snapshot (main process only). Tracks the cgroup usage
        the OOM killer actually watches, plus this rank's own process-tree RSS."""
        parts = []
        used, limit = _cgroup_mem()
        if used is not None:
            gb = used / 1024**3
            if limit:
                parts.append(f"cgroup={gb:.1f}/{limit / 1024**3:.0f}GiB ({100 * used / limit:.0f}%)")
            else:
                parts.append(f"cgroup={gb:.1f}GiB")
        rss = _tree_rss_bytes(self._proc)
        if rss is not None:
            self._peak_tree_rss = max(self._peak_tree_rss, rss)
            parts.append(f"rank0_tree_rss={rss / 1024**3:.1f}GiB (peak {self._peak_tree_rss / 1024**3:.1f})")
        if torch.cuda.is_available():
            parts.append(f"gpu0={torch.cuda.max_memory_allocated() / 1024**3:.1f}GiB")
        return " | ".join(parts)

    def _init_wandb(self, cfg, steps_per_epoch):
        if not self.accelerator.is_main_process:
            return None
        import wandb
        run = wandb.init(
            project=cfg.get("wandb_project", "vjepa_policy"),
            name=cfg.get("wandb_name") or os.path.basename(self.output_dir.rstrip("/")),
            dir=self.output_dir,
            mode="offline",  # offline run; sync later with `wandb sync`
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
        )
        logger.info("wandb offline run at %s", run.dir)
        return run

    @staticmethod
    def _build_scheduler(optimizer, total_steps, warmup):
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup, 1))
        if warmup <= 0:
            return cosine
        warm = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup)
        return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])

    def _save(self):
        rank_rng = None
        if self.cfg.get("save_training_state"):
            local_rng = capture_rng()
            if torch.distributed.is_initialized():
                rank_rng = [None] * self.accelerator.num_processes
                torch.distributed.all_gather_object(rank_rng, local_rng)
            else:
                rank_rng = [local_rng]
        if not self.accelerator.is_main_process:
            if self.cfg.get("save_training_state"):
                self.accelerator.wait_for_everyone()
            return
        core = self.accelerator.unwrap_model(self.model)
        path = os.path.join(self.output_dir, f"checkpoint_step{self.global_step:06d}.pt")
        predictor_block = core.predictor.predictor_blocks[0]
        predictor_attention = getattr(
            predictor_block, "attn", getattr(predictor_block, "self_attn", None)
        )
        encoder_spec = getattr(core, "encoder_spec", None)
        if encoder_spec is None:
            spec = getattr(core.encoder, "spec", None)
            encoder_spec = spec.to_dict() if spec is not None else None
        encoder_blocks = getattr(core.encoder, "blocks", None)
        encoder_attention = None
        if encoder_blocks is not None and len(encoder_blocks) > 0:
            encoder_attention = getattr(encoder_blocks[0], "attn", None)
        latent_layout = getattr(core, "latent_layout", None)
        encoder_interpolate_rope = getattr(encoder_attention, "interpolate_rope", None)
        if encoder_interpolate_rope is None and isinstance(encoder_spec, dict):
            encoder_interpolate_rope = encoder_spec.get("interpolate_rope")
        if encoder_interpolate_rope is None:
            encoder_interpolate_rope = True
        # Encoder is frozen; save only the trained predictor (+ action expert if present).
        payload = {
            "predictor": core.predictor.state_dict(),
            "step": self.global_step,
            "train_precision": self.cfg["mixed_precision"],
            "train_seed": self.cfg.get("train_seed"),
            "data_seed": self.cfg.get("data_seed"),
            "max_steps": self.max_steps,
            "per_device_batch_size": self.cfg["batch_size"],
            "gradient_accumulation_steps": self.cfg["gradient_accumulation_steps"],
            "world_size": self.accelerator.num_processes,
            "dataset_size": self.dataset_size,
            "num_epochs": self.cfg["num_epochs"],
            "global_batch_size": (
                self.cfg["batch_size"]
                * self.accelerator.num_processes
                * self.gradient_accumulation_steps
            ),
            "prepared_microbatches_per_epoch": self.prepared_microbatches_per_epoch,
            "dropped_microbatches_per_epoch": self.dropped_microbatches_per_epoch,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "effective_samples_per_epoch": self.effective_samples_per_epoch,
            "drop_incomplete_accumulation": True,
            "model_topology": {
                "predictor_class": type(core.predictor).__name__,
                "predictor_depth": (
                    len(core.predictor.predictor_blocks)
                    if hasattr(core.predictor, "predictor_blocks")
                    else None
                ),
                "predictor_embed_dim": (
                    int(core.predictor.predictor_embed_dim)
                    if hasattr(core.predictor, "predictor_embed_dim")
                    else int(core.predictor.predictor_embed.out_features)
                    if hasattr(core.predictor, "predictor_embed")
                    and hasattr(core.predictor.predictor_embed, "out_features")
                    else None
                ),
                "predictor_num_heads": (
                    int(core.predictor.predictor_blocks[0].attn.num_heads)
                    if hasattr(core.predictor.predictor_blocks[0], "attn")
                    else int(core.predictor.predictor_blocks[0].self_attn.num_heads)
                    if hasattr(core.predictor, "predictor_blocks")
                    and len(core.predictor.predictor_blocks) > 0
                    else None
                ),
                "predictor_image_height": (
                    int(core.predictor.img_height)
                    if hasattr(core.predictor, "img_height")
                    else None
                ),
                "predictor_image_width": (
                    int(core.predictor.img_width)
                    if hasattr(core.predictor, "img_width")
                    else None
                ),
                "predictor_num_views": int(
                    getattr(core.predictor, "num_views", 1)
                ),
                "predictor_max_views": int(
                    getattr(
                        core.predictor,
                        "max_views",
                        getattr(core.predictor, "num_views", 1),
                    )
                ),
                "latent_grid": (
                    list(getattr(core.predictor, "latent_grid"))
                    if getattr(core.predictor, "latent_grid", None) is not None
                    else None
                ),
                "latent_layout": (
                    latent_layout.to_dict() if latent_layout is not None else None
                ),
                "view_layout": getattr(core, "view_layout", None),
                "camera_keys": list(getattr(core, "camera_keys", ())),
                "encoder": getattr(core, "encoder_name", None),
                "encoder_spec": encoder_spec,
                "encoder_family": getattr(core, "encoder_family", "vjepa2"),
                "encoder_model_name": getattr(core, "encoder_model_name", None),
                "encoder_checkpoint_key": getattr(
                    core, "encoder_checkpoint_key", None
                ),
                "action_indices": (
                    list(core.action_indices)
                    if getattr(core, "action_indices", None) is not None
                    else None
                ),
                "state_indices": (
                    list(core.state_indices)
                    if getattr(core, "state_indices", None) is not None
                    else None
                ),
                "relative_action_indices": (
                    list(core.relative_action_indices)
                    if getattr(core, "relative_action_indices", None) is not None
                    else None
                ),
                "instruction_field": getattr(core, "instruction_field", "task"),
                "proprio_encoding": getattr(core, "proprio_encoding", "identity"),
                "raw_proprio_dim": getattr(core, "raw_proprio_dim", None),
                "encoded_proprio_dim": getattr(core, "encoded_proprio_dim", None),
                "max_state_dim": getattr(core, "max_state_dim", None) or 0,
                "packed_proprio_dim": getattr(
                    core,
                    "packed_proprio_dim",
                    getattr(core.predictor, "proprio_dim", None),
                ),
                "predictor_interpolate_rope": bool(
                    getattr(predictor_attention, "interpolate_rope", False)
                ),
                "predictor_rope_frequency_pairing": (
                    "corrected"
                    if getattr(
                        predictor_attention, "corrected_frequency_pairing", False
                    )
                    else "legacy"
                ),
                "encoder_interpolate_rope": bool(encoder_interpolate_rope),
                "encoder_rope_frequency_pairing": (
                    "corrected"
                    if getattr(
                        encoder_attention, "corrected_frequency_pairing", False
                    )
                    else "legacy"
                ),
                "activation_checkpointing_blocks": int(
                    getattr(core.predictor, "activation_checkpointing_blocks", 0)
                ),
                "action_expert_class": (
                    type(core.action_expert).__name__ if hasattr(core, "action_expert") else None
                ),
                "action_hidden_size": (
                    int(core.action_expert.action_in_proj.out_features)
                    if hasattr(core, "action_expert") else None
                ),
                "action_dim": (
                    int(core.action_expert.action_dim)
                    if hasattr(core, "action_expert") else None
                ),
                "action_chunk_size": (
                    int(core.action_expert.action_chunk_size)
                    if hasattr(core, "action_expert") else None
                ),
                "action_num_layers": (
                    int(core.action_expert.num_layers)
                    if hasattr(core, "action_expert") else None
                ),
                "action_num_heads": (
                    int(core.action_expert.num_heads)
                    if hasattr(core, "action_expert")
                    and hasattr(core.action_expert, "num_heads")
                    else None
                ),
                "action_head_dim": (
                    int(core.action_expert.head_dim)
                    if hasattr(core, "action_expert")
                    and hasattr(core.action_expert, "head_dim")
                    else None
                ),
                "clean_context_attention": bool(
                    getattr(core.predictor, "use_clean_context_attention", False)
                ),
                "read_full_predictor_kv": bool(
                    getattr(getattr(core, "action_expert", None), "read_full_predictor_kv", False)
                ),
            },
            "allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            ),
        }
        if hasattr(core, "action_expert"):
            payload["action_expert"] = core.action_expert.state_dict()
        if hasattr(core, "proprio_encoder"):
            payload["proprio_encoder"] = core.proprio_encoder.state_dict()
        if hasattr(core, "policy_serving"):
            payload["policy_serving"] = core.policy_serving
        if hasattr(core, "predictor_initialization"):
            payload["predictor_initialization"] = core.predictor_initialization
        from vjepa_policy.memory_config import FullHistoryV0Config
        if isinstance(getattr(core, "config", None), FullHistoryV0Config):
            from vjepa_policy.policy_arch import memory_checkpoint_metadata
            payload.update(memory_checkpoint_metadata(core))
        if self.cfg.get("training_stage"):
            payload["training_stage"] = self.cfg["training_stage"]
        atomic_torch_save(payload, path)
        if rank_rng is not None:
            payload["training_state"] = {
                "version": 1, "contract": resume_contract(self),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.accelerator.scaler.state_dict() if self.accelerator.scaler else None,
                "epoch": self._epoch_sampler.epoch,
                "microbatches_in_epoch": self._microbatches_in_epoch,
                "rank_rng": rank_rng,
            }
            atomic_torch_save(payload, os.path.join(self.output_dir, "resume_state.pt"))
            self.accelerator.wait_for_everyone()
        keep = self.cfg.get("keep_last_checkpoints", 0)
        if keep > 0:
            paths = sorted(Path(self.output_dir).glob("checkpoint_step*.pt"))
            protected = {f"checkpoint_step{s:06d}.pt" for s in self.cfg.get("checkpoint_milestones", [])}
            for old in paths[:-keep]:
                if old.name not in protected:
                    old.unlink()
        logger.info("[ckpt] saved %s", path)

    def _stop_requested(self):
        if self.stop_at is None and not self.stop_file and self.stop_after_step is None:
            return False
        local = (self.stop_after_step is not None and self.global_step >= self.stop_after_step)
        local = local or stop_requested(self.stop_at, self.stop_file)
        flag = torch.tensor([int(local)], device=self.accelerator.device)
        # Every rank takes the same branch, even across a deadline boundary.
        return bool(self.accelerator.reduce(flag, reduction="sum").item())

    def train(self):
        core = self.accelerator.unwrap_model(self.model)
        core.encoder.eval()
        core.predictor.train()
        if hasattr(core, "action_expert"):
            core.action_expert.train()
        if hasattr(core, "proprio_encoder"):
            core.proprio_encoder.train()

        if self.cfg["mixed_precision"] == "no":
            non_fp32_parameters = {
                name: parameter.dtype
                for name, parameter in core.named_parameters()
                if parameter.is_floating_point() and parameter.dtype != torch.float32
            }
            if non_fp32_parameters:
                raise RuntimeError(
                    "Strict FP32 requested but model has non-FP32 parameters: "
                    f"{non_fp32_parameters}"
                )

        start = time.perf_counter()
        start_step = self.global_step
        data_iter = iter(self.loader)
        if self._resume_rng is not None:
            restore_rng(self._resume_rng)
            self._resume_rng = None
        microbatches_in_epoch = self._microbatches_in_epoch
        steps_since_recycle = 0
        precision_verified = False
        stopped = self._stop_requested()
        while self.global_step < self.max_steps and not stopped:
            if microbatches_in_epoch >= self.usable_microbatches_per_epoch:
                del data_iter
                gc.collect()
                self._epoch_sampler.start_epoch(self._epoch_sampler.epoch + 1)
                data_iter = iter(self.loader)
                microbatches_in_epoch = 0
                steps_since_recycle = 0
            try:
                batch = next(data_iter)
            except StopIteration:
                raise RuntimeError(
                    "prepared loader ended before the full accumulation windows "
                    f"({microbatches_in_epoch}/{self.usable_microbatches_per_epoch})"
                )
            microbatches_in_epoch += 1
            self._microbatches_in_epoch = microbatches_in_epoch
            if not precision_verified and self.cfg["mixed_precision"] == "no":
                pending = [("batch", batch)]
                non_fp32_inputs = {}
                while pending:
                    name, value = pending.pop()
                    if torch.is_tensor(value):
                        if value.is_floating_point() and value.dtype != torch.float32:
                            non_fp32_inputs[name] = value.dtype
                    elif isinstance(value, (tuple, list)):
                        pending.extend(
                            (f"{name}[{index}]", item) for index, item in enumerate(value)
                        )
                    elif isinstance(value, dict):
                        pending.extend((f"{name}.{key}", item) for key, item in value.items())
                if non_fp32_inputs:
                    raise RuntimeError(
                        "Strict FP32 requested but batch has non-FP32 floating tensors: "
                        f"{non_fp32_inputs}"
                    )
            with self.accelerator.accumulate(self.model):
                loss, metrics = self.model(batch)
                # Check every rank locally. A collective here would force long-
                # history stragglers to synchronize even during no_sync accumulation.
                # torchrun propagates a failing rank's exit to its peers.
                if not torch.isfinite(loss.detach()).item():
                    raise FloatingPointError("Nonfinite training loss")
                if not precision_verified and self.cfg["mixed_precision"] == "no":
                    dtypes = getattr(core, "forward_dtypes", {})
                    non_fp32 = {name: dtype for name, dtype in dtypes.items() if dtype != torch.float32}
                    if non_fp32:
                        raise RuntimeError(f"Strict FP32 forward check failed: {non_fp32}")
                    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                        raise RuntimeError("Strict FP32 requested but TF32 is enabled")
                    precision_verified = True
                    if self.accelerator.is_main_process:
                        logger.info("[precision] verified strict FP32 forward: %s", dtypes)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError("Nonfinite gradient norm")
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    steps_since_recycle += 1

                    if self.global_step % self.log_every == 0:
                        # gather() is a collective: every rank must call it, then log on main only.
                        g = self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        c = self.accelerator.gather(metrics["cos_sim"].float().reshape(1)).mean().item()
                        has_action = "loss_action" in metrics
                        if has_action:
                            gw = self.accelerator.gather(metrics["loss_world"].float().reshape(1)).mean().item()
                            ga = self.accelerator.gather(metrics["loss_action"].float().reshape(1)).mean().item()
                        has_film_diagnostics = hasattr(core, "film_gamma_abs") and hasattr(
                            core, "film_beta_abs"
                        )
                        if has_film_diagnostics:
                            gamma = core.film_gamma_abs.float().cpu()
                            beta = core.film_beta_abs.float().cpu()
                        if self.accelerator.is_main_process:
                            speed = (self.global_step - start_step) / max(time.perf_counter() - start, 1e-6)
                            progress = dict(step=self.global_step, max_steps=self.max_steps,
                                            epoch=self.global_step / self.optimizer_steps_per_epoch,
                                            seconds_per_step=1 / speed,
                                            eta_seconds=(self.max_steps - self.global_step) / speed,
                                            loss=g, grad_norm=float(grad_norm), updated_at=time.time())
                            progress_path = Path(self.output_dir) / "progress.json"
                            progress_path.with_suffix(".tmp").write_text(json.dumps(progress, indent=2) + "\n")
                            progress_path.with_suffix(".tmp").replace(progress_path)
                            extra = f" (world={gw:.4f} action={ga:.4f})" if has_action else ""
                            message = (
                                "step=%d/%d loss=%.4f%s cos_sim=%.4f grad_norm=%.3f "
                                "lr=%.2e %.2f step/s"
                            )
                            values = [
                                self.global_step,
                                self.max_steps,
                                g,
                                extra,
                                c,
                                float(grad_norm),
                                self.optimizer.param_groups[0]["lr"],
                                speed,
                            ]
                            if has_film_diagnostics:
                                message += " | FiLM |gamma|=%.4f |beta|=%.4f"
                                values.extend([gamma.mean().item(), beta.mean().item()])
                            logger.info(message, *values)
                            if has_film_diagnostics:
                                logger.info(
                                    "  FiLM per-layer |gamma|: [%s]",
                                    ", ".join(f"{v:.3f}" for v in gamma.tolist()),
                                )
                                logger.info(
                                    "  FiLM per-layer |beta| : [%s]",
                                    ", ".join(f"{v:.3f}" for v in beta.tolist()),
                                )
                            logger.info("  mem: %s", self._mem_report())
                            if self.wandb_run is not None:
                                payload = {
                                    "train/loss": g, "train/cos_sim": c,
                                    "train/grad_norm": float(grad_norm),
                                    "train/lr": self.optimizer.param_groups[0]["lr"],
                                }
                                if has_film_diagnostics:
                                    payload.update(
                                        {
                                            "film/gamma_abs_mean": gamma.mean().item(),
                                            "film/beta_abs_mean": beta.mean().item(),
                                        }
                                    )
                                if has_action:
                                    payload["train/loss_world"] = gw
                                    payload["train/loss_action"] = ga
                                cg_used, cg_lim = _cgroup_mem()
                                if cg_used is not None:
                                    payload["mem/cgroup_used_gib"] = cg_used / 1024**3
                                    if cg_lim:
                                        payload["mem/cgroup_frac"] = cg_used / cg_lim
                                _rss = _tree_rss_bytes(self._proc)
                                if _rss is not None:
                                    payload["mem/rank0_tree_rss_gib"] = _rss / 1024**3
                                if has_film_diagnostics:
                                    for i in range(gamma.numel()):
                                        payload[f"film/gamma_abs_L{i:02d}"] = gamma[i].item()
                                        payload[f"film/beta_abs_L{i:02d}"] = beta[i].item()
                                self.wandb_run.log(payload, step=self.global_step)
                    if ((self.save_every > 0 and self.global_step % self.save_every == 0)
                            or self.global_step in self.cfg.get("checkpoint_milestones", [])):
                        self._save()
                    stopped = self._stop_requested()
                    if stopped and self.accelerator.is_main_process:
                        logger.info("[stop] saving at optimizer step %d", self.global_step)
                    if self.global_step >= self.max_steps or stopped:
                        break
            # Recreating the iterator on the same prepared DataLoader tears down
            # non-persistent workers without retaining another loader in Accelerate.
            if self.recycle_every > 0 and steps_since_recycle >= self.recycle_every \
                    and self.global_step < self.max_steps:
                del data_iter
                gc.collect()
                batches_per_global_step = (
                    1
                    if getattr(self.accelerator, "split_batches", False)
                    else self.accelerator.num_processes
                )
                self._epoch_sampler.set_position(
                    microbatches_in_epoch
                    * batches_per_global_step
                    * self.cfg["batch_size"]
                )
                data_iter = iter(self.loader)
                steps_since_recycle = 0
                if self.accelerator.is_main_process:
                    logger.info("[worker-cycle] respawned data workers @ step %d | %s",
                                self.global_step, self._mem_report())
        # Release nonpersistent decoder workers before distributed teardown.
        del data_iter
        gc.collect()
        self._save()
        if self.accelerator.is_main_process:
            result = {
                "completed": self.global_step >= self.max_steps,
                "stop_reason": ("max_steps" if self.global_step >= self.max_steps else
                                "stage_boundary" if self.stop_after_step is not None and self.global_step >= self.stop_after_step
                                else "stop_requested"),
                "step": self.global_step, "max_steps": self.max_steps,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "checkpoint": f"checkpoint_step{self.global_step:06d}.pt",
                "optimizer_state_saved": bool(self.cfg.get("save_training_state")),
                "resume_state": "resume_state.pt" if self.cfg.get("save_training_state") else None,
            }
            result_path = os.path.join(self.output_dir, "training_result.json")
            temporary = result_path + ".tmp"
            with open(temporary, "w") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, result_path)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        # All ranks must finish checkpoint/log writes before either exits NCCL.
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        logger.info("[done] training finished at step=%d", self.global_step)
