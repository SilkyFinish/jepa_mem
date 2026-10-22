"""Optimizer-boundary state for resumable jobs on ephemeral compute nodes."""
import os
import random

import numpy as np
import torch


def atomic_torch_save(payload, path):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def resume_contract(trainer):
    keys = ("batch_size", "gradient_accumulation_steps", "mixed_precision", "lr",
            "weight_decay", "max_grad_norm", "train_seed", "data_seed", "history_bucket_size")
    return {
        "config": {key: trainer.cfg.get(key) for key in keys},
        "world_size": trainer.accelerator.num_processes,
        "dataset_size": trainer.dataset_size, "max_steps": trainer.max_steps,
        "steps_per_epoch": trainer.optimizer_steps_per_epoch,
        "data_provenance": getattr(trainer._dataset, "provenance", None),
    }
