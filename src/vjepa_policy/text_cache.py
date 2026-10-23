"""Prompt-indexed filenames shared by frozen text encoders and cache readers."""
import hashlib


def text_cache_filename(prompt, context_length, text_encoder="t5"):
    if text_encoder not in ("t5", "bert"):
        raise ValueError(f"Unsupported text encoder: {text_encoder}")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{digest}.{text_encoder}_len{context_length}.pt"
