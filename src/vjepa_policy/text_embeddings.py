"""Precompute frozen T5 or BERT token embeddings consumed by the policy dataset."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from vjepa_policy.datasets import DEFAULT_PROMPT, read_instructions
from vjepa_policy.text_cache import text_cache_filename


def read_prompts(dataset_dirs, instruction_field="task"):
    prompts = set()
    for dataset_dir in dataset_dirs:
        for instruction in read_instructions(dataset_dir, instruction_field).values():
            prompts.add(DEFAULT_PROMPT.format(task=instruction))
    return sorted(prompts)


def precompute_text_embeddings(
    dataset_dirs,
    cache_dir,
    model_name,
    context_length,
    device="cuda",
    instruction_field="task",
    text_encoder="t5",
):
    encode_prompts(read_prompts(dataset_dirs, instruction_field), cache_dir, model_name,
                   context_length, device, text_encoder)


def encode_prompts(prompts, cache_dir, model_name, context_length, device="cuda", text_encoder="t5"):
    """Encode complete prompts without pooling or updating the frozen encoder."""
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel

    if text_encoder not in ("t5", "bert"):
        raise ValueError(f"Unsupported text encoder: {text_encoder}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = Path(model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=local)
    model_class = T5EncoderModel if text_encoder == "t5" else AutoModel
    model_options = {"local_files_only": local}
    if text_encoder == "bert":
        model_options["add_pooling_layer"] = False
    encoder = model_class.from_pretrained(model_name, **model_options).to(device).eval()
    encoder.requires_grad_(False)
    if text_encoder == "bert" and (encoder.config.model_type != "bert" or encoder.config.hidden_size != 768):
        raise ValueError("BERT full-history features require bert-base with hidden_size=768")
    embedding_dim = encoder.config.d_model if text_encoder == "t5" else encoder.config.hidden_size

    for prompt in tqdm(
        prompts,
        desc="Encoding task prompts",
    ):
        unpadded_tokens = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_length = unpadded_tokens.input_ids.shape[1]
        if prompt_length > context_length:
            raise ValueError(
                f"Prompt requires {prompt_length} tokens but context_length="
                f"{context_length}; increase --context-length to avoid truncation: {prompt!r}"
            )
        output_path = cache_dir / text_cache_filename(prompt, context_length, text_encoder)
        if output_path.is_file():
            cached = torch.load(output_path, map_location="cpu", weights_only=True)
            context, mask = cached["context"], cached["mask"]
            if (context.shape != (context_length, embedding_dim)
                    or mask.shape != (context_length,) or not torch.isfinite(context).all()
                    or mask.sum().item() != prompt_length):
                raise ValueError(f"Invalid existing text cache entry: {output_path}")
            continue
        tokens = tokenizer(
            prompt,
            max_length=context_length,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            context = encoder(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state[0].cpu()
        temporary = output_path.with_suffix(".tmp")
        torch.save(
            {"context": context, "mask": tokens.attention_mask[0].bool()},
            temporary,
        )
        temporary.replace(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-name", default="google/t5-v1_1-xxl")
    parser.add_argument("--text-encoder", choices=("t5", "bert"), default="t5")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction-field", choices=("task", "remarks"), default="task")
    args = parser.parse_args(argv)
    precompute_text_embeddings(
        args.dataset_dirs,
        args.cache_dir,
        args.model_name,
        args.context_length,
        args.device,
        args.instruction_field,
        args.text_encoder,
    )


if __name__ == "__main__":
    main()
