"""Production-width attention capacity diagnostic on one real cached decision.

Run each length in a separate process. Remaining flags are the full_history
training CLI. This does optimizer updates, but is not a task-learning run.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from scripts import train_vjepa_policy as entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blocks", required=True, type=int)
    parser.add_argument("--updates", type=int, default=3)
    options, training_args = parser.parse_known_args(argv)
    if options.blocks < 1 or options.updates < 2:
        parser.error("blocks must be positive and updates >= 2 (include optimizer state)")
    report = dict(status="starting", history_blocks=options.blocks, updates_requested=options.updates)
    path = Path(options.report)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    def probe(args, model, dataset, collator):
        try:
            decision = (options.blocks - 1) * model.config.history_interval_steps
            index = next(i for i, (_, t) in enumerate(dataset.index) if t == decision)
            cpu = collator([dataset[index]])
            assert cpu["history_valid"].sum().item() == options.blocks
            device = torch.device("cuda")
            model.to(device).train()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cpu.items()}
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
            sampled = {name: p.detach().flatten()[:64].clone() for name, p in model.named_parameters()
                       if p.requires_grad}
            report.update(
                backend=model.predictor.attention_backend, device=torch.cuda.get_device_name(),
                torch_version=torch.__version__, precision="bf16", batch_size=1,
                predictor_depth=len(model.predictor.predictor_blocks),
                predictor_width=model.predictor.predictor_embed.out_features,
                action_width=model.action_expert.action_in_proj.out_features,
                activation_checkpointing_blocks=model.predictor.activation_checkpointing_blocks,
                history_tokens=options.blocks * model.predictor.tokens_per_history_block,
                episode_id=dataset.episodes[dataset.index[index][0]]["episode_id"], decision_step=decision,
                trainable_parameters=sum(p.numel() for p in parameters), steps=[],
            )
            save()
            for step in range(options.updates):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                report["phase"] = "forward"
                save()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = model(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                report["phase"] = "backward"
                save()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                report["phase"] = "optimizer"
                save()
                optimizer.step()
                torch.cuda.synchronize()
                report["steps"].append(dict(
                    step=step + 1, seconds=time.perf_counter() - start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    gradient_norm=float(norm), **{k: float(v) for k, v in metrics.items()},
                ))
                save()
            changed = dict(predictor=0, action_expert=0)
            for name, p in model.named_parameters():
                if name in sampled and not torch.equal(sampled[name], p.detach().flatten()[:64]):
                    changed[name.split(".")[0]] += 1
            assert all(changed.values()), changed
            assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
            report.update(status="passed", phase="complete", changed_parameter_tensors=changed,
                          encoder_frozen=True, peak_allocated_gib=max(s["peak_allocated_gib"] for s in report["steps"]),
                          steady_seconds_per_update=sum(s["seconds"] for s in report["steps"][1:]) / (options.updates - 1))
        finally:
            dataset.close()

    original = entry.run_training
    entry.run_training = probe
    save()
    try:
        entry.main(args=entry.parse_args(["--policy-arch", "full_history", *training_args]))
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="oom", error=str(error), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        entry.run_training = original
        save()
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
