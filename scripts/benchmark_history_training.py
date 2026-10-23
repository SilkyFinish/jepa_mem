"""Measure the real Trainer/DDP path on complete native history samples.

Launch with torchrun, passing normal train_vjepa_policy arguments after the
optional --benchmark-stress-updates N. This is a timing diagnostic, not a
convergence experiment. Reports and diagnostic weights go to --output-dir.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from scripts import train_vjepa_policy as entry
from vjepa_policy import train_common
from vjepa_policy.trainer import Trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--benchmark-stress-updates", type=int, default=0)
    parser.add_argument("--benchmark-stress-mixed-episodes", action="store_true",
                        help="Use the final stress update for longest-prefix mixed-episode fallback batches")
    options, rest = parser.parse_known_args(argv)
    if options.benchmark_stress_updates < 0:
        raise ValueError("stress update count must be nonnegative")
    if options.benchmark_stress_mixed_episodes and options.benchmark_stress_updates < 2:
        raise ValueError("Mixed-episode stress requires at least two stress updates")
    args = entry.parse_args(rest)
    if args.policy_arch != "full_history" or not args.history_feature_cache:
        raise ValueError("Timing requires full_history with a frozen feature cache")
    original_run, original_trainer = entry.run_training, train_common.Trainer

    class TimedTrainer(Trainer):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            if not torch.cuda.is_available():
                raise ValueError("This benchmark requires CUDA")
            self.core = self.accelerator.unwrap_model(self.model)
            self.rank = self.accelerator.process_index
            self.report_path = Path(self.output_dir) / f"rank{self.rank}_timing.json"
            if self.report_path.exists():
                raise FileExistsError("Use a fresh benchmark output directory")
            self.micro, self.steps, self.saves = [], [], []
            self.global_batch = self.cfg["batch_size"] * self.accelerator.num_processes * self.gradient_accumulation_steps
            stress = options.benchmark_stress_updates * self.global_batch
            if stress:
                if options.benchmark_stress_updates > self.optimizer_steps_per_epoch:
                    raise ValueError("Stress updates must fit in the first epoch")
                longest = max(range(len(self._dataset)), key=self._dataset.history_lengths.__getitem__)
                episode, decision = self._dataset.index[longest]
                adjacent = [i for i, (e, t) in enumerate(self._dataset.index) if e == episode and t <= decision]
                adjacent = adjacent[-self.cfg["batch_size"]:]
                window = [adjacent[i % len(adjacent)] for i in range(self.cfg["batch_size"])]
                self._epoch_sampler._order[:stress] = torch.tensor(
                    window * (stress // self.cfg["batch_size"]))
                if options.benchmark_stress_mixed_episodes:
                    longest_by_episode = {}
                    for index, (episode, _) in enumerate(self._dataset.index):
                        previous = longest_by_episode.get(episode)
                        if previous is None or self._dataset.history_lengths[index] > self._dataset.history_lengths[previous]:
                            longest_by_episode[episode] = index
                    if len(longest_by_episode) < 2 or self.cfg["batch_size"] < 2:
                        raise ValueError("Mixed-episode stress needs multiple episodes and per-device batch >= 2")
                    indices = sorted(longest_by_episode.values(), key=self._dataset.history_lengths.__getitem__, reverse=True)
                    self._epoch_sampler._order[stress-self.global_batch:stress] = torch.tensor(
                        [indices[i % len(indices)] for i in range(self.global_batch)])
            self.report = dict(
                diagnostic_only=True, status="running", rank=self.rank,
                model_version=self.core.config.model_version, device=torch.cuda.get_device_name(),
                world_size=self.accelerator.num_processes, per_device_batch=self.cfg["batch_size"],
                accumulation=self.gradient_accumulation_steps, global_batch=self.global_batch,
                precision=self.cfg["mixed_precision"], dataset_size=len(self._dataset),
                episodes=len(self._dataset.episodes), selection=self._dataset.selection,
                optimizer_steps_per_epoch=self.optimizer_steps_per_epoch,
                effective_samples_per_epoch=self.effective_samples_per_epoch,
                history_tokens_per_block=self.core.predictor.tokens_per_history_block,
                future_tokens=self.core.predictor.future_tokens,
                stress_updates=options.benchmark_stress_updates,
                mixed_episode_stress=options.benchmark_stress_mixed_episodes,
                shared_stress_layout="adjacent_decisions_at_longest_prefix",
                dataset_history_lengths=self._dataset.history_lengths,
                steps=self.steps, checkpoint_saves=self.saves,
            )
            def before(module, inputs):
                b = inputs[0]
                self.micro.append(dict(history_lengths=b["history_valid"].sum(1).tolist(),
                                       padded_blocks=b["history_valid"].shape[1],
                                       episode_ids=list(b["episode_id"]),
                                       decision_steps=b["decision_step"].tolist()))
            def after(module, inputs, output):
                loss = output[0].detach()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                self.micro[-1]["loss"] = float(loss)
            self.pre = self.core.register_forward_pre_hook(before)
            self.post = self.core.register_forward_hook(after)
            self.opt_hook = self.optimizer.optimizer.register_step_post_hook(self.stepped)
            self.accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.last_time = time.perf_counter()
            self.save_report()

        def save_report(self):
            temporary = self.report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report, indent=2) + "\n")
            temporary.replace(self.report_path)

        def stepped(self, optimizer, positional, keyword):
            torch.cuda.synchronize()
            now = time.perf_counter()
            current = self.micro[-self.gradient_accumulation_steps:]
            self.steps.append(dict(
                step=self.global_step+1, seconds=now-self.last_time,
                history_lengths=[n for b in current for n in b["history_lengths"]],
                padded_blocks=[b["padded_blocks"] for b in current],
                episode_ids=[b["episode_ids"] for b in current],
                decision_steps=[b["decision_steps"] for b in current],
                loss=sum(b["loss"] for b in current)/len(current),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                stress=self.global_step < options.benchmark_stress_updates,
            ))
            if self.global_step + 1 == self.max_steps:
                self.check_final_parameters()
            self.save_report()
            torch.cuda.reset_peak_memory_stats()
            self.last_time = now

        def _save(self):
            started = time.perf_counter()
            super()._save()
            if hasattr(self, "saves") and self.accelerator.is_main_process:
                self.saves.append(dict(step=self.global_step, seconds=time.perf_counter()-started))

        def check_final_parameters(self):
            # Run while NCCL is still alive, before Trainer.end_training().
            parameters = [p for p in self.core.parameters() if p.requires_grad]
            tensors = parameters + [p.grad for p in parameters if p.grad is not None]
            finite = torch.stack([torch.isfinite(p).all() for p in tensors]).all().int()
            if self.accelerator.num_processes > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise RuntimeError("nonfinite trained parameters or gradients")
            sample = torch.cat([p.detach().flatten()[:8].float() for p in parameters])
            if self.accelerator.num_processes > 1:
                gathered = [torch.empty_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, sample)
                if any(not torch.equal(sample, other) for other in gathered):
                    raise RuntimeError("DDP sampled parameters diverged")
            if any(p.requires_grad or p.grad is not None for p in self.core.encoder.parameters()):
                raise RuntimeError("Encoder must remain frozen")
            self.report.update(all_parameters_finite=True, final_gradients_finite=True,
                               sampled_parameters_synchronized=True, encoder_frozen=True)

        def train(self):
            started = time.perf_counter()
            try:
                super().train()
                if self.global_step != self.max_steps:
                    raise RuntimeError("Timing diagnostic stopped before all requested updates")
                self.report.update(status="passed", actual_updates=self.global_step,
                                   training_seconds_including_save=time.perf_counter()-started,
                                   microbatches=len(self.micro))
            except BaseException as error:
                self.report.update(status="failed", error=repr(error))
                raise
            finally:
                self.pre.remove()
                self.post.remove()
                self.opt_hook.remove()
                self.save_report()

    def run(training_args, model, dataset, collator):
        try:
            original_run(training_args, model, dataset, collator)
        finally:
            dataset.close()

    train_common.Trainer = TimedTrainer
    entry.run_training = run
    try:
        entry.main(args=args)
    finally:
        train_common.Trainer = original_trainer
        entry.run_training = original_run


if __name__ == "__main__":
    main()
