"""Simulate Accelerate's eight-rank batches without allocating GPUs."""
import unittest
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard
from scripts.a1008.common import schedule
from vjepa_policy.trainer import _EpochRandomSampler, _full_accumulation_microbatches


class Data:
    share_history_prefixes = True
    def __init__(self, size):
        lengths = (size // 3, size // 3, size - 2 * (size // 3))
        self.index = [(episode, t) for episode, length in enumerate(lengths) for t in range(length)]
        self.history_lengths = [(t + 3) // 4 + 1 for _, t in self.index]
    def __len__(self):
        return len(self.index)


class EightRankSamplingTests(unittest.TestCase):
    def batches(self, sampler, batch):
        base = BatchSampler(sampler, batch_size=batch, drop_last=True)
        return [list(BatchSamplerShard(base, num_processes=8, process_index=rank,
                                      split_batches=False)) for rank in range(8)]

    def test_actual_sharding_matches_sparse_and_dense_update_budgets(self):
        for size in (1013, 17562):
            for batch in (1, 2):
                with self.subTest(size=size, batch=batch):
                    sampler = _EpochRandomSampler(Data(size), seed=7, batch_size=batch, bucket_size=128)
                    ranks = self.batches(sampler, batch)
                    plan = schedule(size, 8, 25, batch)
                    usable = _full_accumulation_microbatches(len(ranks[0]), plan["accumulation"])
                    self.assertTrue(all(len(row) == len(ranks[0]) for row in ranks))
                    self.assertEqual(usable // plan["accumulation"], plan["updates_per_epoch"])
                    consumed = [index for group in zip(*(row[:usable] for row in ranks))
                                for microbatch in group for index in microbatch]
                    self.assertEqual(consumed, list(sampler)[:plan["effective_samples_per_epoch"]])
                    self.assertEqual(len(set(consumed)), plan["effective_samples_per_epoch"])
                    self.assertEqual(size - len(consumed), 5 if size == 1013 else 10)

    def test_consumed_cursor_reconstructs_every_rank_and_epoch_shuffle_changes(self):
        data = Data(17562)
        sampler = _EpochRandomSampler(data, seed=7, batch_size=2, bucket_size=128)
        original = self.batches(sampler, 2)
        sampler.set_position(17 * 8 * 2)
        resumed = self.batches(sampler, 2)
        for before, after in zip(original, resumed):
            self.assertEqual(after, before[17:])
        sampler.start_epoch(1)
        following = self.batches(sampler, 2)
        self.assertNotEqual(original, following)
        consumed = [i for group in zip(*following) for row in group for i in row]
        self.assertEqual(len(consumed), 17552)
        self.assertEqual(len(set(consumed)), 17552)
