"""Checks for the deterministic shard-based Non-IID partition used in Month 3."""

from __future__ import annotations

import unittest

import torch

from clients.noniid_partition import sorted_label_shard_partition_indices


class NonIidPartitionTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self) -> None:
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        first = sorted_label_shard_partition_indices(labels, 4, 1, 42)
        second = sorted_label_shard_partition_indices(labels, 4, 1, 42)
        self.assertEqual(first, second)

        flattened = [position for positions in first.values() for position in positions]
        self.assertEqual(len(flattened), 12)
        self.assertEqual(set(flattened), set(range(12)))

    def test_shards_concentrate_labels_per_client(self) -> None:
        # 4 classes, 20 examples each, 4 clients x 1 shard -> each client's
        # shard should be a contiguous block of the label-sorted sequence,
        # so it is dominated by very few labels instead of being mixed.
        labels = torch.cat([torch.full((20,), value) for value in range(4)])
        partitions = sorted_label_shard_partition_indices(labels, 4, 1, 7)
        for positions in partitions.values():
            client_labels = {int(labels[position]) for position in positions}
            self.assertLessEqual(len(client_labels), 2)

    def test_different_seeds_change_shard_assignment(self) -> None:
        labels = torch.cat([torch.full((20,), value) for value in range(4)])
        first = sorted_label_shard_partition_indices(labels, 4, 1, 1)
        second = sorted_label_shard_partition_indices(labels, 4, 1, 2)
        self.assertNotEqual(first, second)

    def test_invalid_partition_settings_are_rejected(self) -> None:
        labels = torch.arange(10) % 3
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            sorted_label_shard_partition_indices(labels, 6, 2, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            sorted_label_shard_partition_indices(labels, 0, 1, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            sorted_label_shard_partition_indices(labels, 2, 0, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            sorted_label_shard_partition_indices(labels, 2.5, 1, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            sorted_label_shard_partition_indices(labels, 2, True, 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sorted_label_shard_partition_indices(labels, 2, 1, -1)
        with self.assertRaisesRegex(TypeError, "one-dimensional tensor"):
            sorted_label_shard_partition_indices([0, 1, 2], 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
