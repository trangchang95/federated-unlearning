"""Checks for the deterministic IID partition used in Month 2."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from clients.iid_partition import iid_partition_indices, partition_dataset_iid


class IidPartitionTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self) -> None:
        first = iid_partition_indices(23, 5, 42)
        second = iid_partition_indices(23, 5, 42)
        self.assertEqual(first, second)

        flattened = [index for indices in first.values() for index in indices]
        self.assertEqual(len(flattened), 23)
        self.assertEqual(len(set(flattened)), 23)
        self.assertEqual(set(flattened), set(range(23)))
        sizes = [len(indices) for indices in first.values()]
        self.assertEqual(sizes, [5, 5, 5, 4, 4])

    def test_partition_uses_private_rng_and_wraps_dataset(self) -> None:
        torch.manual_seed(1234)
        rng_before = torch.get_rng_state().clone()
        features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        labels = torch.arange(10) % 2
        dataset = TensorDataset(features, labels)
        client_datasets = partition_dataset_iid(dataset, 3, 7)

        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual([len(client_datasets[index]) for index in range(3)], [4, 3, 3])
        recovered_labels = sorted(
            int(client_datasets[client_id][position][1])
            for client_id in client_datasets
            for position in range(len(client_datasets[client_id]))
        )
        self.assertEqual(recovered_labels, sorted(labels.tolist()))

    def test_invalid_partition_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            iid_partition_indices(3, 4, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            iid_partition_indices(0, 1, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            iid_partition_indices(10, 2.5, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            iid_partition_indices(10, True, 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            iid_partition_indices(10, 2, -1)


if __name__ == "__main__":
    unittest.main()
