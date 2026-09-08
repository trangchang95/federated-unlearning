"""Checks for the Dirichlet(alpha) label-skew partition used in Month 3."""

from __future__ import annotations

import unittest

import torch

from clients.dirichlet_partition import dirichlet_label_partition_indices


def _class_counts(positions: tuple[int, ...], labels: torch.Tensor) -> dict[int, int]:
    counts: dict[int, int] = {}
    for position in positions:
        label = int(labels[position])
        counts[label] = counts.get(label, 0) + 1
    return counts


class DirichletPartitionTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self) -> None:
        labels = torch.cat([torch.full((50,), value) for value in range(4)])
        first = dirichlet_label_partition_indices(labels, 5, 0.5, 42)
        second = dirichlet_label_partition_indices(labels, 5, 0.5, 42)
        self.assertEqual(first, second)

        flattened = [position for positions in first.values() for position in positions]
        self.assertEqual(len(flattened), 200)
        self.assertEqual(set(flattened), set(range(200)))

    def test_large_alpha_is_close_to_uniform_per_class(self) -> None:
        labels = torch.cat([torch.full((1000,), value) for value in range(4)])
        partitions = dirichlet_label_partition_indices(labels, 5, 1000.0, 7)
        for client_id, positions in partitions.items():
            counts = _class_counts(positions, labels)
            for class_id in range(4):
                # With alpha=1000 every class should be nearly evenly split;
                # allow generous slack since this is still a random draw.
                self.assertAlmostEqual(counts.get(class_id, 0), 200, delta=40)

    def test_small_alpha_concentrates_classes_on_few_clients(self) -> None:
        # With a large per-class sample size, largest-remainder rounding can
        # still hand a token 1-2 examples to a low-proportion client, so
        # "count of nonzero clients" is not a robust concentration signal.
        # Instead check that most of each class's examples land on a small
        # minority of clients -- the actual definition of label skew.
        labels = torch.cat([torch.full((1000,), value) for value in range(10)])
        partitions = dirichlet_label_partition_indices(labels, 10, 0.1, 3)
        for class_id in range(10):
            counts = sorted(
                (
                    _class_counts(positions, labels).get(class_id, 0)
                    for positions in partitions.values()
                ),
                reverse=True,
            )
            top_two_share = sum(counts[:2]) / sum(counts)
            # A uniform split would give each of 10 clients ~10%, so the top
            # two would hold ~20%. Skew concentrates most of the class there.
            self.assertGreater(top_two_share, 0.45)

    def test_different_seeds_change_the_split(self) -> None:
        labels = torch.cat([torch.full((200,), value) for value in range(4)])
        first = dirichlet_label_partition_indices(labels, 5, 0.5, 1)
        second = dirichlet_label_partition_indices(labels, 5, 0.5, 2)
        self.assertNotEqual(first, second)

    def test_invalid_settings_are_rejected(self) -> None:
        labels = torch.arange(20) % 4
        with self.assertRaisesRegex(ValueError, "positive"):
            dirichlet_label_partition_indices(labels, 5, 0.0, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            dirichlet_label_partition_indices(labels, 0, 0.5, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            dirichlet_label_partition_indices(labels, 2.5, 0.5, 1)
        with self.assertRaisesRegex(TypeError, "integer"):
            dirichlet_label_partition_indices(labels, 2, 0.5, True)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dirichlet_label_partition_indices(labels, 2, 0.5, -1)
        with self.assertRaisesRegex(TypeError, "finite real number"):
            dirichlet_label_partition_indices(labels, 2, float("nan"), 1)
        with self.assertRaisesRegex(TypeError, "one-dimensional tensor"):
            dirichlet_label_partition_indices([0, 1, 2], 1, 0.5, 1)

    def test_too_many_clients_for_severe_skew_raises_a_clear_error(self) -> None:
        # A single example of one label with 50 clients at a tiny alpha will
        # frequently strand most clients with zero examples of that class;
        # with only one class total, that means a client can end up empty.
        labels = torch.zeros(3, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "zero examples"):
            dirichlet_label_partition_indices(labels, 50, 0.01, 0)


if __name__ == "__main__":
    unittest.main()
