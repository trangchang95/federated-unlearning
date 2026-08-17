"""Deterministic IID client partitioning for the first FL experiment."""

from __future__ import annotations

from numbers import Integral

import torch
from torch.utils.data import Dataset, Subset


def iid_partition_indices(
    dataset_size: int, number_of_clients: int, random_seed: int
) -> dict[int, tuple[int, ...]]:
    """Split dataset positions randomly, disjointly, and as evenly as possible."""
    for name, value in (
        ("dataset_size", dataset_size),
        ("number_of_clients", number_of_clients),
        ("random_seed", random_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
    dataset_size = int(dataset_size)
    number_of_clients = int(number_of_clients)
    random_seed = int(random_seed)
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive.")
    if number_of_clients > dataset_size:
        raise ValueError("number_of_clients cannot exceed dataset_size.")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")

    permutation = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(random_seed)
    ).tolist()
    base_size, remainder = divmod(dataset_size, number_of_clients)
    partitions: dict[int, tuple[int, ...]] = {}
    cursor = 0
    for client_id in range(number_of_clients):
        client_size = base_size + (1 if client_id < remainder else 0)
        partitions[client_id] = tuple(permutation[cursor : cursor + client_size])
        cursor += client_size

    if cursor != dataset_size:
        raise RuntimeError("IID partitioning did not consume the complete dataset.")
    return partitions


def partition_dataset_iid(
    dataset: Dataset, number_of_clients: int, random_seed: int
) -> dict[int, Subset]:
    """Create one PyTorch ``Subset`` per client from deterministic positions."""
    indices = iid_partition_indices(len(dataset), number_of_clients, random_seed)
    return {
        client_id: Subset(dataset, list(client_indices))
        for client_id, client_indices in indices.items()
    }
