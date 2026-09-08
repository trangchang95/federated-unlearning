"""Deterministic pathological (shard-based) Non-IID client partitioning.

This is the classic Non-IID scheme from McMahan et al. (2017, Section 3):
sort all training examples by label, cut the sorted sequence into equal-sized
shards, then give each client a small fixed number of shards drawn from a
seeded shuffle of shard order. Because MNIST's ten classes are close to
balanced, a shard is usually dominated by one or two labels, so a client with
``shards_per_client=2`` typically sees mostly one or two digits instead of a
mixed sample -- the "C1 = mostly class A" picture in the plan, in contrast to
``clients/iid_partition.py``'s uniform random split.
"""

from __future__ import annotations

from numbers import Integral

import torch


def sorted_label_shard_partition_indices(
    labels: torch.Tensor,
    number_of_clients: int,
    shards_per_client: int,
    random_seed: int,
) -> dict[int, tuple[int, ...]]:
    """Assign each client ``shards_per_client`` label-sorted shards.

    ``labels`` holds one label per dataset position (position ``i`` is the
    label of example ``i``); the returned positions are indices into that same
    sequence, matching the return convention of
    ``clients.iid_partition.iid_partition_indices``.
    """
    for name, value in (
        ("number_of_clients", number_of_clients),
        ("shards_per_client", shards_per_client),
        ("random_seed", random_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
    number_of_clients = int(number_of_clients)
    shards_per_client = int(shards_per_client)
    random_seed = int(random_seed)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise TypeError("labels must be a one-dimensional tensor.")
    dataset_size = int(labels.shape[0])
    if dataset_size <= 0:
        raise ValueError("labels must not be empty.")
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive.")
    if shards_per_client <= 0:
        raise ValueError("shards_per_client must be positive.")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")

    number_of_shards = number_of_clients * shards_per_client
    if number_of_shards > dataset_size:
        raise ValueError("number_of_clients * shards_per_client cannot exceed dataset_size.")

    label_list = labels.tolist()
    # Sort by (label, position): deterministic on every platform, unlike
    # relying on a sort implementation's tie-breaking behavior for equal keys.
    sorted_positions = [
        position
        for _, position in sorted(
            (label_list[position], position) for position in range(dataset_size)
        )
    ]

    shard_base_size, shard_remainder = divmod(dataset_size, number_of_shards)
    shards: list[tuple[int, ...]] = []
    cursor = 0
    for shard_id in range(number_of_shards):
        shard_size = shard_base_size + (1 if shard_id < shard_remainder else 0)
        shards.append(tuple(sorted_positions[cursor : cursor + shard_size]))
        cursor += shard_size
    if cursor != dataset_size:
        raise RuntimeError("Shard construction did not consume the complete dataset.")

    shard_order = torch.randperm(
        number_of_shards, generator=torch.Generator().manual_seed(random_seed)
    ).tolist()

    partitions: dict[int, tuple[int, ...]] = {}
    for client_id in range(number_of_clients):
        assigned_shard_ids = shard_order[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]
        positions: list[int] = []
        for shard_id in assigned_shard_ids:
            positions.extend(shards[shard_id])
        partitions[client_id] = tuple(sorted(positions))

    total_assigned = sum(len(positions) for positions in partitions.values())
    if total_assigned != dataset_size:
        raise RuntimeError("Non-IID partitioning did not consume the complete dataset.")
    return partitions
