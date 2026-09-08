"""Deterministic Dirichlet(alpha) label-skew client partitioning.

This is the label-distribution-skew scheme from Hsu, Qi & Brown (2019,
"Measuring the Effects of Non-Identical Data Distribution for Federated
Visual Classification"), used throughout the FL literature (including
FedProx's own experiments) as a *tunable* alternative to
``clients/noniid_partition.py``'s all-or-nothing shard scheme:

For each class independently, draw a proportion vector from
``Dirichlet(alpha, alpha, ..., alpha)`` over the ``number_of_clients``
clients, then split that class's examples across clients according to those
proportions. A large alpha (e.g. 100) makes every class's proportions close
to uniform across clients (nearly IID); a small alpha (e.g. 0.1) makes each
class's examples concentrate on very few clients (severe label skew). This
lets Week 10 dial heterogeneity severity, instead of Week 9's single
pathological extreme.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch


def _largest_remainder_allocation(proportions: list[float], total: int) -> list[int]:
    """Round ``proportions * total`` to integers that sum exactly to ``total``."""
    raw = [proportion * total for proportion in proportions]
    base = [int(value) for value in raw]  # each raw value is >= 0, so floor == int()
    shortfall = total - sum(base)
    order = sorted(range(len(proportions)), key=lambda i: raw[i] - base[i], reverse=True)
    for index in order[:shortfall]:
        base[index] += 1
    return base


def dirichlet_label_partition_indices(
    labels: torch.Tensor,
    number_of_clients: int,
    alpha: float,
    random_seed: int,
) -> dict[int, tuple[int, ...]]:
    """Split ``labels`` across clients with per-class Dirichlet(alpha) proportions.

    ``labels`` holds one label per dataset position, matching the return
    convention of ``clients.iid_partition.iid_partition_indices`` and
    ``clients.noniid_partition.sorted_label_shard_partition_indices``.
    """
    if isinstance(number_of_clients, bool) or not isinstance(number_of_clients, Integral):
        raise TypeError("number_of_clients must be an integer.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be an integer.")
    number_of_clients = int(number_of_clients)
    random_seed = int(random_seed)
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(float(alpha)):
        raise TypeError("alpha must be a finite real number.")
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise TypeError("labels must be a one-dimensional tensor.")
    dataset_size = int(labels.shape[0])
    if dataset_size <= 0:
        raise ValueError("labels must not be empty.")
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive.")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")

    label_list = labels.tolist()
    class_ids = sorted(set(label_list))
    positions_by_class: dict[int, list[int]] = {class_id: [] for class_id in class_ids}
    for position, label in enumerate(label_list):
        positions_by_class[label].append(position)

    generator = np.random.default_rng(random_seed)
    partitions: dict[int, list[int]] = {client_id: [] for client_id in range(number_of_clients)}
    concentration = np.full(number_of_clients, alpha)
    for class_id in class_ids:
        positions = positions_by_class[class_id]
        generator.shuffle(positions)
        proportions = generator.dirichlet(concentration).tolist()
        counts = _largest_remainder_allocation(proportions, len(positions))
        cursor = 0
        for client_id, count in enumerate(counts):
            partitions[client_id].extend(positions[cursor : cursor + count])
            cursor += count
        if cursor != len(positions):
            raise RuntimeError("Dirichlet allocation did not consume every class example.")

    empty_clients = [client_id for client_id, positions in partitions.items() if not positions]
    if empty_clients:
        raise RuntimeError(
            f"Clients {empty_clients} received zero examples at alpha={alpha}; "
            "use a larger alpha or a different random_seed."
        )

    total_assigned = sum(len(positions) for positions in partitions.values())
    if total_assigned != dataset_size:
        raise RuntimeError("Dirichlet partitioning did not consume the complete dataset.")
    return {client_id: tuple(sorted(positions)) for client_id, positions in partitions.items()}
