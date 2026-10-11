"""Federated retrain from scratch without some clients: M_retrain^(-A).

M_retrain^(-A) is the gold standard of client-level Federated Unlearning:
the model the federation would have produced if target client A had never
taken part.  It is ordinary FedAvg, run again from the same initial weights
w_0, where client A is removed from the client set before round 1.

Protocol source: ``Ke_hoach_v4_FU.md`` section 6 ("Gold standard":
same config, seed and number of rounds as the original run, only the target
client is dropped).  No paper equation is implemented here; the only formula
is FedAvg aggregation, which stays in ``algorithms/fedavg.py``.

How exclusion works (the whole idea of this file): the excluded clients are
filtered out of ``client_datasets`` BEFORE the unchanged ``FedAvgServer`` is
built.  The server then never trains them and never sees their example
counts, so its aggregation weight for a remaining client k is

    n_k / sum_{j != A} n_j          (renormalised over the remaining clients)

instead of n_k / n.  ``algorithms/fedavg.py`` and ``server/fedavg_server.py``
are not modified.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import torch
from torch import nn
from torch.utils.data import Dataset

from evaluation.efficiency import PhaseTimer
from server.fedavg_server import FedAvgServer, RoundResult


def remaining_client_datasets(
    client_datasets: Mapping[int, Dataset], exclude_clients: Collection[int]
) -> dict[int, Dataset]:
    """Return the clients that stay, keeping their original client IDs.

    IDs are kept (not renumbered 0..K-2) on purpose: ``FedAvgServer`` derives
    each client's local shuffling seed from its ID, so a remaining client
    sees its minibatches in the same order as in the original run.
    """
    excluded = set(exclude_clients)
    unknown = sorted(excluded - set(client_datasets))
    if unknown:
        raise ValueError(f"Cannot exclude unknown client IDs: {unknown}.")
    remaining = {
        client_id: dataset
        for client_id, dataset in client_datasets.items()
        if client_id not in excluded
    }
    if not remaining:
        raise ValueError("Excluding every client leaves nothing to train on.")
    return remaining


def train_fedavg_excluding(
    *,
    model: nn.Module,
    client_datasets: Mapping[int, Dataset],
    exclude_clients: Collection[int],
    number_of_rounds: int,
    client_fraction: float,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    device: torch.device,
    timer: PhaseTimer,
    phase_name: str,
) -> list[RoundResult]:
    """Train ``model`` in place with FedAvg on every non-excluded client.

    ``exclude_clients=()`` gives the original model M (all clients);
    ``exclude_clients=(A,)`` gives M_retrain^(-A).  The caller loads the
    shared initial weights w_0 into ``model`` before calling, so both runs
    start from the same point.

    Only the training rounds are timed (phase ``phase_name``, entered once
    per round); evaluation done by the caller is outside the timer.

    # DEVIATION: with client_fraction < 1 the per-round client sample is
    # drawn from the remaining clients, so it is not "the original sample
    # minus A".  Exact correspondence holds only for client_fraction = 1.0,
    # which is what the thesis configs use.
    """
    server = FedAvgServer(
        global_model=model,
        client_datasets=remaining_client_datasets(client_datasets, exclude_clients),
        client_fraction=client_fraction,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        device=device,
    )
    round_results: list[RoundResult] = []
    for round_number in range(1, number_of_rounds + 1):
        with timer.phase(phase_name):
            round_results.append(server.run_round(round_number))
    return round_results
