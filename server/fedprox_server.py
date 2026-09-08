"""Server-side orchestration for FedProx (Li et al. 2020).

Mirrors ``server/fedavg_server.py`` exactly -- same client selection, same
seeding, same sample-count-weighted aggregation -- except each selected
client trains with ``clients.fedprox_client.train_client_prox`` instead of
plain local SGD, adding one extra hyperparameter, ``mu``. Aggregation itself
is unchanged from FedAvg, per Li et al. (2020): FedProx generalizes what
happens during local training, not how the server combines results.

``server/fedavg_server.py`` (Gate 2 evidence) is intentionally left
untouched; this is a new, separate module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import nn
from torch.utils.data import Dataset

from algorithms.fedavg import (
    clone_model_state,
    model_state_size_bytes,
    weighted_average_model_states,
)
from clients.fedprox_client import LocalTrainingResult, train_client_prox
from server.fedavg_server import client_training_seed, select_client_ids


@dataclass(frozen=True)
class ProxRoundResult:
    """Auditable, non-timing evidence produced by one FedProx round."""

    round_number: int
    selected_client_ids: tuple[int, ...]
    client_example_counts: dict[int, int]
    client_mean_losses: dict[int, float]
    client_optimizer_steps: dict[int, int]
    total_selected_examples: int
    weighted_mean_local_loss: float
    model_payload_bytes: int
    download_bytes: int
    upload_bytes: int
    communication_cost_bytes: int


class FedProxServer:
    """Coordinate clients sequentially using FedProx local training.

    Single-process simulation for learning and reproducible local
    experiments, exactly like ``FedAvgServer``: it models which tensors are
    sent and averaged, not real network transport, parallel execution, or
    privacy. ``random_seed`` controls client selection and local training;
    the caller must separately seed PyTorch *before constructing*
    ``global_model`` so the initial weights are reproducible too.
    """

    def __init__(
        self,
        *,
        global_model: nn.Module,
        client_datasets: Mapping[int, Dataset],
        client_fraction: float,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        mu: float,
        random_seed: int,
        device: torch.device,
    ) -> None:
        if not client_datasets:
            raise ValueError("FedProxServer requires at least one client dataset.")
        if any(
            isinstance(client_id, bool)
            or not isinstance(client_id, Integral)
            or int(client_id) < 0
            for client_id in client_datasets
        ):
            raise TypeError("Every client ID must be a non-negative integer.")
        if any(len(dataset) <= 0 for dataset in client_datasets.values()):
            raise ValueError("Every client must own at least one training example.")
        if not 0 < client_fraction <= 1:
            raise ValueError("client_fraction must be in the interval (0, 1].")
        if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
            raise ValueError("local_epochs must be a positive integer.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number.")
        if isinstance(mu, bool) or not isinstance(mu, Real) or not math.isfinite(float(mu)) or mu < 0:
            raise ValueError("mu must be a finite, non-negative number.")
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, Integral)
            or int(random_seed) < 0
        ):
            raise ValueError("random_seed must be a non-negative integer.")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the server, but CUDA is unavailable.")

        self.global_model = global_model.to(device)
        self.client_datasets = dict(client_datasets)
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        self.random_seed = random_seed
        self.device = device
        self._last_completed_round = 0

    def global_state(self) -> dict[str, torch.Tensor]:
        """Return an independent CPU snapshot of the current global model."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in clone_model_state(self.global_model.state_dict()).items()
        }

    def run_round(self, round_number: int) -> ProxRoundResult:
        """Broadcast, train selected clients with FedProx, aggregate, update global state."""
        if round_number <= self._last_completed_round:
            raise ValueError(
                "round_number must increase; reusing a completed round would reuse its RNG seeds."
            )
        selected_ids = select_client_ids(
            list(self.client_datasets),
            self.client_fraction,
            self.random_seed,
            round_number,
        )
        payload_bytes = model_state_size_bytes(self.global_model.state_dict())

        local_results: list[LocalTrainingResult] = []
        for client_id in selected_ids:
            local_results.append(
                train_client_prox(
                    client_id=client_id,
                    dataset=self.client_datasets[client_id],
                    global_model=self.global_model,
                    batch_size=self.batch_size,
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    mu=self.mu,
                    device=self.device,
                    shuffle_seed=client_training_seed(
                        self.random_seed, round_number, client_id
                    ),
                )
            )

        counts = [result.number_of_examples for result in local_results]
        next_global_state = weighted_average_model_states(
            [result.model_state for result in local_results], counts
        )
        self.global_model.load_state_dict(next_global_state, strict=True)
        self._last_completed_round = round_number

        total_examples = sum(counts)
        weighted_loss = sum(
            result.mean_loss * result.number_of_examples for result in local_results
        ) / total_examples
        download_bytes = payload_bytes * len(selected_ids)
        upload_bytes = payload_bytes * len(selected_ids)

        return ProxRoundResult(
            round_number=round_number,
            selected_client_ids=selected_ids,
            client_example_counts={
                result.client_id: result.number_of_examples for result in local_results
            },
            client_mean_losses={result.client_id: result.mean_loss for result in local_results},
            client_optimizer_steps={
                result.client_id: result.optimizer_steps for result in local_results
            },
            total_selected_examples=total_examples,
            weighted_mean_local_loss=weighted_loss,
            model_payload_bytes=payload_bytes,
            download_bytes=download_bytes,
            upload_bytes=upload_bytes,
            communication_cost_bytes=download_bytes + upload_bytes,
        )
