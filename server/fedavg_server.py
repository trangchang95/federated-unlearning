"""Server-side orchestration for the hand-written Week 6 FedAvg workflow."""

from __future__ import annotations

import math
import random
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
from clients.federated_client import LocalTrainingResult, train_client


def select_client_ids(
    client_ids: list[int], client_fraction: float, random_seed: int, round_number: int
) -> tuple[int, ...]:
    """Select a deterministic random subset for one round.

    The number selected is ``ceil(client_fraction * number_of_clients)``, with
    at least one client.  Sorting the result gives a stable aggregation order,
    which removes an avoidable source of floating-point variation.
    """
    if not client_ids:
        raise ValueError("At least one client is required.")
    if any(
        isinstance(client_id, bool) or not isinstance(client_id, Integral)
        for client_id in client_ids
    ):
        raise TypeError("Every client ID must be an integer.")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("client_ids must be unique.")
    if not 0 < client_fraction <= 1:
        raise ValueError("client_fraction must be in the interval (0, 1].")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("round_number must be a positive integer.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or int(random_seed) < 0
    ):
        raise ValueError("random_seed must be a non-negative integer.")

    selection_count = max(1, math.ceil(client_fraction * len(client_ids)))
    round_seed = (int(random_seed) + round_number * 1_000_003) % (2**63 - 1)
    random_generator = random.Random(round_seed)
    return tuple(sorted(random_generator.sample(sorted(client_ids), selection_count)))


def client_training_seed(random_seed: int, round_number: int, client_id: int) -> int:
    """Derive a reproducible local RNG seed independent of loop order."""
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (random_seed, round_number, client_id)
    ):
        raise TypeError("random_seed, round_number, and client_id must be integers.")
    if random_seed < 0 or round_number < 1 or client_id < 0:
        raise ValueError(
            "random_seed and client_id must be non-negative; round_number must be positive."
        )
    return (
        int(random_seed)
        + round_number * 1_000_003
        + (client_id + 1) * 10_007
    ) % (2**63 - 1)


@dataclass(frozen=True)
class RoundResult:
    """Auditable, non-timing evidence produced by one communication round."""

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


class FedAvgServer:
    """Coordinate clients sequentially while preserving FedAvg semantics.

    This is a single-process simulation for learning and reproducible local
    experiments.  It models which tensors are sent and averaged, but it does
    not claim real network transport, parallel execution, or privacy.

    ``random_seed`` controls client selection and local training. The caller
    must separately seed PyTorch *before constructing* ``global_model`` so the
    initial model weights are reproducible too.
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
        random_seed: int,
        device: torch.device,
    ) -> None:
        if not client_datasets:
            raise ValueError("FedAvgServer requires at least one client dataset.")
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
        self.random_seed = random_seed
        self.device = device
        self._last_completed_round = 0

    def global_state(self) -> dict[str, torch.Tensor]:
        """Return an independent CPU snapshot of the current global model."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in clone_model_state(self.global_model.state_dict()).items()
        }

    def run_round(self, round_number: int) -> RoundResult:
        """Broadcast, train selected clients, aggregate, and update global state."""
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
        # The server does not change the global model until every selected
        # client has trained, so every deep-copied local model begins at w_t.
        for client_id in selected_ids:
            local_results.append(
                train_client(
                    client_id=client_id,
                    dataset=self.client_datasets[client_id],
                    global_model=self.global_model,
                    batch_size=self.batch_size,
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
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
            result.mean_loss * result.number_of_examples
            for result in local_results
        ) / total_examples
        download_bytes = payload_bytes * len(selected_ids)
        upload_bytes = payload_bytes * len(selected_ids)

        return RoundResult(
            round_number=round_number,
            selected_client_ids=selected_ids,
            client_example_counts={
                result.client_id: result.number_of_examples
                for result in local_results
            },
            client_mean_losses={
                result.client_id: result.mean_loss for result in local_results
            },
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
