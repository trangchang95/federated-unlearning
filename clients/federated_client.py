"""One client's local-training responsibility in hand-written FedAvg."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from algorithms.fedavg import clone_model_state


@dataclass(frozen=True)
class LocalTrainingResult:
    """The information one selected client sends back to the server."""

    client_id: int
    number_of_examples: int
    model_state: dict[str, torch.Tensor]
    mean_loss: float
    optimizer_steps: int


def _validate_local_settings(
    client_id: int,
    dataset: Dataset,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    shuffle_seed: int,
) -> None:
    if (
        isinstance(client_id, bool)
        or not isinstance(client_id, Integral)
        or int(client_id) < 0
    ):
        raise TypeError("client_id must be a non-negative integer.")
    if len(dataset) <= 0:
        raise ValueError(f"Client {client_id} has no training examples.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if (
        isinstance(local_epochs, bool)
        or not isinstance(local_epochs, int)
        or local_epochs <= 0
    ):
        raise ValueError("local_epochs must be a positive integer.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a finite positive number.")
    if (
        isinstance(shuffle_seed, bool)
        or not isinstance(shuffle_seed, Integral)
        or int(shuffle_seed) < 0
    ):
        raise ValueError("shuffle_seed must be a non-negative integer.")


def train_client(
    *,
    client_id: int,
    dataset: Dataset,
    global_model: nn.Module,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
    shuffle_seed: int,
) -> LocalTrainingResult:
    """Train a fresh copy of the current global model with local SGD.

    The global model is never optimized here.  A deep copy receives a fresh
    optimizer, trains only on this client's dataset, and returns detached CPU
    tensors.  Creating a new optimizer for every client and round prevents
    optimizer momentum/state from leaking between clients.
    """
    _validate_local_settings(
        client_id,
        dataset,
        batch_size,
        local_epochs,
        learning_rate,
        shuffle_seed,
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for a client, but CUDA is unavailable.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA.")

    local_model = copy.deepcopy(global_model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )

    cuda_devices: list[int] = []
    if device.type == "cuda":
        # torch.manual_seed/manual_seed_all affect CUDA generators beyond the
        # target device. Snapshot all of them so local training cannot advance
        # RNG state owned by the caller or another simulated client.
        cuda_devices = list(range(torch.cuda.device_count()))

    total_loss = 0.0
    processed_examples = 0
    optimizer_steps = 0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(shuffle_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(shuffle_seed)

        local_model.train()
        for _ in range(local_epochs):
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = local_model(features)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                processed_examples += current_batch_size
                optimizer_steps += 1

    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in clone_model_state(local_model.state_dict()).items()
    }
    return LocalTrainingResult(
        client_id=client_id,
        number_of_examples=len(dataset),
        model_state=cpu_state,
        mean_loss=total_loss / processed_examples,
        optimizer_steps=optimizer_steps,
    )
