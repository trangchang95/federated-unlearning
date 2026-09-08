"""One client's local-training responsibility in FedProx (Li et al. 2020).

This mirrors ``clients/federated_client.py`` exactly except for one change:
the local loss adds a proximal term ``(mu / 2) * ||w - w_global||^2`` that
penalizes the local model for drifting away from the model every client
started this round from. Setting ``mu=0`` makes this identical to plain
FedAvg local training, so FedProx is implemented here as a strict
generalization, not a separate code path that could silently diverge from
Week 6/8's hand-written FedAvg semantics.

``clients/federated_client.py`` (Gate 2 evidence) is intentionally left
untouched; this is a new, separate module.
"""

from __future__ import annotations

import copy
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from algorithms.fedavg import clone_model_state
from clients.federated_client import LocalTrainingResult


def _validate_local_settings(
    client_id: int,
    dataset: Dataset,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    shuffle_seed: int,
) -> None:
    # Duplicated from clients/federated_client.py rather than imported: that
    # file is frozen Gate 2 evidence and is not touched even to export a
    # helper, so its content stays byte-identical to the tagged commit.
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


def _validate_mu(mu: float) -> None:
    if isinstance(mu, bool) or not isinstance(mu, Real) or not math.isfinite(float(mu)):
        raise ValueError("mu must be a finite real number.")
    if mu < 0:
        raise ValueError("mu must be non-negative.")


def train_client_prox(
    *,
    client_id: int,
    dataset: Dataset,
    global_model: nn.Module,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    mu: float,
    device: torch.device,
    shuffle_seed: int,
) -> LocalTrainingResult:
    """Train a fresh copy of the global model with local SGD plus a proximal term.

    Identical to ``clients.federated_client.train_client`` except that the
    loss adds ``(mu / 2) * sum((local_param - global_param) ** 2)`` at every
    step, computed against a frozen snapshot of the round-starting global
    weights. ``mu=0`` makes the added term exactly zero for every step,
    reducing this function to plain FedAvg local training.
    """
    _validate_local_settings(
        client_id, dataset, batch_size, local_epochs, learning_rate, shuffle_seed
    )
    _validate_mu(mu)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for a client, but CUDA is unavailable.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA.")

    local_model = copy.deepcopy(global_model).to(device)
    # Frozen reference point: the round's starting global parameters, in the
    # same order local_model.named_parameters() will iterate them.
    global_reference = [
        parameter.detach().clone() for parameter in local_model.parameters()
    ]
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
                task_loss = loss_function(logits, labels)
                proximal_term = sum(
                    (local_parameter - reference_parameter).pow(2).sum()
                    for local_parameter, reference_parameter in zip(
                        local_model.parameters(), global_reference
                    )
                )
                loss = task_loss + (mu / 2) * proximal_term
                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                # Logged loss is the task loss alone, so FedAvg (mu=0) and
                # FedProx mean-loss histories stay directly comparable; the
                # proximal term is an optimization device, not part of the
                # reported task-performance signal.
                total_loss += task_loss.item() * current_batch_size
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
