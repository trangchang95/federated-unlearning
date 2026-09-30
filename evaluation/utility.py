"""Shared classification-utility evaluation for FL experiments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Loss and predictions collected without changing model weights."""

    loss: float
    accuracy: float
    f1_macro: float
    labels: np.ndarray
    predictions: np.ndarray


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> ClassificationEvaluation:
    """Evaluate a classifier and restore its prior train/eval mode afterward.

    ``data_loader`` is normally a ``DataLoader``; any iterable of
    ``(features, labels)`` batches works.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    label_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    number_of_classes: int | None = None

    try:
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            if logits.ndim != 2:
                raise ValueError("Classifier output must have shape [batch, classes].")
            if number_of_classes is None:
                number_of_classes = logits.shape[1]
            elif logits.shape[1] != number_of_classes:
                raise ValueError("Classifier output class count changed between batches.")
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            label_batches.append(labels.cpu().numpy())
            prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        if was_training:
            model.train()
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    if number_of_classes is None:
        raise RuntimeError("Evaluation did not observe a classifier output.")

    all_labels = np.concatenate(label_batches)
    all_predictions = np.concatenate(prediction_batches)
    return ClassificationEvaluation(
        loss=total_loss / total_examples,
        accuracy=float(accuracy_score(all_labels, all_predictions)),
        f1_macro=float(
            f1_score(
                all_labels,
                all_predictions,
                labels=list(range(number_of_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        labels=all_labels,
        predictions=all_predictions,
    )


@dataclass(frozen=True)
class ThreeWayEvaluation:
    """One model evaluated on the three splits an unlearning report needs.

    - ``global_test``: the shared held-out test set.
    - ``target_client``: the data of the client being forgotten.
    - ``non_target_clients``: the data of every other client, **pooled**:
      all non-target examples are evaluated as one set, so each client
      contributes in proportion to its number of examples (not one equal
      vote per client).
    """

    global_test: ClassificationEvaluation
    target_client: ClassificationEvaluation
    non_target_clients: ClassificationEvaluation

    def metrics(self) -> dict[str, dict[str, float | int]]:
        """Scalar metrics per split, JSON-serializable for a result file."""
        return {
            split: {
                "loss": evaluation.loss,
                "accuracy": evaluation.accuracy,
                "f1_macro": evaluation.f1_macro,
                "examples": int(evaluation.labels.shape[0]),
            }
            for split, evaluation in (
                ("global_test", self.global_test),
                ("target_client", self.target_client),
                ("non_target_clients", self.non_target_clients),
            )
        }


def _pooled_batches(
    loaders: Sequence[Iterable[tuple[torch.Tensor, torch.Tensor]]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield every batch of every loader in order; an empty loader raises."""
    for index, loader in enumerate(loaders):
        examples = 0
        for features, labels in loader:
            examples += labels.size(0)
            yield features, labels
        if examples == 0:
            raise ValueError(f"Non-target client loader {index} is empty.")


def evaluate_three_way(
    model: nn.Module,
    global_test_loader: DataLoader,
    target_client_loader: DataLoader,
    non_target_client_loaders: Sequence[DataLoader],
    loss_function: nn.Module,
    device: torch.device,
) -> ThreeWayEvaluation:
    """Evaluate ``model`` on global test, target-client and non-target data.

    Each split is evaluated with ``evaluate_classifier``, so weights are not
    changed and the model's prior train/eval mode is restored.

    The non-target result is **pooled** over all non-target clients: their
    batches are evaluated as a single stream, so loss, accuracy and macro-F1
    are computed over the union of their examples (weighted by examples,
    not averaged per client).  For per-client numbers, call
    ``evaluate_classifier`` on each loader separately.

    Guards against simple wiring mistakes: passing one ``DataLoader``
    instead of a list of them, listing the same loader object twice, and
    reusing the target loader object as a non-target (or any client loader
    as the global test set).  It cannot detect two
    different loaders that share examples; partition checks must do that.
    """
    if isinstance(non_target_client_loaders, DataLoader):
        raise TypeError(
            "non_target_client_loaders must be a sequence of DataLoaders, "
            "not a single DataLoader."
        )
    if not isinstance(non_target_client_loaders, Sequence):
        raise TypeError("non_target_client_loaders must be a sequence.")
    if len(non_target_client_loaders) == 0:
        raise ValueError("At least one non-target client loader is required.")
    if target_client_loader is global_test_loader:
        raise ValueError("The target client loader is the global test loader.")
    if len({id(loader) for loader in non_target_client_loaders}) != len(
        non_target_client_loaders
    ):
        raise ValueError(
            "The same non-target loader appears more than once; pooling "
            "would count its examples twice."
        )
    for index, loader in enumerate(non_target_client_loaders):
        if loader is target_client_loader:
            raise ValueError(
                f"Non-target client loader {index} is the target client loader."
            )
        if loader is global_test_loader:
            raise ValueError(
                f"Non-target client loader {index} is the global test loader."
            )

    return ThreeWayEvaluation(
        global_test=evaluate_classifier(
            model, global_test_loader, loss_function, device
        ),
        target_client=evaluate_classifier(
            model, target_client_loader, loss_function, device
        ),
        non_target_clients=evaluate_classifier(
            model,
            _pooled_batches(non_target_client_loaders),
            loss_function,
            device,
        ),
    )
