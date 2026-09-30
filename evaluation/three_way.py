"""Three-way evaluation for unlearning experiments.

Kept separate from ``evaluation/utility.py`` because that file is pinned by
the Week 8 verifier to its ``month2-week8`` tag revision.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.utility import ClassificationEvaluation, evaluate_classifier


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
