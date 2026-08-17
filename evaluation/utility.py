"""Shared classification-utility evaluation for FL experiments."""

from __future__ import annotations

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
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> ClassificationEvaluation:
    """Evaluate a classifier and restore its prior train/eval mode afterward."""
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
