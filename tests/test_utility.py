"""Checks for shared classifier utility evaluation."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from algorithms.fedavg import clone_model_state
from evaluation.utility import evaluate_classifier


class IdentityClassifier(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class UtilityEvaluationTests(unittest.TestCase):
    def test_evaluation_is_correct_and_restores_training_mode(self) -> None:
        logits = torch.tensor([[4.0, 1.0], [0.5, 3.0], [2.0, 1.0]])
        labels = torch.tensor([0, 1, 0])
        model = IdentityClassifier()
        model.train()
        result = evaluate_classifier(
            model,
            DataLoader(TensorDataset(logits, labels), batch_size=2),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.f1_macro, 1.0)
        self.assertTrue(model.training)

    def test_evaluation_does_not_change_model_parameters(self) -> None:
        model = nn.Linear(2, 2)
        state_before = clone_model_state(model.state_dict())
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        evaluate_classifier(
            model,
            DataLoader(TensorDataset(features, labels), batch_size=2),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state_before[name]))

    def test_loss_is_weighted_by_examples_for_an_uneven_final_batch(self) -> None:
        logits = torch.tensor(
            [[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1, 0])
        loss_function = nn.CrossEntropyLoss()
        expected = loss_function(logits, labels).item()
        result = evaluate_classifier(
            IdentityClassifier(),
            DataLoader(TensorDataset(logits, labels), batch_size=2),
            loss_function,
            torch.device("cpu"),
        )
        self.assertAlmostEqual(result.loss, expected, places=6)

    def test_macro_f1_uses_the_complete_classifier_label_set(self) -> None:
        logits = torch.tensor([[5.0, 0.0], [4.0, 1.0]])
        labels = torch.tensor([0, 0])
        result = evaluate_classifier(
            IdentityClassifier(),
            DataLoader(TensorDataset(logits, labels), batch_size=2),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.f1_macro, 0.5)

    def test_empty_loader_raises_and_restores_the_previous_mode(self) -> None:
        model = IdentityClassifier()
        model.train()
        empty_features = torch.empty((0, 2))
        empty_labels = torch.empty((0,), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_classifier(
                model,
                DataLoader(TensorDataset(empty_features, empty_labels), batch_size=2),
                nn.CrossEntropyLoss(),
                torch.device("cpu"),
            )
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
