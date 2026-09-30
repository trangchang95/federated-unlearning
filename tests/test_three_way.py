"""Checks for three-way (global / target / non-target) evaluation."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from algorithms.fedavg import clone_model_state
from evaluation.three_way import evaluate_three_way


class IdentityClassifier(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class ModeRecordingClassifier(nn.Module):
    """Identity classifier that records whether forward ran in train mode."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_training_flags: list[bool] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_training_flags.append(self.training)
        return features


def logits_loader(
    logits: list[list[float]], labels: list[int], batch_size: int = 2
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
    )


class ThreeWayEvaluationTests(unittest.TestCase):
    """Global test / target client / pooled non-target client evaluation."""

    def setUp(self) -> None:
        # Global test: 3 of 4 correct.
        self.global_logits = [[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [5.0, 0.0]]
        self.global_labels = [0, 1, 0, 1]
        # Target client: 2 of 2 correct.
        self.target_logits = [[4.0, 1.0], [1.0, 4.0]]
        self.target_labels = [0, 1]
        # Non-target client A: 1 of 1 correct (accuracy 1.0).
        self.client_a_logits = [[3.0, 0.0]]
        self.client_a_labels = [0]
        # Non-target client B: 1 of 3 correct (accuracy 1/3).
        self.client_b_logits = [[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]
        self.client_b_labels = [1, 0, 1]
        self.loss_function = nn.CrossEntropyLoss()
        self.device = torch.device("cpu")

    def loaders(self) -> tuple[DataLoader, DataLoader, list[DataLoader]]:
        return (
            logits_loader(self.global_logits, self.global_labels),
            logits_loader(self.target_logits, self.target_labels),
            [
                logits_loader(self.client_a_logits, self.client_a_labels),
                logits_loader(self.client_b_logits, self.client_b_labels),
            ],
        )

    def evaluate(self, model: nn.Module):
        global_loader, target_loader, non_target_loaders = self.loaders()
        return evaluate_three_way(
            model,
            global_loader,
            target_loader,
            non_target_loaders,
            self.loss_function,
            self.device,
        )

    def test_each_split_is_evaluated_on_its_own_data(self) -> None:
        result = self.evaluate(IdentityClassifier())
        self.assertEqual(result.global_test.accuracy, 0.75)
        self.assertEqual(result.target_client.accuracy, 1.0)
        self.assertEqual(result.global_test.labels.tolist(), self.global_labels)
        self.assertEqual(result.target_client.labels.tolist(), self.target_labels)

    def test_non_target_split_is_pooled_and_weighted_by_examples(self) -> None:
        result = self.evaluate(IdentityClassifier())
        # Pooled: 2 correct of 4 examples = 0.5.  An unweighted mean of the
        # two clients' accuracies would be (1.0 + 1/3) / 2 = 0.667 instead.
        self.assertEqual(result.non_target_clients.accuracy, 0.5)

        pooled_logits = torch.tensor(
            self.client_a_logits + self.client_b_logits, dtype=torch.float32
        )
        pooled_labels = torch.tensor(
            self.client_a_labels + self.client_b_labels, dtype=torch.long
        )
        expected_loss = self.loss_function(pooled_logits, pooled_labels).item()
        self.assertAlmostEqual(result.non_target_clients.loss, expected_loss, places=6)

        expected_f1 = f1_score(
            pooled_labels.numpy(),
            pooled_logits.argmax(dim=1).numpy(),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        self.assertAlmostEqual(result.non_target_clients.f1_macro, expected_f1, places=12)
        np.testing.assert_array_equal(
            result.non_target_clients.labels, pooled_labels.numpy()
        )

    def test_metrics_summary_is_json_serializable_with_example_counts(self) -> None:
        summary = json.loads(json.dumps(self.evaluate(IdentityClassifier()).metrics()))
        self.assertEqual(
            list(summary), ["global_test", "target_client", "non_target_clients"]
        )
        self.assertEqual(summary["global_test"]["examples"], 4)
        self.assertEqual(summary["target_client"]["examples"], 2)
        self.assertEqual(summary["non_target_clients"]["examples"], 4)
        self.assertEqual(summary["non_target_clients"]["accuracy"], 0.5)
        for split in summary.values():
            self.assertEqual(set(split), {"loss", "accuracy", "f1_macro", "examples"})

    def test_evaluates_in_eval_mode_and_restores_training_mode(self) -> None:
        model = ModeRecordingClassifier()
        model.train()
        self.evaluate(model)
        self.assertTrue(model.training)
        self.assertTrue(model.forward_training_flags)
        self.assertFalse(any(model.forward_training_flags))

    def test_eval_mode_is_kept_when_the_model_started_in_eval_mode(self) -> None:
        model = ModeRecordingClassifier()
        model.eval()
        self.evaluate(model)
        self.assertFalse(model.training)

    def test_three_way_evaluation_does_not_change_model_parameters(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(2, 2)
        state_before = clone_model_state(model.state_dict())
        self.evaluate(model)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state_before[name]))

    def test_single_dataloader_instead_of_a_list_is_rejected(self) -> None:
        global_loader, target_loader, non_target_loaders = self.loaders()
        with self.assertRaisesRegex(TypeError, "single DataLoader"):
            evaluate_three_way(
                IdentityClassifier(),
                global_loader,
                target_loader,
                non_target_loaders[0],
                self.loss_function,
                self.device,
            )

    def test_empty_non_target_list_is_rejected(self) -> None:
        global_loader, target_loader, _ = self.loaders()
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_three_way(
                IdentityClassifier(),
                global_loader,
                target_loader,
                [],
                self.loss_function,
                self.device,
            )

    def test_target_loader_reused_as_non_target_is_rejected(self) -> None:
        global_loader, target_loader, non_target_loaders = self.loaders()
        with self.assertRaisesRegex(ValueError, "target client loader"):
            evaluate_three_way(
                IdentityClassifier(),
                global_loader,
                target_loader,
                [non_target_loaders[0], target_loader],
                self.loss_function,
                self.device,
            )

    def test_global_loader_reused_as_client_loader_is_rejected(self) -> None:
        global_loader, target_loader, non_target_loaders = self.loaders()
        for target, non_targets in (
            (global_loader, non_target_loaders),
            (target_loader, [global_loader]),
        ):
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "global test loader"):
                    evaluate_three_way(
                        IdentityClassifier(),
                        global_loader,
                        target,
                        non_targets,
                        self.loss_function,
                        self.device,
                    )

    def test_duplicate_non_target_loader_is_rejected(self) -> None:
        global_loader, target_loader, non_target_loaders = self.loaders()
        with self.assertRaisesRegex(ValueError, "more than once"):
            evaluate_three_way(
                IdentityClassifier(),
                global_loader,
                target_loader,
                [non_target_loaders[0], non_target_loaders[0]],
                self.loss_function,
                self.device,
            )

    def test_empty_non_target_loader_raises_and_restores_mode(self) -> None:
        global_loader, target_loader, non_target_loaders = self.loaders()
        empty_loader = DataLoader(
            TensorDataset(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long)),
            batch_size=2,
        )
        model = IdentityClassifier()
        model.train()
        with self.assertRaisesRegex(ValueError, "loader 1 is empty"):
            evaluate_three_way(
                model,
                global_loader,
                target_loader,
                [non_target_loaders[0], empty_loader],
                self.loss_function,
                self.device,
            )
        self.assertTrue(model.training)

if __name__ == "__main__":
    unittest.main()
