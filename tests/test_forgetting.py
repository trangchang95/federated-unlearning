"""Hand-checkable toy cases for the forgetting metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.forgetting import (
    classify_classes,
    compare_predictions,
    compare_to_retrain,
    prediction_agreement,
)
from evaluation.three_way import ThreeWayEvaluation, evaluate_three_way
from evaluation.utility import ClassificationEvaluation


def evaluation(labels: list[int], predictions: list[int]) -> ClassificationEvaluation:
    """A split evaluation holding only what the forgetting metrics read."""
    return ClassificationEvaluation(
        loss=0.0, accuracy=0.0, f1_macro=0.0,
        labels=np.array(labels), predictions=np.array(predictions),
    )


def three_way(global_test, target_client, non_target_clients) -> ThreeWayEvaluation:
    return ThreeWayEvaluation(
        global_test=evaluation(*global_test),
        target_client=evaluation(*target_client),
        non_target_clients=evaluation(*non_target_clients),
    )


class ClassifyClassesTests(unittest.TestCase):
    # Training examples per class (columns: class 0, 1, 2) for three clients.
    counts = {0: [6, 1, 0], 1: [2, 1, 5], 2: [2, 2, 5]}  # class totals: 10, 4, 10

    def test_target_holding_at_least_half_of_a_class_dominates_it(self) -> None:
        result = classify_classes(self.counts, target_client=0, dominance_threshold=0.5)
        # Client 0 holds 6/10 of class 0, 1/4 of class 1 and 0/10 of class 2.
        self.assertEqual(result["target_share"], [0.6, 0.25, 0.0])
        self.assertEqual(result["target_dominant"], [0])
        self.assertEqual(result["shared"], [1, 2])

    def test_exactly_the_threshold_counts_as_dominant(self) -> None:
        # Client 1 holds 5/10 = 0.5 of class 2 (and 1/4 of class 1).
        self.assertEqual(classify_classes(self.counts, 1, 0.5)["target_dominant"], [2])
        # Client 2 holds 2/4 = 0.5 of class 1 and 5/10 = 0.5 of class 2.
        self.assertEqual(classify_classes(self.counts, 2, 0.5)["target_dominant"], [1, 2])

    def test_threshold_comes_from_the_caller(self) -> None:
        # With a 0.25 threshold client 0 also dominates class 1 (1/4).
        self.assertEqual(classify_classes(self.counts, 0, 0.25)["target_dominant"], [0, 1])

    def test_equal_shares_give_no_dominant_class(self) -> None:
        iid_like = {0: [5, 5], 1: [5, 5], 2: [5, 5]}  # every share is 1/3
        result = classify_classes(iid_like, 0, 0.5)
        self.assertEqual(result["target_dominant"], [])
        self.assertEqual(result["shared"], [0, 1])

    def test_class_that_nobody_holds_is_shared(self) -> None:
        result = classify_classes({0: [3, 0], 1: [1, 0]}, 0, 0.5)
        self.assertEqual(result["target_share"], [0.75, 0.0])
        self.assertEqual(result["shared"], [1])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_classes(self.counts, 9, 0.5)
        for threshold in (0.0, 1.5):
            with self.assertRaises(ValueError):
                classify_classes(self.counts, 0, threshold)
        with self.assertRaises(ValueError):  # client 1 has only two class counts
            classify_classes({0: [1, 2, 3], 1: [1, 2]}, 0, 0.5)
        with self.assertRaises(ValueError):  # a count cannot be negative
            classify_classes({0: [1, 2], 1: [-1, 2]}, 0, 0.5)


class PredictionTests(unittest.TestCase):
    def test_agreement_is_the_fraction_of_equal_predictions(self) -> None:
        # Equal at positions 0, 1, 3 -> 3/4.
        self.assertEqual(prediction_agreement(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2])), 0.75)

    def test_agreement_needs_the_same_non_empty_examples(self) -> None:
        with self.assertRaises(ValueError):
            prediction_agreement(np.array([0, 1]), np.array([0]))
        with self.assertRaises(ValueError):
            prediction_agreement(np.array([]), np.array([]))

    def test_accuracy_gap_and_agreement_on_four_examples(self) -> None:
        labels = np.array([0, 0, 1, 1])
        predictions_x = np.array([0, 0, 1, 0])        # 3 of 4 correct
        predictions_retrain = np.array([0, 1, 1, 0])  # 2 of 4 correct
        row = compare_predictions(labels, predictions_x, predictions_retrain)
        self.assertEqual(row["examples"], 4)
        # Integer counts behind the fractions.
        self.assertEqual((row["correct_x"], row["correct_retrain"], row["agree"]), (3, 2, 3))
        self.assertEqual(row["accuracy_x"], 0.75)
        self.assertEqual(row["accuracy_retrain"], 0.5)
        self.assertEqual(row["accuracy_gap"], 0.25)  # signed: M_x minus M_retrain
        # They differ only at position 1; at position 3 both are wrong but agree.
        self.assertEqual(row["agreement"], 0.75)

    def test_empty_group_has_no_metrics(self) -> None:
        empty = np.array([], dtype=int)
        self.assertEqual(
            compare_predictions(empty, empty, empty),
            {"examples": 0, "correct_x": 0, "correct_retrain": 0, "agree": 0,
             "accuracy_x": None, "accuracy_retrain": None,
             "accuracy_gap": None, "agreement": None},
        )


class CompareToRetrainTests(unittest.TestCase):
    def setUp(self) -> None:
        # Target split: classes 0 and 1 are shared, class 2 is target-dominant.
        target_labels = [0, 0, 1, 2, 2]
        self.model_x = three_way(
            ([0, 1], [0, 1]),                    # global: 2/2
            (target_labels, [0, 0, 1, 2, 2]),    # target: 5/5
            ([0, 1, 1, 1], [0, 1, 1, 0]),        # non-target: 3/4
        )
        self.retrain = three_way(
            ([0, 1], [0, 0]),                    # global: 1/2
            (target_labels, [0, 1, 1, 0, 2]),    # target: 3/5
            ([0, 1, 1, 1], [0, 1, 1, 0]),        # non-target: 3/4, same predictions
        )

    def test_rows_for_the_three_splits(self) -> None:
        rows = compare_to_retrain(self.model_x, self.retrain, target_dominant_classes=[2])
        self.assertEqual(rows["global_test"]["accuracy_gap"], 0.5)   # 1.0 - 0.5
        self.assertEqual(rows["global_test"]["agreement"], 0.5)
        self.assertAlmostEqual(rows["target_client"]["accuracy_gap"], 0.4)  # 1.0 - 0.6
        self.assertEqual(rows["target_client"]["agreement"], 0.6)   # differ at positions 1, 3
        self.assertEqual(rows["non_target_clients"]["accuracy_gap"], 0.0)
        self.assertEqual(rows["non_target_clients"]["agreement"], 1.0)

    def test_target_split_is_divided_by_true_label(self) -> None:
        rows = compare_to_retrain(self.model_x, self.retrain, target_dominant_classes=[2])
        dominant = rows["target_client_dominant_classes"]  # the two class-2 examples
        self.assertEqual(dominant["examples"], 2)
        self.assertEqual(dominant["accuracy_x"], 1.0)
        self.assertEqual(dominant["accuracy_retrain"], 0.5)
        self.assertEqual(dominant["agreement"], 0.5)
        shared = rows["target_client_shared_classes"]      # the three class-0/1 examples
        self.assertEqual(shared["examples"], 3)
        self.assertAlmostEqual(shared["accuracy_retrain"], 2 / 3)
        self.assertAlmostEqual(shared["accuracy_gap"], 1 / 3)
        self.assertAlmostEqual(shared["agreement"], 2 / 3)

    def test_no_dominant_class_gives_an_empty_dominant_row(self) -> None:
        rows = compare_to_retrain(self.model_x, self.retrain, target_dominant_classes=[])
        self.assertEqual(rows["target_client_dominant_classes"]["examples"], 0)
        self.assertIsNone(rows["target_client_dominant_classes"]["accuracy_gap"])
        self.assertEqual(rows["target_client_shared_classes"], rows["target_client"])

    def test_a_model_compared_with_itself_has_zero_gap_and_full_agreement(self) -> None:
        for row in compare_to_retrain(self.retrain, self.retrain, [2]).values():
            self.assertEqual(row["accuracy_gap"], 0.0)
            self.assertEqual(row["agreement"], 1.0)

    def test_evaluations_on_different_examples_are_rejected(self) -> None:
        other = three_way(([1, 0], [0, 0]), ([0, 0, 1, 2, 2], [0] * 5), ([0, 1, 1, 1], [0] * 4))
        with self.assertRaises(ValueError):
            compare_to_retrain(self.model_x, other, [2])

    def test_works_on_real_three_way_evaluations(self) -> None:
        # Identity "model": the logits are the features, so predictions are known.
        def loader(logits, labels):
            return DataLoader(TensorDataset(torch.tensor(logits), torch.tensor(labels)), batch_size=2)

        loaders = (
            loader([[5.0, 0.0], [0.0, 5.0]], [0, 0]),   # predictions 0, 1 -> 1/2 correct
            loader([[5.0, 0.0], [5.0, 0.0]], [0, 1]),   # predictions 0, 0 -> 1/2 correct
            [loader([[0.0, 5.0]], [1])],                # prediction 1 -> 1/1 correct
        )
        result = evaluate_three_way(nn.Identity(), *loaders, nn.CrossEntropyLoss(), torch.device("cpu"))
        rows = compare_to_retrain(result, result, target_dominant_classes=[1])
        self.assertEqual(rows["target_client"]["accuracy_x"], result.target_client.accuracy)
        self.assertEqual(rows["target_client_dominant_classes"]["accuracy_x"], 0.0)  # label 1 predicted 0
        self.assertEqual(rows["target_client_shared_classes"]["accuracy_x"], 1.0)


if __name__ == "__main__":
    unittest.main()
