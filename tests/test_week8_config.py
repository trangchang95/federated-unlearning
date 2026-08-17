"""Contract checks for the saved Month 2 Week 8 experiment config.

These tests exercise configuration validation only.  They never load MNIST or
start either training path, so failures identify protocol-label mistakes before
an expensive experiment begins.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.iid.month2_week8_mnist_fedavg import validate_week8_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)


class Week8ConfigValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))

    def changed_config(self, **changes: object) -> dict:
        config = copy.deepcopy(self.canonical)
        config.update(changes)
        return config

    def assert_rejected(self, expected_message: str, **changes: object) -> None:
        with self.assertRaisesRegex(ValueError, expected_message):
            validate_week8_config(self.changed_config(**changes))

    def test_canonical_saved_config_passes_validation(self) -> None:
        validate_week8_config(copy.deepcopy(self.canonical))

    def test_truth_labels_are_enforced(self) -> None:
        cases = (
            ({"dataset": "CIFAR10"}, "supports MNIST only"),
            ({"dataset_version": "unknown"}, "dataset_version does not describe"),
            ({"data_subdirectory": "../outside"}, "must be mnist"),
            ({"model": "CNN"}, "model must be SimpleMLP"),
            ({"architecture": "unknown"}, "architecture does not match"),
            ({"preprocessing": "Normalize"}, "preprocessing does not match"),
            ({"algorithm": "FedProx"}, "algorithm label is incorrect"),
            ({"partition_strategy": "dirichlet_noniid"}, "must use the IID"),
            ({"optimizer": "Adam"}, "must use plain SGD"),
            ({"optimizer_details": "momentum=0.9"}, "optimizer_details do not match"),
            ({"loss_function": "MSELoss"}, "uses CrossEntropyLoss"),
            ({"device": "cuda"}, "runs on CPU only"),
            ({"strict_environment": False}, "strict_environment must be true"),
            ({"environment_manifest": "other.json"}, "Month 2 CPU runtime"),
            ({"communication_cost_unit": "MiB"}, "must be bytes"),
            ({"selection_rule": "floor"}, "selection rule does not match"),
            ({"num_workers": 1}, "num_workers must remain 0"),
        )
        for changes, expected_message in cases:
            with self.subTest(changes=changes):
                self.assert_rejected(expected_message, **changes)

    def test_outcome_placeholders_cannot_pretend_results_exist(self) -> None:
        cases = (
            ({"accuracy": 0.9}, "accuracy and f1 must be null"),
            ({"f1": 0.9}, "accuracy and f1 must be null"),
            ({"unlearning_time": 1.0}, "must be 0.0"),
            ({"communication_cost": 123}, "must be 0"),
        )
        for changes, expected_message in cases:
            with self.subTest(changes=changes):
                self.assert_rejected(expected_message, **changes)

    def test_non_iid_and_unlearning_fields_are_rejected(self) -> None:
        cases = (
            ({"alpha": 0.5}, "alpha must be null"),
            ({"target_client": 0}, "target_client must be null"),
        )
        for changes, expected_message in cases:
            with self.subTest(changes=changes):
                self.assert_rejected(expected_message, **changes)

    def test_partial_client_participation_is_rejected(self) -> None:
        self.assert_rejected("requires full client participation", client_fraction=0.8)

    def test_unmatched_training_budget_is_rejected(self) -> None:
        self.assert_rejected(
            "must equal number_of_rounds \\* local_epochs",
            centralized_epochs=self.canonical["centralized_epochs"] + 1,
        )

    def test_client_count_cannot_exceed_smallest_partition_source(self) -> None:
        self.assert_rejected(
            "exceeds the smallest train/test partition source",
            number_of_clients=self.canonical["expected_test_examples"] + 1,
        )

    def test_positive_integer_fields_reject_zero_and_boolean_values(self) -> None:
        fields = (
            "number_of_clients",
            "local_epochs",
            "batch_size",
            "number_of_rounds",
            "centralized_epochs",
            "hidden_units",
        )
        for field in fields:
            for invalid_value in (0, True):
                with self.subTest(field=field, invalid_value=invalid_value):
                    self.assert_rejected(
                        f"{field} must be a positive integer", **{field: invalid_value}
                    )

    def test_learning_rate_must_be_finite_and_positive(self) -> None:
        for invalid_value in (0.0, -0.1, float("inf"), float("nan"), True, "0.1"):
            with self.subTest(invalid_value=invalid_value):
                self.assert_rejected(
                    "learning_rate must be finite and positive",
                    learning_rate=invalid_value,
                )

    def test_validation_fraction_must_be_strictly_between_zero_and_one(self) -> None:
        for invalid_value in (
            0.0,
            1.0,
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            True,
            "0.15",
        ):
            with self.subTest(invalid_value=invalid_value):
                self.assert_rejected(
                    r"validation_fraction must be in \(0, 1\)",
                    validation_fraction=invalid_value,
                )
        self.assert_rejected(
            "canonical Week 8 train/validation split uses 0.15",
            validation_fraction=0.2,
        )

    def test_seed_fields_reject_negative_and_boolean_values(self) -> None:
        fields = (
            "random_seed",
            "split_seed",
            "train_partition_seed",
            "test_partition_seed",
            "centralized_loader_seed",
            "model_initialization_seed",
        )
        for field in fields:
            for invalid_value in (-1, True):
                with self.subTest(field=field, invalid_value=invalid_value):
                    self.assert_rejected(
                        f"{field} must be a non-negative integer",
                        **{field: invalid_value},
                    )

    def test_output_path_and_revision_must_be_safe_nonempty_strings(self) -> None:
        for field in ("experiment_name", "output_subdirectory", "code_revision"):
            for invalid_value in ("", "   ", None):
                with self.subTest(field=field, invalid_value=invalid_value):
                    self.assert_rejected(
                        f"{field} must be a non-empty string",
                        **{field: invalid_value},
                    )
        self.assert_rejected(
            "must stay inside results", output_subdirectory="../outside"
        )


if __name__ == "__main__":
    unittest.main()
