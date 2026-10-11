"""Hand-checkable toy cases for the noise floor (k retrains, different seeds)."""

from __future__ import annotations

import unittest

from evaluation.forgetting import (
    compare_to_retrain,
    pairwise_retrain_rows,
    summarise_pairs,
    within_noise_floor,
)
from experiments.unlearning.month4_week14_mnist_noise_floor import retrain_seed_plan
from tests.test_forgetting import three_way


TARGET_LABELS = [0, 0, 1, 2, 2]  # classes 0 and 1 shared, class 2 target-dominant
NON_TARGET = ([0, 1, 1, 1], [0, 1, 1, 0])


def retrains():
    """Three toy retrains.  Target predictions: 3/5, 4/5 and 5/5 correct."""
    return [
        three_way(([0, 1], [0, 0]), (TARGET_LABELS, [0, 1, 1, 0, 2]), NON_TARGET),  # R0
        three_way(([0, 1], [0, 1]), (TARGET_LABELS, [0, 0, 1, 0, 2]), NON_TARGET),  # R1
        three_way(([0, 1], [0, 1]), (TARGET_LABELS, [0, 0, 1, 2, 2]), NON_TARGET),  # R2
    ]


def row(examples, correct_x, correct_retrain, agree):
    return {"examples": examples, "correct_x": correct_x, "correct_retrain": correct_retrain, "agree": agree}


class PairwiseTests(unittest.TestCase):
    def test_three_retrains_give_three_pairs(self) -> None:
        pairs = pairwise_retrain_rows(retrains(), target_dominant_classes=[2])
        self.assertEqual([(pair["first"], pair["second"]) for pair in pairs], [(0, 1), (0, 2), (1, 2)])
        # Pair (0, 1) on the target split: R1 has 4 correct, R0 has 3, they agree on 4 of 5.
        target = pairs[0]["rows"]["target_client"]
        self.assertEqual((target["correct_x"], target["correct_retrain"], target["agree"]), (4, 3, 4))

    def test_one_retrain_is_not_enough(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_retrain_rows(retrains()[:1], [2])

    def test_summary_of_the_target_split(self) -> None:
        summary = summarise_pairs(pairwise_retrain_rows(retrains(), [2]))["target_client"]
        # |correct difference| per pair: (0,1) 1, (0,2) 2, (1,2) 1.  agree per pair: 4, 3, 4.
        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["examples"], 5)
        self.assertEqual(summary["max_abs_correct_difference"], 2)
        self.assertEqual(summary["min_agree"], 3)
        self.assertAlmostEqual(summary["abs_accuracy_gap_mean"], (1 + 2 + 1) / 3 / 5)
        self.assertAlmostEqual(summary["abs_accuracy_gap_max"], 2 / 5)
        # population std of (1, 2, 1) is sqrt(2/9); divided by N = 5
        self.assertAlmostEqual(summary["abs_accuracy_gap_std"], (2 / 9) ** 0.5 / 5)
        self.assertAlmostEqual(summary["agreement_mean"], (4 + 3 + 4) / 3 / 5)
        self.assertAlmostEqual(summary["agreement_min"], 3 / 5)

    def test_summary_of_other_rows(self) -> None:
        summary = summarise_pairs(pairwise_retrain_rows(retrains(), [2]))
        # Global test: R0 has 1/2 correct, R1 and R2 have 2/2.
        self.assertEqual(summary["global_test"]["max_abs_correct_difference"], 1)
        self.assertEqual(summary["global_test"]["min_agree"], 1)
        # Non-target predictions are identical in all three retrains.
        self.assertEqual(summary["non_target_clients"]["max_abs_correct_difference"], 0)
        self.assertEqual(summary["non_target_clients"]["min_agree"], 4)
        self.assertEqual(summary["non_target_clients"]["abs_accuracy_gap_std"], 0.0)
        # Class-2 target examples: R0 and R1 have 1/2 correct, R2 has 2/2.
        self.assertEqual(summary["target_client_dominant_classes"]["max_abs_correct_difference"], 1)

    def test_empty_class_group_has_only_counts(self) -> None:
        summary = summarise_pairs(pairwise_retrain_rows(retrains(), target_dominant_classes=[]))
        self.assertEqual(summary["target_client_dominant_classes"], {"pairs": 3, "examples": 0})


class WithinNoiseFloorTests(unittest.TestCase):
    # Floor on a row of 10 examples: retrains differ by at most 2 correct
    # answers and agree on at least 8 examples.
    summary = {"a": {"pairs": 3, "examples": 10, "max_abs_correct_difference": 2, "min_agree": 8}}

    def check(self, correct_x, correct_retrain, agree, rule="max_of_pairs_inclusive"):
        comparison = {"a": row(10, correct_x, correct_retrain, agree)}
        return within_noise_floor(comparison, self.summary, rule)["a"]

    def test_small_gap_and_high_agreement_are_within(self) -> None:
        # |6 - 7| = 1 <= 2 and 9 >= 8; the sign of the gap does not matter.
        self.assertEqual(self.check(6, 7, 9), {"accuracy_gap_within": True, "agreement_within": True})

    def test_large_gap_or_low_agreement_are_outside(self) -> None:
        # |9 - 6| = 3 > 2 and 7 < 8
        self.assertEqual(self.check(9, 6, 7), {"accuracy_gap_within": False, "agreement_within": False})
        # The two answers are independent: same accuracy, different predictions.
        self.assertEqual(self.check(6, 6, 7), {"accuracy_gap_within": True, "agreement_within": False})

    def test_a_tie_is_decided_by_the_rule_not_by_rounding(self) -> None:
        # Exactly at the floor: |8 - 6| = 2 and agree = 8.
        self.assertEqual(self.check(8, 6, 8), {"accuracy_gap_within": True, "agreement_within": True})
        self.assertEqual(
            self.check(8, 6, 8, rule="max_of_pairs_strict"),
            {"accuracy_gap_within": False, "agreement_within": False},
        )

    def test_integer_counts_avoid_a_wrong_float_comparison(self) -> None:
        # M_x: 9 of 10 correct, reference: 6 of 10.  Floor pair: 7 of 10 vs 4 of 10.
        # Both gaps are exactly 3 answers, so M_x is AT the floor (inclusive: within).
        # As floats the two "equal" gaps differ in the last digit, and the
        # comparison gap_x <= floor would wrongly say "outside":
        self.assertGreater(0.9 - 0.6, 0.7 - 0.4)
        summary = {"a": {"pairs": 1, "examples": 10, "max_abs_correct_difference": 7 - 4, "min_agree": 0}}
        comparison = {"a": row(10, 9, 6, 5)}
        self.assertTrue(
            within_noise_floor(comparison, summary, "max_of_pairs_inclusive")["a"]["accuracy_gap_within"]
        )

    def test_empty_row_gives_none(self) -> None:
        result = within_noise_floor(
            {"empty": row(0, 0, 0, 0)}, {"empty": {"pairs": 3, "examples": 0}}, "max_of_pairs_inclusive"
        )
        self.assertEqual(result["empty"], {"accuracy_gap_within": None, "agreement_within": None})

    def test_rule_is_required_and_examples_must_match(self) -> None:
        with self.assertRaises(ValueError):
            self.check(6, 7, 9, rule=None)
        with self.assertRaises(ValueError):
            within_noise_floor({"a": row(9, 6, 7, 8)}, self.summary, "max_of_pairs_inclusive")

    def test_end_to_end_with_the_toy_models(self) -> None:
        # M_x predicts every target example correctly; the reference R0 has 3 of 5.
        model_x = three_way(([0, 1], [0, 1]), (TARGET_LABELS, TARGET_LABELS), NON_TARGET)
        comparison = compare_to_retrain(model_x, retrains()[0], [2])
        summary = summarise_pairs(pairwise_retrain_rows(retrains(), [2]))
        result = within_noise_floor(comparison, summary, "max_of_pairs_inclusive")
        self.assertTrue(result["target_client"]["accuracy_gap_within"])    # |5 - 3| = 2 <= 2
        self.assertTrue(result["target_client"]["agreement_within"])       # agree 3 >= 3
        strict = within_noise_floor(comparison, summary, "max_of_pairs_strict")
        self.assertFalse(strict["target_client"]["accuracy_gap_within"])   # 2 < 2 is false


class RetrainSeedPlanTests(unittest.TestCase):
    def config(self, **changes):
        base = {
            "random_seed": 42, "model_initialization_seed": 42,
            "noise_floor_number_of_retrains": 3, "noise_floor_extra_seeds": [43, 44],
            "noise_floor_seed_mode": "training",
            "noise_floor_decision_rule": "max_of_pairs_inclusive",
        }
        return {**base, **changes}

    def seeds(self, **changes):
        return [
            (entry["index"], entry["training_seed"], entry["initial_weights_seed"])
            for entry in retrain_seed_plan(self.config(**changes))
        ]

    def test_training_mode_changes_only_the_training_seed(self) -> None:
        self.assertEqual(self.seeds(), [(0, 42, 42), (1, 43, 42), (2, 44, 42)])

    def test_training_and_init_mode_changes_both(self) -> None:
        self.assertEqual(
            self.seeds(noise_floor_seed_mode="training_and_init"),
            [(0, 42, 42), (1, 43, 43), (2, 44, 44)],
        )

    def test_each_retrain_has_its_own_checkpoint_name(self) -> None:
        names = [entry["checkpoint"] for entry in retrain_seed_plan(self.config())]
        self.assertEqual(
            names,
            ["retrain_without_target.pt", "retrain_without_target_seed43.pt", "retrain_without_target_seed44.pt"],
        )

    def test_undecided_keys_stop_the_run(self) -> None:
        for key in (
            "noise_floor_number_of_retrains", "noise_floor_extra_seeds",
            "noise_floor_seed_mode", "noise_floor_decision_rule",
        ):
            with self.assertRaises(ValueError):
                retrain_seed_plan(self.config(**{key: None}))

    def test_inconsistent_seeds_are_errors(self) -> None:
        for changes in (
            {"noise_floor_extra_seeds": [43]},              # k = 3 needs two extra seeds
            {"noise_floor_extra_seeds": [43, 43]},          # repeated seed
            {"noise_floor_extra_seeds": [42, 44]},          # same as the first retrain
            {"noise_floor_seed_mode": "partition"},         # unknown mode
            {"noise_floor_number_of_retrains": 1, "noise_floor_extra_seeds": []},  # no pair
        ):
            with self.assertRaises(ValueError):
                retrain_seed_plan(self.config(**changes))


class ConfigGuardTests(unittest.TestCase):
    def test_descriptive_fields_must_name_what_the_code_runs(self) -> None:
        from experiments.unlearning.month4_week14_mnist_federated_retrain import check_supported_config

        config = {
            "dataset": "MNIST", "model": "SimpleMLP", "hidden_units": 128,
            "architecture": "784 -> 128 ReLU -> 10", "number_of_classes": 10,
            "preprocessing": "ToTensor only; pixels scaled to [0, 1]", "optimizer": "SGD",
            "optimizer_details": "plain SGD; momentum=0; weight_decay=0",
            "loss_function": "CrossEntropyLoss",
        }
        check_supported_config(config)  # the supported values pass
        for field, wrong in (("optimizer", "Adam"), ("model", "SmallCNN"), ("architecture", "784 -> 64 ReLU -> 10")):
            with self.assertRaises(ValueError):
                check_supported_config({**config, field: wrong})


if __name__ == "__main__":
    unittest.main()
