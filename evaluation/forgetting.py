"""Forgetting metrics: compare a model M_x with the gold standard M_retrain^(-A).

Protocol source: ``Ke_hoach_v4_FU.md`` section 6 (evaluation protocol).  M_x is any model evaluated the same way as M_retrain^(-A): the
original model M, a second retrain, or later an unlearned model M_FU.

For every split (global test, target client, non-target clients) and for the
target split divided into shared and target-dominant classes, with N examples
(x_i, y_i) and predicted labels p_x(i), p_r(i) of M_x and M_retrain^(-A):

    accuracy_x       = (1/N) * #{i : p_x(i) = y_i}
    accuracy_retrain = (1/N) * #{i : p_r(i) = y_i}
    accuracy_gap     = accuracy_x - accuracy_retrain          (signed)
    agreement        = (1/N) * #{i : p_x(i) = p_r(i)}

Every row also stores the integer counts behind these fractions (examples,
correct_x, correct_retrain, agree).  Decisions (see ``within_noise_floor``)
are taken on the integer counts; the floats are for display only, because
two floats that should be equal can differ in the last digit.

Agreement does not look at the true label: two models that make the same
mistake agree.  Parameter distances are deliberately not computed here.

The accuracies of the three splits come from ``evaluation/three_way.py``;
this module only compares the stored labels and predictions of two such
evaluations.
"""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence

import numpy as np

from evaluation.three_way import ThreeWayEvaluation


SPLITS = ("global_test", "target_client", "non_target_clients")


def classify_classes(
    client_class_counts: Mapping[int, Sequence[int]],
    target_client: int,
    dominance_threshold: float,
) -> dict[str, list]:
    """Split the classes into target-dominant and shared for one target client.

    ``client_class_counts[k][c]`` is n_{k,c}, the number of TRAINING examples
    of class c held by client k.  The target's share of class c is

        share_c = n_{A,c} / sum_k n_{k,c}

    Rule (plan section 2, change 3): class c is target-dominant if
    ``share_c >= dominance_threshold`` (the config value is 0.5: the target
    holds at least half of that class); every other class is shared.  A class
    that no client holds has share 0 and is therefore shared.
    """
    if target_client not in client_class_counts:
        raise ValueError(f"Unknown target client {target_client!r}.")
    if not 0 < dominance_threshold <= 1:
        raise ValueError("dominance_threshold must be in the interval (0, 1].")
    target_counts = client_class_counts[target_client]
    for counts in client_class_counts.values():
        if len(counts) != len(target_counts):
            raise ValueError("Every client needs one count per class.")
        if any(count < 0 for count in counts):
            raise ValueError("Class counts cannot be negative.")
    target_share: list[float] = []
    target_dominant: list[int] = []
    shared: list[int] = []
    for class_id in range(len(target_counts)):
        class_total = sum(counts[class_id] for counts in client_class_counts.values())
        share = target_counts[class_id] / class_total if class_total > 0 else 0.0
        target_share.append(share)
        if share >= dominance_threshold:
            target_dominant.append(class_id)
        else:
            shared.append(class_id)
    return {"target_share": target_share, "target_dominant": target_dominant, "shared": shared}


def prediction_agreement(predictions_x: np.ndarray, predictions_retrain: np.ndarray) -> float:
    """Fraction of examples on which the two models predict the same label."""
    if len(predictions_x) != len(predictions_retrain):
        raise ValueError("The two models must be evaluated on the same examples.")
    if len(predictions_x) == 0:
        raise ValueError("Agreement is undefined on zero examples.")
    same = int((np.asarray(predictions_x) == np.asarray(predictions_retrain)).sum())
    return same / len(predictions_x)


def compare_predictions(
    labels: np.ndarray, predictions_x: np.ndarray, predictions_retrain: np.ndarray
) -> dict[str, float | int | None]:
    """Counts, accuracy of both models, their accuracy gap and their agreement.

    With zero examples (for example: no target-dominant class under IID)
    every metric is ``None``; an empty group has no accuracy.
    """
    number_of_examples = len(labels)
    if number_of_examples == 0:
        return {
            "examples": 0, "correct_x": 0, "correct_retrain": 0, "agree": 0,
            "accuracy_x": None, "accuracy_retrain": None,
            "accuracy_gap": None, "agreement": None,
        }
    correct_x = int((predictions_x == labels).sum())
    correct_retrain = int((predictions_retrain == labels).sum())
    agree = int((predictions_x == predictions_retrain).sum())
    return {
        "examples": number_of_examples,
        "correct_x": correct_x,
        "correct_retrain": correct_retrain,
        "agree": agree,
        "accuracy_x": correct_x / number_of_examples,
        "accuracy_retrain": correct_retrain / number_of_examples,
        "accuracy_gap": (correct_x - correct_retrain) / number_of_examples,
        "agreement": agree / number_of_examples,
    }


def compare_to_retrain(
    evaluation_x: ThreeWayEvaluation,
    evaluation_retrain: ThreeWayEvaluation,
    target_dominant_classes: Collection[int],
) -> dict[str, dict[str, float | int | None]]:
    """Compare M_x with M_retrain^(-A) on every split and class group.

    Returns one ``compare_predictions`` row for each of: ``global_test``,
    ``target_client``, ``non_target_clients``, and the target split restricted
    to ``target_client_dominant_classes`` and ``target_client_shared_classes``
    (grouped by the TRUE label of each target example).

    Both evaluations must come from the same unshuffled loaders, so that
    position i is the same example for both models; equal label arrays are
    checked as a guard.
    """
    rows: dict[str, dict[str, float | int | None]] = {}
    for split in SPLITS:
        x = getattr(evaluation_x, split)
        retrain = getattr(evaluation_retrain, split)
        if not np.array_equal(x.labels, retrain.labels):
            raise ValueError(f"The two evaluations of {split} are not on the same examples.")
        rows[split] = compare_predictions(x.labels, x.predictions, retrain.predictions)

    target_x = evaluation_x.target_client
    target_retrain = evaluation_retrain.target_client
    is_dominant = np.isin(target_x.labels, list(target_dominant_classes))
    for row_name, mask in (
        ("target_client_dominant_classes", is_dominant),
        ("target_client_shared_classes", ~is_dominant),
    ):
        rows[row_name] = compare_predictions(
            target_x.labels[mask], target_x.predictions[mask], target_retrain.predictions[mask]
        )
    return rows



# --- Noise floor (separate change: notes/changes/2026-10-10_noise_floor.md) ---

DECISION_RULES = ("max_of_pairs_inclusive", "max_of_pairs_strict")


def pairwise_retrain_rows(
    retrain_evaluations: Sequence[ThreeWayEvaluation],
    target_dominant_classes: Collection[int],
) -> list[dict]:
    """Compare every pair (i < j) of M_retrain^(-A) runs with ``compare_to_retrain``.

    k retrains give k * (k - 1) / 2 pairs.  In pair (i, j) retrain j plays
    the role of M_x and retrain i is the reference.
    """
    if len(retrain_evaluations) < 2:
        raise ValueError("A noise floor needs at least two retrains.")
    pairs = []
    for first in range(len(retrain_evaluations)):
        for second in range(first + 1, len(retrain_evaluations)):
            rows = compare_to_retrain(
                retrain_evaluations[second], retrain_evaluations[first], target_dominant_classes
            )
            pairs.append({"first": first, "second": second, "rows": rows})
    return pairs


def summarise_pairs(pairs: Sequence[Mapping]) -> dict[str, dict[str, float | int | None]]:
    """Describe, per row, how far apart retrains are from each other.

    All retrains never saw the target client, so their differences are
    run-to-run noise.  With d_p = |correct_second - correct_first| and a_p =
    agree for pair p, on a row of N examples:

        max_abs_correct_difference = max_p d_p        (integer)
        min_agree                  = min_p a_p        (integer)
        abs_accuracy_gap_mean/std/max = mean, std, max of d_p / N
        agreement_mean/std/min        = mean, std, min of a_p / N

    The std is the population standard deviation over the pairs (divide by
    the number of pairs).  The pairs share retrains, so they are not
    independent; the std describes the spread, it is not a standard error.
    This function only describes; ``within_noise_floor`` decides.
    """
    summary: dict[str, dict[str, float | int | None]] = {}
    for row_name in pairs[0]["rows"]:
        rows = [pair["rows"][row_name] for pair in pairs]
        examples = rows[0]["examples"]
        if any(row["examples"] != examples for row in rows):
            raise ValueError(f"Pairs disagree on the number of examples in {row_name}.")
        if examples == 0:
            summary[row_name] = {"pairs": len(rows), "examples": 0}
            continue
        differences = [abs(row["correct_x"] - row["correct_retrain"]) for row in rows]
        agreements = [row["agree"] for row in rows]
        summary[row_name] = {
            "pairs": len(rows),
            "examples": examples,
            "max_abs_correct_difference": max(differences),
            "min_agree": min(agreements),
            "abs_accuracy_gap_mean": statistics.mean(differences) / examples,
            "abs_accuracy_gap_std": statistics.pstdev(differences) / examples,
            "abs_accuracy_gap_max": max(differences) / examples,
            "agreement_mean": statistics.mean(agreements) / examples,
            "agreement_std": statistics.pstdev(agreements) / examples,
            "agreement_min": min(agreements) / examples,
        }
    return summary


def within_noise_floor(
    comparison: Mapping[str, Mapping[str, float | int | None]],
    summary: Mapping[str, Mapping[str, float | int | None]],
    decision_rule: str,
) -> dict[str, dict[str, bool | None]]:
    """Say, per row, whether M_x is as close to a retrain as retrains are to each other.

    ``comparison`` is ``compare_to_retrain(M_x, reference retrain)`` and
    ``summary`` is ``summarise_pairs(...)`` on the same examples.  Both
    comparisons are made on integer counts (same N on both sides):

        "max_of_pairs_inclusive":
            accuracy_gap_within = |correct_x - correct_retrain| <= max_abs_correct_difference
            agreement_within    = agree >= min_agree
        "max_of_pairs_strict": the same with < and >  (a tie is outside)

    ``decision_rule`` has no default: which rule is right is a decision of
    the thesis author, taken from the config.  A row with no examples gives
    ``None`` for both.
    """
    if decision_rule not in DECISION_RULES:
        raise ValueError(f"decision_rule must be one of {DECISION_RULES}, got {decision_rule!r}.")
    result: dict[str, dict[str, bool | None]] = {}
    for row_name, row in comparison.items():
        floor_row = summary[row_name]
        if row["examples"] != floor_row["examples"]:
            raise ValueError(f"{row_name}: comparison and noise floor use different examples.")
        if row["examples"] == 0:
            result[row_name] = {"accuracy_gap_within": None, "agreement_within": None}
            continue
        difference = abs(row["correct_x"] - row["correct_retrain"])
        if decision_rule == "max_of_pairs_inclusive":
            gap_within = difference <= floor_row["max_abs_correct_difference"]
            agreement_within = row["agree"] >= floor_row["min_agree"]
        else:
            gap_within = difference < floor_row["max_abs_correct_difference"]
            agreement_within = row["agree"] > floor_row["min_agree"]
        result[row_name] = {"accuracy_gap_within": gap_within, "agreement_within": agreement_within}
    return result
