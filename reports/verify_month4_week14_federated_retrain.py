"""Consistency check of a saved Week 14 federated-retrain + noise-floor run.

    python reports/verify_month4_week14_federated_retrain.py \
        --config configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json

What PASS means: the two JSON files agree with the config, with each other and
with the saved checkpoints.  Predictions are recomputed from the checkpoints
with a forward pass written out in this file, and every count (examples,
correct, agree), class share, pairwise summary and within-floor flag is
recounted here with plain numpy, without calling ``evaluation/forgetting.py``.

What PASS does NOT mean:
- Nothing is retrained, so it does not show that the checkpoints are what the
  recorded seeds produce, nor that training is reproducible.
- The data and the client partition are rebuilt with the runner's own
  ``prepare_run``; a bug there is shared.  Loss and macro-F1 are recomputed
  with the runner's ``evaluate_three_way`` (only accuracy is independent).
- It says nothing about whether a model has unlearned anything.  Rows on
  which the original model M (never unlearned) is itself inside the noise
  floor are printed as UNINFORMATIVE.
- For a smoke run the worktree may be dirty; then only ``code_sha256``
  identifies the code, not the commit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import load_json_config
from experiments.unlearning.month4_week14_mnist_federated_retrain import (
    current_code_state,
    make_initial_state,
    prepare_run,
)
from experiments.unlearning.month4_week14_mnist_noise_floor import (
    evaluate_checkpoint,
    retrain_seed_plan,
)

ROWS = (
    "global_test", "target_client", "non_target_clients",
    "target_client_dominant_classes", "target_client_shared_classes",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def close(saved: float, recounted: float) -> bool:
    return math.isclose(saved, recounted, rel_tol=0.0, abs_tol=1e-9)


def images_and_labels(datasets: list) -> tuple[torch.Tensor, np.ndarray]:
    """Read every example of the given datasets one by one."""
    images, labels = [], []
    for dataset in datasets:
        for image, label in dataset:
            images.append(image)
            labels.append(int(label))
    return torch.stack(images), np.array(labels)


def predict(state: dict[str, torch.Tensor], images: torch.Tensor) -> np.ndarray:
    """SimpleMLP written out by hand: Flatten -> Linear -> ReLU -> Linear -> argmax."""
    flat = images.reshape(images.shape[0], -1)
    hidden = torch.relu(flat @ state["network.1.weight"].T + state["network.1.bias"])
    logits = hidden @ state["network.3.weight"].T + state["network.3.bias"]
    return logits.argmax(dim=1).numpy()


def count_row(labels: np.ndarray, predictions_x: np.ndarray, predictions_retrain: np.ndarray) -> dict:
    return {
        "examples": int(labels.shape[0]),
        "correct_x": int(np.sum(predictions_x == labels)),
        "correct_retrain": int(np.sum(predictions_retrain == labels)),
        "agree": int(np.sum(predictions_x == predictions_retrain)),
    }


def count_rows(labels: dict, model_x: dict, retrain: dict, dominant_mask: np.ndarray) -> dict:
    """The five rows for one (M_x, reference retrain) pair, as integer counts."""
    rows = {split: count_row(labels[split], model_x[split], retrain[split]) for split in ROWS[:3]}
    for row_name, mask in (("target_client_dominant_classes", dominant_mask), ("target_client_shared_classes", ~dominant_mask)):
        rows[row_name] = count_row(
            labels["target_client"][mask], model_x["target_client"][mask], retrain["target_client"][mask]
        )
    return rows


def require_rows_match(saved_rows: dict, counted_rows: dict, what: str) -> None:
    """Saved integer counts must equal the recount; saved floats must be count / N."""
    for row_name in ROWS:
        saved, counted = saved_rows[row_name], counted_rows[row_name]
        for key in ("examples", "correct_x", "correct_retrain", "agree"):
            require(saved[key] == counted[key], f"{what} {row_name}: {key} is {saved[key]}, recount is {counted[key]}")
        examples = counted["examples"]
        if examples == 0:
            require(saved["accuracy_gap"] is None and saved["agreement"] is None, f"{what} {row_name}: empty row has metrics")
            continue
        require(close(saved["accuracy_x"], counted["correct_x"] / examples), f"{what} {row_name}: accuracy_x")
        require(close(saved["accuracy_retrain"], counted["correct_retrain"] / examples), f"{what} {row_name}: accuracy_retrain")
        require(close(saved["accuracy_gap"], (counted["correct_x"] - counted["correct_retrain"]) / examples), f"{what} {row_name}: accuracy_gap")
        require(close(saved["agreement"], counted["agree"] / examples), f"{what} {row_name}: agreement")


def informative_label(original_is_within: bool) -> str:
    if original_is_within:
        return "UNINFORMATIVE (M, never unlearned, is within the floor)"
    return "informative (M is outside the floor)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = load_json_config(parser.parse_args().config)
    directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    plan = retrain_seed_plan(config)
    names = ["metrics.json", "noise_floor.json", "initial_state.pt", "original_model.pt"]
    for name in names + [entry["checkpoint"] for entry in plan]:
        require((directory / name).is_file(), f"missing artifact {name}")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    floor_file = json.loads((directory / "noise_floor.json").read_text(encoding="utf-8"))

    # 1. Config, seed, device, runtime, commit and code hash are recorded and consistent.
    for saved in (metrics, floor_file):
        require(saved["config"] == config, "saved config differs from the config file")
        require(saved["device"] == config["device"], "device not recorded")
        require(saved["runtime"] == metrics["runtime"], "noise-floor runtime differs from the first run")
        for field, length in (("head_commit", 40), ("code_sha256", 64)):
            value = saved["provenance"][field]
            require(len(value) == length and value == metrics["provenance"][field], f"{field} missing or inconsistent")
    require(metrics["seed"] == config["random_seed"], "seed not recorded")
    require(metrics["runtime"]["device"] == config["device"], "runtime device differs from config")
    clean_then = metrics["provenance"]["worktree_clean_at_start"] and floor_file["provenance"]["worktree_clean_at_start"]
    now = current_code_state()
    same_code_now = now["code_sha256"] == metrics["provenance"]["code_sha256"]
    if config["run_kind"] == "canonical":
        require(clean_then, "canonical run started on a dirty worktree")
        require(now["worktree_clean_at_start"], "worktree is not clean now")
        require(now["head_commit"] == metrics["provenance"]["head_commit"], "HEAD differs from the producing commit")
        require(same_code_now, "code files differ from the ones that produced the run")

    # 2. Partition: rebuilt, disjoint, and class histograms recounted from the raw labels.
    client_datasets, partition_records, test_data = prepare_run(config, download=False)
    require(partition_records == metrics["client_training_partitions"], "saved partition differs from the rebuilt partition")
    target = config["target_client"]
    positions = [position for data in client_datasets.values() for position in data.indices]
    require(len(positions) == len(set(positions)), "two clients share a training example")
    class_counts = {}
    for record in partition_records:
        _images, client_labels = images_and_labels([client_datasets[record["client_id"]]])
        class_counts[record["client_id"]] = np.bincount(client_labels, minlength=config["number_of_classes"])
        require(class_counts[record["client_id"]].tolist() == record["class_histogram"], "class histogram differs from the raw labels")
    total_examples = len(positions)
    target_examples = len(client_datasets[target])
    require(metrics["split_definitions"] == floor_file["split_definitions"], "split definitions differ")

    # 3. Training records: seeds, rounds, who was trained, n, bytes, t_train.
    all_clients = sorted(client_datasets)
    remaining = [client_id for client_id in all_clients if client_id != target]
    require([{k: v for k, v in entry.items() if k != "record"} for entry in floor_file["retrains"]] == plan, "retrain seeds/checkpoints differ from the config")
    records = {"original": (metrics["original"], all_clients, total_examples, config["random_seed"])}
    records[plan[0]["checkpoint"]] = (metrics["retrain_without_target"], remaining, total_examples - target_examples, config["random_seed"])
    for entry in floor_file["retrains"][1:]:
        records[entry["checkpoint"]] = (entry["record"], remaining, total_examples - target_examples, entry["training_seed"])
    initial_state = torch.load(directory / "initial_state.pt", weights_only=True)
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in initial_state.values())
    for name, (record, clients, examples, seed) in records.items():
        require(record["random_seed"] == seed, f"{name}: unexpected training seed")
        require(record["excluded_clients"] == ([] if name == "original" else [target]), f"{name}: wrong excluded clients")
        require(len(record["rounds"]) == config["number_of_rounds"], f"{name}: wrong number of rounds")
        require(record["t_train_seconds"] > 0, f"{name}: t_train was not measured")
        for round_record in record["rounds"]:
            selected = round_record["selected_client_ids"]
            require(name == "original" or target not in selected, f"{name}: the target client was trained")
            require(round_record["communication_cost_bytes"] == 2 * payload_bytes * len(selected), f"{name}: communication bytes mismatch")
            if config["client_fraction"] == 1.0:
                require(selected == clients, f"{name}: unexpected clients in a round")
                require(round_record["total_selected_examples"] == examples, f"{name}: n is not the sum over its clients")

    # 4. Initial weights are the seeded initial model.
    rebuilt = make_initial_state(config, config["model_initialization_seed"])
    require(all(torch.equal(initial_state[key], rebuilt[key]) for key in rebuilt), "initial_state.pt is not the seeded initial model")

    # 5. Predictions from the checkpoints, with the forward pass written in this file.
    split_data = {
        "global_test": [test_data],
        "target_client": [client_datasets[target]],
        "non_target_clients": [client_datasets[client_id] for client_id in remaining],
    }
    images, labels = {}, {}
    for split, datasets in split_data.items():
        images[split], labels[split] = images_and_labels(datasets)
    require(len(labels["target_client"]) == target_examples, "target split size")
    require(len(labels["non_target_clients"]) == total_examples - target_examples, "non-target examples != total - target")
    checkpoint_names = {"original": "original_model.pt", **{entry["checkpoint"]: entry["checkpoint"] for entry in plan}}
    predictions = {}
    for name, file_name in checkpoint_names.items():
        state = torch.load(directory / file_name, map_location="cpu", weights_only=True)
        predictions[name] = {split: predict(state, images[split]) for split in split_data}
        saved_metrics = records[name][0]["metrics"]
        for split in split_data:
            correct = int(np.sum(predictions[name][split] == labels[split]))
            require(saved_metrics[split]["examples"] == len(labels[split]), f"{name} {split}: example count")
            require(close(saved_metrics[split]["accuracy"], correct / len(labels[split])), f"{name} {split}: accuracy is not correct / N")
        # Loss and macro-F1: recomputed with the runner's evaluation code (not independent).
        shared_code = evaluate_checkpoint(config, directory / file_name, client_datasets, test_data).metrics()
        for split in split_data:
            for key in ("loss", "f1_macro"):
                require(close(saved_metrics[split][key], shared_code[split][key]), f"{name} {split}: {key}")

    # 6. Class groups recounted from the raw label counts.
    class_totals = sum(class_counts.values())
    dominant = []
    for class_id in range(config["number_of_classes"]):
        share = class_counts[target][class_id] / class_totals[class_id] if class_totals[class_id] > 0 else 0.0
        require(close(floor_file["class_groups"]["target_share"][class_id], share), f"class {class_id}: target share")
        if share >= config["target_dominant_threshold"]:
            dominant.append(class_id)
    require(floor_file["class_groups"]["target_dominant"] == dominant, "target-dominant classes differ from the recount")
    require(sorted(dominant + floor_file["class_groups"]["shared"]) == list(range(config["number_of_classes"])), "class groups do not cover every class once")
    dominant_mask = np.isin(labels["target_client"], dominant)

    # 7. Every pair of retrains, recounted.
    number_of_retrains = len(plan)
    require(len(floor_file["pairs"]) == number_of_retrains * (number_of_retrains - 1) // 2, "wrong number of pairs")
    counted_pairs = []
    for pair in floor_file["pairs"]:
        require(pair["first"] < pair["second"], "pair order")
        first, second = plan[pair["first"]]["checkpoint"], plan[pair["second"]]["checkpoint"]
        counted = count_rows(labels, predictions[second], predictions[first], dominant_mask)
        require_rows_match(pair["rows"], counted, f"pair ({pair['first']}, {pair['second']})")
        require(counted["target_client_dominant_classes"]["examples"] + counted["target_client_shared_classes"]["examples"] == target_examples, "dominant + shared != target examples")
        counted_pairs.append(counted)
    require(len({(pair["first"], pair["second"]) for pair in floor_file["pairs"]}) == len(counted_pairs), "a pair is repeated")

    # 8. Summary over the pairs, the within-floor flags of M and the positive control.
    rule = config["noise_floor_decision_rule"]
    require(floor_file["decision_rule"] == rule and floor_file["seed_mode"] == config["noise_floor_seed_mode"], "decision rule or seed mode not recorded")
    reference = plan[floor_file["reference_retrain_index"]]["checkpoint"]
    original_rows = count_rows(labels, predictions["original"], predictions[reference], dominant_mask)
    require_rows_match(floor_file["original_vs_reference_retrain"], original_rows, "original vs reference")
    print(f"Decision rule: {rule}; seed mode: {config['noise_floor_seed_mode']}; retrains: {number_of_retrains}; pairs: {len(counted_pairs)}")
    for row_name in ROWS:
        saved = floor_file["noise_floor_summary"][row_name]
        examples = original_rows[row_name]["examples"]
        require(saved["pairs"] == len(counted_pairs) and saved["examples"] == examples, f"{row_name}: summary counts")
        within = floor_file["original_within_noise_floor"][row_name]
        outside = floor_file["positive_control_original_outside_floor"][row_name]
        if examples == 0:
            require(within == {"accuracy_gap_within": None, "agreement_within": None}, f"{row_name}: empty row has a verdict")
            require(outside == {"accuracy_gap": None, "agreement": None}, f"{row_name}: empty row has a positive control")
            print(f"  {row_name}: no examples in this row")
            continue
        differences = np.array([abs(pair[row_name]["correct_x"] - pair[row_name]["correct_retrain"]) for pair in counted_pairs])
        agreements = np.array([pair[row_name]["agree"] for pair in counted_pairs])
        require(saved["max_abs_correct_difference"] == int(differences.max()), f"{row_name}: max difference")
        require(saved["min_agree"] == int(agreements.min()), f"{row_name}: min agree")
        for key, value in (
            ("abs_accuracy_gap_mean", differences.mean() / examples), ("abs_accuracy_gap_std", differences.std() / examples),
            ("abs_accuracy_gap_max", differences.max() / examples), ("agreement_mean", agreements.mean() / examples),
            ("agreement_std", agreements.std() / examples), ("agreement_min", agreements.min() / examples),
        ):
            require(close(saved[key], float(value)), f"{row_name}: {key}")
        difference = abs(original_rows[row_name]["correct_x"] - original_rows[row_name]["correct_retrain"])
        agree = original_rows[row_name]["agree"]
        if rule == "max_of_pairs_inclusive":
            gap_within, agreement_within = difference <= differences.max(), agree >= agreements.min()
        else:
            require(rule == "max_of_pairs_strict", "the verifier does not know this decision rule")
            gap_within, agreement_within = difference < differences.max(), agree > agreements.min()
        require(within == {"accuracy_gap_within": bool(gap_within), "agreement_within": bool(agreement_within)}, f"{row_name}: within-floor flags")
        require(outside == {"accuracy_gap": not gap_within, "agreement": not agreement_within}, f"{row_name}: positive control flags")
        print(f"  {row_name} (N={examples}): accuracy gap {informative_label(gap_within)}; agreement {informative_label(agreement_within)}")

    kind = "SMOKE (not thesis evidence)" if config["run_kind"] == "smoke" else "CANONICAL"
    print(f"Run kind: {kind}; commit {metrics['provenance']['head_commit']}; device {metrics['device']}; seed {metrics['seed']}")
    print(f"Worktree clean when run: {clean_then}; clean now: {now['worktree_clean_at_start']}; code files unchanged since the run: {same_code_now}")
    print(f"Near-empty clients (< {config['near_empty_client_examples']} examples): {metrics['near_empty_clients']}")
    print("PASS (consistency check, nothing retrained): config/seed/device/runtime/commit/code hash recorded; partition, "
          "target exclusion, initial weights, accuracy, class groups, pairwise counts, summary and flags match the checkpoints.")


if __name__ == "__main__":
    main()
