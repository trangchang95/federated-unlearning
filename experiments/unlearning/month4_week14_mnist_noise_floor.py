"""Month 4 Week 14 (plan W3-W4): noise floor from k retrains M_retrain^(-A).

Run AFTER ``month4_week14_mnist_federated_retrain.py`` with the same config.
That first run produced retrain number 0.  This script trains the other
k - 1 retrains without the target client, one after the other, compares
every pair of the k retrains, and writes ``noise_floor.json`` plus one
checkpoint per extra retrain into the same results directory.

Config keys (all required; null stops the run, there is no default):

- ``noise_floor_number_of_retrains``: k, counting retrain 0.
- ``noise_floor_extra_seeds``: the k - 1 seeds of the extra retrains.
- ``noise_floor_seed_mode``: what an extra seed replaces.
    "training"          : only the FedAvg training seed (``random_seed``).
                          With client_fraction = 1.0 every client is selected
                          in every round whatever the seed, and SimpleMLP has
                          no dropout, so the ONLY thing that changes is the
                          order of the minibatches inside each client.  The
                          initial weights are the saved ``initial_state.pt``.
    "training_and_init" : the training seed as above AND new initial weights
                          drawn with the extra seed.
- ``noise_floor_decision_rule``: how the pairwise gaps become a verdict, see
  ``evaluation.forgetting.within_noise_floor``.

Never varied: the MNIST split (``split_seed``) and the client partition
(``train_partition_seed``).  The partition defines which examples the target
client holds; a different partition would be a different unlearning request.

Run from the repository root (one process at a time):
    python experiments/unlearning/month4_week14_mnist_noise_floor.py \
        --config configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.efficiency import PhaseTimer
from evaluation.forgetting import (
    classify_classes,
    compare_to_retrain,
    pairwise_retrain_rows,
    summarise_pairs,
    within_noise_floor,
)
from evaluation.three_way import ThreeWayEvaluation, evaluate_three_way
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from experiments.unlearning.month4_week14_mnist_federated_retrain import (
    SPLIT_DEFINITIONS,
    code_provenance,
    evaluation_loaders,
    make_initial_state,
    prepare_run,
    require_decided,
    train_and_evaluate,
)
from models.simple_mlp import SimpleMLP

NOISE_FLOOR_FIELDS = (
    "noise_floor_number_of_retrains",
    "noise_floor_extra_seeds",
    "noise_floor_seed_mode",
    "noise_floor_decision_rule",
)


def retrain_seed_plan(config: dict) -> list[dict]:
    """List the seeds and checkpoint name of all k retrains; entry 0 is the first run."""
    require_decided(config, NOISE_FLOOR_FIELDS)
    extra_seeds = config["noise_floor_extra_seeds"]
    if len(extra_seeds) != config["noise_floor_number_of_retrains"] - 1 or not extra_seeds:
        raise ValueError("noise_floor_extra_seeds needs k - 1 seeds, and k must be at least 2.")
    if len(set(extra_seeds)) != len(extra_seeds) or config["random_seed"] in extra_seeds:
        raise ValueError("Every retrain needs its own training seed.")
    if config["noise_floor_seed_mode"] not in ("training", "training_and_init"):
        raise ValueError("noise_floor_seed_mode must be 'training' or 'training_and_init'.")
    new_initial_weights = config["noise_floor_seed_mode"] == "training_and_init"
    if new_initial_weights and config["model_initialization_seed"] in extra_seeds:
        raise ValueError("An extra seed equals model_initialization_seed: w_0 would not change.")
    plan = [
        {
            "index": 0,
            "training_seed": config["random_seed"],
            "initial_weights_seed": config["model_initialization_seed"],
            "checkpoint": "retrain_without_target.pt",
        }
    ]
    for index, seed in enumerate(extra_seeds, start=1):
        plan.append(
            {
                "index": index,
                "training_seed": seed,
                "initial_weights_seed": seed if new_initial_weights else config["model_initialization_seed"],
                "checkpoint": f"retrain_without_target_seed{seed}.pt",
            }
        )
    return plan


def evaluate_checkpoint(
    config: dict, checkpoint_path: Path, client_datasets: dict[int, Subset], test_data: Dataset
) -> ThreeWayEvaluation:
    """Load a saved model state and evaluate it on the three splits."""
    model = SimpleMLP(hidden_units=config["hidden_units"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    return evaluate_three_way(
        model, *evaluation_loaders(config, client_datasets, test_data),
        nn.CrossEntropyLoss(), torch.device(config["device"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config_path = parser.parse_args().config
    config = load_json_config(config_path)
    plan = retrain_seed_plan(config)
    provenance = code_provenance(config)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    if (output_directory / "noise_floor.json").exists():
        raise FileExistsError("noise_floor.json already exists; use a new config/directory.")
    first_run = json.loads((output_directory / "metrics.json").read_text(encoding="utf-8"))
    if first_run["config"] != config:
        raise RuntimeError("The first run in this directory used a different config.")
    for field in ("head_commit", "code_sha256"):
        if first_run["provenance"][field] != provenance[field]:
            raise RuntimeError(f"The first run in this directory was made with a different {field}.")
    if first_run["runtime"] != runtime:
        raise RuntimeError("The first run in this directory used a different runtime environment.")

    client_datasets, partition_records, test_data = prepare_run(config, download=False)
    if partition_records != first_run["client_training_partitions"]:
        raise RuntimeError("The client partition differs from the first run.")
    target = config["target_client"]
    saved_initial_state = torch.load(
        output_directory / "initial_state.pt", map_location="cpu", weights_only=True
    )

    # Retrain 0 is read from its checkpoint; the others are trained now, in order.
    timer = PhaseTimer()
    retrain_evaluations = [
        evaluate_checkpoint(config, output_directory / plan[0]["checkpoint"], client_datasets, test_data)
    ]
    new_states = {}
    for entry in plan[1:]:
        if config["noise_floor_seed_mode"] == "training":
            initial_state = saved_initial_state
        else:
            initial_state = make_initial_state(config, entry["initial_weights_seed"])
        run = train_and_evaluate(
            config, client_datasets, test_data, initial_state, (target,),
            entry["training_seed"], timer, f"retrain_without_target_seed{entry['training_seed']}",
        )
        entry["record"] = run["record"]
        retrain_evaluations.append(run["evaluation"])
        new_states[entry["checkpoint"]] = run["state"]

    class_groups = classify_classes(
        {record["client_id"]: record["class_histogram"] for record in partition_records},
        target, config["target_dominant_threshold"],
    )
    dominant = class_groups["target_dominant"]
    pairs = pairwise_retrain_rows(retrain_evaluations, dominant)
    summary = summarise_pairs(pairs)

    # Positive control: the original model M trained on the target client and
    # was not unlearned.  If M is inside the floor on a row, that row cannot
    # tell "unlearned" from "not unlearned" in this setting.
    original_evaluation = evaluate_checkpoint(
        config, output_directory / "original_model.pt", client_datasets, test_data
    )
    original_vs_reference = compare_to_retrain(original_evaluation, retrain_evaluations[0], dominant)
    original_within = within_noise_floor(
        original_vs_reference, summary, config["noise_floor_decision_rule"]
    )
    original_outside = {
        row_name: {
            "accuracy_gap": None if flags["accuracy_gap_within"] is None else not flags["accuracy_gap_within"],
            "agreement": None if flags["agreement_within"] is None else not flags["agreement_within"],
        }
        for row_name, flags in original_within.items()
    }

    write_json(
        output_directory / "noise_floor.json",
        {
            "source_config": str(config_path.resolve().relative_to(PROJECT_ROOT)),
            "config": config,
            "device": config["device"],
            "runtime": runtime,
            "provenance": provenance,
            "seed_mode": config["noise_floor_seed_mode"],
            "decision_rule": config["noise_floor_decision_rule"],
            "held_fixed": ["split_seed", "train_partition_seed", "target_client", "all hyperparameters"],
            "target_client": target,
            "split_definitions": SPLIT_DEFINITIONS,
            "class_groups": class_groups,
            "retrains": plan,
            "pairs": pairs,
            "noise_floor_summary": summary,
            "reference_retrain_index": 0,
            "original_vs_reference_retrain": original_vs_reference,
            "original_within_noise_floor": original_within,
            "positive_control_original_outside_floor": original_outside,
            "timing": timer.to_dict(),
        },
    )
    for checkpoint_name, state in new_states.items():
        torch.save(state, output_directory / checkpoint_name)
    for entry in plan[1:]:
        print(f"t_train retrain seed {entry['training_seed']}: {entry['record']['t_train_seconds']:.2f} s")
    print(f"Outputs: {output_directory / 'noise_floor.json'}")


if __name__ == "__main__":
    main()
