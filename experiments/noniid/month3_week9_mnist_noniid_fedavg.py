"""Month 3 Week 9: IID vs. shard-based Non-IID hand-written FedAvg on MNIST.

Both runs use the same 51,000-example MNIST training split, the same
SimpleMLP architecture and initial weights, and the same K=5, C=1, E=1,
B=128, R=5, SGD(lr=0.1) protocol as Month 2 Week 8's canonical config. The
only thing that differs is how the 51,000 training examples are divided
across the five clients:

- IID: `clients/iid_partition.py`, a uniform random shuffle -- every client's
  class histogram looks like the whole dataset's.
- Non-IID: `clients/noniid_partition.py`, McMahan et al. (2017)'s label-sorted
  shard scheme with two shards per client -- each client is dominated by one
  or two digits.

This is a Week 9 exploratory comparison, not Gate-evidence-grade like Week 8:
it is still 100% config-driven, deterministic, and auto-saves its outputs,
but it does not lock a Git tag or require a student self-check to close a
gate. That heavier apparatus is reserved for the Week 12 benchmark that
actually closes Gate 3.

Run from the repository root:
    python experiments/noniid/month3_week9_mnist_noniid_fedavg.py \
        --config configs/month3_week9_mnist_iid_vs_noniid_fedavg.json
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.fedavg import clone_model_state, model_state_size_bytes
from clients.iid_partition import iid_partition_indices
from clients.noniid_partition import sorted_label_shard_partition_indices
from evaluation.utility import evaluate_classifier
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP
from server.fedavg_server import FedAvgServer


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "month3_week9_mnist_iid_vs_noniid_fedavg.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory's saved results.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def load_and_split_mnist(config: dict) -> tuple[Subset, Dataset]:
    data_directory = PROJECT_ROOT / "data" / config["data_subdirectory"]
    transform = transforms.ToTensor()
    complete_training_data = datasets.MNIST(
        root=data_directory, train=True, transform=transform, download=True
    )
    test_data = datasets.MNIST(
        root=data_directory, train=False, transform=transform, download=True
    )
    validation_size = int(len(complete_training_data) * config["validation_fraction"])
    training_size = len(complete_training_data) - validation_size
    training_data, _validation_data = random_split(
        complete_training_data,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    expected = (
        config["expected_training_examples"],
        config["expected_test_examples"],
    )
    observed = (len(training_data), len(test_data))
    if observed != expected:
        raise RuntimeError(f"MNIST split mismatch: expected {expected}, found {observed}.")
    return training_data, test_data


def class_histogram(labels: torch.Tensor, indices: list[int]) -> list[int]:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    return torch.bincount(selected, minlength=10).tolist()


def build_partitions(
    training_data: Subset, config: dict
) -> tuple[dict[int, Subset], dict[int, Subset], list[dict], list[dict]]:
    """Return IID and Non-IID client Subsets plus their class histograms."""
    number_of_clients = config["number_of_clients"]
    complete_training_data = training_data.dataset
    training_labels = complete_training_data.targets[training_data.indices]

    iid_positions = iid_partition_indices(
        len(training_data), number_of_clients, config["train_partition_seed"]
    )
    noniid_positions = sorted_label_shard_partition_indices(
        training_labels,
        number_of_clients,
        config["shards_per_client"],
        config["train_partition_seed"],
    )

    def to_subsets(positions_by_client: dict[int, tuple[int, ...]]) -> dict[int, Subset]:
        return {
            client_id: Subset(training_data, list(positions))
            for client_id, positions in positions_by_client.items()
        }

    def to_records(positions_by_client: dict[int, tuple[int, ...]]) -> list[dict]:
        records = []
        for client_id, positions in positions_by_client.items():
            records.append(
                {
                    "client_id": client_id,
                    "number_of_examples": len(positions),
                    "class_histogram": class_histogram(training_labels, list(positions)),
                }
            )
        return records

    return (
        to_subsets(iid_positions),
        to_subsets(noniid_positions),
        to_records(iid_positions),
        to_records(noniid_positions),
    )


def run_fedavg(
    client_datasets: dict[int, Subset],
    initial_state: dict[str, torch.Tensor],
    test_data: Dataset,
    config: dict,
    device: torch.device,
) -> dict:
    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    server = FedAvgServer(
        global_model=model,
        client_datasets=client_datasets,
        client_fraction=config["client_fraction"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        random_seed=config["random_seed"],
        device=device,
    )
    loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    history = []
    communication_cost = 0
    for round_number in range(1, config["number_of_rounds"] + 1):
        round_result = server.run_round(round_number)
        communication_cost += round_result.communication_cost_bytes
        evaluation = evaluate_classifier(server.global_model, test_loader, loss_function, device)
        history.append(
            {
                "round": round_number,
                "selected_client_ids": list(round_result.selected_client_ids),
                "weighted_mean_local_loss": round_result.weighted_mean_local_loss,
                "test_loss_after_round": evaluation.loss,
                "test_accuracy_after_round": evaluation.accuracy,
            }
        )

    final_evaluation = evaluate_classifier(server.global_model, test_loader, loss_function, device)
    return {
        "history": history,
        "final_test_loss": final_evaluation.loss,
        "final_test_accuracy": final_evaluation.accuracy,
        "final_test_f1_macro": final_evaluation.f1_macro,
        "final_labels": final_evaluation.labels,
        "final_predictions": final_evaluation.predictions,
        "communication_cost_bytes": communication_cost,
    }


def save_class_distribution_plot(
    iid_records: list[dict], noniid_records: list[dict], output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    class_ids = list(range(10))
    width = 0.15
    for axis, records, title in (
        (axes[0], iid_records, "IID clients (uniform shuffle)"),
        (axes[1], noniid_records, "Non-IID clients (label-sorted shards)"),
    ):
        for record in records:
            offset = (record["client_id"] - (len(records) - 1) / 2) * width
            axis.bar(
                [class_id + offset for class_id in class_ids],
                record["class_histogram"],
                width=width,
                label=f"Client {record['client_id']}",
            )
        axis.set_title(title)
        axis.set_xlabel("Digit class")
        axis.set_xticks(class_ids)
    axes[0].set_ylabel("Training examples")
    axes[1].legend(fontsize=7, ncol=1, loc="upper right")
    figure.suptitle("Client class distributions: IID vs. Non-IID (Month 3, Week 9)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_convergence_plot(iid_result: dict, noniid_result: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    rounds = [entry["round"] for entry in iid_result["history"]]
    axis.plot(
        rounds,
        [entry["test_accuracy_after_round"] for entry in iid_result["history"]],
        marker="o",
        label="IID",
    )
    axis.plot(
        rounds,
        [entry["test_accuracy_after_round"] for entry in noniid_result["history"]],
        marker="o",
        label="Non-IID (2 shards/client)",
    )
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Test accuracy")
    axis.set_title("FedAvg test accuracy per round: IID vs. Non-IID")
    axis.set_xticks(rounds)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_confusion_plot(labels: np.ndarray, predictions: np.ndarray, output_path: Path, title: str) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(range(10)))
    figure, axis = plt.subplots(figsize=(5.4, 5.4))
    display.plot(ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_written_summary(metrics: dict, output_path: Path) -> None:
    iid = metrics["iid"]
    noniid = metrics["noniid"]
    lines = [
        "# Week 9 Saved-Run Summary",
        "",
        "This file was generated automatically from the saved experiment metrics.",
        "",
        "| Partition | Test accuracy | Test macro F1 | Communication bytes |",
        "|---|---:|---:|---:|",
        f"| IID | {iid['final_test_accuracy']:.2%} | {iid['final_test_f1_macro']:.2%} | {iid['communication_cost_bytes']:,} |",
        f"| Non-IID (2 shards/client) | {noniid['final_test_accuracy']:.2%} | {noniid['final_test_f1_macro']:.2%} | {noniid['communication_cost_bytes']:,} |",
        "",
        f"- Accuracy difference (Non-IID minus IID): {(noniid['final_test_accuracy'] - iid['final_test_accuracy']) * 100:+.2f} percentage points.",
        "- Both runs use identical initial weights, K=5, C=1, E=1, B=128, R=5, seed 42; only the client data partition differs.",
        "- This is one fixed-seed, sequential CPU experiment with one Non-IID severity (2 shards/client). It does not establish behavior across Dirichlet alpha values (Week 10) or against FedProx (Week 11).",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    device = torch.device(config["device"])

    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    if output_directory.exists() and any(output_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_directory} already has saved results; pass --overwrite to replace them."
        )

    start_time = time.perf_counter()

    set_random_seed(config["split_seed"])
    training_data, test_data = load_and_split_mnist(config)
    iid_clients, noniid_clients, iid_records, noniid_records = build_partitions(
        training_data, config
    )

    set_random_seed(config["model_initialization_seed"])
    initial_model = SimpleMLP(hidden_units=config["hidden_units"])
    initial_state = {
        name: tensor.clone() for name, tensor in initial_model.state_dict().items()
    }

    iid_result = run_fedavg(iid_clients, copy.deepcopy(initial_state), test_data, config, device)
    noniid_result = run_fedavg(
        noniid_clients, copy.deepcopy(initial_state), test_data, config, device
    )

    elapsed_seconds = time.perf_counter() - start_time

    metrics = {
        "source_config": str(args.config.relative_to(PROJECT_ROOT)),
        "config": config,
        "runtime": runtime,
        "trainable_parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "client_training_partitions": {
            "iid": iid_records,
            "noniid": noniid_records,
        },
        "iid": {
            "history": iid_result["history"],
            "final_test_loss": iid_result["final_test_loss"],
            "final_test_accuracy": iid_result["final_test_accuracy"],
            "final_test_f1_macro": iid_result["final_test_f1_macro"],
            "communication_cost_bytes": iid_result["communication_cost_bytes"],
        },
        "noniid": {
            "history": noniid_result["history"],
            "final_test_loss": noniid_result["final_test_loss"],
            "final_test_accuracy": noniid_result["final_test_accuracy"],
            "final_test_f1_macro": noniid_result["final_test_f1_macro"],
            "communication_cost_bytes": noniid_result["communication_cost_bytes"],
        },
        "accuracy_difference_noniid_minus_iid": (
            noniid_result["final_test_accuracy"] - iid_result["final_test_accuracy"]
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "metrics.json", metrics)
    save_class_distribution_plot(
        iid_records, noniid_records, output_directory / "class_distribution.png"
    )
    save_convergence_plot(iid_result, noniid_result, output_directory / "convergence_comparison.png")
    save_confusion_plot(
        iid_result["final_labels"],
        iid_result["final_predictions"],
        output_directory / "iid_confusion_matrix.png",
        "IID FedAvg confusion matrix",
    )
    save_confusion_plot(
        noniid_result["final_labels"],
        noniid_result["final_predictions"],
        output_directory / "noniid_confusion_matrix.png",
        "Non-IID FedAvg confusion matrix",
    )
    save_written_summary(metrics, output_directory / "comparison_summary.md")

    print(f"Configuration: {args.config}")
    print(f"IID test accuracy: {iid_result['final_test_accuracy']:.4f}")
    print(f"Non-IID test accuracy: {noniid_result['final_test_accuracy']:.4f}")
    print(f"Difference (Non-IID minus IID): {metrics['accuracy_difference_noniid_minus_iid']:+.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
