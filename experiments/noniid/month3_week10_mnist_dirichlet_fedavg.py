"""Month 3 Week 10: Dirichlet(alpha) label-skew hand-written FedAvg on MNIST.

Each config fixes one alpha (1.0, 0.5, or 0.1) and trains hand-written FedAvg
on a Dirichlet(alpha)-skewed 5-client MNIST split, matched against a
centralized SGD baseline trained from the same initial weights on the same
51,000-example split -- the same protocol as Month 2 Week 8's canonical
config (K=5, C=1, E=1, B=128, R=5, SGD lr=0.1), so alpha is the only thing
that changes across the three Week 10 runs.

Like Week 9, this is a config-driven, deterministic, auto-saving exploratory
comparison -- not Gate-evidence-grade like Week 8. The Week 12 benchmark
(FedAvg vs. FedProx across alpha) is what actually closes Gate 3.

Run from the repository root, once per alpha config:
    python experiments/noniid/month3_week10_mnist_dirichlet_fedavg.py \
        --config configs/month3_week10_mnist_alpha1.0_fedavg.json
"""

from __future__ import annotations

import argparse
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

from algorithms.fedavg import model_state_size_bytes
from clients.dirichlet_partition import dirichlet_label_partition_indices
from evaluation.utility import evaluate_classifier
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP
from server.fedavg_server import FedAvgServer


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month3_week10_mnist_alpha1.0_fedavg.json"


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
    expected = (config["expected_training_examples"], config["expected_test_examples"])
    observed = (len(training_data), len(test_data))
    if observed != expected:
        raise RuntimeError(f"MNIST split mismatch: expected {expected}, found {observed}.")
    return training_data, test_data


def class_histogram(labels: torch.Tensor, indices: list[int]) -> list[int]:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    return torch.bincount(selected, minlength=10).tolist()


def build_dirichlet_partitions(
    training_data: Subset, config: dict
) -> tuple[dict[int, Subset], list[dict]]:
    complete_training_data = training_data.dataset
    training_labels = complete_training_data.targets[training_data.indices]
    positions_by_client = dirichlet_label_partition_indices(
        training_labels,
        config["number_of_clients"],
        config["alpha"],
        config["train_partition_seed"],
    )
    client_datasets = {
        client_id: Subset(training_data, list(positions))
        for client_id, positions in positions_by_client.items()
    }
    records = [
        {
            "client_id": client_id,
            "number_of_examples": len(positions),
            "class_histogram": class_histogram(training_labels, list(positions)),
        }
        for client_id, positions in positions_by_client.items()
    ]
    return client_datasets, records


def train_centralized(
    training_data: Subset,
    test_data: Dataset,
    initial_state: dict[str, torch.Tensor],
    config: dict,
    device: torch.device,
) -> dict:
    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    history = []
    model.train()
    for epoch in range(1, config["centralized_epochs"] + 1):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
        evaluation = evaluate_classifier(model, test_loader, loss_function, device)
        history.append({"epoch": epoch, "test_accuracy_after_epoch": evaluation.accuracy})

    final_evaluation = evaluate_classifier(model, test_loader, loss_function, device)
    return {
        "history": history,
        "final_test_loss": final_evaluation.loss,
        "final_test_accuracy": final_evaluation.accuracy,
        "final_test_f1_macro": final_evaluation.f1_macro,
        "final_labels": final_evaluation.labels,
        "final_predictions": final_evaluation.predictions,
    }


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


def save_class_distribution_plot(records: list[dict], alpha: float, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    class_ids = list(range(10))
    width = 0.15
    for record in records:
        offset = (record["client_id"] - (len(records) - 1) / 2) * width
        axis.bar(
            [class_id + offset for class_id in class_ids],
            record["class_histogram"],
            width=width,
            label=f"Client {record['client_id']}",
        )
    axis.set_title(f"Client class distribution: Dirichlet(alpha={alpha})")
    axis.set_xlabel("Digit class")
    axis.set_ylabel("Training examples")
    axis.set_xticks(class_ids)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_convergence_plot(
    centralized_result: dict, fedavg_result: dict, alpha: float, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    rounds = [entry["round"] for entry in fedavg_result["history"]]
    axis.plot(
        rounds,
        [entry["test_accuracy_after_round"] for entry in fedavg_result["history"]],
        marker="o",
        label="FedAvg",
    )
    epochs = [entry["epoch"] for entry in centralized_result["history"]]
    axis.plot(
        epochs,
        [entry["test_accuracy_after_epoch"] for entry in centralized_result["history"]],
        marker="s",
        label="Centralized SGD",
    )
    axis.set_xlabel("Round / epoch")
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"Test accuracy per round: Dirichlet(alpha={alpha})")
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
    centralized = metrics["centralized"]
    fedavg = metrics["fedavg"]
    alpha = metrics["config"]["alpha"]
    lines = [
        f"# Week 10 Saved-Run Summary (alpha={alpha})",
        "",
        "This file was generated automatically from the saved experiment metrics.",
        "",
        "| Method | Test accuracy | Test macro F1 |",
        "|---|---:|---:|",
        f"| Centralized SGD | {centralized['final_test_accuracy']:.2%} | {centralized['final_test_f1_macro']:.2%} |",
        f"| Hand-written FedAvg | {fedavg['final_test_accuracy']:.2%} | {fedavg['final_test_f1_macro']:.2%} |",
        "",
        f"- Accuracy difference (FedAvg minus centralized): {(fedavg['final_test_accuracy'] - centralized['final_test_accuracy']) * 100:+.2f} percentage points.",
        f"- FedAvg communication: {fedavg['communication_cost_bytes']:,} bytes.",
        f"- Client training-set sizes: {[record['number_of_examples'] for record in metrics['client_training_partitions']]}.",
        "- Both paths use identical initial weights, K=5, C=1, E=1, B=128, R=5, seed 42; only the client data partition (Dirichlet alpha) differs from other Week 10 runs.",
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
    client_datasets, partition_records = build_dirichlet_partitions(training_data, config)

    set_random_seed(config["model_initialization_seed"])
    initial_model = SimpleMLP(hidden_units=config["hidden_units"])
    initial_state = {name: tensor.clone() for name, tensor in initial_model.state_dict().items()}

    centralized_result = train_centralized(
        training_data, test_data, {k: v.clone() for k, v in initial_state.items()}, config, device
    )
    fedavg_result = run_fedavg(
        client_datasets, {k: v.clone() for k, v in initial_state.items()}, test_data, config, device
    )

    elapsed_seconds = time.perf_counter() - start_time

    metrics = {
        "source_config": str(args.config.resolve().relative_to(PROJECT_ROOT)),
        "config": config,
        "runtime": runtime,
        "trainable_parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "client_training_partitions": partition_records,
        "centralized": {
            "history": centralized_result["history"],
            "final_test_loss": centralized_result["final_test_loss"],
            "final_test_accuracy": centralized_result["final_test_accuracy"],
            "final_test_f1_macro": centralized_result["final_test_f1_macro"],
        },
        "fedavg": {
            "history": fedavg_result["history"],
            "final_test_loss": fedavg_result["final_test_loss"],
            "final_test_accuracy": fedavg_result["final_test_accuracy"],
            "final_test_f1_macro": fedavg_result["final_test_f1_macro"],
            "communication_cost_bytes": fedavg_result["communication_cost_bytes"],
        },
        "accuracy_difference_fedavg_minus_centralized": (
            fedavg_result["final_test_accuracy"] - centralized_result["final_test_accuracy"]
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "metrics.json", metrics)
    save_class_distribution_plot(
        partition_records, config["alpha"], output_directory / "class_distribution.png"
    )
    save_convergence_plot(
        centralized_result, fedavg_result, config["alpha"], output_directory / "convergence_comparison.png"
    )
    save_confusion_plot(
        centralized_result["final_labels"],
        centralized_result["final_predictions"],
        output_directory / "centralized_confusion_matrix.png",
        f"Centralized confusion matrix (alpha={config['alpha']})",
    )
    save_confusion_plot(
        fedavg_result["final_labels"],
        fedavg_result["final_predictions"],
        output_directory / "fedavg_confusion_matrix.png",
        f"FedAvg confusion matrix (alpha={config['alpha']})",
    )
    save_written_summary(metrics, output_directory / "comparison_summary.md")

    print(f"Configuration: {args.config} (alpha={config['alpha']})")
    print(f"Centralized test accuracy: {centralized_result['final_test_accuracy']:.4f}")
    print(f"FedAvg test accuracy: {fedavg_result['final_test_accuracy']:.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
