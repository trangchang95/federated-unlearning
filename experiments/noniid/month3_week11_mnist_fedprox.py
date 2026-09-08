"""Month 3 Week 11: does FedProx help at the heterogeneity FedAvg struggled with?

Week 10 measured hand-written FedAvg falling to 71.60% test accuracy at
Dirichlet(alpha=0.1) -- the most severe realistic (non-pathological)
heterogeneity tested so far. This experiment reuses that exact same
Dirichlet(alpha=0.1) partition, seeds, and initial weights, and asks whether
FedProx's proximal term (Li et al., 2020) recovers any of that gap, sweeping
mu across a few values to find one worth carrying into the Week 12 benchmark.

mu=0 is trained too, in the same script/config, as the FedAvg reference --
not read from Week 10's separate output file, so this experiment is fully
self-contained and reproducible on its own.

Run from the repository root:
    python experiments/noniid/month3_week11_mnist_fedprox.py \
        --config configs/month3_week11_mnist_fedprox_mu_sweep.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
from server.fedprox_server import FedProxServer


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month3_week11_mnist_fedprox_mu_sweep.json"


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


def evaluate_history(server, test_loader, loss_function, device, rounds: int) -> dict:
    history = []
    communication_cost = 0
    for round_number in range(1, rounds + 1):
        round_result = server.run_round(round_number)
        communication_cost += round_result.communication_cost_bytes
        evaluation = evaluate_classifier(server.global_model, test_loader, loss_function, device)
        history.append({"round": round_number, "test_accuracy_after_round": evaluation.accuracy})
    final_evaluation = evaluate_classifier(server.global_model, test_loader, loss_function, device)
    return {
        "history": history,
        "final_test_loss": final_evaluation.loss,
        "final_test_accuracy": final_evaluation.accuracy,
        "final_test_f1_macro": final_evaluation.f1_macro,
        "communication_cost_bytes": communication_cost,
    }


def run_fedavg(client_datasets, initial_state, test_data, config, device) -> dict:
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
    return evaluate_history(server, test_loader, loss_function, device, config["number_of_rounds"])


def run_fedprox(client_datasets, initial_state, test_data, config, device, mu: float) -> dict:
    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    server = FedProxServer(
        global_model=model,
        client_datasets=client_datasets,
        client_fraction=config["client_fraction"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        mu=mu,
        random_seed=config["random_seed"],
        device=device,
    )
    loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    return evaluate_history(server, test_loader, loss_function, device, config["number_of_rounds"])


def save_convergence_plot(fedavg_result: dict, prox_results: dict[float, dict], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    rounds = [entry["round"] for entry in fedavg_result["history"]]
    axis.plot(
        rounds,
        [entry["test_accuracy_after_round"] for entry in fedavg_result["history"]],
        marker="o",
        label="FedAvg (mu=0)",
    )
    for mu, result in prox_results.items():
        axis.plot(
            rounds,
            [entry["test_accuracy_after_round"] for entry in result["history"]],
            marker="s",
            label=f"FedProx (mu={mu})",
        )
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Test accuracy")
    axis.set_title("FedAvg vs. FedProx test accuracy per round (Dirichlet alpha=0.1)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_written_summary(metrics: dict, output_path: Path) -> None:
    fedavg = metrics["fedavg"]
    lines = [
        "# Week 11 Saved-Run Summary",
        "",
        "This file was generated automatically from the saved experiment metrics.",
        "",
        "| Method | Test accuracy | Test macro F1 |",
        "|---|---:|---:|",
        f"| FedAvg (mu=0) | {fedavg['final_test_accuracy']:.2%} | {fedavg['final_test_f1_macro']:.2%} |",
    ]
    for mu, result in metrics["fedprox"].items():
        lines.append(f"| FedProx (mu={mu}) | {result['final_test_accuracy']:.2%} | {result['final_test_f1_macro']:.2%} |")
    lines.append("")
    best_mu = max(metrics["fedprox"], key=lambda mu: metrics["fedprox"][mu]["final_test_accuracy"])
    best_accuracy = metrics["fedprox"][best_mu]["final_test_accuracy"]
    lines.append(
        f"- Best swept mu: {best_mu} ({best_accuracy:.2%} accuracy, "
        f"{(best_accuracy - fedavg['final_test_accuracy']) * 100:+.2f}pp vs. FedAvg)."
    )
    lines.append(
        "- All runs share the same Dirichlet(alpha=0.1) partition, seeds, and initial weights as Week 10's alpha=0.1 config; only mu differs."
    )
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

    fedavg_result = run_fedavg(
        client_datasets, {k: v.clone() for k, v in initial_state.items()}, test_data, config, device
    )
    fedprox_results = {}
    for mu in config["mu_values"]:
        fedprox_results[mu] = run_fedprox(
            client_datasets,
            {k: v.clone() for k, v in initial_state.items()},
            test_data,
            config,
            device,
            mu,
        )

    elapsed_seconds = time.perf_counter() - start_time

    metrics = {
        "source_config": str(args.config.resolve().relative_to(PROJECT_ROOT)),
        "config": config,
        "runtime": runtime,
        "trainable_parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "client_training_partitions": partition_records,
        "fedavg": {
            "history": fedavg_result["history"],
            "final_test_loss": fedavg_result["final_test_loss"],
            "final_test_accuracy": fedavg_result["final_test_accuracy"],
            "final_test_f1_macro": fedavg_result["final_test_f1_macro"],
            "communication_cost_bytes": fedavg_result["communication_cost_bytes"],
        },
        "fedprox": {
            str(mu): {
                "history": result["history"],
                "final_test_loss": result["final_test_loss"],
                "final_test_accuracy": result["final_test_accuracy"],
                "final_test_f1_macro": result["final_test_f1_macro"],
                "communication_cost_bytes": result["communication_cost_bytes"],
            }
            for mu, result in fedprox_results.items()
        },
        "elapsed_seconds": elapsed_seconds,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "metrics.json", metrics)
    save_convergence_plot(
        fedavg_result, fedprox_results, output_directory / "convergence_comparison.png"
    )
    save_written_summary(
        {"fedavg": metrics["fedavg"], "fedprox": {mu: r for mu, r in fedprox_results.items()}},
        output_directory / "comparison_summary.md",
    )

    print(f"Configuration: {args.config}")
    print(f"FedAvg (mu=0) test accuracy: {fedavg_result['final_test_accuracy']:.4f}")
    for mu, result in fedprox_results.items():
        print(f"FedProx (mu={mu}) test accuracy: {result['final_test_accuracy']:.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
