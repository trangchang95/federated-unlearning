"""Month 3 Week 12: FedProx leg of the FedAvg-vs-FedProx benchmark.

The FedAvg leg of every row in this benchmark already exists as verified
evidence: IID from Week 8/9 (90.99%), and Dirichlet alpha=1.0/0.5/0.1 from
Week 10 (90.59% / 89.83% / 71.60%). This script produces only the missing
FedProx numbers, at a single fixed mu chosen from Week 11's sweep (mu=0.01,
the least-detrimental value tested there), so the final benchmark compares
one FedProx hyperparameter across all four data distributions rather than
re-tuning mu per setting.

`partition_strategy` selects "iid" or "dirichlet"; `alpha` is ignored for
"iid" and required for "dirichlet". Run once per setting:
    python experiments/noniid/month3_week12_fedprox_benchmark.py \
        --config configs/month3_week12_iid_fedprox.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from clients.iid_partition import iid_partition_indices
from evaluation.utility import evaluate_classifier
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP
from server.fedprox_server import FedProxServer


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month3_week12_iid_fedprox.json"


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


def build_partitions(training_data: Subset, config: dict) -> tuple[dict[int, Subset], list[dict]]:
    strategy = config["partition_strategy"]
    if strategy == "iid":
        positions_by_client = iid_partition_indices(
            len(training_data), config["number_of_clients"], config["train_partition_seed"]
        )
    elif strategy == "dirichlet":
        complete_training_data = training_data.dataset
        training_labels = complete_training_data.targets[training_data.indices]
        positions_by_client = dirichlet_label_partition_indices(
            training_labels, config["number_of_clients"], config["alpha"], config["train_partition_seed"]
        )
    else:
        raise ValueError(f"Unknown partition_strategy: {strategy!r}")

    complete_training_data = training_data.dataset
    all_labels = complete_training_data.targets[training_data.indices]
    client_datasets = {
        client_id: Subset(training_data, list(positions))
        for client_id, positions in positions_by_client.items()
    }
    records = [
        {
            "client_id": client_id,
            "number_of_examples": len(positions),
            "class_histogram": class_histogram(all_labels, list(positions)),
        }
        for client_id, positions in positions_by_client.items()
    ]
    return client_datasets, records


def run_fedprox(client_datasets, initial_state, test_data, config, device) -> dict:
    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    server = FedProxServer(
        global_model=model,
        client_datasets=client_datasets,
        client_fraction=config["client_fraction"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        mu=config["mu"],
        random_seed=config["random_seed"],
        device=device,
    )
    loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    history = []
    communication_cost = 0
    round_times = []
    for round_number in range(1, config["number_of_rounds"] + 1):
        round_start = time.perf_counter()
        round_result = server.run_round(round_number)
        round_times.append(time.perf_counter() - round_start)
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
        "training_time_seconds": sum(round_times),
    }


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
    client_datasets, partition_records = build_partitions(training_data, config)

    set_random_seed(config["model_initialization_seed"])
    initial_model = SimpleMLP(hidden_units=config["hidden_units"])
    initial_state = {name: tensor.clone() for name, tensor in initial_model.state_dict().items()}

    fedprox_result = run_fedprox(client_datasets, initial_state, test_data, config, device)
    elapsed_seconds = time.perf_counter() - start_time

    metrics = {
        "source_config": str(args.config.resolve().relative_to(PROJECT_ROOT)),
        "config": config,
        "runtime": runtime,
        "trainable_parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "client_training_partitions": partition_records,
        "fedprox": fedprox_result,
        "elapsed_seconds": elapsed_seconds,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "metrics.json", metrics)

    print(f"Configuration: {args.config}")
    print(f"Partition: {config['partition_strategy']} (alpha={config.get('alpha')})")
    print(f"FedProx (mu={config['mu']}) test accuracy: {fedprox_result['final_test_accuracy']:.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
