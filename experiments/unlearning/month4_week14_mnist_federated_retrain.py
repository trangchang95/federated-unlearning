"""Month 4 Week 14 (plan W3-W4): federated retrain gold standard on MNIST.

Trains two FedAvg models from the SAME initial weights w_0 on the SAME client
partition, with the same seed, rounds and hyperparameters:

- M               : all clients take part (``exclude_clients=()``);
- M_retrain^(-A)  : the target client A is excluded from every round.

Everything comes from the config JSON.  ``run_kind`` is "smoke" or
"canonical"; a canonical run refuses a dirty Git worktree, a smoke run records
the worktree state and must write to a ``smoke_*`` results directory.

Run from the repository root (one process at a time):
    python experiments/unlearning/month4_week14_mnist_federated_retrain.py \
        --config configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.federated_retrain import train_fedavg_excluding
from algorithms.fedavg import clone_model_state, model_state_size_bytes
from clients.dirichlet_partition import dirichlet_label_partition_indices
from clients.iid_partition import iid_partition_indices
from evaluation.efficiency import PhaseTimer
from evaluation.three_way import evaluate_three_way
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP


# Every project file whose code decides a result of the Week 14 scripts.
CODE_FILES = (
    "algorithms/fedavg.py",
    "algorithms/federated_retrain.py",
    "clients/dirichlet_partition.py",
    "clients/federated_client.py",
    "clients/iid_partition.py",
    "evaluation/efficiency.py",
    "evaluation/forgetting.py",
    "evaluation/three_way.py",
    "evaluation/utility.py",
    "experiments/experiment_utils.py",
    "experiments/unlearning/month4_week14_mnist_federated_retrain.py",
    "experiments/unlearning/month4_week14_mnist_noise_floor.py",
    "models/simple_mlp.py",
    "server/fedavg_server.py",
)

# What the three evaluation splits contain (recorded in every result file).
SPLIT_DEFINITIONS = {
    "global_test": "held-out MNIST test examples",
    "target_client": "TRAINING examples of the target client",
    "non_target_clients": "pooled TRAINING examples of all non-target clients",
}


def check_supported_config(config: dict) -> None:
    """Stop if a descriptive config field names something this code does not run.

    The code always uses MNIST, SimpleMLP, plain SGD, CrossEntropyLoss and
    ToTensor; the recorded config must not be able to say otherwise.
    """
    supported = {
        "dataset": "MNIST",
        "model": "SimpleMLP",
        "architecture": f"784 -> {config['hidden_units']} ReLU -> 10",
        "number_of_classes": 10,
        "preprocessing": "ToTensor only; pixels scaled to [0, 1]",
        "optimizer": "SGD",
        "optimizer_details": "plain SGD; momentum=0; weight_decay=0",
        "loss_function": "CrossEntropyLoss",
    }
    for field, value in supported.items():
        if config[field] != value:
            raise ValueError(f"Config field {field!r} must be {value!r}, got {config[field]!r}.")


def require_decided(config: dict, fields: tuple[str, ...]) -> None:
    """Stop if a design decision that belongs to the thesis author is still null."""
    for field in fields:
        if config[field] is None:
            raise ValueError(
                f"Config field {field!r} is null. It is a design decision with no "
                "default; set it in the config before running."
            )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and torch.  DataLoaders use num_workers=0 (no
    worker processes) and get their own seeded ``generator`` in the client."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def current_code_state() -> dict:
    """Read the Git commit, whether the worktree is clean, and a code hash.

    On a dirty worktree (smoke runs only) the commit does not identify the
    code that ran; ``code_sha256`` (SHA-256 over the bytes of ``CODE_FILES``)
    does.
    """
    def git(*arguments: str) -> str:
        # Only the final newline is removed: a porcelain status line can
        # start with a meaningful space (" M file").
        return subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.rstrip("\n")

    code_hash = hashlib.sha256()
    for relative_path in CODE_FILES:
        code_hash.update((PROJECT_ROOT / relative_path).read_bytes())
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "worktree_clean_at_start": not status,
        "worktree_status_at_start": status.splitlines(),
        "code_sha256": code_hash.hexdigest(),
    }


def code_provenance(config: dict) -> dict:
    """Check the smoke/canonical rules, then return ``current_code_state()``."""
    is_smoke_directory = config["output_subdirectory"].startswith("smoke_")
    if config["run_kind"] not in ("smoke", "canonical"):
        raise ValueError("run_kind must be 'smoke' or 'canonical'.")
    if (config["run_kind"] == "smoke") != is_smoke_directory:
        raise ValueError("Exactly the smoke runs must write to a 'smoke_*' directory.")
    state = current_code_state()
    if config["run_kind"] == "canonical" and not state["worktree_clean_at_start"]:
        raise RuntimeError(
            "A canonical run needs a clean worktree:\n" + "\n".join(state["worktree_status_at_start"])
        )
    return state


def load_mnist(config: dict, *, download: bool = True) -> tuple[Subset, Subset]:
    """Return (training data, test data) as Subsets of the MNIST datasets.

    The 15% validation split of earlier weeks is still carved off (and not
    used) so the training examples are the same 51,000 as in Months 2-3.
    ``training_subset_size`` / ``test_subset_size`` keep only the first N
    examples; they are null in canonical configs.
    """
    data_directory = PROJECT_ROOT / "data" / config["data_subdirectory"]
    complete_training_data = datasets.MNIST(
        root=data_directory, train=True, transform=transforms.ToTensor(), download=download
    )
    complete_test_data = datasets.MNIST(
        root=data_directory, train=False, transform=transforms.ToTensor(), download=download
    )
    validation_size = int(len(complete_training_data) * config["validation_fraction"])
    training_split, _unused_validation = random_split(
        complete_training_data,
        [len(complete_training_data) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    training_indices = list(training_split.indices)
    test_indices = list(range(len(complete_test_data)))
    if config["training_subset_size"] is not None:
        training_indices = training_indices[: config["training_subset_size"]]
    if config["test_subset_size"] is not None:
        test_indices = test_indices[: config["test_subset_size"]]
    return Subset(complete_training_data, training_indices), Subset(complete_test_data, test_indices)


def build_client_partition(
    training_data: Subset, config: dict
) -> tuple[dict[int, Subset], list[dict]]:
    """Split the training data across clients; also return per-client class counts."""
    labels = training_data.dataset.targets[torch.tensor(training_data.indices)]
    if config["partition_strategy"] == "iid":
        positions_by_client = iid_partition_indices(
            len(training_data), config["number_of_clients"], config["train_partition_seed"]
        )
    elif config["partition_strategy"] == "dirichlet_label_skew":
        positions_by_client = dirichlet_label_partition_indices(
            labels, config["number_of_clients"], config["alpha"], config["train_partition_seed"]
        )
    else:
        raise ValueError(f"Unknown partition_strategy {config['partition_strategy']!r}.")
    client_datasets = {
        client_id: Subset(training_data, list(positions))
        for client_id, positions in positions_by_client.items()
    }
    records = []
    for client_id, positions in positions_by_client.items():
        histogram = torch.bincount(
            labels[torch.tensor(positions)], minlength=config["number_of_classes"]
        ).tolist()
        records.append(
            {
                "client_id": client_id,
                "number_of_examples": len(positions),
                "class_histogram": histogram,
                "near_empty": len(positions) < config["near_empty_client_examples"],
            }
        )
    return client_datasets, records


def make_initial_state(config: dict, seed: int) -> dict[str, torch.Tensor]:
    """Create the initial weights w_0 that every run of this config starts from."""
    set_random_seed(seed)
    return clone_model_state(SimpleMLP(hidden_units=config["hidden_units"]).state_dict())


def evaluation_loaders(
    config: dict, client_datasets: dict[int, Subset], test_data: Dataset
) -> tuple[DataLoader, DataLoader, list[DataLoader]]:
    """Unshuffled loaders: global test, target client, each non-target client."""
    def loader(dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset, batch_size=config["evaluation_batch_size"], shuffle=False, num_workers=0
        )

    target = config["target_client"]
    non_target = [loader(data) for client_id, data in client_datasets.items() if client_id != target]
    return loader(test_data), loader(client_datasets[target]), non_target


def train_and_evaluate(
    config: dict,
    client_datasets: dict[int, Subset],
    test_data: Dataset,
    initial_state: dict[str, torch.Tensor],
    exclude_clients: tuple[int, ...],
    random_seed: int,
    timer: PhaseTimer,
    phase_name: str,
) -> dict:
    """One FedAvg run from ``initial_state``, then the three-way evaluation."""
    device = torch.device(config["device"])
    model = SimpleMLP(hidden_units=config["hidden_units"])
    model.load_state_dict(clone_model_state(initial_state), strict=True)
    rounds = train_fedavg_excluding(
        model=model,
        client_datasets=client_datasets,
        exclude_clients=exclude_clients,
        number_of_rounds=config["number_of_rounds"],
        client_fraction=config["client_fraction"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        random_seed=random_seed,
        device=device,
        timer=timer,
        phase_name=phase_name,
    )
    evaluation = evaluate_three_way(
        model, *evaluation_loaders(config, client_datasets, test_data),
        nn.CrossEntropyLoss(), device,
    )
    record = {
        "excluded_clients": list(exclude_clients),
        "random_seed": random_seed,
        "rounds": [
            {
                "round": result.round_number,
                "selected_client_ids": list(result.selected_client_ids),
                "total_selected_examples": result.total_selected_examples,
                "weighted_mean_local_loss": result.weighted_mean_local_loss,
                "communication_cost_bytes": result.communication_cost_bytes,
            }
            for result in rounds
        ],
        "communication_cost_bytes": sum(result.communication_cost_bytes for result in rounds),
        "t_train_seconds": timer.seconds(phase_name),
        "metrics": evaluation.metrics(),
    }
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return {"state": state, "evaluation": evaluation, "record": record}


def prepare_run(config: dict, *, download: bool = True) -> tuple[dict[int, Subset], list[dict], Subset]:
    """Data, partition and target check shared by the runner, noise floor and verifier."""
    check_supported_config(config)
    require_decided(config, ("target_client",))
    set_random_seed(config["split_seed"])
    training_data, test_data = load_mnist(config, download=download)
    client_datasets, partition_records = build_client_partition(training_data, config)
    if config["target_client"] not in client_datasets:
        raise ValueError(
            f"target_client {config['target_client']!r} is not an existing client ID."
        )
    return client_datasets, partition_records, test_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config_path = parser.parse_args().config
    config = load_json_config(config_path)
    provenance = code_provenance(config)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"{output_directory} already has results; use a new config/directory.")

    client_datasets, partition_records, test_data = prepare_run(config)
    target = config["target_client"]
    initial_state = make_initial_state(config, config["model_initialization_seed"])
    timer = PhaseTimer()
    original = train_and_evaluate(
        config, client_datasets, test_data, initial_state, (),
        config["random_seed"], timer, "original_training",
    )
    retrain = train_and_evaluate(
        config, client_datasets, test_data, initial_state, (target,),
        config["random_seed"], timer, "retrain_without_target",
    )

    metrics = {
        "source_config": str(config_path.resolve().relative_to(PROJECT_ROOT)),
        "config": config,
        "seed": config["random_seed"],
        "device": config["device"],
        "runtime": runtime,
        "provenance": provenance,
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "client_training_partitions": partition_records,
        "near_empty_clients": [r["client_id"] for r in partition_records if r["near_empty"]],
        "target_client": target,
        "split_definitions": SPLIT_DEFINITIONS,
        "timing": timer.to_dict(),
        "original": original["record"],
        "retrain_without_target": retrain["record"],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(initial_state, output_directory / "initial_state.pt")
    torch.save(original["state"], output_directory / "original_model.pt")
    torch.save(retrain["state"], output_directory / "retrain_without_target.pt")
    write_json(output_directory / "metrics.json", metrics)
    print(f"t_train original: {original['record']['t_train_seconds']:.2f} s")
    print(f"t_train retrain:  {retrain['record']['t_train_seconds']:.2f} s")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
