"""Month 4 Week 13: MNIST client-data deletion with full retraining.

This is the reference retraining baseline: train once on all training data,
remove one deterministic client's training shard, then train a fresh model
from the same initial weights without that shard. It does not implement SISA
or federated unlearning; it establishes the exact-retraining comparison for
later methods.

Run from the repository root:
    python experiments/unlearning/month4_week13_mnist_full_retraining.py \
        --config configs/month4_week13_mnist_full_retraining.json
"""

from __future__ import annotations

import argparse
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

from algorithms.fedavg import model_state_size_bytes
from clients.iid_partition import iid_partition_indices
from evaluation.efficiency import PhaseTimer
from evaluation.three_way import evaluate_three_way
from evaluation.utility import evaluate_classifier
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month4_week13_mnist_full_retraining.json"


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


def validate_code_provenance(config: dict, config_path: Path) -> dict:
    """Require a clean worktree whose HEAD is the config's producing tag."""
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Git provenance check failed: {detail}")
        return completed.stdout.strip()

    try:
        relative_config = config_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("The experiment config must be inside this repository.") from error
    head = git("rev-parse", "HEAD")
    revision = config["code_revision"]
    try:
        tagged_commit = git("rev-parse", "--verify", f"refs/tags/{revision}^{{commit}}")
    except RuntimeError as error:
        raise RuntimeError(
            f"Configured code_revision {revision!r} is not an existing tag. "
            "Commit the finalized Week 13 files and create that tag before training."
        ) from error
    if tagged_commit != head:
        raise RuntimeError(
            f"HEAD ({head}) must equal configured tag {revision!r} ({tagged_commit})."
        )
    for path in (relative_config, config["runner"], config["environment_manifest"]):
        git("ls-files", "--error-unmatch", "--", path)
    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "The Git worktree must be clean before training. Commit the experiment "
            f"files first. Current changes:\n{status}"
        )
    return {
        "configured_revision": revision,
        "resolved_commit": tagged_commit,
        "head_commit": head,
        "worktree_clean_at_start": True,
    }


def load_and_split_mnist(
    config: dict, *, download: bool = True
) -> tuple[Subset, Subset, Dataset]:
    data_directory = PROJECT_ROOT / "data" / config["data_subdirectory"]
    transform = transforms.ToTensor()
    complete_training_data = datasets.MNIST(
        root=data_directory, train=True, transform=transform, download=download
    )
    test_data = datasets.MNIST(
        root=data_directory, train=False, transform=transform, download=download
    )
    validation_size = int(len(complete_training_data) * config["validation_fraction"])
    training_size = len(complete_training_data) - validation_size
    training_data, validation_data = random_split(
        complete_training_data,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    observed = (len(training_data), len(validation_data), len(test_data))
    expected = (
        config["expected_training_examples"],
        config["expected_validation_examples"],
        config["expected_test_examples"],
    )
    if observed != expected:
        raise RuntimeError(f"MNIST split mismatch: expected {expected}, found {observed}.")
    return training_data, validation_data, test_data


def build_client_shards(training_data: Subset, config: dict) -> tuple[dict[int, Subset], list[dict]]:
    complete_training_data = training_data.dataset
    training_labels = complete_training_data.targets[training_data.indices]
    positions_by_client = iid_partition_indices(
        len(training_data), config["number_of_clients"], config["client_partition_seed"]
    )
    clients = {
        client_id: Subset(training_data, list(positions))
        for client_id, positions in positions_by_client.items()
    }
    records = [
        {
            "client_id": client_id,
            "number_of_examples": len(positions),
            "class_histogram": torch.bincount(
                training_labels[torch.tensor(positions, dtype=torch.long)], minlength=10
            ).tolist(),
        }
        for client_id, positions in positions_by_client.items()
    ]
    if set(clients) != set(range(config["number_of_clients"])):
        raise RuntimeError("Client partition IDs do not match the configured client count.")
    if sum(len(client_data) for client_data in clients.values()) != len(training_data):
        raise RuntimeError("Client shards do not cover the training split exactly once.")
    return clients, records


def train_model(
    training_data: Dataset,
    validation_data: Dataset,
    test_data: Dataset,
    target_data: Dataset,
    non_target_data: list[Dataset],
    initial_state: dict[str, torch.Tensor],
    config: dict,
    device: torch.device,
    loader_seed: int,
    timer: PhaseTimer,
    phase_name: str,
) -> dict:
    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(loader_seed),
    )
    test_loader = DataLoader(
        test_data, batch_size=config["evaluation_batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    validation_loader = DataLoader(
        validation_data, batch_size=config["evaluation_batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    target_loader = DataLoader(
        target_data, batch_size=config["evaluation_batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    non_target_loaders = [
        DataLoader(
            client_data, batch_size=config["evaluation_batch_size"], shuffle=False,
            num_workers=config["num_workers"],
        )
        for client_data in non_target_data
    ]

    history = []
    for epoch in range(1, config["training_epochs"] + 1):
        with timer.phase(phase_name):
            model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(features), labels)
                loss.backward()
                optimizer.step()
        validation = evaluate_classifier(model, validation_loader, loss_function, device)
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
                "validation_f1_macro": validation.f1_macro,
            }
        )

    final_evaluation = evaluate_three_way(
        model, test_loader, target_loader, non_target_loaders, loss_function, device
    )
    return {
        "history": history,
        "metrics": final_evaluation.metrics(),
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    provenance = validate_code_provenance(config, args.config)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    if output_directory.exists() and any(output_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_directory} already has saved results; pass --overwrite to replace them."
        )

    set_random_seed(config["split_seed"])
    training_data, validation_data, test_data = load_and_split_mnist(config)
    client_shards, shard_records = build_client_shards(training_data, config)
    target_id = config["target_client"]
    if target_id not in client_shards:
        raise ValueError(f"Unknown target_client {target_id}; configured clients are {list(client_shards)}.")
    target_data = client_shards[target_id]
    non_target_data = [data for client_id, data in client_shards.items() if client_id != target_id]
    retained_data = Subset(
        training_data,
        [
            position
            for client_id, shard in client_shards.items()
            if client_id != target_id
            for position in shard.indices
        ],
    )

    set_random_seed(config["initial_model_seed"])
    initial_model = SimpleMLP(hidden_units=config["hidden_units"])
    initial_state = {
        name: tensor.detach().clone() for name, tensor in initial_model.state_dict().items()
    }
    timer = PhaseTimer()
    original = train_model(
        training_data, validation_data, test_data, target_data, non_target_data, initial_state,
        config, torch.device(config["device"]), config["full_training_loader_seed"],
        timer, "original_full_training",
    )
    retrained = train_model(
        retained_data, validation_data, test_data, target_data, non_target_data, initial_state,
        config, torch.device(config["device"]), config["retraining_loader_seed"],
        timer, "retraining_without_target",
    )

    metrics = {
        "source_config": str(args.config.resolve().relative_to(PROJECT_ROOT)),
        "config": config,
        "runtime": runtime,
        "provenance": provenance,
        "trainable_parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "model_payload_bytes": model_state_size_bytes(initial_state),
        "model_state_schema": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
            }
            for name, tensor in initial_state.items()
        },
        "training_examples_before_deletion": len(training_data),
        "validation_examples": len(validation_data),
        "deleted_target_examples": len(target_data),
        "training_examples_after_deletion": len(retained_data),
        "client_training_partitions": shard_records,
        "timing": timer.to_dict(),
        "original_full_training": {
            "history": original["history"], "metrics": original["metrics"]
        },
        "full_retraining_without_target": {
            "history": retrained["history"], "metrics": retrained["metrics"]
        },
        "accuracy_difference_retrained_minus_original": (
            retrained["metrics"]["global_test"]["accuracy"]
            - original["metrics"]["global_test"]["accuracy"]
        ),
        "accuracy": retrained["metrics"]["global_test"]["accuracy"],
        "f1": retrained["metrics"]["global_test"]["f1_macro"],
        "unlearning_time": timer.seconds("retraining_without_target"),
        "communication_cost": 0,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        original["model_state"], output_directory / "original_full_training.pt"
    )
    torch.save(
        retrained["model_state"], output_directory / "full_retraining_without_target.pt"
    )
    write_json(output_directory / "metrics.json", metrics)
    print(f"Configuration: {args.config}")
    print(f"Original accuracy: {original['metrics']['global_test']['accuracy']:.4f}")
    print(f"Retrained accuracy: {retrained['metrics']['global_test']['accuracy']:.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
