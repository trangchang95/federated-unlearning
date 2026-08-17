"""Month 2 Week 8: matched centralized SGD versus hand-written FedAvg.

Both methods use the same MNIST split, SimpleMLP architecture, initial model
state, SGD learning rate, batch size, and five full data passes. FedAvg uses
five IID clients, full participation, one local epoch, and five rounds in the
default config.

Run from the repository root:
    conda run -n mse-ai python experiments/iid/month2_week8_mnist_fedavg.py \
        --config configs/month2_week8_mnist_iid_fedavg_vs_centralized.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from numbers import Real
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
from evaluation.utility import ClassificationEvaluation, evaluate_classifier
from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP
from server.fedavg_server import FedAvgServer


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Saved JSON config."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Deliberately replace an existing saved run. Without this flag, a "
            "non-empty output directory stops the run before training."
        ),
    )
    return parser.parse_args()


def validate_week8_config(config: dict) -> None:
    """Reject settings that would mislabel this first experiment."""
    if config["dataset"] != "MNIST":
        raise ValueError("The Week 8 runner supports MNIST only.")
    if config["dataset_version"] != "TorchVision MNIST official train/test split":
        raise ValueError("dataset_version does not describe the loaded MNIST files.")
    if config["data_subdirectory"] != "mnist":
        raise ValueError("data_subdirectory must be mnist for this runner.")
    if config["model"] != "SimpleMLP":
        raise ValueError("The Week 8 model must be SimpleMLP.")
    if config["preprocessing"] != "ToTensor only; pixels scaled to [0, 1]":
        raise ValueError("preprocessing does not match the MNIST loader.")
    if config["algorithm"] != "handwritten_fedavg_iid_vs_centralized_sgd":
        raise ValueError("The Week 8 algorithm label is incorrect.")
    if config["partition_strategy"] != "iid_seeded_equal_size":
        raise ValueError("Week 8 must use the IID partition; Non-IID is Month 3.")
    if config["alpha"] is not None:
        raise ValueError("alpha must be null for this IID experiment.")
    if config["target_client"] is not None:
        raise ValueError("target_client must be null before unlearning work begins.")
    if config["optimizer"] != "SGD":
        raise ValueError("Both Week 8 comparison paths must use plain SGD.")
    if config["optimizer_details"] != "plain SGD; momentum=0; weight_decay=0":
        raise ValueError("optimizer_details do not match the two training paths.")
    if config["loss_function"] != "CrossEntropyLoss":
        raise ValueError("The Week 8 runner uses CrossEntropyLoss.")
    if config["device"] != "cpu":
        raise ValueError("The verified Week 8 comparison runs on CPU only.")
    if config["strict_environment"] is not True:
        raise ValueError("strict_environment must be true for the canonical run.")
    if config["runner"] != "experiments/iid/month2_week8_mnist_fedavg.py":
        raise ValueError("runner must name this Week 8 experiment script.")
    if config["environment_manifest"] != "environment/month2_cpu_runtime.json":
        raise ValueError("environment_manifest must name the Month 2 CPU runtime.")
    if config["communication_cost_unit"] != "bytes":
        raise ValueError("communication_cost_unit must be bytes.")
    if config["communication_cost_definition"] != (
        "dense tensor payload: one full model download and upload per selected "
        "client per round"
    ):
        raise ValueError("The communication-cost definition does not match the runner.")
    if config["selection_rule"] != (
        "ceil(client_fraction * number_of_clients), minimum 1"
    ):
        raise ValueError("The client-selection rule does not match FedAvgServer.")
    if config["accuracy"] is not None or config["f1"] is not None:
        raise ValueError("Input accuracy and f1 must be null; the runner measures them.")
    if config["unlearning_time"] != 0.0:
        raise ValueError("unlearning_time must be 0.0 because Week 8 has no unlearning.")
    if config["communication_cost"] != 0:
        raise ValueError(
            "Input communication_cost must be 0; the runner calculates the result."
        )
    if config["num_workers"] != 0:
        raise ValueError("num_workers must remain 0 for deterministic Windows runs.")
    if isinstance(config["client_fraction"], bool) or config["client_fraction"] != 1.0:
        raise ValueError(
            "The first matched Week 8 comparison requires full client participation."
        )
    if (
        isinstance(config["validation_fraction"], bool)
        or not isinstance(config["validation_fraction"], Real)
        or not math.isfinite(float(config["validation_fraction"]))
        or not 0 < config["validation_fraction"] < 1
    ):
        raise ValueError("validation_fraction must be in (0, 1).")
    if config["validation_fraction"] != 0.15:
        raise ValueError("The canonical Week 8 train/validation split uses 0.15.")
    for field in (
        "number_of_clients",
        "local_epochs",
        "batch_size",
        "number_of_rounds",
        "centralized_epochs",
        "hidden_units",
    ):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if config["architecture"] != f"784 -> {config['hidden_units']} ReLU -> 10":
        raise ValueError("architecture does not match SimpleMLP and hidden_units.")
    if (
        isinstance(config["learning_rate"], bool)
        or not isinstance(config["learning_rate"], Real)
        or not math.isfinite(float(config["learning_rate"]))
        or config["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be finite and positive.")
    if config["centralized_epochs"] != (
        config["number_of_rounds"] * config["local_epochs"]
    ):
        raise ValueError(
            "For the matched full-participation comparison, centralized_epochs "
            "must equal number_of_rounds * local_epochs."
        )
    for field in (
        "expected_training_examples",
        "expected_validation_examples",
        "expected_test_examples",
    ):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if (
        config["expected_training_examples"],
        config["expected_validation_examples"],
        config["expected_test_examples"],
    ) != (51_000, 9_000, 10_000):
        raise ValueError("The canonical MNIST split must be 51,000/9,000/10,000.")
    if config["number_of_clients"] > min(
        config["expected_training_examples"], config["expected_test_examples"]
    ):
        raise ValueError(
            "number_of_clients exceeds the smallest train/test partition source."
        )
    for field in (
        "random_seed",
        "split_seed",
        "train_partition_seed",
        "test_partition_seed",
        "centralized_loader_seed",
        "model_initialization_seed",
    ):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer.")
    for field in ("experiment_name", "output_subdirectory", "code_revision"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")
    output_subdirectory = Path(config["output_subdirectory"])
    if output_subdirectory.is_absolute() or ".." in output_subdirectory.parts:
        raise ValueError("output_subdirectory must stay inside results/.")


def _run_git(*arguments: str) -> str:
    """Run one read-only Git command with the repository's safe path declared."""
    command = [
        "git",
        "-c",
        f"safe.directory={PROJECT_ROOT.as_posix()}",
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git provenance check failed: {detail}")
    return completed.stdout.strip()


def validate_code_provenance(config: dict, config_path: Path) -> dict:
    """Require the configured revision to equal a clean, fully tracked HEAD."""
    try:
        relative_config = config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("The experiment config must be inside this repository.") from error

    head_commit = _run_git("rev-parse", "HEAD")
    configured_revision = config["code_revision"]
    try:
        resolved_revision = _run_git(
            "rev-parse", "--verify", f"{configured_revision}^{{commit}}"
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Configured code_revision {configured_revision!r} does not exist. "
            "Commit the finalized experiment and create that tag before training."
        ) from error
    if resolved_revision != head_commit:
        raise RuntimeError(
            "Configured code_revision does not resolve to the current HEAD: "
            f"{resolved_revision} != {head_commit}."
        )

    tracked_dependencies = [
        relative_config,
        config["runner"],
        config["environment_manifest"],
    ]
    for dependency in tracked_dependencies:
        _run_git("ls-files", "--error-unmatch", "--", dependency)

    status = _run_git("status", "--porcelain", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise RuntimeError(
            "The Git worktree is not clean. Commit the exact experiment code before "
            f"training. Current changes:\n{preview}"
        )
    return {
        "configured_revision": configured_revision,
        "resolved_commit": resolved_revision,
        "head_commit": head_commit,
        "worktree_clean_at_start": True,
        "tracked_dependencies": tracked_dependencies,
    }


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def checksum_indices(indices: list[int] | tuple[int, ...]) -> str:
    array = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def checksum_model_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    random_seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        if random_seed is None:
            raise ValueError("A shuffled loader requires an explicit random seed.")
        generator = torch.Generator().manual_seed(random_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def load_and_split_mnist(config: dict) -> tuple[Subset, Subset, Dataset]:
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
    training_data, validation_data = random_split(
        complete_training_data,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    expected = (
        config["expected_training_examples"],
        config["expected_validation_examples"],
        config["expected_test_examples"],
    )
    observed = (len(training_data), len(validation_data), len(test_data))
    if observed != expected:
        raise RuntimeError(f"MNIST split mismatch: expected {expected}, found {observed}.")
    return training_data, validation_data, test_data


def class_histogram(labels: torch.Tensor, indices: list[int]) -> list[int]:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    return torch.bincount(selected, minlength=10).tolist()


def build_client_partitions(
    training_data: Subset,
    test_data: Dataset,
    config: dict,
) -> tuple[dict[int, Subset], dict[int, Subset], list[dict], list[dict]]:
    number_of_clients = config["number_of_clients"]
    training_positions = iid_partition_indices(
        len(training_data), number_of_clients, config["train_partition_seed"]
    )
    test_positions = iid_partition_indices(
        len(test_data), number_of_clients, config["test_partition_seed"]
    )
    client_training_data = {
        client_id: Subset(training_data, list(positions))
        for client_id, positions in training_positions.items()
    }
    client_test_data = {
        client_id: Subset(test_data, list(positions))
        for client_id, positions in test_positions.items()
    }

    complete_training_data = training_data.dataset
    training_records: list[dict] = []
    for client_id, positions in training_positions.items():
        original_indices = [training_data.indices[position] for position in positions]
        training_records.append(
            {
                "client_id": client_id,
                "number_of_examples": len(positions),
                "position_checksum_sha256": checksum_indices(positions),
                "original_index_checksum_sha256": checksum_indices(original_indices),
                "class_histogram": class_histogram(
                    complete_training_data.targets, original_indices
                ),
            }
        )

    test_records: list[dict] = []
    for client_id, positions in test_positions.items():
        test_records.append(
            {
                "client_id": client_id,
                "number_of_examples": len(positions),
                "index_checksum_sha256": checksum_indices(positions),
                "class_histogram": class_histogram(test_data.targets, list(positions)),
            }
        )
    return client_training_data, client_test_data, training_records, test_records


def train_one_centralized_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    steps = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        steps += 1
    return total_loss / total_examples, steps


def train_centralized_reference(
    *,
    model: nn.Module,
    training_data: Dataset,
    validation_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict], float, int]:
    loader = make_loader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        random_seed=config["centralized_loader_seed"],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    history: list[dict] = []
    initial_validation = evaluate_classifier(
        model, validation_loader, loss_function, device
    )
    history.append(
        {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": round(initial_validation.loss, 6),
            "validation_accuracy": round(initial_validation.accuracy, 6),
            "optimizer_steps": 0,
            "selected_as_best_checkpoint": False,
        }
    )

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_history_index: int | None = None
    total_steps = 0
    started = time.perf_counter()
    for epoch in range(1, config["centralized_epochs"] + 1):
        train_loss, steps = train_one_centralized_epoch(
            model, loader, optimizer, loss_function, device
        )
        total_steps += steps
        validation = evaluate_classifier(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "validation_loss": round(validation.loss, 6),
                "validation_accuracy": round(validation.accuracy, 6),
                "optimizer_steps": steps,
                "selected_as_best_checkpoint": False,
            }
        )
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy
            best_state = clone_model_state(model.state_dict())
            best_history_index = len(history) - 1
        print(
            f"Central epoch {epoch:02d}/{config['centralized_epochs']} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {validation.loss:.4f}, val acc {validation.accuracy:.4f}"
        )
    elapsed = time.perf_counter() - started
    if best_state is None or best_history_index is None:
        raise RuntimeError("Centralized training did not produce a checkpoint.")
    history[best_history_index]["selected_as_best_checkpoint"] = True
    return best_state, history, elapsed, total_steps


def train_fedavg(
    *,
    model: nn.Module,
    client_training_data: dict[int, Subset],
    validation_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict], float, int, int]:
    server = FedAvgServer(
        global_model=model,
        client_datasets=client_training_data,
        client_fraction=config["client_fraction"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        random_seed=config["random_seed"],
        device=device,
    )
    loss_function = nn.CrossEntropyLoss()
    history: list[dict] = []
    initial_validation = evaluate_classifier(
        server.global_model, validation_loader, loss_function, device
    )
    history.append(
        {
            "round": 0,
            "selected_client_ids": [],
            "weighted_mean_local_loss": None,
            "validation_loss": round(initial_validation.loss, 6),
            "validation_accuracy": round(initial_validation.accuracy, 6),
            "communication_cost_bytes": 0,
            "selected_as_best_checkpoint": False,
        }
    )

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_history_index: int | None = None
    total_steps = 0
    total_communication = 0
    started = time.perf_counter()
    for round_number in range(1, config["number_of_rounds"] + 1):
        round_result = server.run_round(round_number)
        validation = evaluate_classifier(
            server.global_model, validation_loader, loss_function, device
        )
        round_record = asdict(round_result)
        round_record["round"] = round_record.pop("round_number")
        round_record.update(
            {
                "validation_loss": round(validation.loss, 6),
                "validation_accuracy": round(validation.accuracy, 6),
                "selected_as_best_checkpoint": False,
            }
        )
        history.append(round_record)
        total_steps += sum(round_result.client_optimizer_steps.values())
        total_communication += round_result.communication_cost_bytes
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy
            best_state = server.global_state()
            best_history_index = len(history) - 1
        print(
            f"FedAvg round {round_number:02d}/{config['number_of_rounds']} | "
            f"clients {list(round_result.selected_client_ids)} | "
            f"local loss {round_result.weighted_mean_local_loss:.4f} | "
            f"val loss {validation.loss:.4f}, val acc {validation.accuracy:.4f}"
        )
    elapsed = time.perf_counter() - started
    if best_state is None or best_history_index is None:
        raise RuntimeError("FedAvg training did not produce a checkpoint.")
    history[best_history_index]["selected_as_best_checkpoint"] = True
    return best_state, history, elapsed, total_steps, total_communication


def per_class_accuracy(evaluation: ClassificationEvaluation) -> list[float | None]:
    values: list[float | None] = []
    for class_id in range(10):
        mask = evaluation.labels == class_id
        values.append(
            float((evaluation.predictions[mask] == class_id).mean())
            if mask.any()
            else None
        )
    return values


def client_test_utility(
    model: nn.Module,
    client_test_data: dict[int, Subset],
    config: dict,
    device: torch.device,
) -> list[dict]:
    records: list[dict] = []
    loss_function = nn.CrossEntropyLoss()
    for client_id in sorted(client_test_data):
        evaluation = evaluate_classifier(
            model,
            make_loader(
                client_test_data[client_id],
                batch_size=config["batch_size"],
                shuffle=False,
            ),
            loss_function,
            device,
        )
        records.append(
            {
                "client_id": client_id,
                "number_of_examples": len(client_test_data[client_id]),
                "accuracy": round(evaluation.accuracy, 6),
                "f1_macro": round(evaluation.f1_macro, 6),
            }
        )
    return records


def save_confusion_plot(
    evaluation: ClassificationEvaluation, title: str, output_path: Path
) -> list[list[int]]:
    matrix = confusion_matrix(
        evaluation.labels, evaluation.predictions, labels=range(10)
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=range(10))
    figure, axis = plt.subplots(figsize=(7, 7))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return matrix.tolist()


def save_convergence_plot(
    centralized_history: list[dict],
    fedavg_history: list[dict],
    local_epochs: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    central_x = [row["epoch"] for row in centralized_history]
    fed_x = [row["round"] * local_epochs for row in fedavg_history]
    axes[0].plot(
        central_x,
        [row["validation_accuracy"] for row in centralized_history],
        marker="o",
        label="Centralized SGD",
    )
    axes[0].plot(
        fed_x,
        [row["validation_accuracy"] for row in fedavg_history],
        marker="o",
        label="FedAvg (round × local epochs)",
    )
    axes[0].set(
        xlabel="Cumulative full-data-equivalent passes",
        ylabel="Validation accuracy",
        title="Validation accuracy convergence",
        ylim=(0, 1),
    )
    axes[0].legend()

    axes[1].plot(
        central_x,
        [row["validation_loss"] for row in centralized_history],
        marker="o",
        label="Centralized SGD",
    )
    axes[1].plot(
        fed_x,
        [row["validation_loss"] for row in fedavg_history],
        marker="o",
        label="FedAvg (round × local epochs)",
    )
    axes[1].set(
        xlabel="Cumulative full-data-equivalent passes",
        ylabel="Cross-entropy loss",
        title="Validation loss convergence",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def evaluation_record(
    evaluation: ClassificationEvaluation,
    confusion: list[list[int]],
    client_utility: list[dict],
) -> dict:
    return {
        "test_loss": round(evaluation.loss, 6),
        "accuracy": round(evaluation.accuracy, 6),
        "f1_macro": round(evaluation.f1_macro, 6),
        "confusion_matrix": confusion,
        "per_class_accuracy": [
            round(value, 6) if value is not None else None
            for value in per_class_accuracy(evaluation)
        ],
        "per_client_test_utility": client_utility,
    }


def save_beginner_summary(results: dict, output_path: Path) -> None:
    central = results["centralized"]
    fedavg = results["fedavg"]
    text = f"""# Week 8 Saved-Run Summary

This file was generated automatically from the saved experiment metrics.

| Method | Test accuracy | Macro F1 | Best validation step | Train + validation-loop time |
|---|---:|---:|---:|---:|
| Centralized SGD | {central['accuracy']:.2%} | {central['f1_macro']:.2%} | epoch {central['best_epoch']} | {central['training_and_validation_time_seconds']:.1f}s |
| Hand-written FedAvg | {fedavg['accuracy']:.2%} | {fedavg['f1_macro']:.2%} | round {fedavg['best_round']} | {fedavg['training_and_validation_time_seconds']:.1f}s |

- Accuracy difference (FedAvg minus centralized): {results['accuracy_difference_fedavg_minus_centralized'] * 100:+.2f} percentage points.
- FedAvg estimated communication: {results['communication_cost']:,} bytes ({results['communication_cost'] / (1024 ** 2):.2f} MiB) of dense tensor payload.
- The mandatory top-level accuracy and F1 belong to `{results['primary_method_for_mandatory_metrics']}`.
- Both paths started from checksum `{results['initial_model_checksum_sha256']}`.
- This is one fixed-seed, sequential CPU, IID experiment. Timing is not a real distributed-system benchmark.
- The test set was evaluated only after validation selected each method's checkpoint.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    validate_week8_config(config)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    code_provenance = validate_code_provenance(config, config_path)
    set_random_seed(config["random_seed"])

    device = torch.device(config["device"])
    results_root = (PROJECT_ROOT / "results").resolve()
    output_directory = (results_root / config["output_subdirectory"]).resolve()
    try:
        output_directory.relative_to(results_root)
    except ValueError as error:
        raise ValueError("output_subdirectory escaped the results directory.") from error
    if (
        output_directory.exists()
        and any(output_directory.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. Use --overwrite only "
            "for an intentional exact rerun, or choose a new output_subdirectory."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "source_config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "code_provenance": code_provenance,
    }
    write_json(output_directory / "resolved_config.json", resolved_config)

    training_data, validation_data, test_data = load_and_split_mnist(config)
    (
        client_training_data,
        client_test_data,
        training_partition_records,
        test_partition_records,
    ) = build_client_partitions(training_data, test_data, config)
    validation_loader = make_loader(
        validation_data,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    test_loader = make_loader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    torch.manual_seed(config["model_initialization_seed"])
    initial_model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    initial_state = clone_model_state(initial_model.state_dict())
    initial_checksum = checksum_model_state(initial_state)
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    payload_bytes = model_state_size_bytes(initial_state)

    centralized_model = copy.deepcopy(initial_model)
    fedavg_model = copy.deepcopy(initial_model)
    if checksum_model_state(centralized_model.state_dict()) != initial_checksum:
        raise RuntimeError("Centralized model did not receive the common initialization.")
    if checksum_model_state(fedavg_model.state_dict()) != initial_checksum:
        raise RuntimeError("FedAvg model did not receive the common initialization.")

    print(f"Configuration: {config_path}")
    print(f"Clients: {config['number_of_clients']} IID, C={config['client_fraction']}")
    print(f"Initial model checksum: {initial_checksum}")
    central_state, central_history, central_time, central_steps = (
        train_centralized_reference(
            model=centralized_model,
            training_data=training_data,
            validation_loader=validation_loader,
            config=config,
            device=device,
        )
    )
    fedavg_state, fedavg_history, fedavg_time, fedavg_steps, communication = (
        train_fedavg(
            model=fedavg_model,
            client_training_data=client_training_data,
            validation_loader=validation_loader,
            config=config,
            device=device,
        )
    )

    centralized_model.load_state_dict(central_state, strict=True)
    fedavg_model.load_state_dict(fedavg_state, strict=True)
    loss_function = nn.CrossEntropyLoss()
    centralized_test = evaluate_classifier(
        centralized_model, test_loader, loss_function, device
    )
    fedavg_test = evaluate_classifier(fedavg_model, test_loader, loss_function, device)
    centralized_client_utility = client_test_utility(
        centralized_model, client_test_data, config, device
    )
    fedavg_client_utility = client_test_utility(
        fedavg_model, client_test_data, config, device
    )

    central_confusion = save_confusion_plot(
        centralized_test,
        "MNIST test confusion matrix — centralized SGD",
        output_directory / "centralized_confusion_matrix.png",
    )
    fedavg_confusion = save_confusion_plot(
        fedavg_test,
        "MNIST test confusion matrix — hand-written FedAvg",
        output_directory / "fedavg_confusion_matrix.png",
    )
    save_convergence_plot(
        central_history,
        fedavg_history,
        config["local_epochs"],
        output_directory / "convergence_comparison.png",
    )
    torch.save(central_state, output_directory / "best_centralized_model.pt")
    torch.save(fedavg_state, output_directory / "best_fedavg_model.pt")

    central_best_row = next(
        row for row in central_history if row["selected_as_best_checkpoint"]
    )
    fedavg_best_row = next(
        row for row in fedavg_history if row["selected_as_best_checkpoint"]
    )
    central_record = {
        **evaluation_record(
            centralized_test, central_confusion, centralized_client_utility
        ),
        "best_epoch": central_best_row["epoch"],
        "best_validation_accuracy": central_best_row["validation_accuracy"],
        "training_and_validation_time_seconds": round(central_time, 3),
        "optimizer_steps": central_steps,
        "training_examples_processed": len(training_data)
        * config["centralized_epochs"],
        "communication_cost_bytes": 0,
        "history": central_history,
        "checkpoint_checksum_sha256": checksum_model_state(central_state),
    }
    fedavg_record = {
        **evaluation_record(fedavg_test, fedavg_confusion, fedavg_client_utility),
        "best_round": fedavg_best_row["round"],
        "best_validation_accuracy": fedavg_best_row["validation_accuracy"],
        "training_and_validation_time_seconds": round(fedavg_time, 3),
        "optimizer_steps": fedavg_steps,
        "training_examples_processed": sum(
            sum(row["client_example_counts"].values()) * config["local_epochs"]
            for row in fedavg_history[1:]
        ),
        "communication_cost_bytes": communication,
        "history": fedavg_history,
        "checkpoint_checksum_sha256": checksum_model_state(fedavg_state),
    }

    results = {
        **config,
        "source_config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "runtime": runtime,
        "code_provenance": code_provenance,
        "primary_method_for_mandatory_metrics": "handwritten_fedavg",
        "accuracy": fedavg_record["accuracy"],
        "f1": fedavg_record["f1_macro"],
        "communication_cost": communication,
        "unlearning_time": 0.0,
        "training_examples": len(training_data),
        "validation_examples": len(validation_data),
        "test_examples": len(test_data),
        "trainable_parameters": parameter_count,
        "model_payload_bytes": payload_bytes,
        "initial_model_checksum_sha256": initial_checksum,
        "centralized_initial_checksum_sha256": initial_checksum,
        "fedavg_initial_checksum_sha256": initial_checksum,
        "training_split_checksum_sha256": checksum_indices(training_data.indices),
        "validation_split_checksum_sha256": checksum_indices(validation_data.indices),
        "client_training_partitions": training_partition_records,
        "client_test_partitions": test_partition_records,
        "centralized": central_record,
        "fedavg": fedavg_record,
        "accuracy_difference_fedavg_minus_centralized": round(
            fedavg_record["accuracy"] - central_record["accuracy"], 6
        ),
        "f1_difference_fedavg_minus_centralized": round(
            fedavg_record["f1_macro"] - central_record["f1_macro"], 6
        ),
        "comparison_protocol": {
            "same_initial_model": True,
            "same_training_split": True,
            "same_optimizer": config["optimizer"],
            "same_learning_rate": config["learning_rate"],
            "same_batch_size": config["batch_size"],
            "centralized_full_data_passes": config["centralized_epochs"],
            "fedavg_full_data_passes_when_full_participation": config[
                "number_of_rounds"
            ]
            * config["local_epochs"],
            "test_evaluated_after_validation_selection_only": True,
            "optimizer_updates_identical": False,
            "optimizer_update_difference": (
                "Centralized SGD follows one continuous trajectory; FedAvg follows "
                "separate local trajectories and averages model weights. Partial "
                "final batches also make the optimizer-step counts differ."
            ),
            "timing_scope": "sequential CPU process; not distributed wall-clock time",
        },
    }
    write_json(output_directory / "metrics.json", results)
    save_beginner_summary(results, output_directory / "comparison_summary.md")

    print(
        f"Centralized test: accuracy {central_record['accuracy']:.4f}, "
        f"macro F1 {central_record['f1_macro']:.4f}"
    )
    print(
        f"FedAvg test: accuracy {fedavg_record['accuracy']:.4f}, "
        f"macro F1 {fedavg_record['f1_macro']:.4f}"
    )
    print(f"Estimated FedAvg communication: {communication:,} bytes")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
