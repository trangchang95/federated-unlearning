"""Month 1 / Week 2: learn neural-network training with PyTorch on MNIST.

Concept map for a beginner:

1. Forward propagation: ``model(images)`` produces ten scores per image.
2. Loss: ``CrossEntropyLoss`` measures how wrong those scores are.
3. Backpropagation: ``loss.backward()`` calculates each weight's contribution
   to the error.
4. Optimizer: ``optimizer.step()`` changes the weights to reduce that error.
5. Learning rate: the config controls how large each weight change may be.

Run from the repository root:
    conda run -n mse-ai python experiments/month1_week2_mnist_mlp.py \
        --config configs/month1_week2_mnist_mlp.json
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.simple_mlp import SimpleMLP


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month1_week2_mnist_mlp.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the JSON experiment config.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Make data splitting, initial weights, and batch order repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_data_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download MNIST and construct deterministic train/validation/test loaders."""
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
    split_generator = torch.Generator().manual_seed(config["random_seed"])
    training_data, validation_data = random_split(
        complete_training_data,
        [training_size, validation_size],
        generator=split_generator,
    )

    # num_workers=0 is the safest deterministic choice on Windows.
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
    }
    training_loader = DataLoader(
        training_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["random_seed"]),
        **loader_options,
    )
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return training_loader, validation_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Perform forward propagation, backpropagation, and one optimizer update per batch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    example_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)  # Forward propagation.
        loss = loss_function(logits, labels)  # Measure the current error.
        loss.backward()  # Backpropagation calculates gradients.
        optimizer.step()  # Adam uses those gradients to update the weights.

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        example_count += batch_size

    return total_loss / example_count, correct_predictions / example_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate without calculating gradients, because no weights change here."""
    model.eval()
    total_loss = 0.0
    labels_collected: list[np.ndarray] = []
    predictions_collected: list[np.ndarray] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)

        total_loss += loss.item() * labels.size(0)
        labels_collected.append(labels.cpu().numpy())
        predictions_collected.append(logits.argmax(dim=1).cpu().numpy())

    all_labels = np.concatenate(labels_collected)
    all_predictions = np.concatenate(predictions_collected)
    return total_loss / len(all_labels), all_labels, all_predictions


def save_learning_curve(history: list[dict], output_directory: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss (lower is better)", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy (higher is better)", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_directory / "learning_curve.png", dpi=140)
    plt.close(figure)


def save_confusion_matrix(
    true_labels: np.ndarray, predictions: np.ndarray, output_directory: Path
) -> np.ndarray:
    matrix = confusion_matrix(true_labels, predictions, labels=range(10))
    display = ConfusionMatrixDisplay(matrix, display_labels=range(10))
    figure, axis = plt.subplots(figsize=(7, 7))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("MNIST test confusion matrix (Week 2 MLP)")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=140)
    plt.close(figure)
    return matrix


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    set_random_seed(config["random_seed"])

    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is not available.")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    device = torch.device(config["device"])

    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    training_loader, validation_loader, test_loader = make_data_loaders(config)

    model = SimpleMLP(hidden_units=config["hidden_units"]).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")

    history: list[dict] = []
    best_validation_accuracy = -1.0
    best_model_state: dict | None = None
    training_started = time.perf_counter()

    for epoch in range(1, config["local_epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, training_loader, loss_function, optimizer, device
        )
        validation_loss, validation_labels, validation_predictions = evaluate(
            model, validation_loader, loss_function, device
        )
        validation_accuracy = accuracy_score(validation_labels, validation_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "validation_loss": round(validation_loss, 6),
                "validation_accuracy": round(float(validation_accuracy), 6),
            }
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = float(validation_accuracy)
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{config['local_epochs']} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f}, val acc {validation_accuracy:.4f}"
        )

    training_time = time.perf_counter() - training_started
    if best_model_state is None:
        raise RuntimeError("Training completed without producing a model state.")
    model.load_state_dict(best_model_state)

    test_loss, test_labels, test_predictions = evaluate(
        model, test_loader, loss_function, device
    )
    accuracy = accuracy_score(test_labels, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average="macro", zero_division=0
    )

    save_learning_curve(history, output_directory)
    test_confusion_matrix = save_confusion_matrix(
        test_labels, test_predictions, output_directory
    )
    torch.save(best_model_state, output_directory / "best_model.pt")

    results = {
        **config,
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "runtime": runtime,
        "torch_version": torch.__version__,
        "class_names": [str(digit) for digit in range(10)],
        "confusion_matrix": test_confusion_matrix.tolist(),
        "training_examples": len(training_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "trainable_parameters": parameter_count,
        "training_time_seconds": round(training_time, 3),
        "best_validation_accuracy": round(best_validation_accuracy, 6),
        "test_loss": round(test_loss, 6),
        "accuracy": round(float(accuracy), 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "epoch_history": history,
    }
    write_json(output_directory / "metrics.json", results)

    print(f"Test accuracy: {accuracy:.4f} | macro F1: {f1:.4f}")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
