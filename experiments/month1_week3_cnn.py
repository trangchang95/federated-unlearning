"""Month 1 / Week 3: centralized CNN baseline on MNIST or CIFAR-10.

A convolutional neural network (CNN) differs from Week 2's MLP because it
looks for small local patterns with reusable filters. Early filters often
respond to edges or color changes; later filters combine those signals into
more useful shapes. Pooling reduces the spatial size of those feature maps.

Run one saved configuration from the repository root, for example:
    conda run -n mse-ai python experiments/month1_week3_cnn.py \
        --config configs/month1_week3_mnist_cnn.json
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)
from models.small_cnn import SmallCNN


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month1_week3_mnist_cnn.json"
DATASET_DETAILS = {
    "MNIST": {
        "class_names": [str(digit) for digit in range(10)],
        "dataset_class": datasets.MNIST,
    },
    "CIFAR10": {
        "class_names": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "dataset_class": datasets.CIFAR10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a Week 3 JSON experiment configuration.",
    )
    return parser.parse_args()


def validate_week3_config(config: dict) -> None:
    """Reject dataset/model mismatches before an expensive training run."""
    if config["dataset"] not in DATASET_DETAILS:
        supported = ", ".join(DATASET_DETAILS)
        raise ValueError(f"Unsupported dataset {config['dataset']!r}; choose {supported}.")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(config["convolution_channels"]) != 2:
        raise ValueError("convolution_channels must contain exactly two values.")
    if len(config["normalization_mean"]) != config["input_channels"]:
        raise ValueError("normalization_mean must have one value per input channel.")
    if len(config["normalization_std"]) != config["input_channels"]:
        raise ValueError("normalization_std must have one value per input channel.")


def set_random_seed(seed: int) -> None:
    """Fix model initialization, data split, augmentations, and batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_transforms(config: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and evaluation image transformations."""
    normalize = transforms.Normalize(
        mean=config["normalization_mean"], std=config["normalization_std"]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), normalize])

    training_steps: list = []
    if config["dataset"] == "CIFAR10" and config["data_augmentation"]:
        # These small random changes teach that an object's class should not
        # change merely because it shifts a few pixels or faces the other way.
        training_steps.extend(
            [
                transforms.RandomCrop(config["image_size"], padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    training_steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(training_steps), evaluation_transform


def make_data_loaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create deterministic train/validation/test loaders for either dataset."""
    details = DATASET_DETAILS[config["dataset"]]
    dataset_class = details["dataset_class"]
    data_directory = PROJECT_ROOT / "data" / config["data_subdirectory"]
    training_transform, evaluation_transform = make_transforms(config)

    complete_training_data = dataset_class(
        root=data_directory,
        train=True,
        transform=training_transform,
        download=True,
    )
    # A second view of the same raw samples avoids random augmentation in the
    # validation set. Validation should measure a stable task each epoch.
    complete_validation_data = dataset_class(
        root=data_directory,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_data = dataset_class(
        root=data_directory,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    split_generator = torch.Generator().manual_seed(config["random_seed"])
    shuffled_indices = torch.randperm(
        len(complete_training_data), generator=split_generator
    ).tolist()
    validation_size = int(
        len(complete_training_data) * config["validation_fraction"]
    )
    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]
    training_data = Subset(complete_training_data, training_indices)
    validation_data = Subset(complete_validation_data, validation_indices)

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
    return training_loader, validation_loader, test_loader, details["class_names"]


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    example_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

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
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += loss_function(logits, labels).item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())

    labels_array = np.concatenate(all_labels)
    predictions_array = np.concatenate(all_predictions)
    return total_loss / len(labels_array), labels_array, predictions_array


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
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    dataset_name: str,
    output_directory: Path,
) -> np.ndarray:
    matrix = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    figure, axis = plt.subplots(figsize=(8, 8))
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45)
    axis.set_title(f"{dataset_name} test confusion matrix (Week 3 CNN)")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=140)
    plt.close(figure)
    return matrix


@torch.no_grad()
def save_feature_maps(
    model: SmallCNN,
    data_loader: DataLoader,
    class_names: list[str],
    config: dict,
    device: torch.device,
    output_directory: Path,
) -> dict[str, str]:
    """Show how eight learned filters respond to one real test image."""
    model.eval()
    images, labels = next(iter(data_loader))
    image = images[0]
    logits = model(image.unsqueeze(0).to(device))
    prediction = logits.argmax(dim=1).item()
    first_maps = torch.relu(model.features[0](image.unsqueeze(0).to(device)))
    first_maps = first_maps.squeeze(0).cpu().numpy()

    mean = torch.tensor(config["normalization_mean"]).view(-1, 1, 1)
    standard_deviation = torch.tensor(config["normalization_std"]).view(-1, 1, 1)
    display_image = (image.cpu() * standard_deviation + mean).clamp(0, 1)

    figure, axes = plt.subplots(3, 3, figsize=(8, 8))
    image_axis = axes[0, 0]
    if config["input_channels"] == 1:
        image_axis.imshow(display_image.squeeze(0), cmap="gray")
    else:
        image_axis.imshow(display_image.permute(1, 2, 0))
    image_axis.set_title(
        f"Input: {class_names[labels[0].item()]}\nPredicted: {class_names[prediction]}"
    )
    image_axis.axis("off")

    feature_axes = [axis for row in axes for axis in row][1:]
    for feature_index, axis in enumerate(feature_axes):
        axis.imshow(first_maps[feature_index], cmap="viridis")
        axis.set_title(f"Feature map {feature_index + 1}")
        axis.axis("off")
    figure.suptitle("Responses learned by the first convolution layer")
    figure.tight_layout()
    figure.savefig(output_directory / "feature_maps.png", dpi=140)
    plt.close(figure)
    return {
        "true_label": class_names[labels[0].item()],
        "predicted_label": class_names[prediction],
    }


def calculate_per_class_accuracy(
    true_labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    per_class_accuracy = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = true_labels == class_index
        correct = (predictions[class_mask] == class_index).sum()
        per_class_accuracy[class_name] = round(float(correct / class_mask.sum()), 6)
    return per_class_accuracy


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    validate_week3_config(config)
    set_random_seed(config["random_seed"])

    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is not available.")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    device = torch.device(config["device"])
    print(f"Configuration: {config_path}")
    print(f"Dataset: {config['dataset']} | Device: {device}")

    training_loader, validation_loader, test_loader, class_names = make_data_loaders(
        config
    )
    model = SmallCNN(
        input_channels=config["input_channels"],
        convolution_channels=tuple(config["convolution_channels"]),
        hidden_units=config["hidden_units"],
        dropout=config["dropout"],
        number_of_classes=len(class_names),
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")

    history: list[dict] = []
    best_validation_accuracy = -1.0
    best_epoch = 0
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
            best_epoch = epoch
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
        test_labels,
        test_predictions,
        class_names,
        config["dataset"],
        output_directory,
    )
    feature_map_example = save_feature_maps(
        model, test_loader, class_names, config, device, output_directory
    )
    torch.save(best_model_state, output_directory / "best_model.pt")

    results = {
        **config,
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "runtime": runtime,
        "torch_version": torch.__version__,
        "class_names": class_names,
        "confusion_matrix": test_confusion_matrix.tolist(),
        "feature_map_example": feature_map_example,
        "training_examples": len(training_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "trainable_parameters": parameter_count,
        "training_time_seconds": round(training_time, 3),
        "best_epoch": best_epoch,
        "best_validation_accuracy": round(best_validation_accuracy, 6),
        "test_loss": round(test_loss, 6),
        "accuracy": round(float(accuracy), 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "per_class_accuracy": calculate_per_class_accuracy(
            test_labels, test_predictions, class_names
        ),
        "epoch_history": history,
    }
    write_json(output_directory / "metrics.json", results)

    print(
        f"Best epoch: {best_epoch} | Test accuracy: {accuracy:.4f} | "
        f"macro F1: {f1:.4f}"
    )
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
