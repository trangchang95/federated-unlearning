"""
Week 1 exercise (thesis plan, Month 1 / Week 1): "MNIST classifier".

Week 1 only covers Python, NumPy, Pandas, Matplotlib, Git, and the
train/validation/test idea -- neural networks and CNNs come in Week 2-3.
So the classifier here is intentionally a simple, non-deep-learning
baseline (logistic regression), not a neural net. Its purpose is to
practice the data/eval workflow that every later experiment will reuse.

Run:
    python experiments/month1_week1_mnist_baseline.py

Outputs (written to results/month1_week1/):
    - class_distribution.csv   (Pandas: how many samples per digit)
    - sample_digits.png        (Matplotlib: a few example images)
    - confusion_matrix.png     (Matplotlib: model errors by digit)
    - metrics.json             (accuracy/precision/recall/F1 + run config)
"""

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import (
    load_json_config,
    validate_runtime_environment,
    write_json,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "month1_week1_mnist_logistic_regression.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_mnist(config):
    """Download MNIST (70,000 handwritten digit images, 28x28 pixels each).

    The cache is kept in the repository's gitignored data/ directory, so the
    experiment never writes dataset files into a user's home directory.
    """
    print("Loading MNIST from OpenML (first run downloads ~50MB, cached after)...")
    data_directory = PROJECT_ROOT / "data" / config["data_subdirectory"]
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=data_directory,
    )
    X = mnist.data.astype(np.float32) / 255.0  # pixels 0-255 -> 0-1
    y = mnist.target.astype(np.int64)
    return X, y


def split_data(X, y, config):
    """train/validation/test split, the core Week 1 concept.

    - train: what the model learns from.
    - validation: used to check the model *during* development (not used
      here since logistic regression has no tuning loop yet, but the split
      is created now so later weeks reuse the same convention).
    - test: touched only once, at the end, to report the final number.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config["validation_fraction"] + config["test_fraction"],
        random_state=config["random_seed"],
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config["test_fraction"]
        / (config["validation_fraction"] + config["test_fraction"]),
        random_state=config["random_seed"],
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def explore_with_pandas(y_train, y_val, y_test, output_directory):
    """Pandas practice: tabulate class balance across the three splits."""
    df = pd.DataFrame(
        {
            "train": pd.Series(y_train).value_counts().sort_index(),
            "val": pd.Series(y_val).value_counts().sort_index(),
            "test": pd.Series(y_test).value_counts().sort_index(),
        }
    )
    df.index.name = "digit"
    df.to_csv(output_directory / "class_distribution.csv")
    print("\nClass distribution (samples per digit per split):")
    print(df)
    return df


def plot_sample_digits(X, y, output_directory):
    """Matplotlib practice: show one example image per digit class."""
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        ax = axes[digit // 5, digit % 5]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(str(digit))
        ax.axis("off")
    fig.suptitle("One example per MNIST digit class")
    fig.tight_layout()
    fig.savefig(output_directory / "sample_digits.png", dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, output_directory) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Test set confusion matrix (logistic regression baseline)")
    fig.tight_layout()
    fig.savefig(output_directory / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return cm


def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    runtime = validate_runtime_environment(config, PROJECT_ROOT)
    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    X, y = load_mnist(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    explore_with_pandas(y_train, y_val, y_test, output_directory)
    plot_sample_digits(X, y, output_directory)

    print(
        f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, "
        f"test: {len(X_test)}"
    )

    print("\nTraining logistic regression baseline...")
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=config["max_iterations"], random_state=config["random_seed"]
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")

    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro"
    )
    print(f"Test accuracy: {test_acc:.4f} | macro F1: {f1:.4f}")

    test_confusion_matrix = plot_confusion(y_test, test_pred, output_directory)

    metrics = {
        **config,
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "runtime": runtime,
        "model": f"LogisticRegression(max_iter={config['max_iterations']})",
        "class_names": [str(digit) for digit in range(10)],
        "confusion_matrix": test_confusion_matrix.tolist(),
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "train_time_seconds": round(train_time, 2),
        "val_accuracy": round(float(val_acc), 4),
        "accuracy": round(float(test_acc), 4),
        "test_precision_macro": round(float(precision), 4),
        "test_recall_macro": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }
    write_json(output_directory / "metrics.json", metrics)

    print(f"\nAll outputs written to {output_directory}")


if __name__ == "__main__":
    main()
