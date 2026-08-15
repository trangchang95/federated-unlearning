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

import json
import time
from pathlib import Path

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

RANDOM_SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "month1_week1"


def load_mnist():
    """Download MNIST (70,000 handwritten digit images, 28x28 pixels each).

    fetch_openml caches the download under ~/scikit_learn_data, so this is
    slow only the first time it runs.
    """
    print("Loading MNIST from OpenML (first run downloads ~50MB, cached after)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0  # pixels 0-255 -> 0-1
    y = mnist.target.astype(np.int64)
    return X, y


def split_data(X, y):
    """train/validation/test split, the core Week 1 concept.

    - train: what the model learns from.
    - validation: used to check the model *during* development (not used
      here since logistic regression has no tuning loop yet, but the split
      is created now so later weeks reuse the same convention).
    - test: touched only once, at the end, to report the final number.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def explore_with_pandas(y_train, y_val, y_test):
    """Pandas practice: tabulate class balance across the three splits."""
    df = pd.DataFrame(
        {
            "train": pd.Series(y_train).value_counts().sort_index(),
            "val": pd.Series(y_val).value_counts().sort_index(),
            "test": pd.Series(y_test).value_counts().sort_index(),
        }
    )
    df.index.name = "digit"
    df.to_csv(OUT_DIR / "class_distribution.csv")
    print("\nClass distribution (samples per digit per split):")
    print(df)
    return df


def plot_sample_digits(X, y):
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
    fig.savefig(OUT_DIR / "sample_digits.png", dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Test set confusion matrix (logistic regression baseline)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_mnist()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    explore_with_pandas(y_train, y_val, y_test)
    plot_sample_digits(X, y)

    print(
        f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, "
        f"test: {len(X_test)}"
    )

    print("\nTraining logistic regression baseline...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=300, random_state=RANDOM_SEED)
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

    plot_confusion(y_test, test_pred)

    metrics = {
        "dataset": "MNIST (mnist_784, OpenML)",
        "model": "LogisticRegression(max_iter=300)",
        "random_seed": RANDOM_SEED,
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "train_time_seconds": round(train_time, 2),
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_precision_macro": round(float(precision), 4),
        "test_recall_macro": round(float(recall), 4),
        "test_f1_macro": round(float(f1), 4),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs written to {OUT_DIR}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
