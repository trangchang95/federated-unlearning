"""Build the Week 13 report from the saved, config-driven MNIST run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "results" / "month4_week13_mnist_full_retraining" / "metrics.json"
DEFAULT_REPORT = DEFAULT_METRICS.parent / "week13_report.md"


def render_report(metrics: dict) -> str:
    config = metrics["config"]
    original = metrics["original_full_training"]
    retrained = metrics["full_retraining_without_target"]
    timings = metrics["timing"]["phase_seconds"]
    original_accuracy = original["metrics"]["global_test"]["accuracy"]
    retrained_accuracy = retrained["metrics"]["global_test"]["accuracy"]
    lines = [
        "# Month 4, Week 13 — Full Retraining After Client Data Deletion",
        "",
        "## Question and scope",
        "",
        "This experiment establishes the exact-retraining reference for later unlearning methods. It trains on the complete MNIST training split, removes one deterministic client's training shard, and starts a fresh model from the same initial weights using only the retained examples. It does not implement SISA slicing or federated unlearning.",
        "",
        "SISA (Sharded, Isolated, Sliced, and Aggregated) is the machine-unlearning design studied in the Week 4 SISA note. This week's measured baseline is full retraining, which supplies a gold-standard comparison for a model trained without the target data.",
        "",
        "## Protocol",
        "",
        f"- Dataset: MNIST, official TorchVision train/test split; seed {config['split_seed']} gives {metrics['training_examples_before_deletion']:,} training examples, {metrics['validation_examples']:,} held-out validation examples, and {config['expected_test_examples']:,} official test examples.",
        f"- Client partition: {config['number_of_clients']} deterministic IID shards; target client {config['target_client']} contributes {metrics['deleted_target_examples']:,} examples.",
        f"- Model and optimizer: {config['architecture']}; {config['training_epochs']} epochs of {config['optimizer']} (learning rate {config['learning_rate']}, batch size {config['batch_size']}); CPU, seed {config['random_seed']}.",
        "- Both models start from identical initial weights. The retrained model uses a new optimizer and fresh training, with the target shard excluded.",
        "- Reported wall-clock time covers optimizer training only. Data loading and evaluation are excluded.",
        "",
        "## Measured results",
        "",
        "| Model | Training examples | Global test accuracy | Global test macro F1 | Training time (s) |",
        "|---|---:|---:|---:|---:|",
        f"| Original, all training data | {metrics['training_examples_before_deletion']:,} | {original_accuracy:.4%} | {original['metrics']['global_test']['f1_macro']:.4%} | {timings['original_full_training']:.3f} |",
        f"| Full retraining without target | {metrics['training_examples_after_deletion']:,} | {retrained_accuracy:.4%} | {retrained['metrics']['global_test']['f1_macro']:.4%} | {timings['retraining_without_target']:.3f} |",
        "",
        f"- Global test accuracy difference (retrained minus original): {metrics['accuracy_difference_retrained_minus_original'] * 100:+.2f} percentage points.",
        f"- Target-shard accuracy after retraining: {retrained['metrics']['target_client']['accuracy']:.4%}. This is a diagnostic on examples excluded from training, not a formal privacy guarantee.",
        f"- Pooled non-target-shard accuracy after retraining: {retrained['metrics']['non_target_clients']['accuracy']:.4%}.",
        "",
        "## Interpretation and limits",
        "",
        "The retrained checkpoint is the reference model for later comparisons of forgetting and retained utility. The measured runtime is one run on the recorded CPU environment; it is sensitive to machine load and does not establish a statistically reliable time difference. Accuracy on the deleted shard alone does not prove that the model has forgotten the target in a privacy or causal sense. The experiment also does not compare SISA's shard-local retraining cost with full retraining.",
        "",
        "## Reproduction and artifacts",
        "",
        "From `federated-unlearning/`, run:",
        "",
        "```bash",
        "python experiments/unlearning/month4_week13_mnist_full_retraining.py \\",
        "  --config configs/month4_week13_mnist_full_retraining.json",
        "python reports/build_month4_week13_report.py",
        "python reports/verify_month4_week13_retraining.py",
        "```",
        "",
        f"The saved config is `{metrics['source_config']}`. Machine-readable metrics and the original/retrained checkpoints are saved under `results/{config['output_subdirectory']}/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true", help="Fail if the saved report is stale.")
    args = parser.parse_args()
    with args.metrics.open("r", encoding="utf-8") as metrics_file:
        report = render_report(json.load(metrics_file))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != report:
            raise SystemExit(f"Week 13 report is missing or stale: {args.output}")
        print(f"Week 13 report matches saved metrics: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Built Week 13 report: {args.output}")


if __name__ == "__main__":
    main()
