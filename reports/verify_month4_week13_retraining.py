"""Verify the saved Week 13 full-retraining run without training the model."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import load_json_config
from experiments.unlearning.month4_week13_mnist_full_retraining import (
    build_client_shards,
    load_and_split_mnist,
)
from algorithms.fedavg import model_state_size_bytes
from evaluation.three_way import evaluate_three_way
from models.simple_mlp import SimpleMLP
from reports.build_month4_week13_report import render_report


CONFIG_PATH = PROJECT_ROOT / "configs" / "month4_week13_mnist_full_retraining.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


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
        raise SystemExit(f"FAIL: Git provenance check failed: {detail}")
    return completed.stdout.strip()


def main() -> None:
    config = load_json_config(CONFIG_PATH)
    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    metrics_path = output_directory / "metrics.json"
    report_path = output_directory / "week13_report.md"
    original_path = output_directory / "original_full_training.pt"
    retrained_path = output_directory / "full_retraining_without_target.pt"
    for artifact in (metrics_path, report_path, original_path, retrained_path):
        require(artifact.is_file(), f"missing saved artifact: {artifact.relative_to(PROJECT_ROOT)}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    require(metrics["config"] == config, "saved config differs from the canonical JSON config")
    require(metrics["source_config"] == "configs/month4_week13_mnist_full_retraining.json", "unexpected source config")
    require(metrics["provenance"]["configured_revision"] == config["code_revision"], "configured revision mismatch")
    require(metrics["provenance"]["head_commit"] == metrics["provenance"]["resolved_commit"], "run did not use its producing tag at HEAD")
    require(metrics["provenance"]["worktree_clean_at_start"] is True, "run did not start from a clean worktree")
    require(metrics["runtime"]["device"] == config["device"], "runtime device differs from config")

    current_head = git("rev-parse", "HEAD")
    require(current_head == metrics["provenance"]["head_commit"], "current HEAD differs from the producing commit")
    worktree_status = git("status", "--porcelain", "--untracked-files=all")
    require(not worktree_status, f"worktree is not clean:\n{worktree_status}")

    before = metrics["training_examples_before_deletion"]
    validation_count = metrics["validation_examples"]
    deleted = metrics["deleted_target_examples"]
    after = metrics["training_examples_after_deletion"]
    clients = metrics["client_training_partitions"]
    require(before == config["expected_training_examples"], "unexpected MNIST training split size")
    require(validation_count == config["expected_validation_examples"], "unexpected MNIST validation split size")
    require(after == before - deleted, "post-deletion example count is inconsistent")
    require(deleted == clients[config["target_client"]]["number_of_examples"], "deleted count differs from target shard size")
    require(sum(row["number_of_examples"] for row in clients) == before, "client shards do not cover the training split")
    require(len(clients) == config["number_of_clients"], "client shard count differs from config")
    for row in clients:
        require(sum(row["class_histogram"]) == row["number_of_examples"], "class histogram count mismatch")

    expected_initial = SimpleMLP(hidden_units=config["hidden_units"]).state_dict()
    expected_schema = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in expected_initial.items()
    }
    require(metrics["model_state_schema"] == expected_schema, "saved model state schema differs from the configured model")
    require(
        metrics["model_payload_bytes"] == model_state_size_bytes(expected_initial),
        "saved model payload byte count does not match its state tensors",
    )
    require(
        metrics["trainable_parameters"] == sum(tensor.numel() for tensor in expected_initial.values()),
        "saved parameter count does not match model state tensors",
    )
    checkpoint_states = {}
    for checkpoint_path in (original_path, retrained_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(state.keys() == expected_initial.keys(), f"unexpected state keys in {checkpoint_path.name}")
        require(
            all(state[name].shape == expected_initial[name].shape for name in state),
            f"checkpoint tensor shape mismatch in {checkpoint_path.name}",
        )
        require(
            all(state[name].dtype == expected_initial[name].dtype for name in state),
            f"checkpoint tensor dtype mismatch in {checkpoint_path.name}",
        )
        require(
            model_state_size_bytes(state) == metrics["model_payload_bytes"],
            f"checkpoint payload byte count mismatch in {checkpoint_path.name}",
        )
        checkpoint_states[checkpoint_path.name] = state

    training_data, _validation_data, test_data = load_and_split_mnist(config, download=False)
    client_shards, shard_records = build_client_shards(training_data, config)
    require(shard_records == clients, "saved client partition evidence differs from reconstructed MNIST partition")
    target_data = client_shards[config["target_client"]]
    non_target_data = [
        shard for client_id, shard in client_shards.items()
        if client_id != config["target_client"]
    ]
    test_loader = DataLoader(
        test_data, batch_size=config["evaluation_batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    target_loader = DataLoader(
        target_data, batch_size=config["evaluation_batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    non_target_loaders = [
        DataLoader(
            shard, batch_size=config["evaluation_batch_size"], shuffle=False,
            num_workers=config["num_workers"],
        )
        for shard in non_target_data
    ]
    for model_name, checkpoint_name in (
        ("original_full_training", "original_full_training.pt"),
        ("full_retraining_without_target", "full_retraining_without_target.pt"),
    ):
        model = SimpleMLP(hidden_units=config["hidden_units"])
        model.load_state_dict(checkpoint_states[checkpoint_name], strict=True)
        recomputed = evaluate_three_way(
            model, test_loader, target_loader, non_target_loaders,
            nn.CrossEntropyLoss(), torch.device("cpu"),
        ).metrics()
        for split, fields in metrics[model_name]["metrics"].items():
            require(fields["examples"] == recomputed[split]["examples"], f"{model_name} {split} count differs from checkpoint evaluation")
            for field in ("loss", "accuracy", "f1_macro"):
                require(
                    math.isclose(fields[field], recomputed[split][field], rel_tol=0.0, abs_tol=1e-8),
                    f"{model_name} {split} {field} does not match checkpoint evaluation",
                )

    for model_name in ("original_full_training", "full_retraining_without_target"):
        model = metrics[model_name]
        require(len(model["history"]) == config["training_epochs"], f"{model_name} history length mismatch")
        for epoch_record in model["history"]:
            require(epoch_record["epoch"] >= 1, f"{model_name} has invalid epoch number")
            for field in ("validation_loss", "validation_accuracy", "validation_f1_macro"):
                value = epoch_record[field]
                require(math.isfinite(value), f"{model_name} epoch history has non-finite {field}")
                require(0.0 <= value <= 1.0 if field != "validation_loss" else value >= 0.0, f"{model_name} epoch history {field} is out of range")
        for split in ("global_test", "target_client", "non_target_clients"):
            record = model["metrics"][split]
            require(record["examples"] > 0, f"{model_name} {split} is empty")
            expected_examples = {
                "global_test": config["expected_test_examples"],
                "target_client": deleted,
                "non_target_clients": after,
            }[split]
            require(record["examples"] == expected_examples, f"{model_name} {split} example count mismatch")
            for field in ("accuracy", "f1_macro", "loss"):
                require(math.isfinite(record[field]), f"{model_name} {split} has non-finite {field}")
                require(0.0 <= record[field] <= 1.0 if field != "loss" else record[field] >= 0.0, f"{model_name} {split} {field} is out of range")

    timings = metrics["timing"]["phase_seconds"]
    for phase in ("original_full_training", "retraining_without_target"):
        require(math.isfinite(timings[phase]) and timings[phase] > 0.0, f"invalid timing for {phase}")
        require(metrics["timing"]["phase_counts"][phase] == config["training_epochs"], f"unexpected timer count for {phase}")
    observed_difference = (
        metrics["full_retraining_without_target"]["metrics"]["global_test"]["accuracy"]
        - metrics["original_full_training"]["metrics"]["global_test"]["accuracy"]
    )
    require(
        math.isclose(metrics["accuracy_difference_retrained_minus_original"], observed_difference, abs_tol=1e-12),
        "saved accuracy difference does not match the two measured accuracies",
    )
    require(
        math.isclose(metrics["accuracy"], metrics["full_retraining_without_target"]["metrics"]["global_test"]["accuracy"], abs_tol=1e-12),
        "top-level accuracy does not match retrained global test accuracy",
    )
    require(
        math.isclose(metrics["f1"], metrics["full_retraining_without_target"]["metrics"]["global_test"]["f1_macro"], abs_tol=1e-12),
        "top-level F1 does not match retrained global test F1",
    )
    require(metrics["communication_cost"] == 0, "centralized baseline must not claim network communication")
    require(
        math.isclose(metrics["unlearning_time"], timings["retraining_without_target"], rel_tol=0.0, abs_tol=1e-12),
        "unlearning_time does not equal the retraining phase duration",
    )
    require(report_path.read_text(encoding="utf-8") == render_report(metrics), "Week 13 report is stale relative to metrics")
    print("PASS: Week 13 config, clean Git provenance, partitions, metrics, checkpoints, timings, and report are consistent.")


if __name__ == "__main__":
    main()
