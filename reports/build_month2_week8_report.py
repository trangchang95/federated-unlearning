"""Build and verify the Month 2 Week 8 comparison report.

The report is derived only from the canonical saved ``metrics.json``.  No
experiment result is copied into this source file by hand.

Run from the repository root after the Week 8 experiment finishes:

    conda run -n mse-ai python reports/build_month2_week8_report.py

Verify that the committed report still matches the saved metrics:

    conda run -n mse-ai python reports/build_month2_week8_report.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "month2_week8_mnist_iid_comparison"
    / "metrics.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "month2_week8_centralized_vs_fedavg.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Canonical Week 8 metrics JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Generated Markdown report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; fail unless the existing report matches exactly.",
    )
    return parser.parse_args()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array.")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _fraction(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return result


def _same_number(observed: Any, expected: float, label: str) -> None:
    value = _number(observed, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} is {value}, expected {expected}.")


def _client_mapping(value: Any, label: str) -> dict[int, Any]:
    source = _mapping(value, label)
    result: dict[int, Any] = {}
    for raw_client_id, item in source.items():
        try:
            client_id = int(raw_client_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} has a non-integer client ID.") from error
        if str(client_id) != str(raw_client_id) or client_id < 0:
            raise ValueError(f"{label} has an invalid client ID: {raw_client_id!r}.")
        if client_id in result:
            raise ValueError(f"{label} repeats client {client_id}.")
        result[client_id] = item
    return result


def _relative_to_project(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return None


def _source_link(path: Path) -> str:
    relative = _relative_to_project(path)
    if relative is None:
        return f"`{path.name}`"
    return f"[`{relative}`](../{relative})"


def _recorded_path_link(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("source_config must be a non-empty path string.")
    normalized = Path(raw_path).as_posix()
    return f"[`{normalized}`](../{normalized})"


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _signed_percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} percentage points"


def _format_seed_list(metrics: Mapping[str, Any]) -> str:
    seed_fields = (
        "random_seed",
        "split_seed",
        "train_partition_seed",
        "test_partition_seed",
        "centralized_loader_seed",
        "model_initialization_seed",
    )
    entries: list[str] = []
    for field in seed_fields:
        if field in metrics:
            entries.append(f"`{field}={_integer(metrics[field], field)}`")
    return ", ".join(entries)


def _validate_evaluation_record(
    record: Mapping[str, Any], label: str
) -> tuple[float, float, float]:
    loss = _number(record.get("test_loss"), f"{label}.test_loss", minimum=0.0)
    accuracy = _fraction(record.get("accuracy"), f"{label}.accuracy")
    f1_macro = _fraction(record.get("f1_macro"), f"{label}.f1_macro")
    _fraction(
        record.get("best_validation_accuracy"),
        f"{label}.best_validation_accuracy",
    )
    _number(
        record.get("training_and_validation_time_seconds"),
        f"{label}.training_and_validation_time_seconds",
        minimum=0.0,
    )
    return loss, accuracy, f1_macro


def _validate_centralized_history(
    metrics: Mapping[str, Any], central: Mapping[str, Any]
) -> dict[str, Any]:
    history = _sequence(central.get("history"), "centralized.history")
    epochs = _integer(metrics.get("centralized_epochs"), "centralized_epochs", minimum=1)
    if len(history) != epochs + 1:
        raise ValueError("centralized.history must contain epoch 0 plus every epoch.")

    validated: list[Mapping[str, Any]] = []
    summed_steps = 0
    selected_checkpoint_count = 0
    for expected_epoch, raw_row in enumerate(history):
        row = _mapping(raw_row, f"centralized.history[{expected_epoch}]")
        epoch = _integer(row.get("epoch"), f"centralized.history[{expected_epoch}].epoch")
        if epoch != expected_epoch:
            raise ValueError("centralized.history epoch numbers are not consecutive.")
        _number(
            row.get("validation_loss"),
            f"centralized.history[{expected_epoch}].validation_loss",
            minimum=0.0,
        )
        _fraction(
            row.get("validation_accuracy"),
            f"centralized.history[{expected_epoch}].validation_accuracy",
        )
        steps = _integer(
            row.get("optimizer_steps"),
            f"centralized.history[{expected_epoch}].optimizer_steps",
        )
        if expected_epoch == 0 and (steps != 0 or row.get("train_loss") is not None):
            raise ValueError("Centralized epoch 0 must be an untrained evaluation.")
        selected_as_best = row.get("selected_as_best_checkpoint")
        if not isinstance(selected_as_best, bool):
            raise ValueError("Every centralized history row needs a Boolean checkpoint flag.")
        if expected_epoch == 0 and selected_as_best:
            raise ValueError("The untrained centralized checkpoint cannot be selected.")
        selected_checkpoint_count += int(selected_as_best)
        if expected_epoch > 0:
            _number(
                row.get("train_loss"),
                f"centralized.history[{expected_epoch}].train_loss",
                minimum=0.0,
            )
            summed_steps += steps
        validated.append(row)

    training_examples = _integer(
        metrics.get("training_examples"), "training_examples", minimum=1
    )
    batch_size = _integer(metrics.get("batch_size"), "batch_size", minimum=1)
    steps_per_epoch = math.ceil(training_examples / batch_size)
    expected_steps = steps_per_epoch * epochs
    recorded_steps = _integer(
        central.get("optimizer_steps"), "centralized.optimizer_steps", minimum=1
    )
    if summed_steps != recorded_steps or recorded_steps != expected_steps:
        raise ValueError(
            "Centralized optimizer steps do not match "
            "epochs × ceil(training examples / batch size)."
        )
    expected_processed = training_examples * epochs
    if _integer(
        central.get("training_examples_processed"),
        "centralized.training_examples_processed",
        minimum=1,
    ) != expected_processed:
        raise ValueError("Centralized processed-example count is inconsistent.")

    selected_rows = [
        row for row in validated[1:] if row["selected_as_best_checkpoint"]
    ]
    if selected_checkpoint_count != 1 or len(selected_rows) != 1:
        raise ValueError("Exactly one trained centralized row must be selected.")
    best_row = selected_rows[0]
    maximum_recorded_accuracy = max(
        float(row["validation_accuracy"]) for row in validated[1:]
    )
    if (
        maximum_recorded_accuracy - float(best_row["validation_accuracy"])
        > 1e-6 + 1e-12
    ):
        raise ValueError(
            "A centralized row exceeds the selected checkpoint by more than "
            "six-decimal logging tolerance."
        )
    best_epoch = _integer(central.get("best_epoch"), "centralized.best_epoch", minimum=1)
    if best_epoch != best_row["epoch"]:
        raise ValueError("centralized.best_epoch does not match its validation history.")
    _same_number(
        central.get("best_validation_accuracy"),
        float(best_row["validation_accuracy"]),
        "centralized.best_validation_accuracy",
    )
    return {
        "history": validated,
        "steps_per_epoch": steps_per_epoch,
        "expected_steps": expected_steps,
        "best_row": best_row,
    }


def _validate_fedavg_history(
    metrics: Mapping[str, Any], fedavg: Mapping[str, Any]
) -> dict[str, Any]:
    history = _sequence(fedavg.get("history"), "fedavg.history")
    rounds = _integer(metrics.get("number_of_rounds"), "number_of_rounds", minimum=1)
    local_epochs = _integer(metrics.get("local_epochs"), "local_epochs", minimum=1)
    batch_size = _integer(metrics.get("batch_size"), "batch_size", minimum=1)
    payload_bytes = _integer(
        metrics.get("model_payload_bytes"), "model_payload_bytes", minimum=1
    )
    if len(history) != rounds + 1:
        raise ValueError("fedavg.history must contain round 0 plus every round.")

    validated: list[Mapping[str, Any]] = []
    expected_steps = 0
    expected_processed = 0
    expected_communication = 0
    client_round_participations = 0
    selected_checkpoint_count = 0
    for expected_round, raw_row in enumerate(history):
        row = _mapping(raw_row, f"fedavg.history[{expected_round}]")
        round_number = _integer(
            row.get("round"), f"fedavg.history[{expected_round}].round"
        )
        if round_number != expected_round:
            raise ValueError("fedavg.history round numbers are not consecutive.")
        _number(
            row.get("validation_loss"),
            f"fedavg.history[{expected_round}].validation_loss",
            minimum=0.0,
        )
        _fraction(
            row.get("validation_accuracy"),
            f"fedavg.history[{expected_round}].validation_accuracy",
        )
        selected = _sequence(
            row.get("selected_client_ids"),
            f"fedavg.history[{expected_round}].selected_client_ids",
        )
        if expected_round == 0:
            if row.get("selected_as_best_checkpoint") is not False:
                raise ValueError("The untrained FedAvg checkpoint cannot be selected.")
            if selected or _integer(
                row.get("communication_cost_bytes"),
                "fedavg.history[0].communication_cost_bytes",
            ) != 0:
                raise ValueError("FedAvg round 0 must be an untrained evaluation.")
            validated.append(row)
            continue

        selected_as_best = row.get("selected_as_best_checkpoint")
        if not isinstance(selected_as_best, bool):
            raise ValueError("Every FedAvg history row needs a Boolean checkpoint flag.")
        selected_checkpoint_count += int(selected_as_best)

        selected_ids = [
            _integer(item, f"round {expected_round} selected client", minimum=0)
            for item in selected
        ]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"FedAvg round {expected_round} repeats a client.")
        example_counts = _client_mapping(
            row.get("client_example_counts"),
            f"fedavg.history[{expected_round}].client_example_counts",
        )
        optimizer_steps = _client_mapping(
            row.get("client_optimizer_steps"),
            f"fedavg.history[{expected_round}].client_optimizer_steps",
        )
        if set(selected_ids) != set(example_counts) or set(selected_ids) != set(
            optimizer_steps
        ):
            raise ValueError(
                f"FedAvg round {expected_round} client IDs disagree across fields."
            )
        round_examples = 0
        round_steps = 0
        for client_id in selected_ids:
            examples = _integer(
                example_counts[client_id],
                f"round {expected_round} client {client_id} examples",
                minimum=1,
            )
            steps = _integer(
                optimizer_steps[client_id],
                f"round {expected_round} client {client_id} optimizer steps",
                minimum=1,
            )
            calculated_steps = math.ceil(examples / batch_size) * local_epochs
            if steps != calculated_steps:
                raise ValueError(
                    f"Round {expected_round} client {client_id} optimizer steps "
                    "do not match ceil(examples / B) × E."
                )
            round_examples += examples
            round_steps += steps

        if _integer(
            row.get("total_selected_examples"),
            f"fedavg.history[{expected_round}].total_selected_examples",
            minimum=1,
        ) != round_examples:
            raise ValueError(f"FedAvg round {expected_round} example total disagrees.")
        row_payload = _integer(
            row.get("model_payload_bytes"),
            f"fedavg.history[{expected_round}].model_payload_bytes",
            minimum=1,
        )
        if row_payload != payload_bytes:
            raise ValueError(f"FedAvg round {expected_round} payload size changed.")
        round_communication = payload_bytes * 2 * len(selected_ids)
        if _integer(
            row.get("download_bytes"),
            f"fedavg.history[{expected_round}].download_bytes",
        ) != payload_bytes * len(selected_ids):
            raise ValueError(f"FedAvg round {expected_round} download bytes disagree.")
        if _integer(
            row.get("upload_bytes"),
            f"fedavg.history[{expected_round}].upload_bytes",
        ) != payload_bytes * len(selected_ids):
            raise ValueError(f"FedAvg round {expected_round} upload bytes disagree.")
        if _integer(
            row.get("communication_cost_bytes"),
            f"fedavg.history[{expected_round}].communication_cost_bytes",
        ) != round_communication:
            raise ValueError(f"FedAvg round {expected_round} communication disagrees.")

        expected_steps += round_steps
        expected_processed += round_examples * local_epochs
        expected_communication += round_communication
        client_round_participations += len(selected_ids)
        validated.append(row)

    if _integer(
        fedavg.get("optimizer_steps"), "fedavg.optimizer_steps", minimum=1
    ) != expected_steps:
        raise ValueError("FedAvg optimizer-step total disagrees with its round history.")
    if _integer(
        fedavg.get("training_examples_processed"),
        "fedavg.training_examples_processed",
        minimum=1,
    ) != expected_processed:
        raise ValueError("FedAvg processed-example total disagrees with its history.")
    if _integer(
        fedavg.get("communication_cost_bytes"),
        "fedavg.communication_cost_bytes",
        minimum=1,
    ) != expected_communication:
        raise ValueError("FedAvg communication total disagrees with its history.")

    selected_rows = [
        row for row in validated[1:] if row["selected_as_best_checkpoint"]
    ]
    if selected_checkpoint_count != 1 or len(selected_rows) != 1:
        raise ValueError("Exactly one trained FedAvg row must be selected.")
    best_row = selected_rows[0]
    maximum_recorded_accuracy = max(
        float(row["validation_accuracy"]) for row in validated[1:]
    )
    if (
        maximum_recorded_accuracy - float(best_row["validation_accuracy"])
        > 1e-6 + 1e-12
    ):
        raise ValueError(
            "A FedAvg row exceeds the selected checkpoint by more than "
            "six-decimal logging tolerance."
        )
    best_round = _integer(fedavg.get("best_round"), "fedavg.best_round", minimum=1)
    if best_round != best_row["round"]:
        raise ValueError("fedavg.best_round does not match its validation history.")
    _same_number(
        fedavg.get("best_validation_accuracy"),
        float(best_row["validation_accuracy"]),
        "fedavg.best_validation_accuracy",
    )
    return {
        "history": validated,
        "expected_steps": expected_steps,
        "expected_processed": expected_processed,
        "expected_communication": expected_communication,
        "client_round_participations": client_round_participations,
        "best_row": best_row,
    }


def _validate_per_client_utility(
    central: Mapping[str, Any], fedavg: Mapping[str, Any], number_of_clients: int
) -> list[dict[str, Any]]:
    method_records: list[dict[int, Mapping[str, Any]]] = []
    for method_name, method in (("centralized", central), ("fedavg", fedavg)):
        rows = _sequence(
            method.get("per_client_test_utility"),
            f"{method_name}.per_client_test_utility",
        )
        by_client: dict[int, Mapping[str, Any]] = {}
        for row_index, raw_row in enumerate(rows):
            row = _mapping(raw_row, f"{method_name} client utility row {row_index}")
            client_id = _integer(
                row.get("client_id"), f"{method_name} client_id", minimum=0
            )
            if client_id in by_client:
                raise ValueError(f"{method_name} repeats client {client_id}.")
            _integer(
                row.get("number_of_examples"),
                f"{method_name} client {client_id} examples",
                minimum=1,
            )
            _fraction(row.get("accuracy"), f"{method_name} client {client_id} accuracy")
            _fraction(row.get("f1_macro"), f"{method_name} client {client_id} macro F1")
            by_client[client_id] = row
        method_records.append(by_client)

    central_by_client, fedavg_by_client = method_records
    expected_ids = set(range(number_of_clients))
    if set(central_by_client) != expected_ids or set(fedavg_by_client) != expected_ids:
        raise ValueError("Per-client utility must contain every configured client once.")

    combined: list[dict[str, Any]] = []
    for client_id in sorted(expected_ids):
        central_row = central_by_client[client_id]
        fedavg_row = fedavg_by_client[client_id]
        central_examples = _integer(
            central_row["number_of_examples"], "centralized client examples", minimum=1
        )
        fedavg_examples = _integer(
            fedavg_row["number_of_examples"], "FedAvg client examples", minimum=1
        )
        if central_examples != fedavg_examples:
            raise ValueError(f"Client {client_id} test sizes differ between methods.")
        central_accuracy = float(central_row["accuracy"])
        fedavg_accuracy = float(fedavg_row["accuracy"])
        combined.append(
            {
                "client_id": client_id,
                "examples": central_examples,
                "central_accuracy": central_accuracy,
                "fedavg_accuracy": fedavg_accuracy,
                "accuracy_difference": fedavg_accuracy - central_accuracy,
                "central_f1": float(central_row["f1_macro"]),
                "fedavg_f1": float(fedavg_row["f1_macro"]),
            }
        )
    return combined


def validate_and_derive(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Reject inconsistent evidence and return arithmetic used by the report."""
    if metrics.get("dataset") != "MNIST":
        raise ValueError("The Week 8 report supports the MNIST comparison only.")
    if metrics.get("algorithm") != "handwritten_fedavg_iid_vs_centralized_sgd":
        raise ValueError("The metrics do not identify the Week 8 comparison algorithm.")
    if metrics.get("partition_strategy") != "iid_seeded_equal_size":
        raise ValueError("The Week 8 report must not mislabel a non-IID partition as IID.")
    if metrics.get("alpha") is not None or metrics.get("target_client") is not None:
        raise ValueError("Week 8 is IID FL without unlearning or a target client.")
    if metrics.get("primary_method_for_mandatory_metrics") != "handwritten_fedavg":
        raise ValueError(
            "primary_method_for_mandatory_metrics must identify hand-written FedAvg."
        )

    number_of_clients = _integer(
        metrics.get("number_of_clients"), "number_of_clients", minimum=1
    )
    client_fraction = _number(metrics.get("client_fraction"), "client_fraction")
    if not 0.0 < client_fraction <= 1.0:
        raise ValueError("client_fraction must be in (0, 1].")
    if client_fraction != 1.0:
        raise ValueError("The canonical matched Week 8 report requires C=1.0.")
    local_epochs = _integer(metrics.get("local_epochs"), "local_epochs", minimum=1)
    batch_size = _integer(metrics.get("batch_size"), "batch_size", minimum=1)
    rounds = _integer(metrics.get("number_of_rounds"), "number_of_rounds", minimum=1)
    centralized_epochs = _integer(
        metrics.get("centralized_epochs"), "centralized_epochs", minimum=1
    )
    learning_rate = _number(
        metrics.get("learning_rate"), "learning_rate", minimum=0.0
    )
    if learning_rate == 0.0:
        raise ValueError("learning_rate must be positive.")

    central = _mapping(metrics.get("centralized"), "centralized")
    fedavg = _mapping(metrics.get("fedavg"), "fedavg")
    central_loss, central_accuracy, central_f1 = _validate_evaluation_record(
        central, "centralized"
    )
    fedavg_loss, fedavg_accuracy, fedavg_f1 = _validate_evaluation_record(
        fedavg, "fedavg"
    )
    if _integer(
        central.get("communication_cost_bytes"),
        "centralized.communication_cost_bytes",
    ) != 0:
        raise ValueError("The centralized reference must record zero model transfer.")
    _same_number(metrics.get("unlearning_time"), 0.0, "unlearning_time")
    central_history = _validate_centralized_history(metrics, central)
    fedavg_history = _validate_fedavg_history(metrics, fedavg)

    _same_number(metrics.get("accuracy"), fedavg_accuracy, "top-level accuracy")
    _same_number(metrics.get("f1"), fedavg_f1, "top-level f1")
    communication = fedavg_history["expected_communication"]
    if _integer(
        metrics.get("communication_cost"), "communication_cost", minimum=1
    ) != communication:
        raise ValueError("Top-level communication_cost disagrees with FedAvg.")
    if metrics.get("communication_cost_unit") != "bytes":
        raise ValueError("Week 8 communication must be recorded in bytes.")

    accuracy_difference = fedavg_accuracy - central_accuracy
    f1_difference = fedavg_f1 - central_f1
    _same_number(
        metrics.get("accuracy_difference_fedavg_minus_centralized"),
        accuracy_difference,
        "accuracy difference",
    )
    _same_number(
        metrics.get("f1_difference_fedavg_minus_centralized"),
        f1_difference,
        "F1 difference",
    )

    protocol = _mapping(metrics.get("comparison_protocol"), "comparison_protocol")
    for flag in (
        "same_initial_model",
        "same_training_split",
        "test_evaluated_after_validation_selection_only",
    ):
        if protocol.get(flag) is not True:
            raise ValueError(f"comparison_protocol.{flag} must be true.")
    if protocol.get("same_optimizer") != metrics.get("optimizer"):
        raise ValueError("The comparison protocol optimizer disagrees with the config.")
    if protocol.get("optimizer_updates_identical") is not False:
        raise ValueError("The report must not label the two update trajectories identical.")
    if not isinstance(protocol.get("optimizer_update_difference"), str) or not protocol[
        "optimizer_update_difference"
    ].strip():
        raise ValueError("The optimizer-update difference needs a saved explanation.")
    _same_number(
        protocol.get("same_learning_rate"), learning_rate, "protocol learning rate"
    )
    if _integer(protocol.get("same_batch_size"), "protocol batch size", minimum=1) != batch_size:
        raise ValueError("The comparison protocol batch size disagrees with the config.")
    if _integer(
        protocol.get("centralized_full_data_passes"),
        "centralized full-data passes",
        minimum=1,
    ) != centralized_epochs:
        raise ValueError("The centralized full-data-pass count is inconsistent.")
    fed_full_passes = rounds * local_epochs
    if _integer(
        protocol.get("fedavg_full_data_passes_when_full_participation"),
        "FedAvg full-data passes",
        minimum=1,
    ) != fed_full_passes:
        raise ValueError("The FedAvg full-data-pass count is inconsistent.")
    if centralized_epochs != fed_full_passes:
        raise ValueError("Week 8 requires matched centralized and FedAvg data passes.")

    checksums = (
        metrics.get("initial_model_checksum_sha256"),
        metrics.get("centralized_initial_checksum_sha256"),
        metrics.get("fedavg_initial_checksum_sha256"),
    )
    if any(not isinstance(item, str) or len(item) != 64 for item in checksums):
        raise ValueError("Initial model checksums must be 64-character SHA-256 strings.")
    if len(set(checksums)) != 1:
        raise ValueError("Centralized and FedAvg did not record the same initialization.")

    provenance = _mapping(metrics.get("code_provenance"), "code_provenance")
    configured_revision = provenance.get("configured_revision")
    if configured_revision != metrics.get("code_revision"):
        raise ValueError("Code provenance disagrees with the configured revision.")
    resolved_commit = provenance.get("resolved_commit")
    head_commit = provenance.get("head_commit")
    if (
        not isinstance(resolved_commit, str)
        or len(resolved_commit) != 40
        or resolved_commit != head_commit
    ):
        raise ValueError("Code provenance must resolve the configured revision to HEAD.")
    if provenance.get("worktree_clean_at_start") is not True:
        raise ValueError("The canonical experiment must start from a clean worktree.")
    tracked_dependencies = _sequence(
        provenance.get("tracked_dependencies"), "code_provenance.tracked_dependencies"
    )
    required_dependencies = {
        Path(str(metrics.get("source_config"))).as_posix(),
        Path(str(metrics.get("runner"))).as_posix(),
        Path(str(metrics.get("environment_manifest"))).as_posix(),
    }
    recorded_dependencies = {Path(str(item)).as_posix() for item in tracked_dependencies}
    if not required_dependencies.issubset(recorded_dependencies):
        raise ValueError("Code provenance omits a required tracked dependency.")

    training_examples = _integer(
        metrics.get("training_examples"), "training_examples", minimum=1
    )
    validation_examples = _integer(
        metrics.get("validation_examples"), "validation_examples", minimum=1
    )
    test_examples = _integer(metrics.get("test_examples"), "test_examples", minimum=1)
    if fedavg_history["expected_processed"] != training_examples * fed_full_passes:
        raise ValueError("FedAvg did not process the matched number of training examples.")
    if _integer(
        central.get("training_examples_processed"),
        "centralized.training_examples_processed",
        minimum=1,
    ) != fedavg_history["expected_processed"]:
        raise ValueError("The two methods did not process the same number of examples.")

    per_client = _validate_per_client_utility(central, fedavg, number_of_clients)
    if sum(row["examples"] for row in per_client) != test_examples:
        raise ValueError("Per-client test subsets do not cover the full test set.")

    expected_selected_each_round = max(1, math.ceil(client_fraction * number_of_clients))
    selected_counts = [
        len(row["selected_client_ids"])
        for row in fedavg_history["history"][1:]
    ]
    if any(count != expected_selected_each_round for count in selected_counts):
        raise ValueError("Selected-client counts disagree with ceil(C × K).")

    return {
        "central": central,
        "fedavg": fedavg,
        "central_loss": central_loss,
        "central_accuracy": central_accuracy,
        "central_f1": central_f1,
        "fedavg_loss": fedavg_loss,
        "fedavg_accuracy": fedavg_accuracy,
        "fedavg_f1": fedavg_f1,
        "accuracy_difference": accuracy_difference,
        "f1_difference": f1_difference,
        "central_history": central_history,
        "fedavg_history": fedavg_history,
        "per_client": per_client,
        "number_of_clients": number_of_clients,
        "client_fraction": client_fraction,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "rounds": rounds,
        "centralized_epochs": centralized_epochs,
        "learning_rate": learning_rate,
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "test_examples": test_examples,
        "expected_selected_each_round": expected_selected_each_round,
        "communication": communication,
        "communication_mib": communication / (1024**2),
        "payload_bytes": _integer(
            metrics.get("model_payload_bytes"), "model_payload_bytes", minimum=1
        ),
        "initial_checksum": checksums[0],
        "protocol": protocol,
        "provenance": provenance,
    }


def _result_interpretation(derived: Mapping[str, Any]) -> str:
    accuracy_difference = derived["accuracy_difference"]
    f1_difference = derived["f1_difference"]
    if accuracy_difference < 0:
        direction = (
            "FedAvg underperformed the centralized reference by "
            f"**{abs(accuracy_difference) * 100:.2f} percentage points** in test "
            "accuracy"
        )
    elif accuracy_difference > 0:
        direction = (
            "FedAvg exceeded the centralized reference by "
            f"**{accuracy_difference * 100:.2f} percentage points** in test accuracy"
        )
    else:
        direction = "FedAvg tied the centralized reference in test accuracy"

    return (
        f"{direction}; its macro-F1 difference was "
        f"**{_signed_percentage_points(f1_difference)}**. This is the honest outcome "
        "of this saved run, not evidence that one method is generally superior. "
        "The two optimization procedures are mechanically different, and only one "
        "fixed-seed run is available."
    )


def _convergence_interpretation(derived: Mapping[str, Any]) -> str:
    central_history = derived["central_history"]["history"]
    fedavg_history = derived["fedavg_history"]["history"]
    central_gain = (
        derived["central_history"]["best_row"]["validation_accuracy"]
        - central_history[0]["validation_accuracy"]
    )
    fedavg_gain = (
        derived["fedavg_history"]["best_row"]["validation_accuracy"]
        - fedavg_history[0]["validation_accuracy"]
    )
    if central_gain > 0 and fedavg_gain > 0:
        opening = "Both methods improved from their common untrained starting point"
    else:
        opening = "At least one method did not improve above the untrained starting point"
    return (
        f"{opening}. The best validation gains were "
        f"{_signed_percentage_points(central_gain)} for centralized SGD and "
        f"{_signed_percentage_points(fedavg_gain)} for FedAvg. This supports a narrow "
        f"claim that training progressed; {derived['centralized_epochs']} data passes "
        "are not enough to claim "
        "mathematical convergence or a stable asymptote."
    )


def render_report(
    metrics: Mapping[str, Any],
    *,
    metrics_path: Path,
    metrics_sha256: str,
    resolved_config_path: Path,
    resolved_config_sha256: str,
) -> str:
    """Return deterministic Markdown after validating every reported total."""
    derived = validate_and_derive(metrics)
    central = derived["central"]
    fedavg = derived["fedavg"]
    central_history = derived["central_history"]["history"]
    fedavg_history = derived["fedavg_history"]["history"]
    central_best = derived["central_history"]["best_row"]
    fedavg_best = derived["fedavg_history"]["best_row"]

    per_client_rows = "\n".join(
        "| {client_id} | {examples:,} | {central} | {fedavg} | {difference} | "
        "{central_f1} | {fedavg_f1} |".format(
            client_id=row["client_id"],
            examples=row["examples"],
            central=_percent(row["central_accuracy"]),
            fedavg=_percent(row["fedavg_accuracy"]),
            difference=_signed_percentage_points(row["accuracy_difference"]),
            central_f1=_percent(row["central_f1"]),
            fedavg_f1=_percent(row["fedavg_f1"]),
        )
        for row in derived["per_client"]
    )
    central_client_accuracies = [
        row["central_accuracy"] for row in derived["per_client"]
    ]
    fedavg_client_accuracies = [row["fedavg_accuracy"] for row in derived["per_client"]]

    central_by_pass = {int(row["epoch"]): row for row in central_history}
    fedavg_by_pass = {
        int(row["round"]) * derived["local_epochs"]: row for row in fedavg_history
    }
    convergence_rows: list[str] = []
    for pass_index in sorted(set(central_by_pass) | set(fedavg_by_pass)):
        central_row = central_by_pass.get(pass_index)
        fedavg_row = fedavg_by_pass.get(pass_index)
        central_accuracy = (
            _percent(float(central_row["validation_accuracy"]))
            if central_row is not None
            else "—"
        )
        central_loss = (
            f"{float(central_row['validation_loss']):.6f}"
            if central_row is not None
            else "—"
        )
        fedavg_accuracy = (
            _percent(float(fedavg_row["validation_accuracy"]))
            if fedavg_row is not None
            else "—"
        )
        fedavg_loss = (
            f"{float(fedavg_row['validation_loss']):.6f}"
            if fedavg_row is not None
            else "—"
        )
        convergence_rows.append(
            "| {pass_index} | {central_acc} | {central_loss} | "
            "{fed_acc} | {fed_loss} |".format(
                pass_index=pass_index,
                central_acc=central_accuracy,
                central_loss=central_loss,
                fed_acc=fedavg_accuracy,
                fed_loss=fedavg_loss,
            )
        )

    source_config_link = _recorded_path_link(metrics.get("source_config"))
    metrics_link = _source_link(metrics_path)
    resolved_config_link = _source_link(resolved_config_path)
    metrics_relative = _relative_to_project(metrics_path)
    if metrics_relative is not None:
        result_directory = Path(metrics_relative).parent.as_posix()
        convergence_link = f"[convergence plot](../{result_directory}/convergence_comparison.png)"
        central_confusion_link = (
            f"[centralized confusion matrix](../{result_directory}/centralized_confusion_matrix.png)"
        )
        fedavg_confusion_link = (
            f"[FedAvg confusion matrix](../{result_directory}/fedavg_confusion_matrix.png)"
        )
    else:
        convergence_link = "`convergence_comparison.png`"
        central_confusion_link = "`centralized_confusion_matrix.png`"
        fedavg_confusion_link = "`fedavg_confusion_matrix.png`"

    selected_total = derived["fedavg_history"]["client_round_participations"]
    timing_scope = str(derived["protocol"].get("timing_scope", "not recorded"))
    seed_list = _format_seed_list(metrics)
    code_revision = str(metrics.get("code_revision", "not recorded"))
    resolved_commit = str(derived["provenance"]["resolved_commit"])
    environment_manifest = str(metrics.get("environment_manifest", "not recorded"))
    optimizer = str(metrics.get("optimizer", "not recorded"))
    partition_strategy = str(metrics.get("partition_strategy"))

    return f"""# Month 2, Week 8 — Centralized SGD vs Hand-Written FedAvg

> Generated deterministically from {metrics_link}. Do not copy result numbers into this report by hand.

## How to read the evidence

- **Saved-result fact:** a value stored by the config-driven experiment.
- **Derived fact:** arithmetic recomputed by this builder from saved values.
- **Interpretation:** a deliberately limited explanation, not a new measurement.
- **Limitation:** what this run cannot establish.

## 1. Run identity and integrity

**Saved-result facts**

| Item | Recorded value |
|---|---|
| Dataset | `{metrics['dataset']}` — {derived['training_examples']:,} train / {derived['validation_examples']:,} validation / {derived['test_examples']:,} test examples |
| Config | {source_config_link} |
| Resolved run config | {resolved_config_link} |
| Configured code revision | `{code_revision}` |
| Resolved Git commit / clean HEAD | `{resolved_commit}` / `True` |
| Environment manifest | [`{environment_manifest}`](../{environment_manifest}) |
| Resolved-config SHA-256 | `{resolved_config_sha256}` |
| Metrics SHA-256 | `{metrics_sha256}` |
| Common initial-model SHA-256 | `{derived['initial_checksum']}` |
| Partition | `{partition_strategy}` |
| Fixed seeds | {seed_list} |

**Derived integrity result:** PASS. Before rendering, the builder checked that every resolved input setting is unchanged in the metrics (only the declared `accuracy`, `f1`, and `communication_cost` output placeholders may be filled), the configured Git revision resolved to a clean HEAD, the mandatory top-level metrics belong to `{metrics['primary_method_for_mandatory_metrics']}`, the initialization was common, data-pass counts matched, exactly one validation checkpoint was selected per method, optimizer-step arithmetic was correct, per-client test data covered the test set, and every round's upload/download byte totals agreed. A mismatch makes the builder stop instead of producing a report.

## 2. Beginner protocol card: K, C, E, B, R

**Saved/derived facts**

| Symbol | Meaning | This run |
|---|---|---:|
| `K` | Total number of clients | {derived['number_of_clients']} |
| `C` | Fraction of clients selected each round | {derived['client_fraction']:.2f} ({derived['expected_selected_each_round']} of {derived['number_of_clients']} selected) |
| `E` | Local epochs completed by each selected client before upload | {derived['local_epochs']} |
| `B` | Local/central minibatch size | {derived['batch_size']} |
| `R` | FedAvg communication rounds | {derived['rounds']} |
| `η` | SGD learning rate | {derived['learning_rate']:.6g} |
| Central epochs | Full centralized passes through the training split | {derived['centralized_epochs']} |

With full participation (`C=1`), `R × E = {derived['rounds']} × {derived['local_epochs']} = {derived['rounds'] * derived['local_epochs']}` FedAvg passes over all client training examples. This matches the {derived['centralized_epochs']} centralized passes.

## 3. What is matched, and what is still different

**Saved/derived facts**

| Matched on purpose | Evidence |
|---|---|
| Data | Same MNIST training split; each path processed {central['training_examples_processed']:,} training-example exposures |
| Starting model | All three recorded initialization checksums equal `{derived['initial_checksum']}` |
| Optimization settings | `{optimizer}`, learning rate {derived['learning_rate']:.6g}, batch size {derived['batch_size']} |
| Training budget | {derived['centralized_epochs']} centralized full passes vs {derived['rounds'] * derived['local_epochs']} FedAvg full passes |
| Model selection | Each method's best validation checkpoint was selected before the test set was evaluated |

| Mechanically different | Why it matters |
|---|---|
| Centralized SGD | One model processes the complete shuffled training split and updates after every central minibatch. |
| FedAvg | {derived['number_of_clients']} IID client shards train separate local copies; the server performs a sample-weighted model average once per round. |
| Minibatch boundaries | Splitting {derived['training_examples']:,} examples across clients changes where partial final batches occur, even though total example exposures match. |
| Communication | Centralized simulation records 0 model-transfer bytes; FedAvg counts one download and one upload per selected client per round. |
| Data order and update trajectory | `{derived['protocol']['optimizer_update_difference']}` |

**Interpretation:** this is a controlled *matched-budget comparison*, not a claim that the two training algorithms are numerically identical.

## 4. Final test results

**Saved-result facts**

| Method | Test loss | Test accuracy | Test macro F1 | Best validation checkpoint | Train + validation-loop time |
|---|---:|---:|---:|---:|---:|
| Centralized SGD | {derived['central_loss']:.6f} | {_percent(derived['central_accuracy'])} | {_percent(derived['central_f1'])} | epoch {central['best_epoch']} ({_percent(float(central['best_validation_accuracy']))}) | {float(central['training_and_validation_time_seconds']):.3f} s |
| Hand-written FedAvg | {derived['fedavg_loss']:.6f} | {_percent(derived['fedavg_accuracy'])} | {_percent(derived['fedavg_f1'])} | round {fedavg['best_round']} ({_percent(float(fedavg['best_validation_accuracy']))}) | {float(fedavg['training_and_validation_time_seconds']):.3f} s |

**Derived differences:** FedAvg minus centralized is **{_signed_percentage_points(derived['accuracy_difference'])}** for accuracy and **{_signed_percentage_points(derived['f1_difference'])}** for macro F1.

**Interpretation — report the negative result honestly:** {_result_interpretation(derived)}

The saved visual evidence is the {convergence_link}, {central_confusion_link}, and {fedavg_confusion_link}.

## 5. Validation behavior and convergence

**Saved-result facts**

| Cumulative full-data-equivalent passes | Central validation accuracy | Central validation loss | FedAvg validation accuracy | FedAvg validation loss |
|---:|---:|---:|---:|---:|
{chr(10).join(convergence_rows)}

- Centralized best: epoch {central_best['epoch']}, validation accuracy {_percent(float(central_best['validation_accuracy']))}; final epoch accuracy {_percent(float(central_history[-1]['validation_accuracy']))}.
- FedAvg best: round {fedavg_best['round']} (pass {int(fedavg_best['round']) * derived['local_epochs']}), validation accuracy {_percent(float(fedavg_best['validation_accuracy']))}; final round accuracy {_percent(float(fedavg_history[-1]['validation_accuracy']))}.

**Interpretation:** {_convergence_interpretation(derived)}

## 6. Per-client test utility

These are IID test partitions used to expose whether the overall average hides a weak client. They are not non-IID results.

**Saved/derived facts**

| Client | Test examples | Central accuracy | FedAvg accuracy | FedAvg − central | Central macro F1 | FedAvg macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
{per_client_rows}

- Centralized client accuracy range: {_percent(min(central_client_accuracies))} to {_percent(max(central_client_accuracies))}.
- FedAvg client accuracy range: {_percent(min(fedavg_client_accuracies))} to {_percent(max(fedavg_client_accuracies))}.

**Interpretation:** client-level differences are descriptive checks for this one IID split. They do not estimate performance for a population of real clients and must not be called non-target-client preservation (no unlearning target exists in Week 8).

## 7. Why the optimizer-step totals differ

**Derived facts**

- Centralized: `ceil({derived['training_examples']:,} / {derived['batch_size']}) × {derived['centralized_epochs']} = {derived['central_history']['steps_per_epoch']} × {derived['centralized_epochs']} = {derived['central_history']['expected_steps']:,}` optimizer steps.
- FedAvg: sum `ceil(client examples / {derived['batch_size']}) × E` over every selected client and round = **{derived['fedavg_history']['expected_steps']:,}** optimizer steps.

Thus the saved totals are **{central['optimizer_steps']:,} centralized steps vs {fedavg['optimizer_steps']:,} FedAvg local steps**, even though both paths process {central['training_examples_processed']:,} example exposures. The difference comes from rounding each client's final partial minibatch separately; it is not extra training data.

## 8. Communication accounting

**Derived fact**

`{derived['payload_bytes']:,} payload bytes × 2 directions × {selected_total} client-round participations = {derived['communication']:,} bytes = {derived['communication_mib']:.3f} MiB`.

For this full-participation run, `{selected_total} = K × R = {derived['number_of_clients']} × {derived['rounds']}`. The two directions are one global-model download and one locally trained model upload. `E` does not multiply communication because all {derived['local_epochs']} local epoch(s) occur between those two transfers.

**Limitation:** this is a dense tensor-payload estimate. It excludes protocol headers, serialization overhead, retries, latency, compression, and network contention. Timing was measured as `{timing_scope}`, so the CPU seconds above are not a real distributed-system speed benchmark.

## 9. Conclusion and limits

**Interpretation:** the saved run is sufficient to check that the hand-written multi-client FedAvg workflow trains, logs its mechanics, and can be compared with a deliberately matched centralized reference. {_result_interpretation(derived)}

**Limitations**

- This is **one fixed-seed run**. It provides no across-seed variance, confidence interval, or statistical-equivalence test.
- The IID partition is an introductory Month 2 setting; it says nothing yet about Non-IID client drift.
- The clients run sequentially in one CPU process. This validates learning and accounting logic, not deployment throughput, privacy, or network behavior.
- The result establishes neither Federated Unlearning nor forgetting; those remain later-gate work.

For exact reruns, use the saved and resolved configs, code revision `{code_revision}` (commit `{resolved_commit}`), and environment manifest `{environment_manifest}`. The recorded seed set is {seed_list}.
"""


def load_and_hash_metrics(metrics_path: Path) -> tuple[Mapping[str, Any], str]:
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Week 8 metrics not found: {metrics_path}. Run the saved config first."
        )
    raw_bytes = metrics_path.read_bytes()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {metrics_path}: {error}") from error
    return _mapping(parsed, "metrics root"), hashlib.sha256(raw_bytes).hexdigest()


def load_and_validate_resolved_config(
    metrics_path: Path, metrics: Mapping[str, Any]
) -> tuple[Path, str]:
    """Require the pre-training resolved config and match it to logged metrics."""
    resolved_path = metrics_path.parent / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Resolved Week 8 config not found beside metrics: {resolved_path}"
        )
    raw_bytes = resolved_path.read_bytes()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {resolved_path}: {error}") from error
    resolved = _mapping(parsed, "resolved config root")
    if not resolved:
        raise ValueError("resolved_config.json must not be empty.")
    output_placeholders = {
        "accuracy": None,
        "f1": None,
        "communication_cost": 0,
    }
    for field, value in resolved.items():
        if field not in metrics:
            raise ValueError(f"Resolved-config field {field!r} is absent from metrics.")
        if field in output_placeholders:
            if value != output_placeholders[field]:
                raise ValueError(
                    f"Resolved-config output placeholder {field!r} is invalid."
                )
            continue
        if metrics[field] != value:
            raise ValueError(
                f"Resolved-config field {field!r} changed between preflight and metrics."
            )
    return resolved_path, hashlib.sha256(raw_bytes).hexdigest()


def write_report(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics.resolve()
    output_path = args.output.resolve()
    try:
        metrics, metrics_sha256 = load_and_hash_metrics(metrics_path)
        resolved_config_path, resolved_config_sha256 = (
            load_and_validate_resolved_config(metrics_path, metrics)
        )
        report = render_report(
            metrics,
            metrics_path=metrics_path,
            metrics_sha256=metrics_sha256,
            resolved_config_path=resolved_config_path,
            resolved_config_sha256=resolved_config_sha256,
        )
        if args.check:
            if not output_path.is_file():
                raise FileNotFoundError(f"Generated report not found: {output_path}")
            existing = output_path.read_bytes().decode("utf-8")
            if existing != report:
                raise ValueError(
                    "Generated report is stale: rebuild it from the canonical metrics."
                )
            print(f"PASS: Week 8 report matches {metrics_path}")
            return
        write_report(output_path, report)
        print(f"Wrote deterministic Week 8 report: {output_path}")
        print("PASS: saved-result consistency checks succeeded.")
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
