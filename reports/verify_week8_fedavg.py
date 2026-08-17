"""Verify the Month 2 Week 8 IID MNIST comparison without retraining it.

This script has two deliberate modes:

* before the saved run exists, it checks that the config, exact environment,
  Git revision, and implementation scope are ready, then prints ``PRE-RUN``;
* after the saved run exists, it also checks the resolved pre-training config,
  complete metrics contract, deterministic partitions, histories,
  communication arithmetic, checkpoints, plots, generated beginner summary,
  and deterministic versioned report.

It never downloads MNIST and never starts training. Run it from the repository
root in the recorded Month 2 environment:

    conda run -n mse-ai python reports/verify_week8_fedavg.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import (  # noqa: E402
    REQUIRED_EXPERIMENT_FIELDS,
    REQUIRED_REPRODUCIBILITY_FIELDS,
    load_json_config,
    validate_runtime_environment,
)
from models.simple_mlp import SimpleMLP  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)
EXPECTED_RUNNER = "experiments/iid/month2_week8_mnist_fedavg.py"
EXPECTED_ALGORITHM = "handwritten_fedavg_iid_vs_centralized_sgd"
EXPECTED_DATASET_SIZE = 60_000
EXPECTED_NUMBER_OF_CLASSES = 10
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

# These files determine the scientific result. Documentation may legitimately
# change after the run, but these files must remain byte-equivalent to the Git
# revision named by the saved config.
PROVENANCE_PATHS = (
    "configs/month2_week8_mnist_iid_fedavg_vs_centralized.json",
    "environment/month2_cpu_runtime.json",
    "environment/month2_cpu_requirements.txt",
    EXPECTED_RUNNER,
    "experiments/experiment_utils.py",
    "models/simple_mlp.py",
    "algorithms/fedavg.py",
    "clients/federated_client.py",
    "clients/iid_partition.py",
    "server/fedavg_server.py",
    "evaluation/utility.py",
    "reports/build_month2_week8_report.py",
    "reports/verify_week8_fedavg.py",
)

SCOPE_AUDIT_PATHS = (
    EXPECTED_RUNNER,
    "algorithms/fedavg.py",
    "clients/federated_client.py",
    "clients/iid_partition.py",
    "server/fedavg_server.py",
    "evaluation/utility.py",
)

PROHIBITED_IMPORT_PREFIXES = (
    "flwr",
    "algorithms.fedprox",
    "algorithms.federated_unlearning",
    "algorithms.proposed_method",
    "experiments.noniid",
    "experiments.unlearning",
    "server.flower_compat",
)

EXPECTED_ARTIFACTS = {
    "metrics.json",
    "resolved_config.json",
    "comparison_summary.md",
    "convergence_comparison.png",
    "centralized_confusion_matrix.png",
    "fedavg_confusion_matrix.png",
    "best_centralized_model.pt",
    "best_fedavg_model.pt",
}
VERSIONED_REPORT = PROJECT_ROOT / "reports" / "month2_week8_centralized_vs_fedavg.md"
REPORT_BUILDER = PROJECT_ROOT / "reports" / "build_month2_week8_report.py"


class VerificationError(RuntimeError):
    """One beginner-readable evidence check failed."""


def require(condition: bool, message: str) -> None:
    """Raise a consistent error instead of a hard-to-read assertion trace."""
    if not condition:
        raise VerificationError(message)


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def require_integer(value: object, field: str, *, minimum: int | None = None) -> int:
    require(is_integer(value), f"{field} must be an integer, found {value!r}.")
    result = int(value)
    if minimum is not None:
        require(result >= minimum, f"{field} must be at least {minimum}, found {result}.")
    return result


def require_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    require(is_finite_number(value), f"{field} must be a finite number, found {value!r}.")
    result = float(value)
    if minimum is not None:
        require(result >= minimum, f"{field} must be at least {minimum}, found {result}.")
    if maximum is not None:
        require(result <= maximum, f"{field} must be at most {maximum}, found {result}.")
    return result


def require_close(
    observed: object,
    expected: object,
    field: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    observed_number = require_number(observed, field)
    expected_number = require_number(expected, f"expected value for {field}")
    require(
        math.isclose(observed_number, expected_number, rel_tol=0.0, abs_tol=tolerance),
        f"{field} is {observed_number}, but its recorded inputs imply "
        f"{expected_number} (tolerance {tolerance}).",
    )


def require_sha256(value: object, field: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{field} must be a lowercase 64-character SHA-256 digest.",
    )
    return value


def require_git_commit(value: object, field: str) -> str:
    require(
        isinstance(value, str) and GIT_COMMIT_PATTERN.fullmatch(value) is not None,
        f"{field} must be a complete 40- or 64-character hexadecimal Git commit ID.",
    )
    return value


def require_mapping(value: object, field: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{field} must be a JSON object.")
    return value


def require_list(value: object, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be a JSON list.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Week 8 saved JSON config (default: the versioned project config).",
    )
    return parser.parse_args()


def git_run(arguments: Sequence[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
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
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 and not allow_failure:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no Git detail"
        raise VerificationError(
            f"Git provenance command failed ({' '.join(arguments)}): {detail}"
        )
    return completed


def validate_config(config: Mapping[str, Any], config_path: Path) -> Path:
    """Check that the config describes only the permitted first IID run."""
    expected_config = DEFAULT_CONFIG.resolve()
    require(
        config_path.resolve() == expected_config,
        "Week 8 verification must use the versioned config "
        f"{expected_config.relative_to(PROJECT_ROOT)}, not {config_path}.",
    )
    missing = [
        field
        for field in REQUIRED_EXPERIMENT_FIELDS + REQUIRED_REPRODUCIBILITY_FIELDS
        if field not in config
    ]
    require(not missing, "The config is missing required fields: " + ", ".join(missing))

    require(config.get("dataset") == "MNIST", "Week 8 must use MNIST.")
    require(config.get("data_subdirectory") == "mnist", "data_subdirectory must be mnist.")
    require(
        config.get("dataset_version")
        == "TorchVision MNIST official train/test split",
        "dataset_version must describe the official TorchVision MNIST split.",
    )
    require(config.get("model") == "SimpleMLP", "The Week 8 model must be SimpleMLP.")
    require(
        config.get("preprocessing") == "ToTensor only; pixels scaled to [0, 1]",
        "preprocessing must match the runner's ToTensor-only pipeline.",
    )
    require(config.get("runner") == EXPECTED_RUNNER, f"runner must be {EXPECTED_RUNNER}.")
    require(
        config.get("algorithm") == EXPECTED_ALGORITHM,
        f"algorithm must be {EXPECTED_ALGORITHM}.",
    )
    require(
        config.get("partition_strategy") == "iid_seeded_equal_size",
        "partition_strategy must remain IID; Non-IID belongs to Month 3.",
    )
    require(config.get("alpha") is None, "alpha must be null in an IID run.")
    require(
        config.get("target_client") is None,
        "target_client must be null before Federated Unlearning begins.",
    )
    require(config.get("optimizer") == "SGD", "Both comparison paths must use SGD.")
    require(
        config.get("optimizer_details") == "plain SGD; momentum=0; weight_decay=0",
        "optimizer_details must record plain SGD without momentum or weight decay.",
    )
    require(
        config.get("loss_function") == "CrossEntropyLoss",
        "The recorded loss_function must be CrossEntropyLoss.",
    )
    require(config.get("device") == "cpu", "The verified Week 8 run must use CPU.")
    require(config.get("strict_environment") is True, "strict_environment must be true.")
    require(config.get("num_workers") == 0, "num_workers must be 0 for this run.")
    require(
        not isinstance(config.get("client_fraction"), bool)
        and config.get("client_fraction") == 1.0,
        "The first comparison requires numeric C=1.0.",
    )
    require(config.get("accuracy") is None, "Config accuracy must be null before the run.")
    require(config.get("f1") is None, "Config f1 must be null before the run.")
    require(config.get("unlearning_time") == 0.0, "unlearning_time must be 0.0.")
    require(config.get("communication_cost") == 0, "Config communication_cost is a placeholder 0.")
    require(
        config.get("communication_cost_unit") == "bytes",
        "communication_cost_unit must be bytes.",
    )
    require(
        config.get("communication_cost_definition")
        == "dense tensor payload: one full model download and upload per selected client per round",
        "communication_cost_definition must describe one dense-model download and upload.",
    )
    require(
        config.get("selection_rule")
        == "ceil(client_fraction * number_of_clients), minimum 1",
        "selection_rule must match the hand-written server's ceiling rule.",
    )

    number_of_clients = require_integer(
        config.get("number_of_clients"), "number_of_clients", minimum=2
    )
    require(
        number_of_clients <= 10,
        "The first run must stay modest (at most 10 clients), as required by the plan.",
    )
    local_epochs = require_integer(config.get("local_epochs"), "local_epochs", minimum=1)
    number_of_rounds = require_integer(
        config.get("number_of_rounds"), "number_of_rounds", minimum=1
    )
    centralized_epochs = require_integer(
        config.get("centralized_epochs"), "centralized_epochs", minimum=1
    )
    require(
        centralized_epochs == number_of_rounds * local_epochs,
        "centralized_epochs must equal number_of_rounds * local_epochs so the two "
        "methods receive the same number of full-data passes.",
    )
    require_integer(config.get("batch_size"), "batch_size", minimum=1)
    require_integer(config.get("hidden_units"), "hidden_units", minimum=1)
    require(
        config.get("architecture")
        == f"784 -> {config['hidden_units']} ReLU -> 10",
        "architecture must match SimpleMLP and hidden_units.",
    )
    require_number(config.get("learning_rate"), "learning_rate", minimum=1e-15)

    validation_fraction = require_number(
        config.get("validation_fraction"),
        "validation_fraction",
        minimum=1e-15,
        maximum=1.0 - 1e-15,
    )
    require(
        validation_fraction == 0.15,
        "The canonical Week 8 MNIST validation_fraction must be 0.15.",
    )
    expected_training = require_integer(
        config.get("expected_training_examples"),
        "expected_training_examples",
        minimum=1,
    )
    expected_validation = require_integer(
        config.get("expected_validation_examples"),
        "expected_validation_examples",
        minimum=1,
    )
    expected_test = require_integer(
        config.get("expected_test_examples"), "expected_test_examples", minimum=1
    )
    require(
        (expected_training, expected_validation, expected_test)
        == (51_000, 9_000, 10_000),
        "The canonical MNIST split must be exactly 51,000/9,000/10,000.",
    )
    require(
        expected_training + expected_validation == EXPECTED_DATASET_SIZE,
        "MNIST training and validation sizes must add to the official 60,000 images.",
    )
    require(expected_test == 10_000, "MNIST must retain the official 10,000-image test set.")
    require(
        int(EXPECTED_DATASET_SIZE * validation_fraction) == expected_validation,
        "validation_fraction does not reproduce expected_validation_examples.",
    )

    for field in (
        "random_seed",
        "split_seed",
        "train_partition_seed",
        "test_partition_seed",
        "centralized_loader_seed",
        "model_initialization_seed",
    ):
        require_integer(config.get(field), field, minimum=0)

    for field in ("output_subdirectory", "data_subdirectory"):
        value = config.get(field)
        require(isinstance(value, str) and value.strip(), f"{field} must be a non-empty string.")
        relative = Path(value)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"{field} must stay inside its designated repository directory.",
        )

    output_directory = (PROJECT_ROOT / "results" / config["output_subdirectory"]).resolve()
    require(
        output_directory.is_relative_to((PROJECT_ROOT / "results").resolve()),
        "The output directory escaped the gitignored results/ folder.",
    )
    return output_directory


def validate_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate both the manifest/lock and the currently active environment."""
    manifest_relative = config.get("environment_manifest")
    require(
        manifest_relative == "environment/month2_cpu_runtime.json",
        "Week 8 must use environment/month2_cpu_runtime.json.",
    )
    manifest_path = PROJECT_ROOT / str(manifest_relative)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read the Month 2 environment manifest: {error}") from error
    require(isinstance(manifest, dict), "The environment manifest must be a JSON object.")
    require(manifest.get("device") == "cpu", "The environment manifest must record CPU.")
    require(
        manifest.get("requirements_lock") == "environment/month2_cpu_requirements.txt",
        "The manifest points to the wrong requirements lock.",
    )
    python_version = manifest.get("python_version")
    require(
        isinstance(python_version, str) and re.fullmatch(r"\d+\.\d+\.\d+", python_version),
        "The manifest must record a complete Python version such as 3.10.20.",
    )
    packages = require_mapping(manifest.get("packages"), "manifest.packages")
    require(packages.get("flwr") == "1.30.0", "Month 2 must record Flower 1.30.0.")

    lock_path = PROJECT_ROOT / str(manifest["requirements_lock"])
    require(lock_path.is_file(), f"The dependency lock is missing: {lock_path}.")
    locked_packages: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        require(
            match is not None,
            f"Dependency lock line {line_number} is not an exact name==version pin: {line!r}.",
        )
        name, version = match.groups()
        normalized_name = name.lower().replace("_", "-")
        require(normalized_name not in locked_packages, f"Duplicate lock entry for {name}.")
        locked_packages[normalized_name] = version
    normalized_manifest = {
        str(name).lower().replace("_", "-"): str(version)
        for name, version in packages.items()
    }
    require(
        locked_packages == normalized_manifest,
        "The exact dependency lock and environment manifest package versions differ.",
    )

    try:
        runtime = validate_runtime_environment(dict(config), PROJECT_ROOT)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise VerificationError(
            "The active Python environment does not match the saved Month 2 manifest: "
            f"{error}"
        ) from error
    return runtime


def imported_modules(path: Path) -> tuple[set[str], bool]:
    """Return static imports and whether a dynamic-import call was found."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise VerificationError(f"Cannot parse {path.relative_to(PROJECT_ROOT)}: {error}") from error
    modules: set[str] = set()
    dynamic_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                dynamic_import = True
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                dynamic_import = True
    return modules, dynamic_import


def validate_scope() -> None:
    """Keep Week 8 on hand-written IID FedAvg and before later thesis gates."""
    for relative_path in SCOPE_AUDIT_PATHS:
        path = PROJECT_ROOT / relative_path
        require(path.is_file(), f"Required Week 8 source file is missing: {relative_path}.")
        modules, dynamic_import = imported_modules(path)
        require(
            not dynamic_import,
            f"{relative_path} uses a dynamic import, so its scope cannot be audited reliably.",
        )
        for module in sorted(modules):
            for prohibited in PROHIBITED_IMPORT_PREFIXES:
                require(
                    module != prohibited and not module.startswith(prohibited + "."),
                    f"{relative_path} imports {module}. Week 8 must not use Flower, "
                    "FedProx, Non-IID, or Federated Unlearning code.",
                )

    runner_source = (PROJECT_ROOT / EXPECTED_RUNNER).read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source, filename=EXPECTED_RUNNER)
    called_names = {
        node.func.id
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    require("FedAvgServer" in called_names, "The runner no longer calls the hand-written FedAvgServer.")
    require(
        ("datasets", "MNIST") in attribute_calls,
        "The runner no longer loads the official TorchVision MNIST dataset.",
    )
    require(
        ("torch", "save") in attribute_calls,
        "The runner must save validation-selected model checkpoints automatically.",
    )


def validate_git_provenance(config: Mapping[str, Any], *, post_run: bool) -> tuple[str, str]:
    """Resolve the named tag and compare every result-producing source file."""
    revision = config.get("code_revision")
    require(isinstance(revision, str) and revision.strip(), "code_revision must name a Git tag.")
    require(
        re.fullmatch(r"[A-Za-z0-9._-]+", revision) is not None,
        "code_revision contains characters that are unsafe in a tag name.",
    )
    resolved_process = git_run(
        ["rev-parse", "--verify", f"refs/tags/{revision}^{{commit}}"],
        allow_failure=True,
    )
    require(
        resolved_process.returncode == 0,
        f"Git tag {revision!r} does not exist yet. Commit the finalized Week 8 "
        "implementation and create that tag before running the experiment.",
    )
    resolved_revision = resolved_process.stdout.strip()
    require_git_commit(resolved_revision, f"resolved Git tag {revision}")
    head_revision = git_run(["rev-parse", "HEAD"]).stdout.strip()
    require_git_commit(head_revision, "current Git HEAD")

    for relative_path in PROVENANCE_PATHS:
        at_revision = git_run(
            ["cat-file", "-e", f"{resolved_revision}:{relative_path}"],
            allow_failure=True,
        )
        require(
            at_revision.returncode == 0,
            f"{relative_path} is not stored in the commit named by code_revision.",
        )
        comparison = git_run(
            ["diff", "--quiet", resolved_revision, "--", relative_path],
            allow_failure=True,
        )
        require(
            comparison.returncode == 0,
            f"{relative_path} differs from tag {revision}. The saved result would no "
            "longer match its claimed code revision.",
        )

    if not post_run:
        require(
            head_revision == resolved_revision,
            f"PRE-RUN requires HEAD to equal tag {revision}; HEAD is {head_revision[:12]} "
            f"but the tag is {resolved_revision[:12]}.",
        )
        status = git_run(["status", "--porcelain=v1", "--untracked-files=all"]).stdout.strip()
        require(
            not status,
            "PRE-RUN requires a clean Git worktree so the producing code is unambiguous. "
            "Commit the intended changes first.",
        )

    for ignored_path in (
        f"results/{Path(str(config['output_subdirectory'])).as_posix()}/metrics.json",
        f"data/{Path(str(config['data_subdirectory'])).as_posix()}/MNIST/raw/train-images-idx3-ubyte",
    ):
        ignored = git_run(["check-ignore", "-q", ignored_path], allow_failure=True)
        require(
            ignored.returncode == 0,
            f"{ignored_path} is not gitignored; datasets and run outputs must not be committed.",
        )
    return resolved_revision, head_revision


def checksum_indices(indices: Sequence[int]) -> str:
    array = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def checksum_model_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def deterministic_partitions(
    dataset_size: int, number_of_clients: int, random_seed: int
) -> dict[int, list[int]]:
    permutation = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(random_seed)
    ).tolist()
    base_size, remainder = divmod(dataset_size, number_of_clients)
    partitions: dict[int, list[int]] = {}
    cursor = 0
    for client_id in range(number_of_clients):
        size = base_size + (1 if client_id < remainder else 0)
        partitions[client_id] = permutation[cursor : cursor + size]
        cursor += size
    require(cursor == dataset_size, "Internal verifier error while rebuilding IID partitions.")
    return partitions


def validate_histogram(value: object, field: str, expected_total: int) -> list[int]:
    histogram = require_list(value, field)
    require(
        len(histogram) == EXPECTED_NUMBER_OF_CLASSES,
        f"{field} must have one count for each digit 0-9.",
    )
    counts = [require_integer(item, f"{field}[{index}]", minimum=0) for index, item in enumerate(histogram)]
    require(sum(counts) == expected_total, f"{field} sums to {sum(counts)}, expected {expected_total}.")
    return counts


def validate_partitions(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[int, int], list[int]]:
    """Rebuild every saved split/checksum from seeds, without loading MNIST."""
    training_size = int(config["expected_training_examples"])
    validation_size = int(config["expected_validation_examples"])
    test_size = int(config["expected_test_examples"])
    number_of_clients = int(config["number_of_clients"])

    split_permutation = torch.randperm(
        EXPECTED_DATASET_SIZE,
        generator=torch.Generator().manual_seed(int(config["split_seed"])),
    ).tolist()
    training_original_indices = split_permutation[:training_size]
    validation_original_indices = split_permutation[
        training_size : training_size + validation_size
    ]
    require(
        len(set(training_original_indices).intersection(validation_original_indices)) == 0,
        "Rebuilt training and validation splits overlap.",
    )
    require(
        len(set(training_original_indices + validation_original_indices))
        == EXPECTED_DATASET_SIZE,
        "Rebuilt training and validation splits do not cover all 60,000 images.",
    )
    require(
        metrics.get("training_split_checksum_sha256")
        == checksum_indices(training_original_indices),
        "training_split_checksum_sha256 does not match split_seed and split sizes.",
    )
    require(
        metrics.get("validation_split_checksum_sha256")
        == checksum_indices(validation_original_indices),
        "validation_split_checksum_sha256 does not match split_seed and split sizes.",
    )

    expected_training_partitions = deterministic_partitions(
        training_size, number_of_clients, int(config["train_partition_seed"])
    )
    expected_test_partitions = deterministic_partitions(
        test_size, number_of_clients, int(config["test_partition_seed"])
    )
    flattened_training = [item for part in expected_training_partitions.values() for item in part]
    flattened_test = [item for part in expected_test_partitions.values() for item in part]
    require(
        len(flattened_training) == len(set(flattened_training)) == training_size,
        "Rebuilt client training partitions are not disjoint and complete.",
    )
    require(
        len(flattened_test) == len(set(flattened_test)) == test_size,
        "Rebuilt client test partitions are not disjoint and complete.",
    )

    training_records = require_list(
        metrics.get("client_training_partitions"), "client_training_partitions"
    )
    test_records = require_list(
        metrics.get("client_test_partitions"), "client_test_partitions"
    )
    require(
        len(training_records) == number_of_clients,
        "There must be one saved training-partition record per client.",
    )
    require(
        len(test_records) == number_of_clients,
        "There must be one saved test-partition record per client.",
    )

    client_training_sizes: dict[int, int] = {}
    aggregate_test_histogram = [0] * EXPECTED_NUMBER_OF_CLASSES
    for client_id in range(number_of_clients):
        training_record = require_mapping(
            training_records[client_id], f"client_training_partitions[{client_id}]"
        )
        test_record = require_mapping(
            test_records[client_id], f"client_test_partitions[{client_id}]"
        )
        require(
            training_record.get("client_id") == client_id,
            f"Training partition record {client_id} has the wrong client_id.",
        )
        require(
            test_record.get("client_id") == client_id,
            f"Test partition record {client_id} has the wrong client_id.",
        )
        training_positions = expected_training_partitions[client_id]
        test_positions = expected_test_partitions[client_id]
        expected_training_count = len(training_positions)
        expected_test_count = len(test_positions)
        require(
            training_record.get("number_of_examples") == expected_training_count,
            f"Client {client_id} training count does not match the seeded IID split.",
        )
        require(
            test_record.get("number_of_examples") == expected_test_count,
            f"Client {client_id} test count does not match the seeded IID split.",
        )
        require(
            training_record.get("position_checksum_sha256")
            == checksum_indices(training_positions),
            f"Client {client_id} training-position checksum is wrong.",
        )
        original_indices = [training_original_indices[position] for position in training_positions]
        require(
            training_record.get("original_index_checksum_sha256")
            == checksum_indices(original_indices),
            f"Client {client_id} original-training-index checksum is wrong.",
        )
        require(
            test_record.get("index_checksum_sha256") == checksum_indices(test_positions),
            f"Client {client_id} test-index checksum is wrong.",
        )
        validate_histogram(
            training_record.get("class_histogram"),
            f"client_training_partitions[{client_id}].class_histogram",
            expected_training_count,
        )
        test_histogram = validate_histogram(
            test_record.get("class_histogram"),
            f"client_test_partitions[{client_id}].class_histogram",
            expected_test_count,
        )
        aggregate_test_histogram = [
            total + count for total, count in zip(aggregate_test_histogram, test_histogram)
        ]
        client_training_sizes[client_id] = expected_training_count

    require(sum(client_training_sizes.values()) == training_size, "Client training counts do not cover the training split.")
    require(sum(aggregate_test_histogram) == test_size, "Client test histograms do not cover the test split.")
    return client_training_sizes, aggregate_test_histogram


def macro_f1_from_confusion(matrix: list[list[int]]) -> float:
    per_class: list[float] = []
    for class_id in range(EXPECTED_NUMBER_OF_CLASSES):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in range(EXPECTED_NUMBER_OF_CLASSES) if row != class_id)
        false_negative = sum(matrix[class_id][column] for column in range(EXPECTED_NUMBER_OF_CLASSES) if column != class_id)
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(per_class) / len(per_class)


def validate_evaluation_record(
    record: Mapping[str, Any],
    field: str,
    *,
    test_size: int,
    aggregate_test_histogram: list[int],
    number_of_clients: int,
) -> None:
    require_number(record.get("test_loss"), f"{field}.test_loss", minimum=0.0)
    accuracy = require_number(record.get("accuracy"), f"{field}.accuracy", minimum=0.0, maximum=1.0)
    f1_macro = require_number(record.get("f1_macro"), f"{field}.f1_macro", minimum=0.0, maximum=1.0)
    require(
        accuracy >= 0.80,
        f"{field}.accuracy is below 0.80. The run may be real, but it needs investigation before Gate 2 can pass.",
    )

    raw_confusion = require_list(record.get("confusion_matrix"), f"{field}.confusion_matrix")
    require(len(raw_confusion) == EXPECTED_NUMBER_OF_CLASSES, f"{field}.confusion_matrix must be 10x10.")
    matrix: list[list[int]] = []
    for row_index, raw_row in enumerate(raw_confusion):
        row = require_list(raw_row, f"{field}.confusion_matrix[{row_index}]")
        require(len(row) == EXPECTED_NUMBER_OF_CLASSES, f"{field}.confusion_matrix row {row_index} must have 10 values.")
        matrix.append(
            [
                require_integer(value, f"{field}.confusion_matrix[{row_index}][{column_index}]", minimum=0)
                for column_index, value in enumerate(row)
            ]
        )
    row_totals = [sum(row) for row in matrix]
    require(row_totals == aggregate_test_histogram, f"{field} confusion-matrix row totals do not match saved client test histograms.")
    require(sum(row_totals) == test_size, f"{field} confusion matrix contains the wrong number of test predictions.")
    accuracy_from_confusion = sum(matrix[index][index] for index in range(EXPECTED_NUMBER_OF_CLASSES)) / test_size
    require_close(accuracy, round(accuracy_from_confusion, 6), f"{field}.accuracy from confusion matrix", tolerance=5e-7)
    require_close(f1_macro, round(macro_f1_from_confusion(matrix), 6), f"{field}.f1_macro from confusion matrix", tolerance=5e-7)

    raw_per_class = require_list(record.get("per_class_accuracy"), f"{field}.per_class_accuracy")
    require(len(raw_per_class) == EXPECTED_NUMBER_OF_CLASSES, f"{field}.per_class_accuracy must contain 10 values.")
    for class_id, value in enumerate(raw_per_class):
        expected = matrix[class_id][class_id] / row_totals[class_id]
        require_close(value, round(expected, 6), f"{field}.per_class_accuracy[{class_id}]", tolerance=5e-7)

    client_records = require_list(record.get("per_client_test_utility"), f"{field}.per_client_test_utility")
    require(len(client_records) == number_of_clients, f"{field} must report utility for every client.")
    weighted_accuracy = 0.0
    total_client_examples = 0
    for client_id in range(number_of_clients):
        client = require_mapping(client_records[client_id], f"{field}.per_client_test_utility[{client_id}]")
        require(client.get("client_id") == client_id, f"{field} client-utility record {client_id} has the wrong ID.")
        count = require_integer(client.get("number_of_examples"), f"{field}.client[{client_id}].number_of_examples", minimum=1)
        client_accuracy = require_number(client.get("accuracy"), f"{field}.client[{client_id}].accuracy", minimum=0.0, maximum=1.0)
        require_number(client.get("f1_macro"), f"{field}.client[{client_id}].f1_macro", minimum=0.0, maximum=1.0)
        weighted_accuracy += count * client_accuracy
        total_client_examples += count
    require(total_client_examples == test_size, f"{field} per-client test counts do not sum to the full test set.")
    require_close(accuracy, weighted_accuracy / test_size, f"{field} weighted client accuracy", tolerance=1e-6)


def validate_history(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    client_training_sizes: Mapping[int, int],
) -> None:
    centralized = require_mapping(metrics.get("centralized"), "centralized")
    fedavg = require_mapping(metrics.get("fedavg"), "fedavg")
    training_size = int(config["expected_training_examples"])
    batch_size = int(config["batch_size"])
    centralized_epochs = int(config["centralized_epochs"])
    number_of_rounds = int(config["number_of_rounds"])
    local_epochs = int(config["local_epochs"])
    number_of_clients = int(config["number_of_clients"])

    central_history = require_list(centralized.get("history"), "centralized.history")
    require(len(central_history) == centralized_epochs + 1, "Centralized history must contain epoch 0 plus every trained epoch.")
    central_steps_per_epoch = math.ceil(training_size / batch_size)
    central_step_total = 0
    central_validation_accuracies: list[float] = []
    central_selected_rows: list[int] = []
    for epoch, raw_row in enumerate(central_history):
        row = require_mapping(raw_row, f"centralized.history[{epoch}]")
        require(row.get("epoch") == epoch, f"Centralized history row {epoch} has the wrong epoch number.")
        validation_loss = require_number(row.get("validation_loss"), f"centralized.history[{epoch}].validation_loss", minimum=0.0)
        del validation_loss
        validation_accuracy = require_number(row.get("validation_accuracy"), f"centralized.history[{epoch}].validation_accuracy", minimum=0.0, maximum=1.0)
        central_validation_accuracies.append(validation_accuracy)
        selected_as_best = row.get("selected_as_best_checkpoint")
        require(
            isinstance(selected_as_best, bool),
            f"centralized.history[{epoch}].selected_as_best_checkpoint must be Boolean.",
        )
        if selected_as_best:
            central_selected_rows.append(epoch)
        if epoch == 0:
            require(row.get("train_loss") is None, "Centralized epoch 0 train_loss must be null.")
            require(row.get("optimizer_steps") == 0, "Centralized epoch 0 must have zero optimizer steps.")
        else:
            require_number(row.get("train_loss"), f"centralized.history[{epoch}].train_loss", minimum=0.0)
            require(row.get("optimizer_steps") == central_steps_per_epoch, f"Centralized epoch {epoch} has the wrong optimizer-step count.")
            central_step_total += central_steps_per_epoch
    require(centralized.get("optimizer_steps") == central_step_total, "centralized.optimizer_steps does not equal the history sum.")
    require(centralized.get("training_examples_processed") == training_size * centralized_epochs, "centralized.training_examples_processed is wrong.")
    require(
        len(central_selected_rows) == 1 and central_selected_rows[0] > 0,
        "Exactly one trained centralized epoch must carry the checkpoint-selection flag.",
    )
    central_best_index = central_selected_rows[0]
    require(
        not any(
            accuracy > central_validation_accuracies[central_best_index] + 1e-6
            for accuracy in central_validation_accuracies[1:]
        ),
        "A centralized validation row is more than 1e-6 better than the flagged "
        "checkpoint. (The tolerance allows a tie created by six-decimal logging.)",
    )
    require(centralized.get("best_epoch") == central_best_index, "centralized.best_epoch is not the first epoch with best validation accuracy.")
    require_close(centralized.get("best_validation_accuracy"), central_validation_accuracies[central_best_index], "centralized.best_validation_accuracy", tolerance=5e-7)
    require(
        central_validation_accuracies[central_best_index] > central_validation_accuracies[0],
        "Centralized validation accuracy did not improve beyond the untrained model.",
    )

    fed_history = require_list(fedavg.get("history"), "fedavg.history")
    require(len(fed_history) == number_of_rounds + 1, "FedAvg history must contain round 0 plus every communication round.")
    fed_step_total = 0
    fed_communication_total = 0
    fed_examples_processed = 0
    fed_validation_accuracies: list[float] = []
    fed_selected_rows: list[int] = []
    selected_ids = list(range(number_of_clients))
    expected_key_set = {str(client_id) for client_id in selected_ids}
    payload_bytes = require_integer(metrics.get("model_payload_bytes"), "model_payload_bytes", minimum=1)
    for round_number, raw_row in enumerate(fed_history):
        row = require_mapping(raw_row, f"fedavg.history[{round_number}]")
        require(row.get("round") == round_number, f"FedAvg history row {round_number} has the wrong round number.")
        require(
            "round_number" not in row,
            f"FedAvg history row {round_number} must use the normalized key 'round'.",
        )
        validation_accuracy = require_number(row.get("validation_accuracy"), f"fedavg.history[{round_number}].validation_accuracy", minimum=0.0, maximum=1.0)
        require_number(row.get("validation_loss"), f"fedavg.history[{round_number}].validation_loss", minimum=0.0)
        fed_validation_accuracies.append(validation_accuracy)
        selected_as_best = row.get("selected_as_best_checkpoint")
        require(
            isinstance(selected_as_best, bool),
            f"fedavg.history[{round_number}].selected_as_best_checkpoint must be Boolean.",
        )
        if selected_as_best:
            fed_selected_rows.append(round_number)
        if round_number == 0:
            require(row.get("selected_client_ids") == [], "FedAvg round 0 must select no clients.")
            require(row.get("weighted_mean_local_loss") is None, "FedAvg round 0 local loss must be null.")
            require(row.get("communication_cost_bytes") == 0, "FedAvg round 0 communication must be zero.")
            continue

        require(row.get("selected_client_ids") == selected_ids, f"FedAvg round {round_number} must select every client in sorted order.")
        example_counts = require_mapping(row.get("client_example_counts"), f"fedavg.history[{round_number}].client_example_counts")
        client_losses = require_mapping(row.get("client_mean_losses"), f"fedavg.history[{round_number}].client_mean_losses")
        client_steps = require_mapping(row.get("client_optimizer_steps"), f"fedavg.history[{round_number}].client_optimizer_steps")
        for name, mapping in (("example counts", example_counts), ("mean losses", client_losses), ("optimizer steps", client_steps)):
            require(set(mapping.keys()) == expected_key_set, f"FedAvg round {round_number} {name} do not cover exactly all clients.")

        weighted_loss_numerator = 0.0
        total_examples = 0
        for client_id in selected_ids:
            key = str(client_id)
            count = client_training_sizes[client_id]
            require(example_counts[key] == count, f"FedAvg round {round_number}, client {client_id} has the wrong example count.")
            mean_loss = require_number(client_losses[key], f"fedavg round {round_number} client {client_id} mean loss", minimum=0.0)
            expected_steps = math.ceil(count / batch_size) * local_epochs
            require(client_steps[key] == expected_steps, f"FedAvg round {round_number}, client {client_id} has the wrong optimizer-step count.")
            weighted_loss_numerator += count * mean_loss
            total_examples += count
            fed_step_total += expected_steps
        require(row.get("total_selected_examples") == total_examples == training_size, f"FedAvg round {round_number} does not process the complete training split.")
        require_close(row.get("weighted_mean_local_loss"), weighted_loss_numerator / total_examples, f"fedavg.history[{round_number}].weighted_mean_local_loss", tolerance=1e-10)
        require(row.get("model_payload_bytes") == payload_bytes, f"FedAvg round {round_number} has the wrong model payload size.")
        expected_one_direction = payload_bytes * number_of_clients
        require(row.get("download_bytes") == expected_one_direction, f"FedAvg round {round_number} download bytes are wrong.")
        require(row.get("upload_bytes") == expected_one_direction, f"FedAvg round {round_number} upload bytes are wrong.")
        expected_round_communication = 2 * expected_one_direction
        require(row.get("communication_cost_bytes") == expected_round_communication, f"FedAvg round {round_number} communication bytes are wrong.")
        fed_communication_total += expected_round_communication
        fed_examples_processed += total_examples * local_epochs

    require(fedavg.get("optimizer_steps") == fed_step_total, "fedavg.optimizer_steps does not equal all client-step counts.")
    require(fedavg.get("training_examples_processed") == fed_examples_processed, "fedavg.training_examples_processed is wrong.")
    require(fedavg.get("communication_cost_bytes") == fed_communication_total, "fedavg.communication_cost_bytes does not equal the round sum.")
    require(metrics.get("communication_cost") == fed_communication_total, "Top-level communication_cost does not equal FedAvg communication.")
    require(
        len(fed_selected_rows) == 1 and fed_selected_rows[0] > 0,
        "Exactly one trained FedAvg round must carry the checkpoint-selection flag.",
    )
    fed_best_index = fed_selected_rows[0]
    require(
        not any(
            accuracy > fed_validation_accuracies[fed_best_index] + 1e-6
            for accuracy in fed_validation_accuracies[1:]
        ),
        "A FedAvg validation row is more than 1e-6 better than the flagged checkpoint. "
        "(The tolerance allows a tie created by six-decimal logging.)",
    )
    require(fedavg.get("best_round") == fed_best_index, "fedavg.best_round is not the first round with best validation accuracy.")
    require_close(fedavg.get("best_validation_accuracy"), fed_validation_accuracies[fed_best_index], "fedavg.best_validation_accuracy", tolerance=5e-7)
    require(
        fed_validation_accuracies[fed_best_index] > fed_validation_accuracies[0],
        "FedAvg validation accuracy did not improve beyond the untrained model.",
    )
    require_close(central_history[0]["validation_loss"], fed_history[0]["validation_loss"], "common initial validation loss", tolerance=0.0)
    require_close(central_history[0]["validation_accuracy"], fed_history[0]["validation_accuracy"], "common initial validation accuracy", tolerance=0.0)
    require(
        centralized.get("training_examples_processed") == fedavg.get("training_examples_processed"),
        "The comparison is not exposure-matched: the two methods processed different numbers of examples.",
    )


def expected_initial_state(config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["model_initialization_seed"]))
        model = SimpleMLP(hidden_units=int(config["hidden_units"]))
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def load_and_validate_checkpoint(
    path: Path,
    expected_state: Mapping[str, torch.Tensor],
    expected_checksum: object,
    field: str,
) -> str:
    require(path.is_file(), f"Missing checkpoint: {path.name}.")
    require(path.stat().st_size > 0, f"Checkpoint {path.name} is empty.")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # PyTorch emits several exception types for corrupt archives.
        raise VerificationError(f"Cannot load checkpoint {path.name}: {error}") from error
    require(isinstance(state, Mapping), f"Checkpoint {path.name} must contain a state dictionary.")
    require(tuple(state.keys()) == tuple(expected_state.keys()), f"Checkpoint {path.name} has unexpected tensor keys or order.")
    for name, expected_tensor in expected_state.items():
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor), f"Checkpoint {path.name} entry {name} is not a tensor.")
        require(tensor.shape == expected_tensor.shape, f"Checkpoint {path.name} tensor {name} has the wrong shape.")
        require(tensor.dtype == expected_tensor.dtype, f"Checkpoint {path.name} tensor {name} has the wrong dtype.")
        require(torch.isfinite(tensor).all().item(), f"Checkpoint {path.name} tensor {name} contains NaN or infinity.")
    checksum = checksum_model_state(state)
    require(checksum == require_sha256(expected_checksum, field), f"Checkpoint {path.name} does not match {field}.")
    model = SimpleMLP(hidden_units=expected_state["network.1.bias"].numel())
    model.load_state_dict(state, strict=True)
    return checksum


def validate_png(path: Path, expected_size: tuple[int, int]) -> None:
    require(path.is_file(), f"Missing plot: {path.name}.")
    require(path.stat().st_size > 0, f"Plot {path.name} is empty.")
    try:
        with Image.open(path) as image:
            require(image.format == "PNG", f"{path.name} is not a PNG image.")
            require(image.size == expected_size, f"{path.name} has size {image.size}, expected {expected_size}.")
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
            require(
                ImageChops.difference(rgb, background).getbbox() is not None,
                f"{path.name} is a valid but visually blank single-color image.",
            )
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"Cannot validate plot {path.name}: {error}") from error


def validate_artifacts(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    output_directory: Path,
) -> None:
    present_files = {path.name for path in output_directory.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_ARTIFACTS - present_files)
    require(not missing, "The run is missing required artifacts: " + ", ".join(missing))

    resolved_config_path = output_directory / "resolved_config.json"
    try:
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read resolved_config.json: {error}") from error
    expected_resolved_config = {
        **dict(config),
        "source_config": "configs/month2_week8_mnist_iid_fedavg_vs_centralized.json",
        "code_provenance": metrics.get("code_provenance"),
    }
    require(
        resolved_config == expected_resolved_config,
        "resolved_config.json must be the exact config-plus-provenance snapshot "
        "written before dataset loading and training.",
    )

    initial_state = expected_initial_state(config)
    expected_parameter_count = sum(tensor.numel() for tensor in initial_state.values())
    expected_payload_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in initial_state.values()
    )
    require(metrics.get("trainable_parameters") == expected_parameter_count, "trainable_parameters does not match SimpleMLP.")
    require(metrics.get("model_payload_bytes") == expected_payload_bytes, "model_payload_bytes does not match the dense SimpleMLP state.")
    expected_initial_checksum = checksum_model_state(initial_state)
    initial_checksum = require_sha256(metrics.get("initial_model_checksum_sha256"), "initial_model_checksum_sha256")
    require(initial_checksum == expected_initial_checksum, "The saved initial-model checksum cannot be reproduced from model_initialization_seed.")
    require(metrics.get("centralized_initial_checksum_sha256") == initial_checksum, "The centralized model did not record the common initialization.")
    require(metrics.get("fedavg_initial_checksum_sha256") == initial_checksum, "The FedAvg model did not record the common initialization.")

    centralized = require_mapping(metrics.get("centralized"), "centralized")
    fedavg = require_mapping(metrics.get("fedavg"), "fedavg")
    central_checksum = load_and_validate_checkpoint(
        output_directory / "best_centralized_model.pt",
        initial_state,
        centralized.get("checkpoint_checksum_sha256"),
        "centralized.checkpoint_checksum_sha256",
    )
    fedavg_checksum = load_and_validate_checkpoint(
        output_directory / "best_fedavg_model.pt",
        initial_state,
        fedavg.get("checkpoint_checksum_sha256"),
        "fedavg.checkpoint_checksum_sha256",
    )
    require(central_checksum != initial_checksum, "The centralized checkpoint is still the untrained initial model.")
    require(fedavg_checksum != initial_checksum, "The FedAvg checkpoint is still the untrained initial model.")

    validate_png(output_directory / "centralized_confusion_matrix.png", (980, 980))
    validate_png(output_directory / "fedavg_confusion_matrix.png", (980, 980))
    validate_png(output_directory / "convergence_comparison.png", (1400, 560))

    summary_path = output_directory / "comparison_summary.md"
    require(summary_path.is_file() and summary_path.stat().st_size > 0, "comparison_summary.md is missing or empty.")
    summary = summary_path.read_text(encoding="utf-8")
    required_fragments = (
        "# Week 8 Saved-Run Summary",
        f"| Centralized SGD | {float(centralized['accuracy']):.2%} | {float(centralized['f1_macro']):.2%} | epoch {centralized['best_epoch']} | {float(centralized['training_and_validation_time_seconds']):.1f}s |",
        f"| Hand-written FedAvg | {float(fedavg['accuracy']):.2%} | {float(fedavg['f1_macro']):.2%} | round {fedavg['best_round']} | {float(fedavg['training_and_validation_time_seconds']):.1f}s |",
        f"{float(metrics['accuracy_difference_fedavg_minus_centralized']) * 100:+.2f} percentage points",
        f"{int(metrics['communication_cost']):,} bytes",
        "`handwritten_fedavg`",
        initial_checksum,
        "one fixed-seed, sequential CPU, IID experiment",
        "test set was evaluated only after validation selected",
    )
    for fragment in required_fragments:
        require(fragment in summary, f"comparison_summary.md is missing the generated fact: {fragment!r}.")


def validate_top_level_results(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    resolved_revision: str,
) -> None:
    for field in REQUIRED_EXPERIMENT_FIELDS + REQUIRED_REPRODUCIBILITY_FIELDS:
        require(field in metrics, f"metrics.json is missing required field {field!r}.")
    dynamic_fields = {"accuracy", "f1", "communication_cost", "unlearning_time"}
    for field, expected in config.items():
        if field not in dynamic_fields:
            require(metrics.get(field) == expected, f"metrics.json field {field!r} differs from its saved config.")
    require(metrics.get("source_config") == "configs/month2_week8_mnist_iid_fedavg_vs_centralized.json", "source_config does not identify the versioned Week 8 config.")
    require(metrics.get("runtime") == runtime, "The saved runtime snapshot differs from the currently validated exact environment.")
    require(metrics.get("training_examples") == config["expected_training_examples"], "training_examples differs from the config.")
    require(metrics.get("validation_examples") == config["expected_validation_examples"], "validation_examples differs from the config.")
    require(metrics.get("test_examples") == config["expected_test_examples"], "test_examples differs from the config.")
    require(metrics.get("unlearning_time") == 0.0, "Week 8 unlearning_time must remain 0.0.")
    require(
        metrics.get("primary_method_for_mandatory_metrics") == "handwritten_fedavg",
        "primary_method_for_mandatory_metrics must explicitly identify the "
        "hand-written FedAvg result stored in top-level accuracy/F1.",
    )

    provenance = require_mapping(metrics.get("code_provenance"), "code_provenance")
    expected_tracked_dependencies = [
        "configs/month2_week8_mnist_iid_fedavg_vs_centralized.json",
        EXPECTED_RUNNER,
        "environment/month2_cpu_runtime.json",
    ]
    require(
        provenance.get("configured_revision") == config["code_revision"],
        "code_provenance.configured_revision differs from the saved config.",
    )
    require(
        require_git_commit(provenance.get("resolved_commit"), "code_provenance.resolved_commit")
        == resolved_revision,
        "The recorded producing commit no longer matches the configured Git tag.",
    )
    require(
        provenance.get("head_commit") == resolved_revision,
        "The recorded HEAD was not the tagged producing commit when training began.",
    )
    require(
        provenance.get("worktree_clean_at_start") is True,
        "The run did not record a clean Git worktree at training start.",
    )
    require(
        provenance.get("tracked_dependencies") == expected_tracked_dependencies,
        "code_provenance.tracked_dependencies does not list the config, runner, and environment manifest exactly.",
    )

    centralized = require_mapping(metrics.get("centralized"), "centralized")
    fedavg = require_mapping(metrics.get("fedavg"), "fedavg")
    require_close(metrics.get("accuracy"), fedavg.get("accuracy"), "top-level accuracy", tolerance=0.0)
    require_close(metrics.get("f1"), fedavg.get("f1_macro"), "top-level f1", tolerance=0.0)
    require(centralized.get("communication_cost_bytes") == 0, "Centralized communication cost must be zero in this local baseline.")
    require_number(
        centralized.get("training_and_validation_time_seconds"),
        "centralized.training_and_validation_time_seconds",
        minimum=1e-9,
    )
    require_number(
        fedavg.get("training_and_validation_time_seconds"),
        "fedavg.training_and_validation_time_seconds",
        minimum=1e-9,
    )
    expected_accuracy_difference = round(float(fedavg["accuracy"]) - float(centralized["accuracy"]), 6)
    expected_f1_difference = round(float(fedavg["f1_macro"]) - float(centralized["f1_macro"]), 6)
    require_close(metrics.get("accuracy_difference_fedavg_minus_centralized"), expected_accuracy_difference, "accuracy_difference_fedavg_minus_centralized", tolerance=5e-7)
    require_close(metrics.get("f1_difference_fedavg_minus_centralized"), expected_f1_difference, "f1_difference_fedavg_minus_centralized", tolerance=5e-7)

    protocol = require_mapping(metrics.get("comparison_protocol"), "comparison_protocol")
    expected_protocol = {
        "same_initial_model": True,
        "same_training_split": True,
        "same_optimizer": config["optimizer"],
        "same_learning_rate": config["learning_rate"],
        "same_batch_size": config["batch_size"],
        "centralized_full_data_passes": config["centralized_epochs"],
        "fedavg_full_data_passes_when_full_participation": config["number_of_rounds"] * config["local_epochs"],
        "test_evaluated_after_validation_selection_only": True,
        "optimizer_updates_identical": False,
        "optimizer_update_difference": (
            "Centralized SGD follows one continuous trajectory; FedAvg follows "
            "separate local trajectories and averages model weights. Partial "
            "final batches also make the optimizer-step counts differ."
        ),
        "timing_scope": "sequential CPU process; not distributed wall-clock time",
    }
    require(dict(protocol) == expected_protocol, "comparison_protocol does not match the implemented fair-comparison rules.")


def validate_no_nonfinite_json(value: object, field: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{field} contains NaN or infinity.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_no_nonfinite_json(item, f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_no_nonfinite_json(item, f"{field}.{key}")
        return
    raise VerificationError(f"{field} contains an unsupported value type {type(value).__name__}.")


def validate_versioned_report(output_directory: Path) -> None:
    """Ask the deterministic report builder to prove the report is not stale."""
    require(REPORT_BUILDER.is_file(), "The deterministic Week 8 report builder is missing.")
    require(
        VERSIONED_REPORT.is_file() and VERSIONED_REPORT.stat().st_size > 0,
        "The versioned Week 8 report is missing. Build it with "
        "`python reports/build_month2_week8_report.py`, then rerun this verifier.",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPORT_BUILDER),
            "--metrics",
            str(output_directory / "metrics.json"),
            "--output",
            str(VERSIONED_REPORT),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no report-builder detail"
        raise VerificationError(
            "The versioned Week 8 report is stale or inconsistent with metrics.json: "
            + detail
        )
    report_text = VERSIONED_REPORT.read_text(encoding="utf-8")
    for required_heading in (
        "# Month 2, Week 8",
        "## 4. Final test results",
        "## 5. Validation behavior and convergence",
        "## 6. Per-client test utility",
        "## 8. Communication accounting",
        "one fixed-seed run",
    ):
        require(
            required_heading in report_text,
            f"The versioned Week 8 report is missing required text {required_heading!r}.",
        )


def verify_post_run(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    output_directory: Path,
    resolved_revision: str,
) -> None:
    metrics_path = output_directory / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read metrics.json: {error}") from error
    require(isinstance(metrics, dict), "metrics.json must contain one JSON object.")
    validate_no_nonfinite_json(metrics)
    validate_top_level_results(metrics, config, runtime, resolved_revision)
    client_training_sizes, aggregate_test_histogram = validate_partitions(metrics, config)
    centralized = require_mapping(metrics.get("centralized"), "centralized")
    fedavg = require_mapping(metrics.get("fedavg"), "fedavg")
    validate_evaluation_record(
        centralized,
        "centralized",
        test_size=int(config["expected_test_examples"]),
        aggregate_test_histogram=aggregate_test_histogram,
        number_of_clients=int(config["number_of_clients"]),
    )
    validate_evaluation_record(
        fedavg,
        "fedavg",
        test_size=int(config["expected_test_examples"]),
        aggregate_test_histogram=aggregate_test_histogram,
        number_of_clients=int(config["number_of_clients"]),
    )
    validate_history(metrics, config, client_training_sizes)
    validate_artifacts(metrics, config, output_directory)
    validate_versioned_report(output_directory)


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.resolve()
        try:
            config = load_json_config(config_path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as error:
            raise VerificationError(f"Cannot load the Week 8 config: {error}") from error
        output_directory = validate_config(config, config_path)
        runtime = validate_environment(config)
        validate_scope()

        metrics_path = output_directory / "metrics.json"
        post_run = metrics_path.is_file()
        resolved_revision, head_revision = validate_git_provenance(
            config, post_run=post_run
        )

        if not post_run:
            partial_files = (
                sorted(path.name for path in output_directory.iterdir() if path.is_file())
                if output_directory.is_dir()
                else []
            )
            require(
                not partial_files,
                "The Week 8 output directory contains partial files but no metrics.json: "
                + ", ".join(partial_files)
                + ". Treat this as an interrupted run, not a clean pre-run state.",
            )
            print(
                "PRE-RUN: Week 8 config, exact environment, hand-written IID scope, "
                f"and Git tag {config['code_revision']} ({resolved_revision[:12]}) are ready."
            )
            print(
                "No metrics.json exists yet, so no MNIST accuracy, convergence, or Gate 2 "
                "completion has been verified. Run the saved config, then run this verifier again."
            )
            return 0

        verify_post_run(config, runtime, output_directory, resolved_revision)
        head_note = (
            ""
            if head_revision == resolved_revision
            else f" Current HEAD is {head_revision[:12]}, while producing code remains identical to tag {config['code_revision']}."
        )
        print(
            "PASS: Week 8 IID MNIST FedAvg evidence is internally consistent: config, "
            "environment, Git provenance, splits, client partitions, histories, optimizer "
            "steps, communication bytes, metrics, checkpoints, plots, and generated summary "
            f"all passed.{head_note}"
        )
        print(
            "This verifies one fixed-seed sequential CPU experiment. It does not establish "
            "multi-seed statistical equivalence or real distributed-network timing."
        )
        return 0
    except VerificationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # Keep unexpected failures readable for a beginner.
        print(
            "FAIL: The verifier encountered an unexpected error. This is a verification "
            f"problem, not evidence that Gate 2 passed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
