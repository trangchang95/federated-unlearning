"""Small helpers shared by the thesis experiment scripts.

The thesis rules require every experiment to be driven by a saved config and
to write the same core fields.  Keeping that check here makes it harder for a
later experiment to silently omit an important setting such as the random
seed or the learning rate.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENT_FIELDS = (
    "dataset",
    "number_of_clients",
    "client_fraction",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "alpha",
    "random_seed",
    "algorithm",
    "target_client",
    "number_of_rounds",
    "accuracy",
    "f1",
    "unlearning_time",
    "communication_cost",
)

REQUIRED_REPRODUCIBILITY_FIELDS = (
    "code_revision",
    "dataset_version",
    "device",
    "environment_manifest",
    "runner",
    "strict_environment",
)


def load_json_config(config_path: Path) -> dict[str, Any]:
    """Load and validate one JSON experiment configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_fields = REQUIRED_EXPERIMENT_FIELDS + REQUIRED_REPRODUCIBILITY_FIELDS
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(
            "Experiment config is missing required fields: " + ", ".join(missing)
        )
    project_root = config_path.resolve().parent.parent
    for path_field in ("environment_manifest", "runner"):
        referenced_path = project_root / config[path_field]
        if not referenced_path.is_file():
            raise FileNotFoundError(
                f"Config field {path_field!r} points to a missing file: {referenced_path}"
            )
    return config


def validate_runtime_environment(config: dict[str, Any], project_root: Path) -> dict:
    """Validate and return the exact runtime snapshot named by the config.

    The saved config points to a versioned environment manifest. In strict
    mode, a run stops before training if Python or a required package differs.
    This avoids silently producing a different result on an unrecorded setup.
    """
    manifest_path = project_root / config["environment_manifest"]
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    current_packages: dict[str, str] = {}
    for package_name, expected_version in manifest["packages"].items():
        try:
            installed_version = version(package_name)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package {package_name!r} is not installed. "
                f"Recreate the environment from {manifest['requirements_lock']}."
            ) from error
        current_packages[package_name] = installed_version
        if config["strict_environment"] and installed_version != expected_version:
            raise RuntimeError(
                f"Environment mismatch for {package_name}: expected {expected_version}, "
                f"found {installed_version}. Recreate the environment from "
                f"{manifest['requirements_lock']}."
            )

    current_python = platform.python_version()
    if config["strict_environment"] and current_python != manifest["python_version"]:
        raise RuntimeError(
            f"Python mismatch: expected {manifest['python_version']}, found {current_python}."
        )

    return {
        "python_version": current_python,
        "platform": platform.platform(),
        "device": config["device"],
        "packages": current_packages,
        "environment_manifest": config["environment_manifest"],
    }


def write_json(output_path: Path, content: dict[str, Any]) -> None:
    """Write JSON in a stable, human-readable format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(content, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
