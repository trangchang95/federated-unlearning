"""Compare two Week 8 metrics files while ignoring only measured CPU timing.

This is intentionally stricter than comparing final accuracy alone. Splits,
partitions, histories, losses, metrics, communication totals, and checkpoint
checksums must all be exactly equal across the two fixed-seed CPU runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = PROJECT_ROOT / "results" / "month2_week8_mnist_iid_comparison"
IGNORED_TIMING_PATHS = (
    ("centralized", "training_and_validation_time_seconds"),
    ("fedavg", "training_and_validation_time_seconds"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first",
        type=Path,
        default=DEFAULT_DIRECTORY / "metrics_first_run.json",
    )
    parser.add_argument(
        "--second", type=Path, default=DEFAULT_DIRECTORY / "metrics.json"
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metrics root must be a JSON object: {path}")
    return value


def remove_expected_timing(
    metrics: dict[str, Any], label: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for method, field in IGNORED_TIMING_PATHS:
        record = metrics.get(method)
        if not isinstance(record, dict) or field not in record:
            raise ValueError(f"{label} is missing {method}.{field}.")
        value = record.pop(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{label} has invalid {method}.{field}: {value!r}.")
        result[method] = float(value)
    return result


def differences(left: Any, right: Any, path: str = "metrics") -> list[str]:
    if type(left) is not type(right):
        return [
            f"{path}: type {type(left).__name__} != {type(right).__name__}"
        ]
    if isinstance(left, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{path}.{key}: missing from one run")
            else:
                found.extend(differences(left[key], right[key], f"{path}.{key}"))
        return found
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        found = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found.extend(differences(left_item, right_item, f"{path}[{index}]"))
        return found
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def main() -> None:
    args = parse_args()
    first = copy.deepcopy(load_metrics(args.first.resolve()))
    second = copy.deepcopy(load_metrics(args.second.resolve()))
    first_times = remove_expected_timing(first, "first run")
    second_times = remove_expected_timing(second, "second run")
    found = differences(first, second)
    if found:
        preview = "\n".join(found[:50])
        raise SystemExit(
            "FAIL: deterministic Week 8 fields differ across reruns:\n" + preview
        )
    print(
        "PASS: every deterministic Week 8 JSON field is exactly identical "
        "across the two runs."
    )
    for method in ("centralized", "fedavg"):
        print(
            "Ignored expected timing variation: "
            f"{method} {first_times[method]:.3f}s -> {second_times[method]:.3f}s"
        )


if __name__ == "__main__":
    main()
