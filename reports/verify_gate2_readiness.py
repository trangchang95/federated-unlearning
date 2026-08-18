"""Audit whether Month 2 Gate 2 has enough evidence to be reviewed.

This verifier does not answer the learning questions and does not run either
controlled experiment.  It distinguishes three states:

* ``FAIL`` (exit 1): existing evidence is contradictory or invalid;
* ``WAITING`` (exit 2): valid evidence is still missing; and
* ``READY``/``PASS`` (exit 0): technical evidence and the recorded human
  review are complete.

Run from the repository root with the verified Month 2 environment:

    conda run -n mse-ai python reports/verify_gate2_readiness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_utils import validate_runtime_environment  # noqa: E402
from experiments.iid.month2_week8_mnist_fedavg import (  # noqa: E402
    validate_week8_config,
)
from reports.build_month2_week8_report import (  # noqa: E402
    load_and_hash_metrics,
    load_and_validate_resolved_config,
    render_report,
    validate_and_derive,
)
from reports.verify_week8_fedavg import (  # noqa: E402
    VerificationError as Week8VerificationError,
    checksum_model_state,
    expected_initial_state,
    load_and_validate_checkpoint,
    validate_partitions,
    validate_png,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "gate2_variant_evidence.json"
CANONICAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)
SELF_CHECK_PATH = PROJECT_ROOT / "reports" / "gate2_self_check.md"
README_PATH = PROJECT_ROOT / "README.md"
PROGRESS_PATH = PROJECT_ROOT / "PROGRESS.md"
CANONICAL_VERIFIER = PROJECT_ROOT / "reports" / "verify_week8_fedavg.py"
CANONICAL_METRICS = (
    PROJECT_ROOT
    / "results"
    / "month2_week8_mnist_iid_comparison"
    / "metrics.json"
)
CANONICAL_TAG = "month2-week8"
CANONICAL_PASS_TEXT = "PASS: Week 8 IID MNIST FedAvg evidence"
SAFE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

AUTHORITATIVE_EVIDENCE_TOOL_PATHS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "tests" / "test_gate2_readiness.py",
    PROJECT_ROOT / "reports" / "build_month2_week8_report.py",
    CANONICAL_VERIFIER,
)

QUESTION_PREFIXES = {
    1: "Trace one complete FedAvg round.",
    2: "Why must every selected client in one round start",
    3: "Explain `K`, `C`, `E`, `B`, and `R`",
    4: "The two paths process the same 255,000 example exposures",
    5: "State the measured accuracy result",
    6: "Both validation-accuracy curves rise",
    7: "Reconstruct the communication estimate",
    8: "The report includes five IID client test partitions.",
    9: "**Change the number of clients.**",
    10: "**Change local epochs.**",
}

# These files determine the scientific result.  Variant configs may differ,
# but the implementation must remain byte-identical to the canonical run.
SCIENTIFIC_IMPLEMENTATION_PATHS = (
    "environment/month2_cpu_runtime.json",
    "environment/month2_cpu_requirements.txt",
    "experiments/iid/month2_week8_mnist_fedavg.py",
    "experiments/experiment_utils.py",
    "models/simple_mlp.py",
    "algorithms/fedavg.py",
    "clients/federated_client.py",
    "clients/iid_partition.py",
    "server/fedavg_server.py",
    "evaluation/utility.py",
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

VARIANT_RULES = {
    "number_of_clients_k10": {
        "required": {
            "number_of_clients": 10,
            "client_fraction": 1.0,
            "local_epochs": 1,
            "batch_size": 128,
            "number_of_rounds": 5,
            "centralized_epochs": 5,
        },
        "allowed_differences": {
            "experiment_name",
            "output_subdirectory",
            "code_revision",
            "number_of_clients",
        },
    },
    "local_epochs_e2": {
        "required": {
            "number_of_clients": 5,
            "client_fraction": 1.0,
            "local_epochs": 2,
            "batch_size": 128,
            "number_of_rounds": 5,
            "centralized_epochs": 10,
        },
        "allowed_differences": {
            "experiment_name",
            "output_subdirectory",
            "code_revision",
            "local_epochs",
            "centralized_epochs",
        },
    },
}


class Gate2EvidenceError(RuntimeError):
    """Existing Gate 2 evidence is internally inconsistent."""


@dataclass(frozen=True)
class SelfCheckEvidence:
    answers_present: tuple[bool, ...]
    proposals_present: tuple[bool, bool]
    saved_results_present: tuple[bool, bool]
    proposal_review_date: str | None
    proposal_review_result: str
    review_date: str | None
    review_result: str
    pre_run_snapshot_sha256: str

    @property
    def all_student_text_present(self) -> bool:
        return (
            all(self.answers_present)
            and all(self.proposals_present)
            and all(self.saved_results_present)
        )

    @property
    def review_passed(self) -> bool:
        return self.review_result == "PASS" and self.review_date is not None

    @property
    def proposal_review_passed(self) -> bool:
        return (
            self.proposal_review_result == "APPROVED TO RUN"
            and self.proposal_review_date is not None
        )


@dataclass(frozen=True)
class VariantEvidence:
    name: str
    config_path: Path
    metrics_path: Path | None
    report_path: Path | None
    config: Mapping[str, Any]
    resolved_commit: str
    pre_run_snapshot_sha256: str


@dataclass(frozen=True)
class VariantCollection:
    variants: tuple[VariantEvidence, ...]
    waiting: tuple[str, ...]

    @property
    def all_configs_present(self) -> bool:
        return len(self.variants) == len(VARIANT_RULES)

    @property
    def all_results_present(self) -> bool:
        return self.all_configs_present and all(
            variant.metrics_path is not None and variant.report_path is not None
            for variant in self.variants
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Gate2EvidenceError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Versioned Gate 2 evidence manifest.",
    )
    parser.add_argument(
        "--allow-waiting",
        action="store_true",
        help=(
            "Return exit 0 for the expected WAITING state so Conda does not print "
            "an alarming error footer. FAIL still exits 1; READY/PASS still print "
            "their distinct status. Omit this flag in automated gate checks."
        ),
    )
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Gate2EvidenceError(f"{label} is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate2EvidenceError(f"Cannot read {label} {path}: {error}") from error
    require(isinstance(value, Mapping), f"{label} must contain one JSON object.")
    return value


def git_run(
    arguments: Sequence[str], *, allow_failure: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 and not allow_failure:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no Git detail"
        raise Gate2EvidenceError(
            f"Git evidence check failed ({' '.join(arguments)}): {detail}"
        )
    return completed


def resolve_manifest_path(
    raw_value: Any,
    *,
    label: str,
    allowed_root: Path,
    suffix: str | None = None,
) -> Path | None:
    if raw_value is None:
        return None
    require(
        isinstance(raw_value, str) and raw_value.strip(),
        f"{label} must be null or a non-empty repository-relative path.",
    )
    relative = Path(raw_value)
    require(not relative.is_absolute(), f"{label} must be repository-relative.")
    require(".." not in relative.parts, f"{label} must not contain '..'.")
    resolved = (PROJECT_ROOT / relative).resolve()
    root = allowed_root.resolve()
    require(
        resolved.is_relative_to(root),
        f"{label} must stay inside {root.relative_to(PROJECT_ROOT)}.",
    )
    if suffix is not None:
        require(resolved.suffix == suffix, f"{label} must end in {suffix}.")
    return resolved


def normalized_output_key(config: Mapping[str, Any], label: str) -> tuple[Path, str]:
    raw_value = config.get("output_subdirectory")
    require(
        isinstance(raw_value, str) and raw_value.strip(),
        f"{label} output_subdirectory must be a non-empty string.",
    )
    relative = Path(raw_value)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{label} output_subdirectory must stay inside results/.",
    )
    results_root = (PROJECT_ROOT / "results").resolve()
    resolved = (results_root / relative).resolve()
    require(
        resolved.is_relative_to(results_root) and resolved != results_root,
        f"{label} output_subdirectory must be a distinct folder inside results/.",
    )
    return resolved, os.path.normcase(str(resolved))


def _content_present(value: str) -> bool:
    without_comments = re.sub(r"(?s)<!--.*?-->", " ", value)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_comments)
    cleaned = re.sub(r"[`*_\[\](){}#>~]", " ", without_tags).strip()
    if not cleaned:
        return False
    tokens = re.findall(r"\w+", cleaned, flags=re.UNICODE)
    if len(tokens) < 3:
        return False
    normalized_tokens = [token.casefold() for token in tokens]
    if normalized_tokens[0] in {"todo", "tbd", "placeholder", "awaiting"}:
        return False
    normalized = " ".join(normalized_tokens)
    return normalized not in {
        "not yet reviewed",
        "not yet answered",
        "answer goes here",
    }


def _question_block(text: str, number: int) -> str:
    prefix = QUESTION_PREFIXES[number]
    match = re.search(
        rf"(?m)^[ \t]*{number}\.\s+{re.escape(prefix)}.*$", text
    )
    require(match is not None, f"Gate 2 self-check question {number} is missing.")
    if number == 8:
        following = re.search(r"(?m)^## Part B\s+", text[match.end():])
        require(following is not None, "Gate 2 Part B section is missing.")
        end = match.end() + following.start()
    elif number < 10:
        next_prefix = QUESTION_PREFIXES[number + 1]
        following = re.search(
            rf"(?m)^[ \t]*{number + 1}\.\s+{re.escape(next_prefix)}.*$",
            text[match.end():],
        )
        require(
            following is not None,
            f"Gate 2 self-check question {number + 1} is missing.",
        )
        end = match.end() + following.start()
    else:
        following = re.search(
            r"(?m)^## Pre-run proposal review\s*$", text[match.end():]
        )
        require(following is not None, "Gate 2 pre-run review section is missing.")
        end = match.end() + following.start()
    return text[match.start():end]


def _marker_content(block: str, marker: str, *, next_marker: str | None = None) -> str:
    match = re.search(
        rf"(?m)^[ \t]*\*\*{re.escape(marker)}:\*\*[ \t]*(?:\r?\n)?",
        block,
    )
    require(match is not None, f"Self-check marker **{marker}:** is missing.")
    content_start = match.end()
    if next_marker is None:
        content_end = len(block)
    else:
        following = re.search(
            rf"(?m)^[ \t]*\*\*{re.escape(next_marker)}:\*\*",
            block[content_start:],
        )
        require(
            following is not None,
            f"Self-check marker **{next_marker}:** is missing.",
        )
        content_end = content_start + following.start()
    return block[content_start:content_end].strip()


def _review_marker(
    block: str,
    *,
    label: str,
    allowed_results: set[str],
) -> tuple[str | None, str]:
    date_matches = list(
        re.finditer(r"(?m)^\*\*Review date:\*\*[ \t]*(.*?)[ \t]*$", block)
    )
    result_matches = list(
        re.finditer(r"(?m)^\*\*Result:\*\*[ \t]*(.*?)[ \t]*$", block)
    )
    require(
        len(date_matches) == 1 and len(result_matches) == 1,
        f"{label} must contain exactly one **Review date:** and **Result:** marker.",
    )
    date_match = date_matches[0]
    result_match = result_matches[0]
    raw_date = date_match.group(1).strip()
    review_date: str | None = None
    if _content_present(raw_date):
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise Gate2EvidenceError(
                f"{label} date must use YYYY-MM-DD, or remain NOT YET REVIEWED."
            ) from error
        require(parsed_date <= date.today(), f"{label} date cannot be in the future.")
        review_date = raw_date
    raw_result = result_match.group(1).strip().upper()
    require(
        raw_result in allowed_results,
        f"{label} result must be one of: {', '.join(sorted(allowed_results))}.",
    )
    if raw_result in {"PASS", "APPROVED TO RUN"}:
        require(review_date is not None, f"{label} {raw_result} requires a review date.")
    if raw_result == "NOT YET REVIEWED":
        require(review_date is None, f"{label} cannot have a date while still unreviewed.")
    return review_date, raw_result


def parse_self_check_text(text: str) -> SelfCheckEvidence:
    answer_texts: list[str] = []
    for number in range(1, 9):
        block = _question_block(text, number)
        answer = _marker_content(block, "Answer")
        answer_texts.append(answer)

    part_b_texts: list[tuple[str, str]] = []
    for number in (9, 10):
        block = _question_block(text, number)
        proposal = _marker_content(
            block,
            "Proposed changes and prediction",
            next_marker="Saved run/result after review",
        )
        saved = _marker_content(block, "Saved run/result after review")
        part_b_texts.append((proposal, saved))

    proposal_review_match = re.search(
        r"(?ms)^## Pre-run proposal review\s*(.*?)(?=^## Passing standard\s*$)",
        text,
    )
    require(
        proposal_review_match is not None,
        "The pre-run proposal-review section is missing.",
    )
    proposal_review_block = proposal_review_match.group(1)
    proposal_review_date, proposal_review_result = _review_marker(
        proposal_review_block,
        label="Pre-run proposal review",
        allowed_results={"NOT YET REVIEWED", "REVISE", "APPROVED TO RUN"},
    )

    review_match = re.search(r"(?ms)^## Review result\s*(.*)\Z", text)
    require(review_match is not None, "The Gate 2 review section is missing.")
    review_date, review_result = _review_marker(
        review_match.group(1),
        label="Final Gate 2 review",
        allowed_results={"NOT YET REVIEWED", "REVISE", "PASS"},
    )
    proposal_texts, saved_result_texts = zip(*part_b_texts)
    snapshot_payload = {
        "answers": [value.strip().replace("\r\n", "\n") for value in answer_texts],
        "proposals": [
            value.strip().replace("\r\n", "\n") for value in proposal_texts
        ],
        "proposal_review": proposal_review_block.strip().replace("\r\n", "\n"),
    }
    snapshot_bytes = json.dumps(
        snapshot_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SelfCheckEvidence(
        answers_present=tuple(_content_present(value) for value in answer_texts),
        proposals_present=(
            _content_present(proposal_texts[0]),
            _content_present(proposal_texts[1]),
        ),
        saved_results_present=(
            _content_present(saved_result_texts[0]),
            _content_present(saved_result_texts[1]),
        ),
        proposal_review_date=proposal_review_date,
        proposal_review_result=proposal_review_result,
        review_date=review_date,
        review_result=review_result,
        pre_run_snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
    )


def parse_self_check(path: Path) -> SelfCheckEvidence:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise Gate2EvidenceError(f"Cannot read the Gate 2 self-check: {error}") from error
    return parse_self_check_text(text)


def read_gate2_checkbox() -> bool:
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(r"(?m)^- \[([ xX])\] Gate 2 \(Month 2\):", text)
    require(match is not None, "README.md is missing the Gate 2 status checkbox.")
    return match.group(1).lower() == "x"


def run_canonical_verifier() -> None:
    completed = subprocess.run(
        [sys.executable, str(CANONICAL_VERIFIER)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    require(
        completed.returncode == 0 and CANONICAL_PASS_TEXT in completed.stdout,
        "The canonical Week 8 post-run verifier did not return its PASS state. "
        f"A PRE-RUN message is not enough. Detail: {combined or 'no output'}",
    )


def require_committed_current(paths: Sequence[Path], label: str) -> None:
    for path in paths:
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise Gate2EvidenceError(f"{label} path is outside the repository: {path}") from error
        tracked = git_run(
            ["ls-files", "--error-unmatch", "--", relative], allow_failure=True
        )
        require(tracked.returncode == 0, f"{label} is not Git-tracked: {relative}.")
        unchanged = git_run(
            ["diff", "--quiet", "HEAD", "--", relative], allow_failure=True
        )
        require(
            unchanged.returncode == 0,
            f"{label} has uncommitted changes: {relative}.",
        )


def validate_variant_delta(
    name: str,
    config: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> None:
    require(name in VARIANT_RULES, f"Unknown Gate 2 variant {name!r}.")
    try:
        validate_week8_config(dict(config))
    except (KeyError, TypeError, ValueError) as error:
        raise Gate2EvidenceError(f"Variant {name} has an invalid runner config: {error}") from error

    rules = VARIANT_RULES[name]
    for field, expected in rules["required"].items():
        require(
            config.get(field) == expected,
            f"Variant {name} requires {field}={expected!r}; found {config.get(field)!r}.",
        )
    require(
        set(config) == set(canonical),
        f"Variant {name} must have exactly the canonical config fields.",
    )
    changed = {field for field in canonical if config[field] != canonical[field]}
    unexpected = changed - set(rules["allowed_differences"])
    require(
        not unexpected,
        f"Variant {name} changes unrelated fields: {', '.join(sorted(unexpected))}.",
    )
    for identity_field in ("experiment_name", "output_subdirectory", "code_revision"):
        require(
            config[identity_field] != canonical[identity_field],
            f"Variant {name} must use a new {identity_field}.",
        )
    _, output_key = normalized_output_key(config, f"Variant {name}")
    _, canonical_output_key = normalized_output_key(canonical, "Canonical config")
    require(
        output_key != canonical_output_key,
        f"Variant {name} output directory aliases the canonical result directory.",
    )


def resolve_variant_revision(
    name: str, config_path: Path, config: Mapping[str, Any]
) -> tuple[str, str]:
    revision = config.get("code_revision")
    require(
        isinstance(revision, str) and SAFE_TAG_PATTERN.fullmatch(revision),
        f"Variant {name} code_revision must be a safe Git tag name.",
    )
    completed = git_run(
        ["rev-parse", "--verify", f"refs/tags/{revision}^{{commit}}"],
        allow_failure=True,
    )
    require(
        completed.returncode == 0,
        f"Variant {name} tag {revision!r} does not exist yet.",
    )
    commit = completed.stdout.strip()
    require(
        COMMIT_PATTERN.fullmatch(commit) is not None,
        f"Variant {name} tag did not resolve to a 40-character Git commit.",
    )
    relative_config = config_path.relative_to(PROJECT_ROOT).as_posix()
    stored = git_run(["cat-file", "-e", f"{commit}:{relative_config}"], allow_failure=True)
    require(
        stored.returncode == 0,
        f"Variant {name} config is not stored in its producing tag.",
    )
    config_comparison = git_run(
        ["diff", "--quiet", commit, "--", relative_config],
        allow_failure=True,
    )
    require(
        config_comparison.returncode == 0,
        f"Variant {name} config differs from the version stored in its producing tag.",
    )

    canonical_commit = git_run(
        ["rev-parse", "--verify", f"refs/tags/{CANONICAL_TAG}^{{commit}}"]
    ).stdout.strip()
    for relative_path in SCIENTIFIC_IMPLEMENTATION_PATHS:
        comparison = git_run(
            ["diff", "--quiet", canonical_commit, commit, "--", relative_path],
            allow_failure=True,
        )
        require(
            comparison.returncode == 0,
            f"Variant {name} changes result-producing file {relative_path}; "
            "Gate 2 variants may change only their configs.",
        )

    tagged_self_check = git_run(
        ["show", f"{commit}:{SELF_CHECK_PATH.relative_to(PROJECT_ROOT).as_posix()}"],
        allow_failure=True,
    )
    require(
        tagged_self_check.returncode == 0,
        f"Variant {name} producing tag does not contain the Gate 2 self-check.",
    )
    tagged_learning = parse_self_check_text(tagged_self_check.stdout)
    require(
        all(tagged_learning.answers_present)
        and all(tagged_learning.proposals_present)
        and tagged_learning.proposal_review_passed,
        f"Variant {name} tag must contain Questions 1-8, both predictions, and "
        "the pre-run approval before training.",
    )
    return commit, tagged_learning.pre_run_snapshot_sha256


def _validated_confusion_matrix(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_row_totals: Sequence[int],
    expected_client_counts: Sequence[int],
) -> None:
    raw_matrix = record.get("confusion_matrix")
    require(
        isinstance(raw_matrix, list) and len(raw_matrix) == 10,
        f"{label}.confusion_matrix must be 10x10.",
    )
    matrix: list[list[int]] = []
    for row_index, raw_row in enumerate(raw_matrix):
        require(
            isinstance(raw_row, list) and len(raw_row) == 10,
            f"{label}.confusion_matrix row {row_index} must have 10 entries.",
        )
        row: list[int] = []
        for value in raw_row:
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{label}.confusion_matrix contains a non-negative-integer violation.",
            )
            row.append(value)
        matrix.append(row)
    row_totals = [sum(row) for row in matrix]
    require(
        row_totals == list(expected_row_totals),
        f"{label}.confusion_matrix row totals disagree with the test partitions.",
    )
    total = sum(row_totals)
    accuracy = sum(matrix[index][index] for index in range(10)) / total
    require(
        math.isclose(float(record.get("accuracy")), round(accuracy, 6), abs_tol=5e-7),
        f"{label}.accuracy disagrees with its confusion matrix.",
    )
    f1_values: list[float] = []
    for class_id in range(10):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in range(10) if row != class_id)
        false_negative = sum(matrix[class_id][column] for column in range(10) if column != class_id)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    macro_f1 = sum(f1_values) / 10
    require(
        math.isclose(float(record.get("f1_macro")), round(macro_f1, 6), abs_tol=5e-7),
        f"{label}.f1_macro disagrees with its confusion matrix.",
    )
    per_class = record.get("per_class_accuracy")
    require(
        isinstance(per_class, list) and len(per_class) == 10,
        f"{label}.per_class_accuracy must contain ten values.",
    )
    for class_id, total_for_class in enumerate(row_totals):
        expected = matrix[class_id][class_id] / total_for_class
        require(
            math.isclose(float(per_class[class_id]), round(expected, 6), abs_tol=5e-7),
            f"{label}.per_class_accuracy[{class_id}] disagrees with the matrix.",
        )

    client_rows = record.get("per_client_test_utility")
    require(
        isinstance(client_rows, list)
        and len(client_rows) == len(expected_client_counts),
        f"{label}.per_client_test_utility must contain every client once.",
    )
    weighted_accuracy = 0.0
    total_client_examples = 0
    for client_id, (raw_row, expected_count) in enumerate(
        zip(client_rows, expected_client_counts)
    ):
        require(isinstance(raw_row, Mapping), f"{label} client row must be an object.")
        require(raw_row.get("client_id") == client_id, f"{label} client IDs are wrong.")
        require(
            raw_row.get("number_of_examples") == expected_count,
            f"{label} client {client_id} test count is wrong.",
        )
        client_accuracy = float(raw_row.get("accuracy"))
        client_f1 = float(raw_row.get("f1_macro"))
        require(
            0.0 <= client_accuracy <= 1.0 and 0.0 <= client_f1 <= 1.0,
            f"{label} client utility must stay in [0, 1].",
        )
        weighted_accuracy += expected_count * client_accuracy
        total_client_examples += expected_count
    require(
        total_client_examples == total,
        f"{label} per-client test counts do not cover the test set.",
    )
    require(
        math.isclose(
            float(record.get("accuracy")),
            weighted_accuracy / total,
            abs_tol=1e-6,
        ),
        f"{label} global accuracy disagrees with weighted client accuracy.",
    )


def validate_no_nonfinite_json(value: Any, label: str = "metrics") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} contains NaN or infinity.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_no_nonfinite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_no_nonfinite_json(item, f"{label}.{key}")
        return
    raise Gate2EvidenceError(f"{label} contains unsupported type {type(value).__name__}.")


def render_beginner_summary(metrics: Mapping[str, Any]) -> str:
    central = metrics["centralized"]
    fedavg = metrics["fedavg"]
    return f"""# Week 8 Saved-Run Summary

This file was generated automatically from the saved experiment metrics.

| Method | Test accuracy | Macro F1 | Best validation step | Train + validation-loop time |
|---|---:|---:|---:|---:|
| Centralized SGD | {central['accuracy']:.2%} | {central['f1_macro']:.2%} | epoch {central['best_epoch']} | {central['training_and_validation_time_seconds']:.1f}s |
| Hand-written FedAvg | {fedavg['accuracy']:.2%} | {fedavg['f1_macro']:.2%} | round {fedavg['best_round']} | {fedavg['training_and_validation_time_seconds']:.1f}s |

- Accuracy difference (FedAvg minus centralized): {metrics['accuracy_difference_fedavg_minus_centralized'] * 100:+.2f} percentage points.
- FedAvg estimated communication: {metrics['communication_cost']:,} bytes ({metrics['communication_cost'] / (1024 ** 2):.2f} MiB) of dense tensor payload.
- The mandatory top-level accuracy and F1 belong to `{metrics['primary_method_for_mandatory_metrics']}`.
- Both paths started from checksum `{metrics['initial_model_checksum_sha256']}`.
- This is one fixed-seed, sequential CPU, IID experiment. Timing is not a real distributed-system benchmark.
- The test set was evaluated only after validation selected each method's checkpoint.
"""


def validate_generic_artifacts(
    metrics: Mapping[str, Any], config: Mapping[str, Any], output_directory: Path
) -> None:
    present = {path.name for path in output_directory.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_ARTIFACTS - present)
    require(not missing, "Variant output is missing: " + ", ".join(missing))

    initial_state = expected_initial_state(config)
    expected_parameters = sum(tensor.numel() for tensor in initial_state.values())
    expected_payload = sum(
        tensor.numel() * tensor.element_size() for tensor in initial_state.values()
    )
    require(
        metrics.get("trainable_parameters") == expected_parameters,
        "Variant trainable_parameters does not match SimpleMLP.",
    )
    require(
        metrics.get("model_payload_bytes") == expected_payload,
        "Variant model_payload_bytes does not match SimpleMLP.",
    )
    initial_checksum = checksum_model_state(initial_state)
    require(
        metrics.get("initial_model_checksum_sha256") == initial_checksum,
        "Variant initial-model checksum cannot be rebuilt from its seed.",
    )
    for field in (
        "centralized_initial_checksum_sha256",
        "fedavg_initial_checksum_sha256",
    ):
        require(
            metrics.get(field) == initial_checksum,
            f"Variant {field} does not record the common initialization.",
        )

    centralized = metrics.get("centralized")
    fedavg = metrics.get("fedavg")
    require(isinstance(centralized, Mapping), "Variant centralized record is missing.")
    require(isinstance(fedavg, Mapping), "Variant fedavg record is missing.")
    try:
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
        validate_png(output_directory / "centralized_confusion_matrix.png", (980, 980))
        validate_png(output_directory / "fedavg_confusion_matrix.png", (980, 980))
        validate_png(output_directory / "convergence_comparison.png", (1400, 560))
    except Week8VerificationError as error:
        raise Gate2EvidenceError(str(error)) from error
    require(central_checksum != initial_checksum, "Centralized variant checkpoint is untrained.")
    require(fedavg_checksum != initial_checksum, "FedAvg variant checkpoint is untrained.")
    summary = output_directory / "comparison_summary.md"
    require(
        summary.read_text(encoding="utf-8") == render_beginner_summary(metrics),
        "Variant comparison_summary.md is stale or was edited by hand.",
    )


def derive_expected_accounting(
    config: Mapping[str, Any],
    client_training_sizes: Mapping[int, int],
    payload_bytes: int,
) -> dict[str, int]:
    training_examples = int(config["expected_training_examples"])
    batch_size = int(config["batch_size"])
    local_epochs = int(config["local_epochs"])
    rounds = int(config["number_of_rounds"])
    centralized_epochs = int(config["centralized_epochs"])
    selected_clients = math.ceil(
        float(config["client_fraction"]) * int(config["number_of_clients"])
    )
    require(
        selected_clients == len(client_training_sizes),
        "Gate 2 accounting expects full participation in every round.",
    )
    return {
        "central_steps": math.ceil(training_examples / batch_size)
        * centralized_epochs,
        "fedavg_steps": sum(
            math.ceil(size / batch_size) * local_epochs
            for size in client_training_sizes.values()
        )
        * rounds,
        "exposures": training_examples * centralized_epochs,
        "communication": payload_bytes * 2 * selected_clients * rounds,
    }


def validate_variant_result(
    variant: VariantEvidence,
    canonical_metrics: Mapping[str, Any],
) -> None:
    require(variant.metrics_path is not None, "Internal error: metrics path is absent.")
    require(variant.report_path is not None, "Internal error: report path is absent.")
    metrics_path = variant.metrics_path
    report_path = variant.report_path
    expected_metrics = (
        PROJECT_ROOT / "results" / str(variant.config["output_subdirectory"]) / "metrics.json"
    ).resolve()
    require(
        metrics_path.resolve() == expected_metrics,
        f"Variant {variant.name} metrics path must be {expected_metrics.relative_to(PROJECT_ROOT)}.",
    )
    require(metrics_path.is_file(), f"Variant {variant.name} metrics are missing.")
    require(report_path.is_file(), f"Variant {variant.name} report is missing.")

    try:
        metrics, metrics_sha256 = load_and_hash_metrics(metrics_path)
        validate_no_nonfinite_json(metrics)
        resolved_path, resolved_sha256 = load_and_validate_resolved_config(
            metrics_path, metrics
        )
        derived = validate_and_derive(metrics)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise Gate2EvidenceError(
            f"Variant {variant.name} saved-result validation failed: {error}"
        ) from error

    config_relative = variant.config_path.relative_to(PROJECT_ROOT).as_posix()
    require(
        metrics.get("source_config") == config_relative,
        f"Variant {variant.name} metrics identify the wrong source config.",
    )
    resolved = load_json_object(resolved_path, "resolved variant config")
    expected_resolved = {
        **dict(variant.config),
        "source_config": config_relative,
        "code_provenance": metrics.get("code_provenance"),
    }
    require(
        dict(resolved) == expected_resolved,
        f"Variant {variant.name} resolved_config is not the exact pre-training snapshot.",
    )

    provenance = metrics.get("code_provenance")
    require(isinstance(provenance, Mapping), f"Variant {variant.name} provenance is missing.")
    require(
        provenance.get("configured_revision") == variant.config["code_revision"],
        f"Variant {variant.name} provenance names the wrong tag.",
    )
    require(
        provenance.get("resolved_commit") == variant.resolved_commit
        and provenance.get("head_commit") == variant.resolved_commit,
        f"Variant {variant.name} was not produced at its tagged commit.",
    )
    require(
        provenance.get("worktree_clean_at_start") is True,
        f"Variant {variant.name} did not record a clean worktree at training start.",
    )
    expected_dependencies = [
        config_relative,
        str(variant.config["runner"]),
        str(variant.config["environment_manifest"]),
    ]
    require(
        provenance.get("tracked_dependencies") == expected_dependencies,
        f"Variant {variant.name} provenance dependency list is incomplete.",
    )
    runtime = validate_runtime_environment(dict(variant.config), PROJECT_ROOT)
    require(
        metrics.get("runtime") == runtime,
        f"Variant {variant.name} runtime does not match the exact Month 2 environment.",
    )

    for field in (
        "training_split_checksum_sha256",
        "validation_split_checksum_sha256",
        "initial_model_checksum_sha256",
        "model_payload_bytes",
        "trainable_parameters",
    ):
        require(
            metrics.get(field) == canonical_metrics.get(field),
            f"Variant {variant.name} changed canonical invariant {field}.",
        )
    if variant.name == "local_epochs_e2":
        for field in ("client_training_partitions", "client_test_partitions"):
            require(
                metrics.get(field) == canonical_metrics.get(field),
                f"E=2 variant changed {field} even though K and seeds are unchanged.",
            )

    try:
        client_sizes, test_histogram = validate_partitions(metrics, variant.config)
    except Week8VerificationError as error:
        raise Gate2EvidenceError(f"Variant {variant.name}: {error}") from error
    test_partition_records = metrics.get("client_test_partitions")
    require(
        isinstance(test_partition_records, list),
        f"Variant {variant.name} client test partitions are missing.",
    )
    test_client_counts = [
        int(record["number_of_examples"]) for record in test_partition_records
    ]
    for method in ("centralized", "fedavg"):
        record = metrics.get(method)
        require(isinstance(record, Mapping), f"Variant {variant.name} lacks {method} results.")
        _validated_confusion_matrix(
            record,
            label=f"{variant.name}.{method}",
            expected_row_totals=test_histogram,
            expected_client_counts=test_client_counts,
        )

    expected = derive_expected_accounting(
        variant.config, client_sizes, int(metrics["model_payload_bytes"])
    )
    require(
        derived["central"]["optimizer_steps"] == expected["central_steps"],
        f"Variant {variant.name} centralized optimizer-step arithmetic is wrong.",
    )
    require(
        derived["fedavg"]["optimizer_steps"] == expected["fedavg_steps"],
        f"Variant {variant.name} FedAvg optimizer-step arithmetic is wrong.",
    )
    require(
        derived["central"]["training_examples_processed"]
        == derived["fedavg"]["training_examples_processed"]
        == expected["exposures"],
        f"Variant {variant.name} exposure accounting is wrong.",
    )
    require(
        derived["communication"] == expected["communication"],
        f"Variant {variant.name} communication arithmetic is wrong.",
    )

    validate_generic_artifacts(metrics, variant.config, metrics_path.parent)
    expected_report = render_report(
        metrics,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha256,
        resolved_config_path=resolved_path,
        resolved_config_sha256=resolved_sha256,
    )
    try:
        observed_report = report_path.read_text(encoding="utf-8")
    except OSError as error:
        raise Gate2EvidenceError(
            f"Cannot read variant {variant.name} report: {error}"
        ) from error
    require(
        observed_report == expected_report,
        f"Variant {variant.name} report is stale or contains hand-copied numbers.",
    )


def load_variant_configs(
    manifest: Mapping[str, Any], canonical: Mapping[str, Any]
) -> VariantCollection:
    raw_variants = manifest.get("variants")
    require(isinstance(raw_variants, Mapping), "Manifest variants must be a JSON object.")
    require(
        set(raw_variants) == set(VARIANT_RULES),
        "Manifest must contain exactly the K=10 and E=2 variant entries.",
    )
    variants: list[VariantEvidence] = []
    waiting: list[str] = []
    for name in VARIANT_RULES:
        record = raw_variants[name]
        require(isinstance(record, Mapping), f"Manifest variant {name} must be an object.")
        require(
            set(record) == {"config", "metrics", "report"},
            f"Manifest variant {name} must contain config, metrics, and report only.",
        )
        config_path = resolve_manifest_path(
            record.get("config"),
            label=f"{name}.config",
            allowed_root=PROJECT_ROOT / "configs",
            suffix=".json",
        )
        metrics_path = resolve_manifest_path(
            record.get("metrics"),
            label=f"{name}.metrics",
            allowed_root=PROJECT_ROOT / "results",
            suffix=".json",
        )
        report_path = resolve_manifest_path(
            record.get("report"),
            label=f"{name}.report",
            allowed_root=PROJECT_ROOT / "reports",
            suffix=".md",
        )
        if config_path is None:
            require(
                metrics_path is None and report_path is None,
                f"Manifest {name} cannot name results before its config.",
            )
            waiting.append(f"create, review, commit, and tag the {name} config")
            continue
        require(config_path.is_file(), f"Manifest config does not exist: {config_path}")
        config = load_json_object(config_path, f"{name} config")
        validate_variant_delta(name, config, canonical)
        commit, snapshot_sha256 = resolve_variant_revision(name, config_path, config)
        variants.append(
            VariantEvidence(
                name=name,
                config_path=config_path,
                metrics_path=metrics_path,
                report_path=report_path,
                config=config,
                resolved_commit=commit,
                pre_run_snapshot_sha256=snapshot_sha256,
            )
        )
        if metrics_path is None:
            require(report_path is None, f"Manifest {name} report cannot precede metrics.")
            waiting.append(f"run the tagged {name} config and record its metrics path")
        elif report_path is None:
            waiting.append(f"build and record the deterministic {name} report")

    configured = [variant for variant in variants]
    if len(configured) == 2:
        experiment_names = {variant.config["experiment_name"] for variant in configured}
        require(len(experiment_names) == 2, "Gate 2 variants repeat experiment_name.")
        output_keys = {
            normalized_output_key(variant.config, f"Variant {variant.name}")[1]
            for variant in configured
        }
        require(len(output_keys) == 2, "Gate 2 variant output directories alias each other.")
        canonical_output_key = normalized_output_key(canonical, "Canonical config")[1]
        require(
            canonical_output_key not in output_keys,
            "A Gate 2 variant would overwrite the canonical result directory.",
        )
        report_keys = {
            os.path.normcase(str(variant.report_path.resolve()))
            for variant in configured
            if variant.report_path is not None
        }
        named_reports = sum(variant.report_path is not None for variant in configured)
        require(
            len(report_keys) == named_reports,
            "Gate 2 variant report paths alias each other.",
        )
    return VariantCollection(variants=tuple(variants), waiting=tuple(waiting))


def manifest_status_check(
    manifest: Mapping[str, Any], *, collection: VariantCollection
) -> None:
    require(manifest.get("schema_version") == 1, "Unsupported Gate 2 manifest schema.")
    status = manifest.get("status")
    require(
        status in {"awaiting_student", "configs_ready", "complete"},
        "Manifest status must be awaiting_student, configs_ready, or complete.",
    )
    if status == "awaiting_student":
        require(
            not collection.variants,
            "Manifest says awaiting_student but already names variant configs.",
        )
    elif status == "configs_ready":
        require(collection.all_configs_present, "configs_ready requires both tagged configs.")
        require(
            collection.waiting,
            "configs_ready is stale because all artifacts are already named.",
        )
    else:
        require(
            collection.all_results_present and not collection.waiting,
            "complete requires all variant paths.",
        )


def progress_records_gate2_closure(review_date: str | None) -> bool:
    if review_date is None:
        return False
    text = PROGRESS_PATH.read_text(encoding="utf-8")
    dated_sections = re.finditer(
        rf"(?ms)^## {re.escape(review_date)} .*?(?=^## \d{{4}}-\d{{2}}-\d{{2}}|\Z)",
        text,
    )
    marker = re.compile(
        r"(?m)^\*\*Gate 2 status:\*\*[ \t]*(OPEN|CLOSED)[ \t]*$"
    )
    dated_decisions = [
        match.group(1)
        for section in dated_sections
        for match in marker.finditer(section.group(0))
    ]
    all_decisions = [match.group(1) for match in marker.finditer(text)]
    return (
        bool(dated_decisions)
        and dated_decisions[-1] == "CLOSED"
        and all_decisions[-1] == "CLOSED"
    )


def decide_gate_state(
    self_check: SelfCheckEvidence,
    *,
    all_artifacts_valid: bool,
    gate_checked: bool,
    progress_closed: bool,
    initial_waiting: Sequence[str] = (),
) -> tuple[str, list[str]]:
    waiting = list(initial_waiting)
    if self_check.proposal_review_date and self_check.review_date:
        require(
            date.fromisoformat(self_check.review_date)
            >= date.fromisoformat(self_check.proposal_review_date),
            "Final Gate 2 review date cannot precede the pre-run proposal review date.",
        )
    missing_answers = [
        str(index)
        for index, present in enumerate(self_check.answers_present, start=1)
        if not present
    ]
    if missing_answers:
        waiting.append("answer self-check questions " + ", ".join(missing_answers))
    missing_proposals = [
        str(index)
        for index, present in zip((9, 10), self_check.proposals_present)
        if not present
    ]
    if missing_proposals:
        waiting.append("write predictions for questions " + ", ".join(missing_proposals))
    if self_check.proposal_review_passed:
        require(
            all(self_check.answers_present) and all(self_check.proposals_present),
            "Pre-run proposal review is approved before Questions 1-8 and both "
            "predictions are present.",
        )
    else:
        waiting.append("obtain pre-run approval for both proposed config changes")
    missing_saved = [
        str(index)
        for index, present in zip((9, 10), self_check.saved_results_present)
        if not present
    ]
    if missing_saved:
        waiting.append("interpret saved results for questions " + ", ".join(missing_saved))
    if not self_check.review_passed:
        waiting.append("obtain a dated human conceptual review marked PASS")

    if self_check.review_passed:
        require(
            self_check.all_student_text_present
            and self_check.proposal_review_passed
            and all_artifacts_valid,
            "The self-check is marked PASS before all student text and variant evidence exist.",
        )
    if gate_checked:
        require(
            self_check.review_passed
            and self_check.proposal_review_passed
            and self_check.all_student_text_present
            and all_artifacts_valid,
            "README marks Gate 2 complete before its required evidence passed.",
        )
        require(
            progress_closed,
            "README marks Gate 2 complete without the exact dated PROGRESS.md "
            "closure marker.",
        )
        return "PASS", ["Gate 2 is closed with canonical, variant, review, and progress evidence."]
    if waiting:
        return "WAITING", waiting
    return "READY", [
        "All Gate 2 evidence is valid; update README.md and PROGRESS.md to close the gate."
    ]


def verify(manifest_path: Path) -> tuple[str, list[str]]:
    require(
        manifest_path.resolve() == DEFAULT_MANIFEST.resolve(),
        "Only reports/gate2_variant_evidence.json is authoritative for Gate 2.",
    )
    manifest = load_json_object(manifest_path, "Gate 2 evidence manifest")
    run_canonical_verifier()

    canonical_path = resolve_manifest_path(
        manifest.get("canonical_config"),
        label="canonical_config",
        allowed_root=PROJECT_ROOT / "configs",
        suffix=".json",
    )
    self_check_path = resolve_manifest_path(
        manifest.get("student_self_check"),
        label="student_self_check",
        allowed_root=PROJECT_ROOT / "reports",
        suffix=".md",
    )
    require(canonical_path is not None and canonical_path.is_file(), "Canonical config is missing.")
    require(self_check_path is not None and self_check_path.is_file(), "Student self-check is missing.")
    require(
        canonical_path.resolve() == CANONICAL_CONFIG_PATH.resolve(),
        "Manifest canonical_config must identify the protected Week 8 config.",
    )
    require(
        self_check_path.resolve() == SELF_CHECK_PATH.resolve(),
        "Manifest student_self_check must identify reports/gate2_self_check.md.",
    )
    canonical = load_json_object(canonical_path, "canonical config")
    self_check = parse_self_check(self_check_path)
    gate_checked = read_gate2_checkbox()
    canonical_metrics = load_json_object(CANONICAL_METRICS, "canonical metrics")

    collection = load_variant_configs(manifest, canonical)
    manifest_status_check(manifest, collection=collection)
    if collection.variants:
        tagged_snapshots = {
            variant.pre_run_snapshot_sha256 for variant in collection.variants
        }
        require(
            tagged_snapshots == {self_check.pre_run_snapshot_sha256},
            "The current answers, predictions, or pre-run review differ from the "
            "snapshot approved in the producing tag(s).",
        )
    waiting = list(collection.waiting)
    complete_variants = [
        variant
        for variant in collection.variants
        if variant.metrics_path is not None and variant.report_path is not None
    ]
    for variant in complete_variants:
        validate_variant_result(variant, canonical_metrics)
    if manifest.get("status") == "complete":
        require_committed_current(
            [manifest_path, *(variant.report_path for variant in complete_variants if variant.report_path)],
            "Complete variant evidence",
        )

    all_artifacts_valid = collection.all_results_present and len(complete_variants) == 2
    progress_closed = progress_records_gate2_closure(self_check.review_date)
    status, details = decide_gate_state(
        self_check,
        all_artifacts_valid=all_artifacts_valid,
        gate_checked=gate_checked,
        progress_closed=progress_closed,
        initial_waiting=waiting,
    )
    if self_check.review_passed:
        require_committed_current(
            [self_check_path, manifest_path], "Final Gate 2 review evidence"
        )
    if status in {"READY", "PASS"}:
        require_committed_current(
            AUTHORITATIVE_EVIDENCE_TOOL_PATHS,
            "Authoritative Gate 2 verification tool",
        )
    if status == "PASS":
        require_committed_current(
            [README_PATH, PROGRESS_PATH], "Gate 2 closure evidence"
        )
    return status, details


def main() -> int:
    args = parse_args()
    try:
        status, details = verify(args.manifest.resolve())
        if status == "WAITING":
            print("WAITING: Gate 2 remains open; valid required evidence is still missing.")
            for detail in details:
                print(f"- {detail}")
            print(
                "The canonical Week 8 result remains verified. No Non-IID, FedProx, "
                "or Federated Unlearning work is authorized yet."
            )
            return 0 if args.allow_waiting else 2
        print(f"{status}: {details[0]}")
        return 0
    except Gate2EvidenceError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            "FAIL: Unexpected readiness-check error; this is not evidence that Gate 2 "
            f"passed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
