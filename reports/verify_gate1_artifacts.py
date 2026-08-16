"""Verify the technical evidence for Gate 1 without rerunning model training.

This checker does not pass Gate 1 for the student. It proves only that the
Month 1 files, recorded runs, report, and literature artifacts are present and
internally consistent. The student's own explanation is a separate gate
condition in the thesis roadmap.
"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXPERIMENT_FIELDS = {
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
}
REQUIRED_REPRODUCIBILITY_FIELDS = {
    "code_revision",
    "dataset_version",
    "device",
    "environment_manifest",
    "runner",
    "strict_environment",
}
CONFIG_TO_RESULT_EXCEPTIONS = {"accuracy", "f1"}
LITERATURE_COLUMNS = [
    "Paper",
    "Problem",
    "Method",
    "Assumption",
    "Dataset",
    "Metric",
    "Result",
    "Limitation",
    "Possible Gap",
]
SIX_QUESTIONS = [
    "Problem",
    "Gap",
    "Idea",
    "Assumption",
    "Evaluation",
    "Limitation",
]

RUN_SPECS = (
    {
        "config": "configs/month1_week1_mnist_logistic_regression.json",
        "artifacts": (
            "class_distribution.csv",
            "confusion_matrix.png",
            "metrics.json",
            "sample_digits.png",
        ),
    },
    {
        "config": "configs/month1_week2_mnist_mlp.json",
        "artifacts": (
            "best_model.pt",
            "confusion_matrix.png",
            "learning_curve.png",
            "metrics.json",
        ),
    },
    {
        "config": "configs/month1_week3_mnist_cnn.json",
        "artifacts": (
            "best_model.pt",
            "confusion_matrix.png",
            "feature_maps.png",
            "learning_curve.png",
            "metrics.json",
        ),
    },
    {
        "config": "configs/month1_week3_cifar10_cnn.json",
        "artifacts": (
            "best_model.pt",
            "confusion_matrix.png",
            "feature_maps.png",
            "learning_curve.png",
            "metrics.json",
        ),
    },
)

VERSIONED_MONTH1_PATHS = (
    ".gitattributes",
    ".gitignore",
    "PROGRESS.md",
    "README.md",
    "requirements.txt",
    "configs/month1_week1_mnist_logistic_regression.json",
    "configs/month1_week2_mnist_mlp.json",
    "configs/month1_week3_mnist_cnn.json",
    "configs/month1_week3_cifar10_cnn.json",
    "environment/month1_cpu_requirements.txt",
    "environment/month1_cpu_runtime.json",
    "experiments/experiment_utils.py",
    "experiments/month1_week1_mnist_baseline.py",
    "experiments/month1_week2_mnist_mlp.py",
    "experiments/month1_week3_cnn.py",
    "literature/literature_matrix.md",
    "literature/sisa_2021_six_questions.md",
    "models/simple_mlp.py",
    "models/small_cnn.py",
    "output/pdf/month1_experiment_report.pdf",
    "reports/build_month1_report.py",
    "reports/gate1_evidence.md",
    "reports/gate1_self_check.md",
    "reports/gate1_study_guide.md",
    "reports/verify_gate1_artifacts.py",
    "reports/verify_month1_report.py",
)


class GateEvidenceError(AssertionError):
    """Raised when a required Gate 1 artifact is absent or inconsistent."""


class EvidenceRecorder:
    """Collect readable pass messages while failing immediately on a gap."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise GateEvidenceError(message)

    def pass_check(self, message: str) -> None:
        self.messages.append(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalized_relative_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_run(spec: dict, evidence: EvidenceRecorder) -> dict:
    config_path = PROJECT_ROOT / spec["config"]
    evidence.require(config_path.is_file(), f"Missing config: {config_path}")
    config = load_json(config_path)

    missing_config_fields = (
        REQUIRED_EXPERIMENT_FIELDS | REQUIRED_REPRODUCIBILITY_FIELDS
    ) - set(config)
    evidence.require(
        not missing_config_fields,
        f"{config_path.name} lacks required fields: {sorted(missing_config_fields)}",
    )
    evidence.require(
        isinstance(config["random_seed"], int),
        f"{config_path.name} must record an integer random_seed",
    )
    evidence.require(
        config["device"] == "cpu",
        f"{config_path.name} must explicitly reproduce the verified CPU run",
    )
    evidence.require(
        config["strict_environment"] is True,
        f"{config_path.name} must enable strict environment validation",
    )
    for referenced_field in ("runner", "environment_manifest"):
        referenced_path = PROJECT_ROOT / config[referenced_field]
        evidence.require(
            referenced_path.is_file(),
            f"{config_path.name} points to a missing {referenced_field}: {referenced_path}",
        )

    output_directory = PROJECT_ROOT / "results" / config["output_subdirectory"]
    evidence.require(
        output_directory.is_dir(),
        f"Missing result directory for {config_path.name}: {output_directory}",
    )
    for artifact_name in spec["artifacts"]:
        artifact_path = output_directory / artifact_name
        evidence.require(
            artifact_path.is_file() and artifact_path.stat().st_size > 0,
            f"Missing or empty artifact: {artifact_path}",
        )

    metrics_path = output_directory / "metrics.json"
    metrics = load_json(metrics_path)
    missing_metric_fields = (
        REQUIRED_EXPERIMENT_FIELDS | REQUIRED_REPRODUCIBILITY_FIELDS
    ) - set(metrics)
    evidence.require(
        not missing_metric_fields,
        f"{metrics_path} lacks required fields: {sorted(missing_metric_fields)}",
    )

    for key, configured_value in config.items():
        if key in CONFIG_TO_RESULT_EXCEPTIONS:
            continue
        evidence.require(
            metrics.get(key) == configured_value,
            f"{metrics_path}: {key!r} differs from its source config",
        )

    recorded_source = normalized_relative_path(metrics.get("source_config", ""))
    evidence.require(
        recorded_source == normalized_relative_path(spec["config"]),
        f"{metrics_path} records the wrong source_config: {recorded_source!r}",
    )
    for metric_name in ("accuracy", "f1"):
        metric_value = metrics[metric_name]
        evidence.require(
            isinstance(metric_value, (int, float)) and 0.0 <= metric_value <= 1.0,
            f"{metrics_path}: {metric_name} must be a measured value in [0, 1]",
        )

    environment_manifest = load_json(PROJECT_ROOT / config["environment_manifest"])
    runtime = metrics.get("runtime", {})
    evidence.require(
        runtime.get("python_version") == environment_manifest["python_version"],
        f"{metrics_path} does not record the pinned Python version",
    )
    evidence.require(
        runtime.get("device") == config["device"],
        f"{metrics_path} runtime device differs from its config",
    )
    evidence.require(
        runtime.get("packages") == environment_manifest["packages"],
        f"{metrics_path} package snapshot differs from its environment manifest",
    )

    class_names = metrics.get("class_names", [])
    confusion = metrics.get("confusion_matrix", [])
    evidence.require(
        len(class_names) == 10
        and len(confusion) == len(class_names)
        and all(len(row) == len(class_names) for row in confusion),
        f"{metrics_path} must store a 10-by-10 numeric confusion matrix",
    )
    expected_test_examples = metrics.get("test_examples", metrics.get("split", {}).get("test"))
    evidence.require(
        sum(sum(row) for row in confusion) == expected_test_examples,
        f"{metrics_path} confusion counts do not sum to the test-set size",
    )
    if "feature_maps.png" in spec["artifacts"]:
        feature_example = metrics.get("feature_map_example", {})
        evidence.require(
            feature_example.get("true_label") in class_names
            and feature_example.get("predicted_label") in class_names,
            f"{metrics_path} does not identify the saved feature-map example",
        )

    if "dataset_archive" in config:
        archive = config["dataset_archive"]
        archive_path = PROJECT_ROOT / "data" / config["data_subdirectory"] / archive["filename"]
        evidence.require(archive_path.is_file(), f"Missing dataset archive: {archive_path}")
        evidence.require(
            archive_path.stat().st_size == archive["size_bytes"],
            f"Dataset archive size differs from {config_path.name}",
        )
        evidence.require(
            file_md5(archive_path).lower() == archive["md5"].lower(),
            f"Dataset archive MD5 differs from {config_path.name}",
        )

    evidence.require(
        metrics["number_of_clients"] == 1
        and metrics["number_of_rounds"] == 0
        and metrics["target_client"] is None
        and metrics["communication_cost"] == 0
        and metrics["unlearning_time"] == 0.0,
        f"{metrics_path} must explicitly identify this as centralized-only work",
    )

    evidence.pass_check(
        f"{config['experiment_name']}: config, required fields, and artifacts agree "
        f"(accuracy={metrics['accuracy']:.4f}, macro F1={metrics['f1']:.4f})"
    )
    return metrics


def verify_source_files(evidence: EvidenceRecorder) -> None:
    source_paths = (
        "experiments/experiment_utils.py",
        "experiments/month1_week1_mnist_baseline.py",
        "experiments/month1_week2_mnist_mlp.py",
        "experiments/month1_week3_cnn.py",
        "models/simple_mlp.py",
        "models/small_cnn.py",
        "reports/build_month1_report.py",
        "reports/verify_month1_report.py",
        "reports/verify_gate1_artifacts.py",
    )
    for relative_path in source_paths:
        path = PROJECT_ROOT / relative_path
        evidence.require(path.is_file(), f"Missing source file: {path}")
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    evidence.pass_check(f"All {len(source_paths)} Gate 1 Python sources compile")


def verify_checkpoints(metrics_by_name: dict[str, dict], evidence: EvidenceRecorder) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    from models.simple_mlp import SimpleMLP
    from models.small_cnn import SmallCNN

    checkpoint_specs = (
        (
            "month1_week2_mnist_mlp",
            "month1_week2",
            lambda metrics: SimpleMLP(hidden_units=metrics["hidden_units"]),
            torch.zeros(2, 1, 28, 28),
        ),
        (
            "month1_week3_mnist_cnn",
            "month1_week3_mnist_cnn",
            lambda metrics: SmallCNN(
                input_channels=metrics["input_channels"],
                convolution_channels=tuple(metrics["convolution_channels"]),
                hidden_units=metrics["hidden_units"],
                dropout=metrics["dropout"],
            ),
            torch.zeros(2, 1, 28, 28),
        ),
        (
            "month1_week3_cifar10_cnn",
            "month1_week3_cifar10_cnn",
            lambda metrics: SmallCNN(
                input_channels=metrics["input_channels"],
                convolution_channels=tuple(metrics["convolution_channels"]),
                hidden_units=metrics["hidden_units"],
                dropout=metrics["dropout"],
            ),
            torch.zeros(2, 3, 32, 32),
        ),
    )

    for experiment_name, output_subdirectory, build_model, example_batch in checkpoint_specs:
        metrics = metrics_by_name[experiment_name]
        checkpoint_path = PROJECT_ROOT / "results" / output_subdirectory / "best_model.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(metrics)
        model.load_state_dict(state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        evidence.require(
            parameter_count == metrics["trainable_parameters"],
            f"{checkpoint_path} parameter count differs from metrics.json",
        )
        model.eval()
        with torch.no_grad():
            output = model(example_batch)
        evidence.require(
            tuple(output.shape) == (2, 10),
            f"{checkpoint_path} does not produce ten class scores per example",
        )
    evidence.pass_check("All three neural checkpoints load strictly and pass shape checks")


def verify_environment_lock(evidence: EvidenceRecorder) -> None:
    manifest_path = PROJECT_ROOT / "environment" / "month1_cpu_runtime.json"
    manifest = load_json(manifest_path)
    lock_path = PROJECT_ROOT / manifest["requirements_lock"]
    evidence.require(lock_path.is_file(), f"Missing environment lock: {lock_path}")
    locked_versions = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package_name, package_version = stripped.split("==", maxsplit=1)
        locked_versions[package_name] = package_version
    evidence.require(
        locked_versions == manifest["packages"],
        "Environment manifest package versions differ from the exact requirements lock",
    )
    evidence.pass_check(
        f"Python {manifest['python_version']} CPU manifest and {len(locked_versions)} exact package versions agree"
    )


def verify_literature(evidence: EvidenceRecorder) -> None:
    note_path = PROJECT_ROOT / "literature" / "sisa_2021_six_questions.md"
    matrix_path = PROJECT_ROOT / "literature" / "literature_matrix.md"
    evidence.require(note_path.is_file(), f"Missing SISA note: {note_path}")
    evidence.require(matrix_path.is_file(), f"Missing literature matrix: {matrix_path}")

    note_text = note_path.read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+\d+\.\s+(.+?)\s*$", note_text, flags=re.MULTILINE)
    evidence.require(
        headings == SIX_QUESTIONS,
        f"SISA note headings must be exactly {SIX_QUESTIONS}; found {headings}",
    )

    matrix_text = matrix_path.read_text(encoding="utf-8")
    header_match = re.search(r"^\|\s*Paper\s*\|.*Possible Gap\s*\|\s*$", matrix_text, re.MULTILINE)
    evidence.require(header_match is not None, "Literature matrix header row is missing")
    header_cells = [cell.strip() for cell in header_match.group(0).strip("|").split("|")]
    evidence.require(
        header_cells == LITERATURE_COLUMNS,
        f"Literature matrix columns differ from the plan: {header_cells}",
    )
    evidence.require(
        "Bourtoule et al." in matrix_text and "SISA" in matrix_text,
        "Literature matrix does not contain the SISA paper row",
    )
    evidence.pass_check("SISA note uses exactly six questions and the matrix has all required columns")


def verify_report(metrics_by_name: dict[str, dict], evidence: EvidenceRecorder) -> None:
    report_path = PROJECT_ROOT / "output" / "pdf" / "month1_experiment_report.pdf"
    evidence.require(report_path.is_file(), f"Missing Month 1 report: {report_path}")
    reader = PdfReader(str(report_path))
    evidence.require(
        5 <= len(reader.pages) <= 8,
        f"Month 1 report must contain 5–8 pages; found {len(reader.pages)}",
    )
    all_text_parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        evidence.require(
            len(page_text.strip()) >= 80,
            f"Report page {page_number} contains too little extractable text",
        )
        evidence.require(
            f"Page {page_number} of {len(reader.pages)}" in page_text,
            f"Report page {page_number} lacks the correct page label",
        )
        all_text_parts.append(page_text)

    all_text = "\n".join(all_text_parts)
    for metrics in metrics_by_name.values():
        displayed_accuracy = f"{metrics['accuracy'] * 100:.2f}%"
        evidence.require(
            displayed_accuracy in all_text,
            f"Report does not contain measured accuracy {displayed_accuracy}",
        )
    evidence.require("SISA" in all_text, "Report does not contain the SISA concept review")
    evidence.require(
        "Gate 1 status" in all_text,
        "Report does not explain that the student Gate 1 condition remains open",
    )
    cifar_metrics = metrics_by_name["month1_week3_cifar10_cnn"]
    confusion = cifar_metrics["confusion_matrix"]
    largest_error_count = max(
        confusion[row][column]
        for row in range(len(confusion))
        for column in range(len(confusion))
        if row != column
    )
    evidence.require(
        str(largest_error_count) in all_text,
        "Report does not contain the largest confusion count from metrics.json",
    )
    report_source = (PROJECT_ROOT / "reports" / "build_month1_report.py").read_text(
        encoding="utf-8"
    )
    for stale_literal in ("92.19%", "96.80%", "98.60%", "71.38%", "233 of 1,000"):
        evidence.require(
            stale_literal not in report_source,
            f"Report source still hard-codes measured text: {stale_literal}",
        )
    evidence.pass_check(
        f"Month 1 report has {len(reader.pages)} pages and contains all four measured accuracies"
    )


def verify_repository_status(evidence: EvidenceRecorder) -> None:
    readme_path = PROJECT_ROOT / "README.md"
    progress_path = PROJECT_ROOT / "PROGRESS.md"
    guide_path = PROJECT_ROOT / "reports" / "gate1_study_guide.md"
    evidence_path = PROJECT_ROOT / "reports" / "gate1_evidence.md"
    check_path = PROJECT_ROOT / "reports" / "gate1_self_check.md"
    gitignore_path = PROJECT_ROOT / ".gitignore"

    for path in (
        readme_path,
        progress_path,
        guide_path,
        evidence_path,
        check_path,
        gitignore_path,
    ):
        evidence.require(path.is_file(), f"Missing Gate 1 documentation: {path}")
    evidence.require(
        (PROJECT_ROOT.parent / ".git").exists() or (PROJECT_ROOT / ".git").exists(),
        "The thesis workspace is not a Git repository",
    )

    readme = readme_path.read_text(encoding="utf-8")
    for week in range(1, 5):
        evidence.require(f"- [x] Week {week}:" in readme, f"README does not mark Week {week} complete")
    evidence.require(
        "- [ ] Gate 1 (Month 1):" in readme,
        "Gate 1 must remain unchecked until the student's explanation is reviewed",
    )

    gitignore = gitignore_path.read_text(encoding="utf-8")
    evidence.require("data/*" in gitignore, "data/ is not ignored as required")
    evidence.require("results/*" in gitignore, "results/ is not ignored as required")

    status = run_git("status", "--porcelain", "--untracked-files=all")
    evidence.require(
        status == "",
        "Gate 1 evidence is not fully committed; git status is:\n" + status,
    )
    for relative_path in VERSIONED_MONTH1_PATHS:
        try:
            run_git("ls-files", "--error-unmatch", relative_path)
        except subprocess.CalledProcessError as error:
            raise GateEvidenceError(
                f"Required Month 1 artifact is not tracked by Git: {relative_path}"
            ) from error

    revisions = {
        load_json(PROJECT_ROOT / spec["config"])["code_revision"] for spec in RUN_SPECS
    }
    evidence.require(
        len(revisions) == 1,
        f"Month 1 configs name inconsistent code revisions: {sorted(revisions)}",
    )
    revision = revisions.pop()
    evidence.require(
        run_git("rev-parse", f"{revision}^{{commit}}") == run_git("rev-parse", "HEAD"),
        f"Code revision tag {revision!r} does not identify the committed Gate 1 evidence",
    )
    evidence.pass_check(
        f"All Month 1 evidence is committed, the tree is clean, and tag {revision!r} identifies HEAD"
    )


def main() -> None:
    evidence = EvidenceRecorder()
    metrics_by_name: dict[str, dict] = {}

    for spec in RUN_SPECS:
        metrics = verify_run(spec, evidence)
        metrics_by_name[metrics["experiment_name"]] = metrics
    verify_source_files(evidence)
    verify_checkpoints(metrics_by_name, evidence)
    verify_environment_lock(evidence)
    verify_literature(evidence)
    verify_report(metrics_by_name, evidence)
    verify_repository_status(evidence)

    print("Gate 1 technical evidence verification passed")
    for message in evidence.messages:
        print(f"  [PASS] {message}")
    print()
    print("Remaining condition: the student must answer the Gate 1 self-check in their own words.")
    print("This script intentionally does not check or close that human-understanding condition.")


if __name__ == "__main__":
    main()
