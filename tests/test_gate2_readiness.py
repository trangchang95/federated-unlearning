"""Fast checks for the Gate 2 evidence state machine.

These tests do not load MNIST, train a model, or create variant evidence.  They
exercise the student-form parser and the controlled-config rules.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reports.verify_gate2_readiness as readiness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)
SELF_CHECK = PROJECT_ROOT / "reports" / "gate2_self_check.md"
MANIFEST = PROJECT_ROOT / "reports" / "gate2_variant_evidence.json"


class Gate2ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))

    def variant(self, **changes: object) -> dict:
        config = copy.deepcopy(self.canonical)
        config.update(
            {
                "experiment_name": "gate2_variant",
                "output_subdirectory": "gate2_variant_output",
                "code_revision": "gate2-variants",
                **changes,
            }
        )
        return config

    def complete_self_check(self, **changes: object) -> readiness.SelfCheckEvidence:
        values = {
            "answers_present": (True,) * 8,
            "proposals_present": (True, True),
            "saved_results_present": (True, True),
            "proposal_review_date": "2026-08-18",
            "proposal_review_result": "APPROVED TO RUN",
            "review_date": "2026-08-18",
            "review_result": "PASS",
            "pre_run_snapshot_sha256": "a" * 64,
        }
        values.update(changes)
        return readiness.SelfCheckEvidence(**values)

    def approved_pre_run_text(self) -> str:
        text = SELF_CHECK.read_text(encoding="utf-8")
        for question in range(1, 9):
            text = text.replace(
                "**Answer:**\n\n",
                f"**Answer:** Question {question} has a substantive student explanation.\n\n",
                1,
            )
        for question in (9, 10):
            text = text.replace(
                "**Proposed changes and prediction:**\n\n",
                "**Proposed changes and prediction:** "
                f"Question {question} predicts the controlled arithmetic before running.\n\n",
                1,
            )
        text = text.replace(
            "**Review date:** NOT YET REVIEWED",
            "**Review date:** 2026-08-18",
            1,
        )
        text = text.replace(
            "**Result:** NOT YET REVIEWED",
            "**Result:** APPROVED TO RUN",
            1,
        )
        return text

    def test_current_self_check_is_unambiguously_waiting(self) -> None:
        evidence = readiness.parse_self_check(SELF_CHECK)
        self.assertEqual(evidence.answers_present, (False,) * 8)
        self.assertEqual(evidence.proposals_present, (False, False))
        self.assertEqual(evidence.saved_results_present, (False, False))
        self.assertIsNone(evidence.proposal_review_date)
        self.assertEqual(evidence.proposal_review_result, "NOT YET REVIEWED")
        self.assertIsNone(evidence.review_date)
        self.assertEqual(evidence.review_result, "NOT YET REVIEWED")

    def test_numbered_text_inside_an_answer_does_not_hide_the_next_question(self) -> None:
        text = SELF_CHECK.read_text(encoding="utf-8")
        text = text.replace(
            "   **Answer:**",
            "   **Answer:** The client takes these steps.\n\n"
            "   2. This is a numbered detail inside Question 1.",
            1,
        )
        evidence = readiness.parse_self_check_text(text)
        self.assertTrue(evidence.answers_present[0])
        self.assertFalse(evidence.answers_present[1])

    def test_pre_run_snapshot_changes_if_an_approved_prediction_changes(self) -> None:
        approved = self.approved_pre_run_text()
        first = readiness.parse_self_check_text(approved)
        self.assertTrue(first.proposal_review_passed)
        changed = approved.replace(
            "predicts the controlled arithmetic",
            "predicts different arithmetic",
            1,
        )
        second = readiness.parse_self_check_text(changed)
        self.assertNotEqual(
            first.pre_run_snapshot_sha256, second.pre_run_snapshot_sha256
        )

    def test_duplicate_review_markers_are_rejected(self) -> None:
        text = SELF_CHECK.read_text(encoding="utf-8").replace(
            "**Review date:** NOT YET REVIEWED",
            "**Review date:** NOT YET REVIEWED\n\n"
            "**Review date:** NOT YET REVIEWED",
            1,
        )
        with self.assertRaisesRegex(readiness.Gate2EvidenceError, "exactly one"):
            readiness.parse_self_check_text(text)

    def test_current_manifest_names_no_student_evidence(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "awaiting_student")
        for record in manifest["variants"].values():
            self.assertEqual(record, {"config": None, "metrics": None, "report": None})

    def test_k10_changes_only_the_controlled_client_count(self) -> None:
        config = self.variant(number_of_clients=10)
        readiness.validate_variant_delta(
            "number_of_clients_k10", config, self.canonical
        )

    def test_e2_requires_a_matched_ten_epoch_central_budget(self) -> None:
        config = self.variant(local_epochs=2, centralized_epochs=10)
        readiness.validate_variant_delta("local_epochs_e2", config, self.canonical)
        config["centralized_epochs"] = 5
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError,
            "centralized_epochs must equal number_of_rounds",
        ):
            readiness.validate_variant_delta(
                "local_epochs_e2", config, self.canonical
            )

    def test_unrelated_hyperparameter_change_is_rejected(self) -> None:
        config = self.variant(number_of_clients=10, learning_rate=0.05)
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError, "changes unrelated fields: learning_rate"
        ):
            readiness.validate_variant_delta(
                "number_of_clients_k10", config, self.canonical
            )

    def test_results_root_cannot_be_used_as_variant_output(self) -> None:
        config = self.variant(number_of_clients=10, output_subdirectory=".")
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError, "distinct folder inside results"
        ):
            readiness.validate_variant_delta(
                "number_of_clients_k10", config, self.canonical
            )

    def test_path_alias_cannot_overwrite_the_canonical_output(self) -> None:
        canonical_output = self.canonical["output_subdirectory"]
        config = self.variant(
            number_of_clients=10,
            output_subdirectory=f"{canonical_output}/.",
        )
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError, "aliases the canonical"
        ):
            readiness.validate_variant_delta(
                "number_of_clients_k10", config, self.canonical
            )

    def test_equivalent_variant_output_spellings_have_one_normalized_key(self) -> None:
        first = self.variant(number_of_clients=10, output_subdirectory="gate2/foo")
        second = self.variant(number_of_clients=10, output_subdirectory="gate2/foo/.")
        self.assertEqual(
            readiness.normalized_output_key(first, "first")[1],
            readiness.normalized_output_key(second, "second")[1],
        )

    def test_manifest_path_cannot_escape_its_allowed_directory(self) -> None:
        with self.assertRaisesRegex(readiness.Gate2EvidenceError, "must not contain"):
            readiness.resolve_manifest_path(
                "configs/../README.md",
                label="test path",
                allowed_root=PROJECT_ROOT / "configs",
            )

    def test_only_the_versioned_manifest_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "alternate.json"
            alternate.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                readiness.Gate2EvidenceError, "Only reports/gate2_variant_evidence"
            ):
                readiness.verify(alternate)

    def test_state_machine_rejects_a_premature_final_pass(self) -> None:
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError, "marked PASS before all"
        ):
            readiness.decide_gate_state(
                self.complete_self_check(),
                all_artifacts_valid=False,
                gate_checked=False,
                progress_closed=False,
            )

    def test_state_machine_distinguishes_waiting_ready_and_pass(self) -> None:
        blank = readiness.parse_self_check(SELF_CHECK)
        status, details = readiness.decide_gate_state(
            blank,
            all_artifacts_valid=False,
            gate_checked=False,
            progress_closed=False,
            initial_waiting=("create the variants",),
        )
        self.assertEqual(status, "WAITING")
        self.assertIn("create the variants", details)

        complete = self.complete_self_check()
        status, _ = readiness.decide_gate_state(
            complete,
            all_artifacts_valid=True,
            gate_checked=False,
            progress_closed=False,
        )
        self.assertEqual(status, "READY")
        with self.assertRaisesRegex(readiness.Gate2EvidenceError, "closure marker"):
            readiness.decide_gate_state(
                complete,
                all_artifacts_valid=True,
                gate_checked=True,
                progress_closed=False,
            )
        status, _ = readiness.decide_gate_state(
            complete,
            all_artifacts_valid=True,
            gate_checked=True,
            progress_closed=True,
        )
        self.assertEqual(status, "PASS")

    def test_final_review_cannot_predate_proposal_approval(self) -> None:
        with self.assertRaisesRegex(
            readiness.Gate2EvidenceError, "cannot precede"
        ):
            readiness.decide_gate_state(
                self.complete_self_check(review_date="2026-08-17"),
                all_artifacts_valid=True,
                gate_checked=False,
                progress_closed=False,
            )

    @patch("reports.verify_gate2_readiness.git_run")
    def test_tagged_pre_run_approval_is_parsed_and_hashed(self, git_run) -> None:
        approved = self.approved_pre_run_text()

        def fake_git(arguments, *, allow_failure=False):
            del allow_failure
            if arguments[:2] == ["rev-parse", "--verify"]:
                commit = "b" * 40 if "month2-week8" in arguments[2] else "a" * 40
                return subprocess.CompletedProcess(arguments, 0, stdout=commit + "\n", stderr="")
            if arguments[0] == "show":
                return subprocess.CompletedProcess(arguments, 0, stdout=approved, stderr="")
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

        git_run.side_effect = fake_git
        commit, snapshot = readiness.resolve_variant_revision(
            "number_of_clients_k10",
            CANONICAL_CONFIG,
            self.variant(number_of_clients=10),
        )
        self.assertEqual(commit, "a" * 40)
        self.assertEqual(
            snapshot,
            readiness.parse_self_check_text(approved).pre_run_snapshot_sha256,
        )

    def test_allow_waiting_changes_only_the_waiting_exit_code(self) -> None:
        for allow_waiting, expected in ((False, 2), (True, 0)):
            with self.subTest(allow_waiting=allow_waiting), patch.object(
                readiness,
                "parse_args",
                return_value=argparse.Namespace(
                    manifest=readiness.DEFAULT_MANIFEST,
                    allow_waiting=allow_waiting,
                ),
            ), patch.object(
                readiness,
                "verify",
                return_value=("WAITING", ["student evidence is missing"]),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(readiness.main(), expected)

    def test_placeholder_wrappers_do_not_count_as_student_answers(self) -> None:
        for value in (
            "<!-- TODO -->",
            "<p>TODO: answer later</p>",
            "[TODO]",
            "TODO: answer later",
            "x",
            "answer goes here",
        ):
            with self.subTest(value=value):
                self.assertFalse(readiness._content_present(value))
        self.assertTrue(
            readiness._content_present("Optimizer step changes the model weights.")
        )

    def test_readme_gate_checkbox_is_currently_open(self) -> None:
        self.assertFalse(readiness.read_gate2_checkbox())

    def test_checked_readme_marker_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text(
                "## Status\n\n- [x] Gate 2 (Month 2): FedAvg from scratch\n",
                encoding="utf-8",
            )
            with patch.object(readiness, "README_PATH", readme):
                self.assertTrue(readiness.read_gate2_checkbox())

    def test_progress_needs_an_explicit_gate_closure_statement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "PROGRESS.md"
            progress.write_text(
                "## 2026-08-18 — Week 8 complete\n\n"
                "Gate 2 remains open; the technical experiment is complete.\n",
                encoding="utf-8",
            )
            with patch.object(readiness, "PROGRESS_PATH", progress):
                self.assertFalse(
                    readiness.progress_records_gate2_closure("2026-08-18")
                )
            progress.write_text(
                "## 2026-08-18 — Gate review\n\nGate 2 passed after review.\n",
                encoding="utf-8",
            )
            with patch.object(readiness, "PROGRESS_PATH", progress):
                self.assertFalse(
                    readiness.progress_records_gate2_closure("2026-08-18")
                )
            progress.write_text(
                "## 2026-08-18 — Gate review\n\n"
                "**Gate 2 status:** CLOSED\n",
                encoding="utf-8",
            )
            with patch.object(readiness, "PROGRESS_PATH", progress):
                self.assertTrue(
                    readiness.progress_records_gate2_closure("2026-08-18")
                )
            progress.write_text(
                "## 2026-08-18 — Gate review\n\n"
                "**Gate 2 status:** CLOSED\n\n"
                "## 2026-08-19 — Correction\n\n"
                "**Gate 2 status:** OPEN\n",
                encoding="utf-8",
            )
            with patch.object(readiness, "PROGRESS_PATH", progress):
                self.assertFalse(
                    readiness.progress_records_gate2_closure("2026-08-18")
                )

    def test_accounting_is_derived_from_each_variant_config(self) -> None:
        k10 = self.variant(number_of_clients=10)
        k10_counts = {client_id: 5_100 for client_id in range(10)}
        self.assertEqual(
            readiness.derive_expected_accounting(k10, k10_counts, 407_080),
            {
                "central_steps": 1_995,
                "fedavg_steps": 2_000,
                "exposures": 255_000,
                "communication": 40_708_000,
            },
        )
        e2 = self.variant(local_epochs=2, centralized_epochs=10)
        e2_counts = {client_id: 10_200 for client_id in range(5)}
        self.assertEqual(
            readiness.derive_expected_accounting(e2, e2_counts, 407_080),
            {
                "central_steps": 3_990,
                "fedavg_steps": 4_000,
                "exposures": 510_000,
                "communication": 20_354_000,
            },
        )

    def test_low_accuracy_confusion_record_is_checked_not_rejected_by_threshold(self) -> None:
        matrix = [[0] * 10 for _ in range(10)]
        for row in matrix:
            row[0] = 1
        record = {
            "accuracy": 0.1,
            "f1_macro": 0.018182,
            "confusion_matrix": matrix,
            "per_class_accuracy": [1.0] + [0.0] * 9,
            "per_client_test_utility": [
                {
                    "client_id": client_id,
                    "number_of_examples": 1,
                    "accuracy": 1.0 if client_id == 0 else 0.0,
                    "f1_macro": 0.0,
                }
                for client_id in range(10)
            ],
        }
        readiness._validated_confusion_matrix(
            record,
            label="negative result",
            expected_row_totals=[1] * 10,
            expected_client_counts=[1] * 10,
        )


if __name__ == "__main__":
    unittest.main()
