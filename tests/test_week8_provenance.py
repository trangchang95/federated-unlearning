"""Synthetic checks for Week 8 Git provenance enforcement."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.iid.month2_week8_mnist_fedavg import validate_code_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "month2_week8_mnist_iid_fedavg_vs_centralized.json"
)


class Week8ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    @patch("experiments.iid.month2_week8_mnist_fedavg._run_git")
    def test_clean_matching_revision_is_recorded(self, run_git) -> None:
        commit = "a" * 40
        run_git.side_effect = [commit, commit, "", "", "", ""]
        result = validate_code_provenance(self.config, CONFIG_PATH)
        self.assertEqual(result["resolved_commit"], commit)
        self.assertEqual(result["head_commit"], commit)
        self.assertTrue(result["worktree_clean_at_start"])
        self.assertEqual(len(result["tracked_dependencies"]), 3)

    @patch("experiments.iid.month2_week8_mnist_fedavg._run_git")
    def test_missing_configured_revision_is_rejected(self, run_git) -> None:
        run_git.side_effect = ["a" * 40, RuntimeError("unknown revision")]
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            validate_code_provenance(self.config, CONFIG_PATH)

    @patch("experiments.iid.month2_week8_mnist_fedavg._run_git")
    def test_revision_not_at_head_is_rejected(self, run_git) -> None:
        run_git.side_effect = ["a" * 40, "b" * 40]
        with self.assertRaisesRegex(RuntimeError, "does not resolve to the current HEAD"):
            validate_code_provenance(self.config, CONFIG_PATH)

    @patch("experiments.iid.month2_week8_mnist_fedavg._run_git")
    def test_dirty_worktree_is_rejected(self, run_git) -> None:
        commit = "a" * 40
        run_git.side_effect = [
            commit,
            commit,
            "",
            "",
            "",
            " M experiments/iid/month2_week8_mnist_fedavg.py",
        ]
        with self.assertRaisesRegex(RuntimeError, "worktree is not clean"):
            validate_code_provenance(self.config, CONFIG_PATH)

    def test_config_outside_repository_is_rejected(self) -> None:
        outside = PROJECT_ROOT.parent / "outside.json"
        with self.assertRaisesRegex(ValueError, "inside this repository"):
            validate_code_provenance(self.config, outside)


if __name__ == "__main__":
    unittest.main()
