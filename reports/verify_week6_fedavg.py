"""Verify the Month 2 Week 6 hand-written FedAvg foundation.

This verifier uses only synthetic unit tests. It does not download MNIST or
run a thesis experiment.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEK6_MODULES = (
    PROJECT_ROOT / "algorithms" / "fedavg.py",
    PROJECT_ROOT / "clients" / "federated_client.py",
    PROJECT_ROOT / "server" / "fedavg_server.py",
)
FORBIDDEN_IMPORT_ROOTS = {
    "flwr",
    "flower",
    "fedprox",
    "federated_unlearning",
    "proposed_method",
}


def imported_roots(path: Path) -> set[str]:
    """Return top-level module names imported by one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def verify_scope() -> None:
    """Prove that Week 6 did not bypass the hand-written implementation."""
    for path in WEEK6_MODULES:
        if not path.is_file():
            raise AssertionError(f"Missing Week 6 module: {path}")
        forbidden = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            raise AssertionError(
                f"{path.relative_to(PROJECT_ROOT)} imports blocked module(s): "
                + ", ".join(sorted(forbidden))
            )

    # A framework dependency becomes legitimate in Week 7. Keep this verifier
    # useful in later months by checking the Week 6 implementation modules,
    # rather than permanently forbidding Flower elsewhere in the repository.


def run_tests() -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern="test_fedavg.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("The synthetic FedAvg unit suite failed.")
    if result.testsRun < 13:
        raise AssertionError(
            f"Expected at least 13 Week 6 tests, but only {result.testsRun} ran."
        )
    return result


def main() -> None:
    verify_scope()
    result = run_tests()
    print(
        "PASS: Week 6 hand-written FedAvg — "
        f"{result.testsRun} synthetic tests and Week 6 import-scope audit succeeded."
    )


if __name__ == "__main__":
    main()
