"""Verify the Month 2 Week 7 Flower compatibility evidence."""

from __future__ import annotations

import json
import sys
import unittest
from importlib.metadata import version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_FLOWER_VERSION = "1.30.0"


def verify_environment_record() -> None:
    manifest_path = PROJECT_ROOT / "environment" / "month2_cpu_runtime.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["packages"]["flwr"]
    installed = version("flwr")
    if recorded != EXPECTED_FLOWER_VERSION or installed != recorded:
        raise AssertionError(
            f"Flower version mismatch: expected/recorded {recorded}, installed {installed}."
        )

    active_requirement = f"flwr=={EXPECTED_FLOWER_VERSION}"
    requirements = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if active_requirement not in requirements:
        raise AssertionError(f"Missing exact active requirement: {active_requirement}")


def run_tests() -> unittest.result.TestResult:
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTests(
        loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_fedavg.py")
    )
    suite.addTests(
        loader.discover(
            str(PROJECT_ROOT / "tests"), pattern="test_flower_compat.py"
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Week 6/7 compatibility tests failed.")
    if result.testsRun < 20:
        raise AssertionError(f"Expected at least 20 tests, but {result.testsRun} ran.")
    return result


def main() -> None:
    verify_environment_record()
    result = run_tests()
    print(
        "PASS: Week 7 Flower compatibility — "
        f"version {EXPECTED_FLOWER_VERSION}, {result.testsRun} tests succeeded."
    )


if __name__ == "__main__":
    main()
